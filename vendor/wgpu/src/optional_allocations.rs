//! Native extension for disposable render work. A denied reservation unwinds only
//! the optional scope, before invoking the backend. The owner must discard that
//! scope's mutable state and retry on its mandatory path. Ordinary errors and
//! panics retain their normal behavior. No device error or invalid handle is made.
extern crate alloc;
use alloc::{boxed::Box, sync::Arc};
use core::sync::atomic::{AtomicU64, Ordering};
use std::{cell::RefCell, panic::{catch_unwind, resume_unwind, AssertUnwindSafe}};

/// A capacity refusal, caught only by the enclosing optional scope.
#[derive(Debug)]
pub struct Refused;
impl core::fmt::Display for Refused {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result { f.write_str("optional GPU allocation exceeds available memory") }
}
impl core::error::Error for Refused {}

/// Shared reservation for all optional consumers of one device.
pub struct Budget {
    limit: u64,
    used: AtomicU64,
    available: Box<dyn Fn() -> Option<u64> + Send + Sync>,
    headroom: AtomicU64,
    spent: AtomicU64,
    host_reserved: AtomicU64,
    volatile: bool,
}
impl core::fmt::Debug for Budget {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Budget").field("limit", &self.limit).field("used", &self.used).finish()
    }
}
impl Budget {
    /// `limit` covers optional resources; `available` reports current device and
    /// unified-memory headroom, excluding a reserve for native/staging allocations.
    pub fn new(limit: u64, available: impl Fn() -> Option<u64> + Send + Sync + 'static) -> Arc<Self> {
        Self::make(limit, available, true)
    }
    /// On a discrete GPU, reserve a native-headroom epoch and debit every
    /// allocation against it. The owner refreshes before each frame. Released
    /// handles do not mint new headroom until the driver confirms it at refresh.
    pub fn new_batched(limit: u64, available: impl Fn() -> Option<u64> + Send + Sync + 'static) -> Arc<Self> {
        Self::make(limit, available, false)
    }
    fn make(limit: u64, available: impl Fn() -> Option<u64> + Send + Sync + 'static, volatile: bool) -> Arc<Self> {
        Arc::new(Self { limit, used: AtomicU64::new(0), available: Box::new(available),
            headroom: AtomicU64::new(0), spent: AtomicU64::new(0), host_reserved: AtomicU64::new(0), volatile })
    }
    /// Obtain fresh native headroom at a frame boundary. Allocation admission is
    /// held through backend creation, so a refresh cannot credit uncommitted GPU
    /// requests from another worker. CPU reservations remain debited explicitly.
    pub fn refresh(&self) -> Result<(), Refused> {
        let _gate = ALLOCATING.lock().unwrap_or_else(|poison| poison.into_inner());
        self.refresh_locked()
    }
    fn refresh_locked(&self) -> Result<(), Refused> {
        let Some(free) = (self.available)() else {
            self.headroom.store(0, Ordering::Release);
            self.spent.store(0, Ordering::Release);
            return Err(Refused);
        };
        let free = free.saturating_sub(self.host_reserved.load(Ordering::Acquire));
        self.headroom.store(free, Ordering::Release);
        self.spent.store(0, Ordering::Release);
        if free == 0 { Err(Refused) } else { Ok(()) }
    }
    /// Reserved bytes retained by live resources (cloned handles count once).
    pub fn used(&self) -> u64 { self.used.load(Ordering::Acquire) }
    /// Reserve non-resource capacity (for example unified-memory worker state).
    /// Keep the returned ownership token until that state has been dropped.
    pub fn reserve_capacity(self: &Arc<Self>, bytes: u64) -> Result<Arc<Lease>, Refused> {
        let _gate = ALLOCATING.lock().unwrap_or_else(|poison| poison.into_inner());
        let mut lease = self.reserve(bytes)?;
        self.host_reserved.fetch_add(lease.bytes, Ordering::AcqRel);
        lease.host = true;
        Ok(Arc::new(lease))
    }
    /// Run disposable work. On refusal all scope-local resources are dropped.
    /// Keep recoverable job inputs outside the closure. Mutex-protected shared
    /// publication must be transactional and tolerate poisoning on cancellation.
    pub fn run<T>(self: &Arc<Self>, work: impl FnOnce() -> T) -> Result<T, Refused> {
        if cfg!(panic = "abort") { return Err(Refused); }
        self.refresh()?;
        struct Restore(Option<Arc<Budget>>);
        impl Drop for Restore {
            fn drop(&mut self) { ACTIVE.with(|active| *active.borrow_mut() = self.0.take()); }
        }
        let _restore = Restore(ACTIVE.with(|active| active.replace(Some(self.clone()))));
        match catch_unwind(AssertUnwindSafe(work)) {
            Ok(value) => Ok(value),
            Err(error) if error.is::<Refused>() => Err(Refused),
            Err(error) => resume_unwind(error),
        }
    }
    fn reserve(self: &Arc<Self>, bytes: u64) -> Result<Lease, Refused> {
        // Conservative allocation granularity includes image tiling and small
        // dedicated allocations. Checked arithmetic refuses oversized requests.
        let bytes = bytes.checked_add(65535).ok_or(Refused)? / 65536 * 65536;
        if self.volatile { self.refresh_locked()?; }
        let fits_epoch = || self.spent.load(Ordering::Acquire).checked_add(bytes)
            .is_some_and(|spent| spent <= self.headroom.load(Ordering::Acquire));
        if !fits_epoch() {
            self.refresh_locked()?;
            if !fits_epoch() { return Err(Refused); }
        }
        self.used.fetch_update(Ordering::AcqRel, Ordering::Acquire, |used|
            used.checked_add(bytes).filter(|&next| next <= self.limit)).map_err(|_| Refused)?;
        self.spent.fetch_add(bytes, Ordering::AcqRel);
        Ok(Lease { budget: self.clone(), bytes, host: false })
    }
}
std::thread_local! {
    static ACTIVE: RefCell<Option<Arc<Budget>>> = const { RefCell::new(None) };
}
#[derive(Debug)]
/// Ownership of reserved optional capacity. Clones share one charge.
pub struct Lease { budget: Arc<Budget>, bytes: u64, host: bool }
impl Drop for Lease {
    fn drop(&mut self) {
        if self.host { self.budget.host_reserved.fetch_sub(self.bytes, Ordering::AcqRel); }
        self.budget.used.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
// Hold this through the backend allocation, so two threads cannot both spend
// the same native headroom while either allocation has not reached the driver.
static ALLOCATING: std::sync::Mutex<()> = std::sync::Mutex::new(());
pub(crate) struct Admission {
    pub(crate) lease: Option<Arc<Lease>>,
    _gate: Option<std::sync::MutexGuard<'static, ()>>,
}
pub(crate) fn admit(bytes: u64) -> Admission {
    let Some(budget) = ACTIVE.with(|active| active.borrow().clone()) else {
        return Admission { lease: None, _gate: None };
    };
    let gate = ALLOCATING.lock().unwrap_or_else(|poison| poison.into_inner());
    match budget.reserve(bytes) {
        Ok(lease) => Admission { lease: Some(Arc::new(lease)), _gate: Some(gate) },
        Err(error) => resume_unwind(Box::new(error)),
    }
}
#[cfg(test)]
fn reserve(bytes: u64) -> Option<Arc<Lease>> { admit(bytes).lease }
pub(crate) fn texture_bytes(desc: &wgpu_types::TextureDescriptor<Option<&str>, &[wgpu_types::TextureFormat]>) -> u64 {
    let (bw, bh) = desc.format.block_dimensions();
    let block = desc.format.block_copy_size(None).unwrap_or(8) as u64;
    (0..desc.mip_level_count).try_fold(0u64, |bytes, level| {
        let size = desc.mip_level_size(level)?;
        let level = u64::from(size.width.div_ceil(bw)).checked_mul(u64::from(size.height.div_ceil(bh)))?
            .checked_mul(u64::from(size.depth_or_array_layers))?.checked_mul(block)?
            .checked_mul(u64::from(desc.sample_count))?;
        bytes.checked_add(level)
    }).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn retained_clones_and_cancelled_growth() {
        let budget = Budget::new(131072, || Some(u64::MAX));
        let a = budget.run(|| reserve(1).unwrap()).unwrap();
        let b = a.clone();
        assert_eq!(budget.used(), 65536);
        drop(a);
        assert_eq!(budget.used(), 65536);
        assert!(budget.run(|| { let _temporary = reserve(1); reserve(1) }).is_err());
        assert_eq!(budget.used(), 65536);
        drop(b);
        assert_eq!(budget.used(), 0);
        assert!(reserve(u64::MAX).is_none(), "mandatory work stays outside the scope");
    }
    #[test]
    fn overflow_and_unknown_headroom_refuse_before_allocation() {
        let budget = Budget::new(u64::MAX, || Some(u64::MAX));
        assert!(budget.run(|| reserve(u64::MAX)).is_err());
        let unknown = Budget::new(u64::MAX, || None);
        assert!(unknown.run(|| reserve(1)).is_err());
        assert_eq!(unknown.used(), 0);
    }
    #[test]
    fn changing_native_pressure_cancels_and_restores_the_mandatory_path() {
        let free = Arc::new(AtomicU64::new(131072));
        let source = free.clone();
        let budget = Budget::new(131072, move || Some(source.load(Ordering::Acquire)));
        let retained = budget.run(|| reserve(1).unwrap()).unwrap();
        free.store(0, Ordering::Release);
        let job = std::cell::Cell::new(7);
        assert!(budget.run(|| { job.set(8); reserve(1) }).is_err());
        assert_eq!(job.get(), 7, "entry refusal leaves the recoverable job untouched");
        assert!(reserve(1).is_none(), "the mandatory retry has no optional admission scope");
        assert_eq!(budget.used(), 65536);
        drop(retained);
        assert_eq!(budget.used(), 0);
    }
    #[test]
    fn dropped_handles_do_not_reuse_unconfirmed_native_capacity() {
        let free = Arc::new(AtomicU64::new(65536));
        let source = free.clone();
        let budget = Budget::new_batched(65536, move || Some(source.load(Ordering::Acquire)));
        assert!(budget.run(|| {
            drop(reserve(1));
            free.store(0, Ordering::Release); // a native reference still owns those bytes
            reserve(1)
        }).is_err());
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn pending_host_state_and_gpu_resources_share_one_limit() {
        let budget = Budget::new_batched(262144, || Some(262144));
        let host = budget.reserve_capacity(1).unwrap();
        let gpu = budget.run(|| (reserve(131072), reserve(65536))).unwrap();
        assert_eq!(budget.used(), 262144);
        drop(host);
        assert_eq!(budget.used(), 196608);
        drop(gpu);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn ordinary_panics_are_not_capacity_refusals() {
        let budget = Budget::new(65536, || Some(65536));
        let result = catch_unwind(AssertUnwindSafe(|| budget.run(|| panic!("programmer error"))));
        assert!(result.is_err());
        assert!(reserve(1).is_none());
    }

    #[test]
    fn concurrent_reservations_do_not_overcommit() {
        let budget = Budget::new(65536, || Some(u64::MAX));
        let barrier = Arc::new(std::sync::Barrier::new(8));
        let leases: std::vec::Vec<_> = std::thread::scope(|scope| {
            let jobs: std::vec::Vec<_> = (0..8).map(|_| {
                let budget = budget.clone(); let barrier = barrier.clone();
                scope.spawn(move || { barrier.wait(); budget.run(|| reserve(1)) })
            }).collect();
            jobs.into_iter().map(|job| job.join().unwrap()).collect()
        });
        assert_eq!(leases.iter().filter(|v| v.is_ok()).count(), 1);
        assert_eq!(budget.used(), 65536);
        drop(leases);
        assert_eq!(budget.used(), 0);
    }
}
