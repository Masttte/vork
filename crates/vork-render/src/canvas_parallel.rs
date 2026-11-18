//! Memory admission for optional canvas replicas.
use std::sync::Arc;

/// A qualification-only total-memory ceiling. Native budgets remain authoritative;
/// this can only make admission more conservative, including on a larger test GPU.
fn ceiling() -> Option<u64> {
    std::env::var("VORK_CANVAS_MEMORY_BUDGET").ok()?.parse().ok()
}
pub(crate) fn headroom(device: &wgpu::Device) -> Option<(u64, u64)> {
    let (used, mut free) = crate::residency::camera_usage_headroom(device)?;
    if let Some(cap) = ceiling() { free = free.min(cap.saturating_sub(used)); }
    #[cfg(target_os = "macos")]
    { free = free.min(unified_free()?); }
    Some((used, free))
}
#[cfg(target_os = "macos")]
fn unified_free() -> Option<u64> {
    // XNU's free_count already includes speculative pages (osfmk/kern/host.c).
    // Purgeable pages can be discarded without application paging. Inactive
    // application pages and swap are not capacity for replication.
    unsafe {
        // One process-lifetime send right; repeated probes must not leak ports.
        static HOST: std::sync::OnceLock<libc::mach_port_t> = std::sync::OnceLock::new();
        let host = *HOST.get_or_init(|| libc::mach_host_self());
        let mut stats: libc::vm_statistics64 = std::mem::zeroed();
        let mut count = libc::HOST_VM_INFO64_COUNT;
        let status = libc::host_statistics64(host, libc::HOST_VM_INFO64,
            &mut stats as *mut _ as *mut libc::integer_t, &mut count);
        let page_size = libc::sysconf(libc::_SC_PAGESIZE);
        if status != libc::KERN_SUCCESS || page_size <= 0 { return None; }
        (u64::from(stats.free_count) + u64::from(stats.purgeable_count)).checked_mul(page_size as u64)
    }
}
/// Unified-memory CPU histories must be admitted before constructing them.
/// A whole-process resident set deliberately overestimates one worker history;
/// decoded assets, fonts, code and installation are actually shared.
pub(crate) fn host_working_set() -> Option<u64> {
    #[cfg(target_os = "macos")]
    unsafe {
        let mut info: libc::mach_task_basic_info = std::mem::zeroed();
        let mut count = libc::MACH_TASK_BASIC_INFO_COUNT;
        if libc::task_info(libc::mach_task_self(), libc::MACH_TASK_BASIC_INFO,
            &mut info as *mut _ as *mut libc::integer_t, &mut count) != libc::KERN_SUCCESS { return None; }
        return Some(info.resident_size);
    }
    #[cfg(not(target_os = "macos"))]
    None
}
/// Leave native command buffers, staging and other applications room to grow.
/// The mandatory serial target/ring are never charged to optional capacity.
pub(crate) fn budget(device: &wgpu::Device, discrete: bool) -> Option<Arc<wgpu::optional_allocations::Budget>> {
    let (initial_used, free) = headroom(device)?;
    #[cfg(target_os = "macos")]
    let initial_rss = host_working_set()?;
    // HAL Performance allocation sizes permit a 256 MiB device block and a
    // 128 MiB host block (vendor/wgpu-hal/src/lib.rs::AllocationSizes). Retain
    // room for both plus command/descriptor allocation, even at tight headroom.
    let reserve = (free / 8).max(512 * 1024 * 1024);
    let device = device.clone();
    let available = move || {
        let (used, current_free) = headroom(&device)?;
        let fixed = free.saturating_sub(used.saturating_sub(initial_used));
        // host_statistics is rate-limited by XNU. Do not regain credit from a
        // cached free-page report while our own CPU/GPU working set grows.
        #[cfg(target_os = "macos")]
        let fixed = fixed.saturating_sub(host_working_set()?.saturating_sub(initial_rss));
        Some(current_free.min(fixed).saturating_sub(reserve))
    };
    Some(if discrete {
        wgpu::optional_allocations::Budget::new_batched(free.saturating_sub(reserve), available)
    } else {
        // Unified memory can change during CPU evaluation/decode within a frame.
        wgpu::optional_allocations::Budget::new(free.saturating_sub(reserve), available)
    })
}
/// Conservatively reserve the output ring before native encoder open. The
/// max-buffer limit remains a separate single-allocation check, never capacity.
pub(crate) fn ring_limit(free: Option<u64>, bytes_per_slot: u64, serial: usize, wanted: usize) -> usize {
    let Some(free) = free else { return serial; };
    let optional = free.saturating_sub(free / 8) / bytes_per_slot.max(1);
    wanted.min(usize::try_from(optional).unwrap_or(usize::MAX)).max(serial)
}

/// A short export can have fewer slots than twice its worker count even on
/// main. Only reduce that count when memory actually shortened the ring.
pub(crate) fn replicas_for_ring(wanted: usize, admitted: usize, full: usize) -> usize {
    if admitted == full { wanted } else { wanted.min(admitted / 2).max(1) }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn memory_does_not_change_the_count_for_short_exports() {
        assert_eq!(replicas_for_ring(8, 9, 9), 8);
        assert_eq!(replicas_for_ring(3, 3, 3), 3);
        assert_eq!(replicas_for_ring(8, 8, 16), 4);
        assert_eq!(replicas_for_ring(8, 3, 16), 1);
    }
    #[test]
    fn unknown_and_tight_budgets_keep_the_serial_ring() {
        assert_eq!(ring_limit(None, 1024, 3, 16), 3);
        assert_eq!(ring_limit(Some(1024), 1024, 3, 16), 3);
        assert_eq!(ring_limit(Some(8192), 1024, 3, 16), 7);
        assert_eq!(ring_limit(Some(u64::MAX), u64::MAX, 3, 16), 3);
    }
}

// Compile the excluded dependency's allocation contracts in the renderer tests.
#[cfg(test)]
#[path = "../../../vendor/wgpu/src/optional_allocations.rs"]
mod allocation_contract;
