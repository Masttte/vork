//! Opt-in wall-clock stage totals for the export path (`MODA_RENDER_PROFILE=1`): the perf-export bench
//! (scripts/bench/perf-export) reads them to split an export into evaluation, scene build, planning, GPU work,
//! readback, pixel-graph work and the encoder's back-pressure.
//!
//! Off by default: [`span`] checks flags but reads no clock and allocates nothing. Bench totals are process-global and
//! inclusive (a span inside another counts in both); names say where they nest (`frame.draw`, `frame.draw.plan`).
//!
//! `MODA_RENDER_PROFILE=gpu` also times every GPU pass the export thread records (`gpu_profile`'s timestamps) and adds
//! each pass's GPU time to `gpu:<stage>`, the innermost span open when the pass was recorded ([`current`]), plus
//! `gpu.busy` for all of them. A frame's timestamps are resolved at its end ([`crate::gpu_profile::resolve_into_prof`]),
//! which waits for the GPU: wall times in this mode lose the CPU/GPU overlap, the `gpu:` totals don't.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Mutex;
use std::time::Instant;

static STATE: AtomicU8 = AtomicU8::new(0); // 0 unknown, 1 off, 2 on
static TOTALS: Mutex<Vec<(&'static str, u128, u64)>> = Mutex::new(Vec::new());

/// Whether profiling is on (`MODA_RENDER_PROFILE` set to anything but `0` or empty).
#[inline]
pub fn on() -> bool {
    match STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let on = std::env::var("MODA_RENDER_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0");
            STATE.store(if on { 2 } else { 1 }, Ordering::Relaxed);
            on
        }
    }
}

/// Whether the profile is `MODA_RENDER_PROFILE=gpu` (the module docs).
pub fn gpu() -> bool {
    on() && std::env::var("MODA_RENDER_PROFILE").is_ok_and(|v| v == "gpu")
}

thread_local! {
    // The spans open on this thread, outermost first: what [`current`] names a GPU pass after.
    static OPEN: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

/// The innermost span open on this thread (`"(none)"` outside every span, or with profiling off).
pub fn current() -> &'static str {
    OPEN.with(|open| open.borrow().last().copied().unwrap_or("(none)"))
}

/// A timed stage: its wall time is added to `name`'s total when it drops.
pub struct Span {
    name: &'static str,
    start: Option<Instant>,
    // Its place in this thread's open spans; dropping it closes it and any span opened after it.
    depth: usize,
    global: bool,
}

/// Time a stage until the returned guard drops.
#[inline]
pub fn span(name: &'static str) -> Span {
    let global = on();
    let capture = CAPTURE.with(|c| c.borrow().is_some());
    if !global && !capture {
        return Span {
            name,
            start: None,
            depth: 0,
            global,
        };
    }
    let depth = OPEN.with(|open| {
        let mut open = open.borrow_mut();
        open.push(name);
        CHILDREN.with(|c| c.borrow_mut().push(0.));
        open.len() - 1
    });
    Span {
        name,
        start: Some(Instant::now()),
        depth,
        global,
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some(s) = self.start {
            let ns = s.elapsed().as_nanos();
            if self.global {
                add(self.name, ns);
            }
            CAPTURE.with(|c| {
                if let Some(capture) = c.borrow_mut().as_mut() {
                    let totals = &mut capture.totals;
                    let ms = ns as f64 / 1e6;
                    *totals.entry(self.name).or_default() += ms;
                    let child =
                        CHILDREN.with(|c| c.borrow().get(self.depth).copied().unwrap_or(0.));
                    let category = OPEN
                        .with(|open| {
                            open.borrow()
                                .iter()
                                .rev()
                                .find_map(|name| (capture.classify)(name))
                        })
                        .unwrap_or("$other_host");
                    *totals.entry(category).or_default() += (ms - child).max(0.);
                }
            });
            CHILDREN.with(|c| {
                let mut c = c.borrow_mut();
                c.truncate(self.depth);
                if let Some(parent) = c.last_mut() {
                    *parent += ns as f64 / 1e6;
                }
            });
            OPEN.with(|open| open.borrow_mut().truncate(self.depth));
        }
    }
}

/// Add `ns` to `name`'s total (one more call).
pub fn add(name: &'static str, ns: u128) {
    add_work(name, ns, 1);
}

pub fn add_work(name: &'static str, ns: u128, calls: u64) {
    let mut t = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    match t.iter_mut().find(|(n, _, _)| *n == name) {
        Some(e) => {
            e.1 += ns;
            e.2 += calls;
        }
        None => t.push((name, ns, calls)),
    }
}

/// The totals so far as one JSON object: `{ "<stage>": { "ms": …, "calls": … }, … }`, in first-seen order.
pub fn report_json() -> String {
    let t = TOTALS.lock().unwrap_or_else(|e| e.into_inner());
    let parts: Vec<String> = t
        .iter()
        .map(|(n, ns, c)| format!("\"{n}\":{{\"ms\":{:.1},\"calls\":{c}}}", *ns as f64 / 1e6))
        .collect();
    format!("{{{}}}", parts.join(","))
}

/// With `MODA_RENDER_PROFILE=sync`, wait for the GPU to finish what was submitted, so the enclosing span holds its
/// GPU time rather than leaving it to the next readback's wait. A no-op otherwise.
pub fn sync(device: &wgpu::Device) {
    if on() && !std::thread::panicking() && std::env::var("MODA_RENDER_PROFILE").is_ok_and(|v| v == "sync") {
        crate::scratch::flush();
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    }
}

/// [`sync`] when dropped: declared after a [`span`], it drops first, so the span holds the GPU time of what it
/// submitted (with `MODA_RENDER_PROFILE=sync`; a no-op otherwise).
pub struct SyncOnDrop<'a>(pub &'a wgpu::Device);

impl Drop for SyncOnDrop<'_> {
    fn drop(&mut self) {
        sync(self.0);
    }
}

/// Request-local host receipts. Independent of the process-global bench profiler;
/// no log, lock or clock reads when disabled. The guard restores a surrounding capture.
pub type HostTimings = std::collections::BTreeMap<&'static str, f64>;
thread_local! {
    static CAPTURE: RefCell<Option<Captured>> = const { RefCell::new(None) };
}
struct Captured {
    totals: HostTimings,
    classify: fn(&str) -> Option<&'static str>,
}
pub struct Capture(Option<Captured>);
pub fn capture(classify: fn(&str) -> Option<&'static str>) -> Capture {
    Capture(CAPTURE.with(|c| {
        c.replace(Some(Captured {
            totals: HostTimings::new(),
            classify,
        }))
    }))
}
pub fn take_capture() -> HostTimings {
    CAPTURE.with(|c| {
        c.borrow_mut()
            .as_mut()
            .map(|c| std::mem::take(&mut c.totals))
            .unwrap_or_default()
    })
}
impl Drop for Capture {
    fn drop(&mut self) {
        CAPTURE.with(|c| {
            c.replace(self.0.take());
        });
    }
}

pub fn capturing() -> bool {
    CAPTURE.with(|c| c.borrow().is_some())
}
pub fn capture_add(name: &'static str, ms: f64) {
    CAPTURE.with(|c| {
        if let Some(t) = c.borrow_mut().as_mut() {
            *t.totals.entry(name).or_default() += ms;
        }
    });
}

thread_local! { static CHILDREN: RefCell<Vec<f64>> = const { RefCell::new(Vec::new()) }; }

#[cfg(test)]
mod capture_tests {
    use super::*;
    #[test]
    fn request_capture_is_local_resets_and_partitions_nested_spans() {
        let _capture = capture(|name| {
            Some(match name {
                "canvas.evaluation" => "$evaluation",
                "canvas.input_decode" => "$input_decode",
                "canvas.plan" => "$planning",
                "canvas.plan.coverage" => "$coverage_planning",
                _ => return None,
            })
        });
        {
            let _evaluation = span("canvas.evaluation");
            let _decode = span("canvas.input_decode");
        }
        let first = take_capture();
        let inclusive = first["canvas.evaluation"];
        let exclusive = first["$evaluation"] + first["$input_decode"];
        assert!((inclusive - exclusive).abs() < 1e-6);
        assert!(take_capture().is_empty());
        std::thread::spawn(|| {
            assert!(!capturing());
            assert!(take_capture().is_empty());
        })
        .join()
        .unwrap();
        {
            let _plan = span("canvas.plan");
            let _coverage = span("canvas.plan.coverage");
        }
        let second = take_capture();
        assert!(
            (second["canvas.plan"] - second["$planning"] - second["$coverage_planning"]).abs()
                < 1e-6
        );
        assert!(!second.contains_key("canvas.evaluation"));
    }
}
