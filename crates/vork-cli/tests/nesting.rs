//! Exercise process-start stack policy, rather than supplying a large test thread.
use std::{
    io::Write,
    process::{Command, Stdio},
};

fn scene(groups: usize) -> String {
    format!(
        r##"{{"version":2,"composition":{{"space":"2d","viewBox":[0,0,16,16]}},"nodes":[{}{{"type":"path","d":"M0 0H8V8Z","fill":"#fff"}}{}]}}"##,
        r#"{"children":["#.repeat(groups),
        "]}".repeat(groups),
    )
}

fn check(document: &str, stack: Option<(u64, u64)>, code: i32) {
    let mut command = Command::new(env!("CARGO_BIN_EXE_vork"));
    command
        .args(["check", "-"])
        .env("VORK_GPU", "not-a-device")
        .env(
            "VK_ICD_FILENAMES",
            "/missing/nesting-must-not-open-vulkan.json",
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    if let Some((soft, hard)) = stack {
        use std::os::unix::process::CommandExt;
        // SAFETY: the child only changes its own limit through an async-signal-safe
        // syscall before exec; it does not allocate or touch the parent's limit.
        unsafe {
            command.pre_exec(move || {
                let limit = libc::rlimit {
                    rlim_cur: soft as _,
                    rlim_max: hard as _,
                };
                if libc::setrlimit(libc::RLIMIT_STACK, &limit) != 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    assert!(stack.is_none());
    let mut child = command.spawn().unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(document.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let error = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(code), "{error}");
    if code == 3 {
        assert!(error.contains("more than 2127 deep"), "{error}");
    }
}

#[test]
fn deep_scene_and_refusal_with_inherited_stack() {
    check(&scene(1061), None, 0);
    check(&scene(1063), None, 3);
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn deep_scene_and_refusal_with_low_soft_or_hard_stack() {
    // Every case must use the independent command stack. Neither a low soft
    // limit nor an unraiseable hard cap may constrain scene reads or refusals.
    let mut inherited = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: the pointer is to this frame's initialized limit.
    assert_eq!(
        unsafe { libc::getrlimit(libc::RLIMIT_STACK, &mut inherited) },
        0
    );
    for (soft, hard) in [
        (1 << 20, inherited.rlim_max as u64),
        (1 << 20, 1 << 20),
        (128 << 10, 128 << 10),
    ] {
        check(&scene(1061), Some((soft, hard)), 0);
        check(&scene(1063), Some((soft, hard)), 3);
    }
}
