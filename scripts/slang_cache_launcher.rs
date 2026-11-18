//! Build-only compiler proxy. The packaged toolchain is always the native one.
use std::{env, path::PathBuf, process::{Command, exit}};
fn main() {
    let args: Vec<_> = env::args_os().skip(1).collect();
    let real = env::var_os("VORK_BUILD_REAL_SLANGC").expect("build compiler configuration");
    let output = args.windows(2).find(|p| p[0] == "-o").map(|p| PathBuf::from(&p[1]));
    let fixed = args.iter().any(|a| a == "-reflection-json")
        && env::current_dir().ok().and_then(|p| p.file_name().map(|n| n == "vork-render")).unwrap_or(false)
        && env::var_os("OUT_DIR").is_some_and(|dir| output.as_ref().is_some_and(|p| p.parent() == Some(PathBuf::from(dir).join("slang").as_path())));
    let mut command = if fixed {
        let mut command = Command::new(env::var_os("VORK_BUILD_PYTHON").expect("build Python"));
        command.arg(env::var_os("VORK_BUILD_CACHE_SCRIPT").expect("cache script"))
            .arg(env::var_os("VORK_SLANG_BUILD_CACHE").expect("build cache"))
            .arg(&real);
        command
    } else { Command::new(real) };
    let status = command.args(&args).status().expect("start shader compiler");
    exit(status.code().unwrap_or(1));
}
