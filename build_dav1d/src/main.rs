use std::fs;
use std::path;
use std::process::Command;

fn main() -> std::io::Result<()> {
  let dav1d_dir = path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .parent()
    .unwrap()
    .join("dav1d");
  let dav1d_build_dir = dav1d_dir.join("build");

  if fs::exists(&dav1d_build_dir)? {
    fs::remove_dir_all(&dav1d_build_dir)?;
  }

  fs::create_dir_all(&dav1d_build_dir)?;

  let meson_args = "-Denable_asm=true";

  // macOS x86_64 on CI doesn't support AVX
  #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
  let meson_args = "-Denable_asm=false";

  Command::new("meson")
    .current_dir(&dav1d_build_dir)
    .args(&[
      "setup",
      "..",
      "--default-library=static",
      "--buildtype=release",
      "-Denable_tools=false",
      "-Denable_tests=false",
      meson_args,
    ])
    .status()?;
  Command::new("ninja")
    .current_dir(&dav1d_build_dir)
    .status()?;
  Command::new("meson")
    .current_dir(&dav1d_build_dir)
    .arg("install")
    .status()?;
  Ok(())
}
