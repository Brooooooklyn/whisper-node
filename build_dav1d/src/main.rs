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

  Command::new("meson")
    .current_dir(&dav1d_build_dir)
    .args(&[
      "setup",
      "..",
      "--default-library=static",
      "--buildtype=release",
      "-Denable_tools=false",
      "-Denable_tests=false",
      "-Denable_asm=true",
    ])
    .status()?;
  Command::new("ninja")
    .current_dir(&dav1d_build_dir)
    .status()?;
  if cfg!(target_os = "linux") {
    Command::new("sudo")
      .current_dir(&dav1d_build_dir)
      .args(["meson", "install"])
      .status()?;
  } else {
    Command::new("meson")
      .current_dir(&dav1d_build_dir)
      .arg("install")
      .status()?;
  }
  Ok(())
}
