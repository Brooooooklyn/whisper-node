// Thanks to https://github.com/tazz4843/whisper-rs/blob/master/sys/build.rs

use std::path;

fn main() {
  let target = std::env::var("TARGET").unwrap();
  let whisper_dir = path::Path::new(env!("CARGO_MANIFEST_DIR")).join("whisper.cpp");

  let mut cmake_config = cmake::Config::new(whisper_dir);
  cmake_config
    .profile("Release")
    .define("BUILD_SHARED_LIBS", "OFF")
    .define("WHISPER_ALL_WARNINGS", "OFF")
    .define("WHISPER_ALL_WARNINGS_3RD_PARTY", "OFF")
    .define("WHISPER_BUILD_TESTS", "OFF")
    .define("WHISPER_BUILD_EXAMPLES", "OFF")
    .build_arg("-Wno-dev")
    .very_verbose(true);
  if target.contains("apple") {
    // Enable coreml on arm64 macOS
    if target.contains("aarch64") {
      println!("cargo:rustc-link-lib=framework=Accelerate");
      println!("cargo:rustc-link-lib=framework=Foundation");
      println!("cargo:rustc-link-lib=framework=CoreML");
      println!("cargo:rustc-link-lib=static=whisper.coreml");
    }
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalKit");

    cmake_config
      .define("GGML_METAL", "ON")
      .define("GGML_METAL_NDEBUG", "ON")
      .define("GGML_METAL_EMBED_LIBRARY", "ON");
  }

  // #[cfg(feature = "rocm")]
  {
    use std::env;
    use std::path::PathBuf;

    if target.contains("linux") {
      println!("cargo:rerun-if-env-changed=HIP_PATH");

      let hip_path = match env::var("HIP_PATH") {
        Ok(path) => PathBuf::from(path),
        Err(_) => PathBuf::from("/opt/rocm"),
      };
      let hip_lib_path = hip_path.join("lib");
      println!("cargo:rustc-link-search={}", hip_lib_path.display());
      println!("cargo:rustc-link-lib=hipblas");
      println!("cargo:rustc-link-lib=rocblas");
      println!("cargo:rustc-link-lib=amdhip64");
    }

    cmake_config
      .define("GGML_HIPBLAS", "ON")
      .define("GGML_OPENMP", "OFF")
      .define("CMAKE_C_COMPILER", "hipcc")
      .define("CMAKE_CXX_COMPILER", "hipcc")
      .cflag("--emit-static-lib")
      .cxxflag("--emit-static-lib");
    println!("cargo:rerun-if-env-changed=AMDGPU_TARGETS");
    cmake_config.define(
      "AMDGPU_TARGETS",
      env::var("AMDGPU_TARGETS").unwrap_or("gfx908".to_string()),
    );

    println!("cargo:rerun-if-env-changed=HIP_PATH");

    let hip_path = match env::var("HIP_PATH") {
      Ok(path) => PathBuf::from(path),
      Err(_) => PathBuf::from("/opt/rocm"),
    };
    let hip_lib_path = hip_path.join("lib");
    println!("cargo:rustc-link-search={}", hip_lib_path.display());
    println!("cargo:rustc-link-lib=amdhip64");
    println!("cargo:rustc-link-lib=hipblas");
  }

  let output_path = cmake_config.build();

  println!("cargo:rustc-link-search={}/lib", output_path.display());
  println!("cargo:rustc-link-lib=static=whisper");
  println!("cargo:rustc-link-lib=static=ggml");

  napi_build::setup();
}
