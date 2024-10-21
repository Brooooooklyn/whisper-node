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
    .define("GGML_OPENMP", "OFF")
    .build_arg("-Wno-dev")
    .very_verbose(true)
    .pic(true);
  if target.contains("apple") {
    println!("cargo:rustc-link-lib=framework=CoreML");
    cmake_config
      .define("GGML_ACCELERATE", "1")
      .define("WHISPER_COREML", "ON")
      .define("WHISPER_COREML_ALLOW_FALLBACK", "ON");
    cmake_config
      .define("GGML_METAL", "ON")
      .define("GGML_METAL_NDEBUG", "ON")
      .define("GGML_METAL_EMBED_LIBRARY", "ON")
      .define("GGML_HIPBLAS", "OFF");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=MetalKit");
    println!("cargo:rustc-link-lib=static=whisper.coreml");
    println!("cargo:rustc-link-lib=c++");
  }

  #[cfg(feature = "rocm")]
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
  if target.contains("apple") {
    // libwhisper.coreml.a is in build/src
    println!(
      "cargo:rustc-link-search={}/build/src",
      output_path.display()
    );
  }
  println!("cargo:rustc-link-lib=static=ggml");
  println!("cargo:rustc-link-lib=static=whisper");

  napi_build::setup();
}
