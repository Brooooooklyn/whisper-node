#![allow(dead_code)]
#![allow(non_camel_case_types)]
use std::ffi::{c_int, c_void};

pub const GGML_FILE_MAGIC: u32 = 1734831468;
pub const GGML_FILE_VERSION: u32 = 1;
pub const GGML_QNT_VERSION: u32 = 2;
pub const GGML_QNT_VERSION_FACTOR: u32 = 1000;
pub const GGML_MAX_DIMS: u32 = 4;
pub const GGML_MAX_PARAMS: u32 = 2048;
pub const GGML_MAX_CONTEXTS: u32 = 64;
pub const GGML_MAX_SRC: u32 = 10;
pub const GGML_MAX_NAME: u32 = 64;
pub const GGML_MAX_OP_PARAMS: u32 = 64;
pub const GGML_DEFAULT_N_THREADS: u32 = 4;
pub const GGML_DEFAULT_GRAPH_SIZE: u32 = 2048;
pub const GGML_MEM_ALIGN: u32 = 16;
pub const GGML_EXIT_SUCCESS: u32 = 0;
pub const GGML_EXIT_ABORTED: u32 = 1;
pub const GGUF_MAGIC: &[u8; 5] = b"GGUF\0";
pub const GGUF_VERSION: u32 = 3;
pub const GGUF_DEFAULT_ALIGNMENT: u32 = 32;
pub const GGML_KQ_MASK_PAD: u32 = 32;
pub const GGML_N_TASKS_MAX: i32 = -1;
pub const WHISPER_SAMPLE_RATE: u32 = 16000;
pub const WHISPER_N_FFT: u32 = 400;
pub const WHISPER_N_FFT_HALF: u32 = 201;
pub const WHISPER_HOP_LENGTH: u32 = 160;
pub const WHISPER_CHUNK_SIZE: u32 = 30;
pub const WHISPER_N_SAMPLES: u32 = 480000;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_context {
  _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_state {
  _unused: [u8; 0],
}
pub type whisper_pos = i32;
pub type whisper_token = i32;
pub type whisper_seq_id = i32;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_ahead {
  pub n_text_layer: c_int,
  pub n_head: c_int,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_aheads {
  pub n_heads: usize,
  pub heads: *const whisper_ahead,
}

pub mod whisper_alignment_heads_preset {
  use std::ffi::c_uint;

  pub type WhisperAlignmentHeadsPreset = c_uint;

  pub const WHISPER_AHEADS_NONE: c_uint = 0;
  pub const WHISPER_AHEADS_N_TOP_MOST: c_uint = 1;
  pub const WHISPER_AHEADS_CUSTOM: c_uint = 2;
  pub const WHISPER_AHEADS_TINY_EN: c_uint = 3;
  pub const WHISPER_AHEADS_TINY: c_uint = 4;
  pub const WHISPER_AHEADS_BASE_EN: c_uint = 5;
  pub const WHISPER_AHEADS_BASE: c_uint = 6;
  pub const WHISPER_AHEADS_SMALL_EN: c_uint = 7;
  pub const WHISPER_AHEADS_SMALL: c_uint = 8;
  pub const WHISPER_AHEADS_MEDIUM_EN: c_uint = 9;
  pub const WHISPER_AHEADS_MEDIUM: c_uint = 10;
  pub const WHISPER_AHEADS_LARGE_V1: c_uint = 11;
  pub const WHISPER_AHEADS_LARGE_V2: c_uint = 12;
  pub const WHISPER_AHEADS_LARGE_V3: c_uint = 13;
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_context_params {
  pub use_gpu: bool,
  pub flash_attn: bool,
  pub gpu_device: ::std::os::raw::c_int,
  pub dtw_token_timestamps: bool,
  pub dtw_aheads_preset: whisper_alignment_heads_preset::WhisperAlignmentHeadsPreset,
  pub dtw_n_top: ::std::os::raw::c_int,
  pub dtw_aheads: whisper_aheads,
  pub dtw_mem_size: usize,
}

impl Default for whisper_context_params {
  fn default() -> Self {
    unsafe { whisper_context_default_params() }
  }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_token_data {
  pub id: whisper_token,
  pub tid: whisper_token,
  pub p: f32,
  pub plog: f32,
  pub pt: f32,
  pub ptsum: f32,
  pub t0: i64,
  pub t1: i64,
  pub t_dtw: i64,
  pub vlen: f32,
}

#[link(name = "whisper", kind = "static")]
extern "C" {
  pub fn whisper_context_default_params() -> whisper_context_params;

  pub fn whisper_init_from_buffer_with_params(
    buffer: *const c_void,
    buffer_size: usize,
    params: whisper_context_params,
  ) -> *mut whisper_context;
}
