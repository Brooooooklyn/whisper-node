#![allow(dead_code)]
#![allow(non_camel_case_types)]
use std::{
  ffi::{c_char, c_float, c_int, c_long, c_uint, c_void},
  os::raw::c_longlong,
};

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

pub mod whisper_sampling_strategy {
  use std::ffi::c_uint;

  pub type WhisperSamplingStrategy = c_uint;

  pub const WHISPER_SAMPLING_GREEDY: c_uint = 0;
  pub const WHISPER_SAMPLING_BEAM_SEARCH: c_uint = 1;
}

pub mod ggml_log_level {
  use std::ffi::c_int;

  pub type GgmlLogLevel = c_int;

  pub const GGML_LOG_LEVEL_NONE: c_int = 0;
  pub const GGML_LOG_LEVEL_INFO: c_int = 1;
  pub const GGML_LOG_LEVEL_WARN: c_int = 2;
  pub const GGML_LOG_LEVEL_ERROR: c_int = 3;
  pub const GGML_LOG_LEVEL_DEBUG: c_int = 4;
  pub const GGML_LOG_LEVEL_CONT: c_int = 5; // continue previous log
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
  pub p: c_float,
  pub plog: c_float,
  pub pt: c_float,
  pub ptsum: c_float,
  pub t0: c_long,
  pub t1: c_long,
  pub t_dtw: c_long,
  pub vlen: c_float,
}

// Type definition for ggml abort callback
pub type ggml_abort_callback = Option<extern "C" fn(user_data: *mut c_void) -> bool>;
pub type ggml_log_callback = Option<
  extern "C" fn(
    level: ggml_log_level::GgmlLogLevel,
    message: *const c_char,
    user_data: *mut c_void,
  ),
>;

// Grammar element type enum
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq)]
pub enum WhisperGrammarElementType {
  End = 0,          // end of rule definition
  Alt = 1,          // start of alternate definition for rule
  RuleRef = 2,      // non-terminal element: reference to rule
  Char = 3,         // terminal element: character (code point)
  CharNot = 4,      // inverse char(s) ([^a], [^a-b] [^abc])
  CharRngUpper = 5, // modifies preceding Char to be inclusive range
  CharAlt = 6,      // modifies preceding Char to add alternate char
}

// Grammar element struct
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct whisper_grammar_element {
  pub type_: WhisperGrammarElementType,
  pub value: c_uint, // Unicode code point or rule ID
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct whisper_full_params {
  pub strategy: whisper_sampling_strategy::WhisperSamplingStrategy,
  pub n_threads: c_int,
  pub n_max_text_ctx: c_int,
  pub offset_ms: c_int,
  pub duration_ms: c_int,

  pub translate: bool,
  pub no_context: bool,
  pub no_timestamps: bool,
  pub single_segment: bool,
  pub print_special: bool,
  pub print_progress: bool,
  pub print_realtime: bool,
  pub print_timestamps: bool,

  // Token-level timestamps
  pub token_timestamps: bool,
  pub thold_pt: c_float,
  pub thold_ptsum: c_float,
  pub max_len: c_int,
  pub split_on_word: bool,
  pub max_tokens: c_int,

  // Speed-up techniques
  pub debug_mode: bool,
  pub audio_ctx: c_int,

  // Speaker detection
  pub tdrz_enable: bool,

  // Regex pattern for token suppression
  pub suppress_regex: *const c_char,

  // Initial prompts
  pub initial_prompt: *const c_char,
  pub prompt_tokens: *const whisper_token,
  pub prompt_n_tokens: c_int,

  // Language
  pub language: *const c_char,
  pub detect_language: bool,

  // Decoding parameters
  pub suppress_blank: bool,
  pub suppress_non_speech_tokens: bool,
  pub temperature: c_float,
  pub max_initial_ts: c_float,
  pub length_penalty: c_float,

  // Fallback parameters
  pub temperature_inc: c_float,
  pub entropy_thold: c_float,
  pub logprob_thold: c_float,
  pub no_speech_thold: c_float,

  // Greedy decoding parameters
  pub greedy: WhisperGreedyParams,

  // Beam search parameters
  pub beam_search: whisper_beam_search_params,

  // Callbacks
  pub new_segment_callback: whisper_new_segment_callback,
  pub new_segment_callback_user_data: *mut c_void,
  pub progress_callback: whisper_progress_callback,
  pub progress_callback_user_data: *mut c_void,
  pub encoder_begin_callback: whisper_encoder_begin_callback,
  pub encoder_begin_callback_user_data: *mut c_void,
  pub abort_callback: ggml_abort_callback,
  pub abort_callback_user_data: *mut c_void,
  pub logits_filter_callback: whisper_logits_filter_callback,
  pub logits_filter_callback_user_data: *mut c_void,

  // Grammar
  pub grammar_rules: *const *const whisper_grammar_element,
  pub n_grammar_rules: usize,
  pub i_start_rule: usize,
  pub grammar_penalty: c_float,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct WhisperGreedyParams {
  pub best_of: c_int,
}

#[repr(C)]
#[derive(Debug, Clone)]
pub struct whisper_beam_search_params {
  pub beam_size: c_int,
  pub patience: c_float,
}

// Function callback types
pub type whisper_new_segment_callback = Option<
  extern "C" fn(
    ctx: *mut whisper_context,
    state: *mut whisper_state,
    n_new: c_int,
    user_data: *mut c_void,
  ),
>;

pub type whisper_progress_callback = Option<
  extern "C" fn(
    ctx: *mut whisper_context,
    state: *mut whisper_state,
    progress: c_int,
    user_data: *mut c_void,
  ),
>;

pub type whisper_encoder_begin_callback = Option<
  extern "C" fn(
    ctx: *mut whisper_context,
    state: *mut whisper_state,
    user_data: *mut c_void,
  ) -> bool,
>;

pub type whisper_logits_filter_callback = Option<
  extern "C" fn(
    ctx: *mut whisper_context,
    state: *mut whisper_state,
    tokens: *const whisper_token_data,
    n_tokens: c_int,
    logits: *mut c_float,
    user_data: *mut c_void,
  ),
>;

#[link(name = "whisper", kind = "static")]
extern "C" {
  pub fn whisper_lang_max_id() -> c_int;
  pub fn whisper_lang_id(lang: *const c_char) -> c_int;
  pub fn whisper_lang_str(id: c_int) -> *const c_char;
  pub fn whisper_lang_str_full(id: c_int) -> *const c_char;

  pub fn whisper_context_default_params() -> whisper_context_params;
  pub fn whisper_full_default_params(
    strategy: whisper_sampling_strategy::WhisperSamplingStrategy,
  ) -> whisper_full_params;
  pub fn whisper_full_default_params_by_ref(
    strategy: whisper_sampling_strategy::WhisperSamplingStrategy,
  ) -> *mut whisper_full_params;
  pub fn whisper_init_from_buffer_with_params(
    buffer: *const c_void,
    buffer_size: usize,
    params: whisper_context_params,
  ) -> *mut whisper_context;
  // mel length
  pub fn whisper_n_len(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_n_len_from_state(state: *mut whisper_state) -> c_int;
  pub fn whisper_n_vocab(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_n_text_ctx(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_n_audio_ctx(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_is_multilingual(ctx: *mut whisper_context) -> c_int;

  pub fn whisper_model_n_vocab(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_model_n_audio_ctx(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_model_n_audio_state(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_model_n_audio_head(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_model_n_audio_layer(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_model_n_text_ctx(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_model_n_text_state(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_model_n_text_head(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_model_n_text_layer(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_model_n_mels(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_model_ftype(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_model_type(ctx: *mut whisper_context) -> c_int;

  pub fn whisper_full_lang_id(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_full_lang_id_from_state(state: *mut whisper_state) -> c_int;
  pub fn whisper_token_count(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_full_n_segments(ctx: *mut whisper_context) -> c_int;
  pub fn whisper_full_n_segments_from_state(state: *mut whisper_state) -> c_int;
  pub fn whisper_full_get_segment_t0(ctx: *mut whisper_context, i_segment: c_int) -> c_longlong;
  pub fn whisper_full_get_segment_t1(ctx: *mut whisper_context, i_segment: c_int) -> c_longlong;
  pub fn whisper_full_get_segment_text(
    ctx: *mut whisper_context,
    i_segment: c_int,
  ) -> *const c_char;
  #[must_use]
  pub fn whisper_full(
    ctx: *mut whisper_context,
    params: whisper_full_params,
    samples: *const c_float,
    n_samples: c_int,
  ) -> c_int;
  pub fn whisper_full_with_state(
    ctx: *mut whisper_context,
    state: *mut whisper_state,
    params: whisper_full_params,
    samples: *const c_float,
    n_samples: c_int,
  ) -> c_int;
  pub fn whisper_init_state(ctx: *mut whisper_context) -> *mut whisper_state;

  pub fn whisper_free(ctx: *mut whisper_context);
  pub fn whisper_free_params(params: *mut whisper_full_params);
  pub fn whisper_free_state(state: *mut whisper_state);

  pub fn whisper_log_set(log_callback: ggml_log_callback, user_data: *mut c_void);
}
