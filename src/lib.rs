#![deny(clippy::all)]

use std::{borrow::Cow, ffi::CString, fs::File, io::Read, ptr, sync::atomic::Ordering};

use napi::{
  bindgen_prelude::*,
  module_init,
  threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode},
};
use napi_derive::napi;

pub use audio_decode::{decode_audio, decode_audio_async};
use context_params::WhisperContextParams;
use full_params::{WhisperCallbackUserData, WhisperFullParams};
pub use state::WhisperState;
pub use video::split_audio_from_video;

mod audio_decode;
mod context_params;
mod full_params;
mod state;
mod sys;
mod video;

#[cfg(not(target_arch = "arm"))]
#[global_allocator]
static ALLOC: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub(crate) const WHISPER_SAMPLE_RATE: u32 = 16000;

#[module_init]
fn init() {
  unsafe { sys::whisper_log_set(Some(whisper_logger_callback), ptr::null_mut()) };
}

#[napi]
pub struct Whisper {
  inner: *mut sys::whisper_context,
  callback_user_data: *mut WhisperCallbackUserData,
}

impl Drop for Whisper {
  fn drop(&mut self) {
    unsafe {
      sys::whisper_free(self.inner);
    }
  }
}

#[napi]
impl Whisper {
  #[napi]
  /// Largest language id (i.e. number of available languages - 1)
  pub fn max_lang_id() -> i32 {
    unsafe { sys::whisper_lang_max_id() }
  }

  #[napi]
  /// Return the id of the specified language, returns -1 if not found
  /// Examples:
  ///   "de" -> 2
  ///   "german" -> 2
  pub fn lang_id(lang: String) -> Result<i32> {
    let c_lang = CString::new(lang)?;
    Ok(unsafe { sys::whisper_lang_id(c_lang.as_ptr().cast()) })
  }

  #[napi]
  /// Return the short string of the specified language id (e.g. 2 -> "de"), returns None if not found
  pub fn lang(id: i32) -> Option<RawCString> {
    let lang = unsafe { sys::whisper_lang_str(id) };
    if lang.is_null() {
      return None;
    }
    Some(RawCString::new(lang, NAPI_AUTO_LENGTH))
  }

  #[napi]
  /// Return the short string of the specified language name (e.g. 2 -> "german"), returns nullptr if not found
  pub fn lang_full(id: i32) -> Option<RawCString> {
    let lang = unsafe { sys::whisper_lang_str_full(id) };
    if lang.is_null() {
      return None;
    }
    Some(RawCString::new(lang, NAPI_AUTO_LENGTH))
  }

  #[napi(constructor)]
  pub fn new(model: Either<&[u8], String>, params: Option<WhisperContextParams>) -> Result<Self> {
    let model = match model {
      Either::A(buf) => Cow::Borrowed(buf),
      Either::B(filepath) => {
        let mut file = File::open(filepath)?;
        let mut buf = Vec::with_capacity(file.metadata()?.len() as usize);
        file.read_to_end(&mut buf)?;
        Cow::Owned(buf)
      }
    };
    let inner = unsafe {
      sys::whisper_init_from_buffer_with_params(
        model.as_ptr().cast_mut().cast(),
        model.len(),
        params.map(Into::into).unwrap_or_default(),
      )
    };
    if inner.is_null() {
      return Err(Error::new(
        Status::InvalidArg,
        "Failed to initialize Whisper model from buffer",
      ));
    }

    Ok(Self {
      inner,
      callback_user_data: ptr::null_mut(),
    })
  }

  #[napi(getter)]
  /// mel length
  pub fn get_n_len(&self) -> i32 {
    unsafe { sys::whisper_n_len(self.inner) }
  }

  #[napi(getter)]
  pub fn get_n_vocab(&self) -> i32 {
    unsafe { sys::whisper_n_vocab(self.inner) }
  }

  #[napi(getter)]
  pub fn get_n_text(&self) -> i32 {
    unsafe { sys::whisper_n_text_ctx(self.inner) }
  }

  #[napi(getter)]
  pub fn get_n_audio(&self) -> i32 {
    unsafe { sys::whisper_n_audio_ctx(self.inner) }
  }

  #[napi(getter)]
  pub fn get_is_multilingual(&self) -> i32 {
    unsafe { sys::whisper_is_multilingual(self.inner) }
  }

  #[napi(getter)]
  pub fn get_model_n_vocab(&self) -> i32 {
    unsafe { sys::whisper_model_n_vocab(self.inner) }
  }

  #[napi(getter)]
  pub fn get_model_n_audio_ctx(&self) -> i32 {
    unsafe { sys::whisper_model_n_audio_ctx(self.inner) }
  }

  #[napi(getter)]
  pub fn get_model_n_audio_state(&self) -> i32 {
    unsafe { sys::whisper_model_n_audio_state(self.inner) }
  }

  #[napi(getter)]
  pub fn get_model_n_audio_head(&self) -> i32 {
    unsafe { sys::whisper_model_n_audio_head(self.inner) }
  }

  #[napi(getter)]
  pub fn get_model_n_audio_layer(&self) -> i32 {
    unsafe { sys::whisper_model_n_audio_layer(self.inner) }
  }

  #[napi(getter)]
  pub fn get_model_n_text_ctx(&self) -> i32 {
    unsafe { sys::whisper_model_n_text_ctx(self.inner) }
  }

  #[napi(getter)]
  pub fn get_model_n_text_state(&self) -> i32 {
    unsafe { sys::whisper_model_n_text_state(self.inner) }
  }

  #[napi(getter)]
  pub fn get_model_n_text_head(&self) -> i32 {
    unsafe { sys::whisper_model_n_text_head(self.inner) }
  }

  #[napi(getter)]
  pub fn get_model_n_text_layer(&self) -> i32 {
    unsafe { sys::whisper_model_n_text_layer(self.inner) }
  }

  #[napi(getter)]
  pub fn get_model_n_mels(&self) -> i32 {
    unsafe { sys::whisper_model_n_mels(self.inner) }
  }

  #[napi(getter)]
  pub fn get_model_ftype(&self) -> i32 {
    unsafe { sys::whisper_model_ftype(self.inner) }
  }

  #[napi(getter)]
  pub fn get_model_type(&self) -> i32 {
    unsafe { sys::whisper_model_type(self.inner) }
  }

  #[napi(getter)]
  /// Language id associated with the context's default state
  pub fn get_full_lang_id(&self) -> i32 {
    unsafe { sys::whisper_full_lang_id(self.inner) }
  }

  #[napi(getter)]
  pub fn get_state(&self, env: &Env, mut this: This) -> Option<ClassInstance<WhisperState>> {
    const STATE_PROPERTY_KEY: &str = "_state";
    if self.callback_user_data.is_null() {
      return None;
    }
    let callback_user_data = Box::leak(unsafe { Box::from_raw(self.callback_user_data) });
    let state_ptr = callback_user_data.state.load(Ordering::Relaxed);
    if state_ptr.is_null() {
      return None;
    }
    if let Ok(state) = this.get_named_property_unchecked(STATE_PROPERTY_KEY) {
      return Some(state);
    }
    let whisper_state = WhisperState { inner: state_ptr }.into_instance(env).ok()?;
    whisper_state
      .assign_to_this_with_attributes(STATE_PROPERTY_KEY, PropertyAttributes::Default, &mut this)
      .ok()
  }

  #[napi]
  /// Return the number of tokens in the provided text
  pub fn count(&self) -> u32 {
    unsafe { sys::whisper_token_count(self.inner) as u32 }
  }

  #[napi]
  pub fn full(&mut self, parmas: &mut WhisperFullParams, samples: &[f32]) -> Result<String> {
    self.callback_user_data = parmas.callback_user_data;
    let status = unsafe {
      sys::whisper_full(
        self.inner,
        parmas.inner.clone(),
        samples.as_ptr().cast(),
        samples.len() as i32,
      )
    };
    if status != 0 {
      return Err(Error::new(
        Status::GenericFailure,
        format!("Failed to run full whisper model: {status}"),
      ));
    }
    self.callback_user_data = ptr::null_mut();
    let segments_length = unsafe { sys::whisper_full_n_segments(self.inner) };
    let mut output = String::with_capacity(1024);
    for i in 0..segments_length {
      let segment = unsafe { sys::whisper_full_get_segment_text(self.inner, i) };
      let text = unsafe { std::ffi::CStr::from_ptr(segment) };
      if let Ok(s) = text.to_str() {
        output.push_str(s);
      }
    }
    Ok(output)
  }
}

#[napi]
pub enum WhisperLogLevel {
  None = 0,
  Info = 1,
  Warn = 2,
  Error = 3,
  Debug = 4,
  Cont = 5, // continue previous log
}

impl From<sys::ggml_log_level::GgmlLogLevel> for WhisperLogLevel {
  fn from(level: sys::ggml_log_level::GgmlLogLevel) -> Self {
    match level {
      sys::ggml_log_level::GGML_LOG_LEVEL_NONE => WhisperLogLevel::None,
      sys::ggml_log_level::GGML_LOG_LEVEL_INFO => WhisperLogLevel::Info,
      sys::ggml_log_level::GGML_LOG_LEVEL_WARN => WhisperLogLevel::Warn,
      sys::ggml_log_level::GGML_LOG_LEVEL_ERROR => WhisperLogLevel::Error,
      sys::ggml_log_level::GGML_LOG_LEVEL_DEBUG => WhisperLogLevel::Debug,
      sys::ggml_log_level::GGML_LOG_LEVEL_CONT => WhisperLogLevel::Cont,
      _ => WhisperLogLevel::None,
    }
  }
}

type LoggerCallback =
  ThreadsafeFunction<(WhisperLogLevel, String), (), (WhisperLogLevel, String), false, true>;

#[napi]
pub fn setup_logger(callback: Function<(WhisperLogLevel, String), ()>) -> Result<()> {
  let logger = callback
    .build_threadsafe_function::<(WhisperLogLevel, String)>()
    .callee_handled::<false>()
    .weak::<true>()
    .build_callback(|ctx| Ok((ctx.value.0, ctx.value.1)))?;
  unsafe {
    sys::whisper_log_set(
      Some(whisper_logger_callback),
      Box::into_raw(Box::new(logger)).cast(),
    )
  };
  Ok(())
}

extern "C" fn whisper_logger_callback(
  level: sys::ggml_log_level::GgmlLogLevel,
  message: *const std::ffi::c_char,
  user_data: *mut std::ffi::c_void,
) {
  if user_data.is_null() {
    return;
  }
  let logger = unsafe { Box::from_raw(user_data.cast::<LoggerCallback>()) };
  let message = unsafe { std::ffi::CStr::from_ptr(message) };
  if let Ok(s) = message.to_str() {
    logger.call(
      (level.into(), s.trim().to_string()),
      ThreadsafeFunctionCallMode::NonBlocking,
    );
  }
}
