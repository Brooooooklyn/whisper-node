use std::{
  ffi::{c_int, c_void, CString},
  ptr,
  sync::atomic::{AtomicPtr, Ordering},
};

use napi::{
  bindgen_prelude::*,
  threadsafe_function::{ThreadsafeFunction, ThreadsafeFunctionCallMode},
};
use napi_derive::napi;

use crate::{
  sys::{
    whisper_beam_search_params, whisper_context, whisper_full_default_params, whisper_full_params,
    whisper_sampling_strategy, whisper_state,
  },
  WhisperState,
};

const ON_ENCODER_BEGIN_CB_NAME: &str = "_onEncoderBegin";
const ON_NEW_SEGMENT_CB_NAME: &str = "_onNewSegment";
const ON_PROGRESS_CB_NAME: &str = "_onProgress";
const ON_ABORT_CB_NAME: &str = "_onAbort";

#[napi]
#[derive(Debug, Clone, Copy)]
pub enum WhisperSamplingStrategy {
  Greedy = 0,
  BeamSearch = 1,
}

impl From<WhisperSamplingStrategy> for whisper_sampling_strategy::WhisperSamplingStrategy {
  fn from(strategy: WhisperSamplingStrategy) -> Self {
    match strategy {
      WhisperSamplingStrategy::Greedy => whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY,
      WhisperSamplingStrategy::BeamSearch => {
        whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH
      }
    }
  }
}

impl From<whisper_sampling_strategy::WhisperSamplingStrategy> for WhisperSamplingStrategy {
  fn from(strategy: whisper_sampling_strategy::WhisperSamplingStrategy) -> Self {
    match strategy {
      whisper_sampling_strategy::WHISPER_SAMPLING_GREEDY => WhisperSamplingStrategy::Greedy,
      whisper_sampling_strategy::WHISPER_SAMPLING_BEAM_SEARCH => {
        WhisperSamplingStrategy::BeamSearch
      }
      _ => unreachable!("Unknown sampling strategy"),
    }
  }
}

#[napi(object)]
pub struct WhisperGreedyParams {
  pub best_of: i32,
}

#[napi(object)]
pub struct WhisperBeamSearchParams {
  pub beam_size: i32,
  pub patience: f64,
}

impl From<WhisperBeamSearchParams> for whisper_beam_search_params {
  fn from(params: WhisperBeamSearchParams) -> Self {
    Self {
      beam_size: params.beam_size,
      patience: params.patience as f32,
    }
  }
}

#[napi(object)]
#[derive(Debug)]
pub struct Segment {
  pub text: String,
  pub start: u32,
  pub end: u32,
}

type OnStartCallback = ThreadsafeFunction<WhisperState, (), WhisperState, false>;
type SegmentCallback = ThreadsafeFunction<Segment, (), Segment, false>;
type ProgressCallback = ThreadsafeFunction<u32, (), u32, false>;
type AbortCallback = ThreadsafeFunction<(), (), (), false>;

pub(crate) struct WhisperCallbackUserData {
  pub(crate) encoder_begin_callback: AtomicPtr<OnStartCallback>,
  pub(crate) new_segment_callback: AtomicPtr<SegmentCallback>,
  progress_callback: AtomicPtr<ProgressCallback>,
  abort_callback: AtomicPtr<AbortCallback>,
  pub(crate) state: AtomicPtr<whisper_state>,
}

#[napi]
/// Parameters for the whisper_full() function
pub struct WhisperFullParams {
  pub(crate) inner: whisper_full_params,
  suppress_regex: String,
  language: Option<CString>,
  initial_prompt: Option<CString>,
  pub(crate) callback_user_data: *mut WhisperCallbackUserData,
}

impl Drop for WhisperFullParams {
  fn drop(&mut self) {
    unsafe {
      let WhisperCallbackUserData {
        encoder_begin_callback,
        new_segment_callback,
        progress_callback,
        abort_callback,
        ..
      } = *Box::from_raw(self.callback_user_data);
      let new_segment_callback_ptr = new_segment_callback.load(Ordering::Relaxed);
      if !new_segment_callback_ptr.is_null() {
        drop(Box::from_raw(new_segment_callback_ptr));
      }
      let encoder_begin_callback_ptr = encoder_begin_callback.load(Ordering::Relaxed);
      if !encoder_begin_callback_ptr.is_null() {
        drop(Box::from_raw(encoder_begin_callback_ptr));
      }
      let progress_callback_ptr = progress_callback.load(Ordering::Relaxed);
      if !progress_callback_ptr.is_null() {
        drop(Box::from_raw(progress_callback_ptr));
      }
      let abort_callback_ptr = abort_callback.load(Ordering::Relaxed);
      if !abort_callback_ptr.is_null() {
        drop(Box::from_raw(abort_callback_ptr));
      }
    }
  }
}

#[napi]
impl WhisperFullParams {
  #[napi(constructor)]
  pub fn new(sampling_strategy: WhisperSamplingStrategy) -> Result<Self> {
    let mut params = unsafe { whisper_full_default_params(sampling_strategy.into()) };

    params.new_segment_callback = Some(whisper_new_segment_callback);

    let callback_user_data = Box::new(WhisperCallbackUserData {
      new_segment_callback: AtomicPtr::new(ptr::null_mut()),
      encoder_begin_callback: AtomicPtr::new(ptr::null_mut()),
      progress_callback: AtomicPtr::new(ptr::null_mut()),
      abort_callback: AtomicPtr::new(ptr::null_mut()),
      state: AtomicPtr::new(ptr::null_mut()),
    });
    let callback_user_data_ptr = Box::into_raw(callback_user_data);
    params.new_segment_callback_user_data = callback_user_data_ptr.cast();
    params.encoder_begin_callback = Some(whisper_encoder_begin_callback);
    params.encoder_begin_callback_user_data = callback_user_data_ptr.cast();
    params.progress_callback = Some(whisper_progress_callback);
    params.progress_callback_user_data = callback_user_data_ptr.cast();
    params.abort_callback = Some(whisper_abort_callback);
    params.abort_callback_user_data = callback_user_data_ptr.cast();
    Ok(Self {
      inner: params,
      suppress_regex: String::new(),
      language: None,
      initial_prompt: None,
      callback_user_data: callback_user_data_ptr,
    })
  }

  #[napi(getter)]
  pub fn get_strategy(&self) -> WhisperSamplingStrategy {
    self.inner.strategy.into()
  }

  #[napi(setter)]
  pub fn set_strategy(&mut self, strategy: WhisperSamplingStrategy) {
    self.inner.strategy = strategy.into();
  }

  #[napi(getter)]
  pub fn get_n_threads(&self) -> i32 {
    self.inner.n_threads
  }

  #[napi(setter)]
  pub fn set_n_threads(&mut self, n_threads: i32) {
    self.inner.n_threads = n_threads;
  }

  #[napi(getter)]
  pub fn get_n_max_text_ctx(&self) -> i32 {
    self.inner.n_max_text_ctx
  }

  #[napi(setter)]
  pub fn set_n_max_text_ctx(&mut self, value: i32) {
    self.inner.n_max_text_ctx = value;
  }

  #[napi(getter)]
  pub fn get_offset_ms(&self) -> i32 {
    self.inner.offset_ms
  }

  #[napi(setter)]
  pub fn set_offset_ms(&mut self, value: i32) {
    self.inner.offset_ms = value;
  }

  #[napi(getter)]
  pub fn get_duration_ms(&self) -> i32 {
    self.inner.duration_ms
  }

  #[napi(setter)]
  pub fn set_duration_ms(&mut self, value: i32) {
    self.inner.duration_ms = value;
  }

  #[napi(getter)]
  pub fn get_translate(&self) -> bool {
    self.inner.translate
  }

  #[napi(setter)]
  pub fn set_translate(&mut self, value: bool) {
    self.inner.translate = value;
  }

  #[napi(getter)]
  pub fn get_no_context(&self) -> bool {
    self.inner.no_context
  }

  #[napi(setter)]
  pub fn set_no_context(&mut self, value: bool) {
    self.inner.no_context = value;
  }

  #[napi(getter)]
  pub fn get_no_timestamps(&self) -> bool {
    self.inner.no_timestamps
  }

  #[napi(setter)]
  pub fn set_no_timestamps(&mut self, value: bool) {
    self.inner.no_timestamps = value;
  }

  #[napi(getter)]
  pub fn get_single_segment(&self) -> bool {
    self.inner.single_segment
  }

  #[napi(setter)]
  pub fn set_single_segment(&mut self, value: bool) {
    self.inner.single_segment = value;
  }

  #[napi(getter)]
  pub fn get_print_special(&self) -> bool {
    self.inner.print_special
  }

  #[napi(setter)]
  pub fn set_print_special(&mut self, value: bool) {
    self.inner.print_special = value;
  }

  #[napi(getter)]
  pub fn get_print_progress(&self) -> bool {
    self.inner.print_progress
  }

  #[napi(setter)]
  pub fn set_print_progress(&mut self, value: bool) {
    self.inner.print_progress = value;
  }

  #[napi(getter)]
  pub fn get_print_realtime(&self) -> bool {
    self.inner.print_realtime
  }

  #[napi(setter)]
  pub fn set_print_realtime(&mut self, value: bool) {
    self.inner.print_realtime = value;
  }

  #[napi(getter)]
  pub fn get_print_timestamps(&self) -> bool {
    self.inner.print_timestamps
  }

  #[napi(setter)]
  pub fn set_print_timestamps(&mut self, value: bool) {
    self.inner.print_timestamps = value;
  }

  #[napi(getter)]
  pub fn get_token_timestamps(&self) -> bool {
    self.inner.token_timestamps
  }

  #[napi(setter)]
  pub fn set_token_timestamps(&mut self, value: bool) {
    self.inner.token_timestamps = value;
  }

  #[napi(getter)]
  pub fn get_thold_pt(&self) -> f64 {
    self.inner.thold_pt as f64
  }

  #[napi(setter)]
  pub fn set_thold_pt(&mut self, value: f64) {
    self.inner.thold_pt = value as f32;
  }

  #[napi(getter)]
  pub fn get_thold_ptsum(&self) -> f64 {
    self.inner.thold_ptsum as f64
  }

  #[napi(setter)]
  pub fn set_thold_ptsum(&mut self, value: f64) {
    self.inner.thold_ptsum = value as f32;
  }

  #[napi(getter)]
  pub fn get_max_len(&self) -> i32 {
    self.inner.max_len
  }

  #[napi(setter)]
  pub fn set_max_len(&mut self, value: i32) {
    self.inner.max_len = value;
  }

  #[napi(getter)]
  pub fn get_split_on_word(&self) -> bool {
    self.inner.split_on_word
  }

  #[napi(setter)]
  pub fn set_split_on_word(&mut self, value: bool) {
    self.inner.split_on_word = value;
  }

  #[napi(getter)]
  pub fn get_max_tokens(&self) -> i32 {
    self.inner.max_tokens
  }

  #[napi(setter)]
  pub fn set_max_tokens(&mut self, value: i32) {
    self.inner.max_tokens = value;
  }

  #[napi(getter)]
  pub fn get_debug_mode(&self) -> bool {
    self.inner.debug_mode
  }

  #[napi(setter)]
  pub fn set_debug_mode(&mut self, value: bool) {
    self.inner.debug_mode = value;
  }

  #[napi(getter)]
  pub fn get_audio_ctx(&self) -> i32 {
    self.inner.audio_ctx
  }

  #[napi(setter)]
  pub fn set_audio_ctx(&mut self, value: i32) {
    self.inner.audio_ctx = value;
  }

  #[napi(getter)]
  pub fn get_tdrz_enable(&self) -> bool {
    self.inner.tdrz_enable
  }

  #[napi(setter)]
  pub fn set_tdrz_enable(&mut self, value: bool) {
    self.inner.tdrz_enable = value;
  }

  #[napi(getter)]
  pub fn get_suppress_regex(&self) -> RawCString {
    RawCString::new(self.inner.suppress_regex, NAPI_AUTO_LENGTH)
  }

  #[napi(setter)]
  pub fn set_suppress_regex(&mut self, value: String) {
    self.inner.suppress_regex = value.as_ptr().cast();
    self.suppress_regex = value;
  }

  #[napi(getter)]
  pub fn get_language(&self) -> RawCString {
    RawCString::new(self.inner.language, NAPI_AUTO_LENGTH)
  }

  #[napi(setter)]
  pub fn set_language(&mut self, value: String) {
    let c_value = CString::new(value.as_str()).unwrap();
    self.inner.language = c_value.as_ptr().cast();
    self.language = Some(c_value);
  }

  #[napi(getter)]
  pub fn get_detect_language(&self) -> bool {
    self.inner.detect_language
  }

  #[napi(setter)]
  pub fn set_detect_language(&mut self, value: bool) {
    self.inner.detect_language = value;
  }

  #[napi(getter)]
  pub fn get_initial_prompt(&self) -> RawCString {
    RawCString::new(self.inner.initial_prompt, NAPI_AUTO_LENGTH)
  }

  #[napi(setter)]
  pub fn set_initial_prompt(&mut self, value: String) {
    let c_value = CString::new(value.as_str()).unwrap();
    self.inner.initial_prompt = c_value.as_ptr().cast();
    self.initial_prompt = Some(c_value);
  }

  #[napi(getter)]
  pub fn get_on_encoder_begin(&self, this: This) -> Result<Function<Segment, ()>> {
    this.get_named_property_unchecked(ON_ENCODER_BEGIN_CB_NAME)
  }

  #[napi(getter)]
  pub fn get_suppress_blank(&self) -> bool {
    self.inner.suppress_blank
  }

  #[napi(setter)]
  pub fn set_suppress_blank(&mut self, value: bool) {
    self.inner.suppress_blank = value;
  }

  #[napi(getter)]
  pub fn get_suppress_non_speech_tokens(&self) -> bool {
    self.inner.suppress_non_speech_tokens
  }

  #[napi(setter)]
  pub fn set_suppress_non_speech_tokens(&mut self, value: bool) {
    self.inner.suppress_non_speech_tokens = value;
  }

  #[napi(getter)]
  pub fn get_temperature(&self) -> f64 {
    self.inner.temperature as f64
  }

  #[napi(setter)]
  pub fn set_temperature(&mut self, value: f64) {
    self.inner.temperature = value as f32;
  }

  #[napi(getter)]
  pub fn get_max_initial_ts(&self) -> f64 {
    self.inner.max_initial_ts as f64
  }

  #[napi(setter)]
  pub fn set_max_initial_ts(&mut self, value: f64) {
    self.inner.max_initial_ts = value as f32;
  }

  #[napi(getter)]
  pub fn get_length_penalty(&self) -> f64 {
    self.inner.length_penalty as f64
  }

  #[napi(setter)]
  pub fn set_length_penalty(&mut self, value: f64) {
    self.inner.length_penalty = value as f32;
  }

  #[napi(setter, return_if_invalid)]
  pub fn set_on_encoder_begin(
    &mut self,
    mut this: This,
    callback: Function<WhisperState, ()>,
  ) -> Result<()> {
    let tsfn = callback
      .build_threadsafe_function::<WhisperState>()
      .callee_handled::<false>()
      .build()?;
    if this
      .get_named_property_unchecked::<Option<Function<WhisperState, ()>>>(ON_ENCODER_BEGIN_CB_NAME)?
      .is_some()
    {
      this.set_named_property(ON_ENCODER_BEGIN_CB_NAME, callback)?;
    } else {
      this.define_properties(&[Property::new(ON_ENCODER_BEGIN_CB_NAME)?
        .with_property_attributes(PropertyAttributes::Writable)
        .with_value(&callback)])?;
    }
    let callback_user_data = Box::leak(unsafe { Box::from_raw(self.callback_user_data) });
    let prev_tsfn_ptr = callback_user_data
      .encoder_begin_callback
      .load(Ordering::Relaxed);
    if !prev_tsfn_ptr.is_null() {
      let prev_tsfn = unsafe { Box::from_raw(prev_tsfn_ptr) };
      drop(prev_tsfn);
    }
    callback_user_data
      .encoder_begin_callback
      .store(Box::into_raw(Box::new(tsfn)), Ordering::Relaxed);
    Ok(())
  }

  #[napi(getter)]
  pub fn get_on_progress(&self, this: This) -> Result<Function<u32, ()>> {
    this.get_named_property_unchecked(ON_PROGRESS_CB_NAME)
  }

  #[napi(setter, return_if_invalid)]
  pub fn set_on_progress(&mut self, mut this: This, callback: Function<u32, ()>) -> Result<()> {
    let tsfn = callback
      .build_threadsafe_function::<u32>()
      .callee_handled::<false>()
      .build()?;
    if this
      .get_named_property_unchecked::<Option<Function<u32, ()>>>(ON_PROGRESS_CB_NAME)?
      .is_some()
    {
      this.set_named_property(ON_PROGRESS_CB_NAME, callback)?;
    } else {
      this.define_properties(&[Property::new(ON_PROGRESS_CB_NAME)?
        .with_property_attributes(PropertyAttributes::Writable)
        .with_value(&callback)])?;
    }
    let callback_user_data = Box::leak(unsafe { Box::from_raw(self.callback_user_data) });
    let prev_tsfn_ptr = callback_user_data.progress_callback.load(Ordering::Relaxed);
    if !prev_tsfn_ptr.is_null() {
      let prev_tsfn = unsafe { Box::from_raw(prev_tsfn_ptr) };
      drop(prev_tsfn);
    }
    callback_user_data
      .progress_callback
      .store(Box::into_raw(Box::new(tsfn)), Ordering::Relaxed);
    Ok(())
  }

  #[napi(getter)]
  pub fn get_on_new_segment(&self, this: This) -> Result<Function<Segment, ()>> {
    this.get_named_property_unchecked(ON_NEW_SEGMENT_CB_NAME)
  }

  #[napi(setter, return_if_invalid)]
  pub fn set_on_new_segment(
    &mut self,
    mut this: This,
    callback: Function<Segment, ()>,
  ) -> Result<()> {
    let tsfn = callback
      .build_threadsafe_function::<Segment>()
      .callee_handled::<false>()
      .build()?;
    if this
      .get_named_property_unchecked::<Option<Function<Segment, ()>>>(ON_NEW_SEGMENT_CB_NAME)?
      .is_some()
    {
      this.set_named_property(ON_NEW_SEGMENT_CB_NAME, callback)?;
    } else {
      this.define_properties(&[Property::new(ON_NEW_SEGMENT_CB_NAME)?
        .with_property_attributes(PropertyAttributes::Writable)
        .with_value(&callback)])?;
    }
    let callback_user_data = Box::leak(unsafe { Box::from_raw(self.callback_user_data) });
    let prev_tsfn_ptr = callback_user_data
      .new_segment_callback
      .load(Ordering::Relaxed);
    if !prev_tsfn_ptr.is_null() {
      let prev_tsfn = unsafe { Box::from_raw(prev_tsfn_ptr) };
      drop(prev_tsfn);
    }
    callback_user_data
      .new_segment_callback
      .store(Box::into_raw(Box::new(tsfn)), Ordering::Relaxed);
    Ok(())
  }

  #[napi(getter)]
  pub fn get_on_abort(&self, this: This) -> Result<Function<(), ()>> {
    this.get_named_property_unchecked(ON_ABORT_CB_NAME)
  }

  #[napi(setter, return_if_invalid)]
  pub fn set_on_abort(&mut self, mut this: This, callback: Function<(), ()>) -> Result<()> {
    let tsfn = callback
      .build_threadsafe_function::<()>()
      .callee_handled::<false>()
      .build()?;
    if this
      .get_named_property_unchecked::<Option<Function<(), ()>>>(ON_ABORT_CB_NAME)?
      .is_some()
    {
      this.set_named_property(ON_ABORT_CB_NAME, callback)?;
    } else {
      this.define_properties(&[Property::new(ON_ABORT_CB_NAME)?
        .with_property_attributes(PropertyAttributes::Writable)
        .with_value(&callback)])?;
    }
    let callback_user_data = Box::leak(unsafe { Box::from_raw(self.callback_user_data) });
    let prev_tsfn_ptr = callback_user_data.abort_callback.load(Ordering::Relaxed);
    if !prev_tsfn_ptr.is_null() {
      let prev_tsfn = unsafe { Box::from_raw(prev_tsfn_ptr) };
      drop(prev_tsfn);
    }
    callback_user_data
      .abort_callback
      .store(Box::into_raw(Box::new(tsfn)), Ordering::Relaxed);
    Ok(())
  }
}

extern "C" fn whisper_encoder_begin_callback(
  _: *mut whisper_context,
  state: *mut whisper_state,
  user_data: *mut c_void,
) -> bool {
  if user_data.is_null() {
    return true;
  }
  let callback_user_data: &mut WhisperCallbackUserData =
    Box::leak(unsafe { Box::from_raw(user_data.cast()) });
  let js_callback_ptr = callback_user_data
    .encoder_begin_callback
    .load(Ordering::Relaxed);
  callback_user_data.state.store(state, Ordering::Relaxed);
  if js_callback_ptr.is_null() {
    return true;
  }
  let js_callback = Box::leak(unsafe { Box::from_raw(js_callback_ptr) });
  js_callback.call(
    WhisperState { inner: state },
    ThreadsafeFunctionCallMode::NonBlocking,
  );
  true
}

extern "C" fn whisper_progress_callback(
  _ctx: *mut whisper_context,
  state: *mut whisper_state,
  progress: c_int,
  user_data: *mut c_void,
) {
  if user_data.is_null() {
    return;
  }
  let callback_user_data: &mut WhisperCallbackUserData =
    Box::leak(unsafe { Box::from_raw(user_data.cast()) });
  callback_user_data.state.store(state, Ordering::Relaxed);
  let js_callback_ptr = callback_user_data.progress_callback.load(Ordering::Relaxed);
  if !js_callback_ptr.is_null() {
    let js_callback = Box::leak(unsafe { Box::from_raw(js_callback_ptr) });
    js_callback.call(progress as _, ThreadsafeFunctionCallMode::NonBlocking);
  }
}

extern "C" fn whisper_new_segment_callback(
  ctx: *mut whisper_context,
  state: *mut whisper_state,
  n_new: c_int,
  user_data: *mut c_void,
) {
  use crate::sys;

  if user_data.is_null() {
    return;
  }
  let callback_user_data: &mut WhisperCallbackUserData =
    Box::leak(unsafe { Box::from_raw(user_data.cast()) });
  let js_callback_ptr = callback_user_data
    .new_segment_callback
    .load(Ordering::Relaxed);
  callback_user_data.state.store(state, Ordering::Relaxed);
  if js_callback_ptr.is_null() {
    return;
  }

  let js_callback = Box::leak(unsafe { Box::from_raw(js_callback_ptr) });

  let n_segments = unsafe { sys::whisper_full_n_segments(ctx) };
  let s0 = n_segments - n_new;

  // Process each new segment with more careful text handling
  for i in s0..n_segments {
    let t0 = unsafe { sys::whisper_full_get_segment_t0(ctx, i) };
    let t1 = unsafe { sys::whisper_full_get_segment_t1(ctx, i) };

    // Get the segment text
    let text_ptr = unsafe { sys::whisper_full_get_segment_text(ctx, i) };
    if text_ptr.is_null() {
      continue;
    }

    // Convert text and clean it up
    if let Ok(text) = unsafe { std::ffi::CStr::from_ptr(text_ptr).to_str() } {
      let text = text.trim();
      js_callback.call(
        Segment {
          text: text.to_string(),
          start: t0 as u32,
          end: t1 as u32,
        },
        ThreadsafeFunctionCallMode::NonBlocking,
      );
    }
  }
}

extern "C" fn whisper_abort_callback(user_data: *mut c_void) -> bool {
  if user_data.is_null() {
    return false;
  }

  let callback_user_data =
    Box::leak(unsafe { Box::from_raw(user_data.cast::<WhisperCallbackUserData>()) });
  let js_callback_ptr = callback_user_data.abort_callback.load(Ordering::Relaxed);
  if js_callback_ptr.is_null() {
    return false;
  }
  let js_callback = Box::leak(unsafe { Box::from_raw(js_callback_ptr) });
  js_callback.call((), ThreadsafeFunctionCallMode::NonBlocking);
  false
}
