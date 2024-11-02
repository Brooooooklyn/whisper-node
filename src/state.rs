use napi_derive::napi;

use crate::sys;

#[napi]
pub struct WhisperState {
  pub(crate) inner: *mut sys::whisper_state,
}

#[napi]
impl WhisperState {
  #[napi(getter)]
  /// Language id associated with the provided state
  pub fn get_full_lang_id(&self) -> i32 {
    unsafe { sys::whisper_full_lang_id_from_state(self.inner) }
  }

  #[napi(getter)]
  /// mel length
  pub fn get_n_len(&self) -> i32 {
    unsafe { sys::whisper_n_len_from_state(self.inner) }
  }
}
