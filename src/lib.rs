#![deny(clippy::all)]

use napi_derive::napi;

mod sys;

#[napi]
pub struct Whisper {
  inner: *mut sys::whisper_context,
}

#[napi]
impl Whisper {
  #[napi(constructor)]
  pub fn new(model: &[u8]) -> Self {
    Self {
      inner: unsafe {
        sys::whisper_init_from_buffer_with_params(
          model.as_ptr().cast_mut().cast(),
          model.len(),
          Default::default(),
        )
      },
    }
  }
}
