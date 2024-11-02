use napi_derive::napi;

use crate::sys;

#[napi]
pub enum WhisperAlignmentHeadsPreset {
  None = 0,
  NTopMost = 1,
  Custom = 2,
  TinyEn = 3,
  Tiny = 4,
  BaseEn = 5,
  Base = 6,
  SmallEn = 7,
  Small = 8,
  MediumEn = 9,
  Medium = 10,
  LargeV1 = 11,
  LargeV2 = 12,
  LargeV3 = 13,
}

#[napi(object)]
pub struct WhisperContextParams {
  pub use_gpu: bool,
  pub flash_attn: bool,
  // CUDA device
  pub gpu_device: u32,
  /// [EXPERIMENTAL] Token-level timestamps with DTW
  pub dtw_token_timestamps: bool,
  pub dtw_aheads_preset: WhisperAlignmentHeadsPreset,
  pub dtw_n_top: u32,
}

impl From<WhisperContextParams> for sys::whisper_context_params {
  fn from(params: WhisperContextParams) -> Self {
    Self {
      use_gpu: params.use_gpu,
      flash_attn: params.flash_attn,
      gpu_device: params.gpu_device as i32,
      dtw_token_timestamps: params.dtw_token_timestamps,
      dtw_aheads_preset: params.dtw_aheads_preset as u32,
      dtw_n_top: params.dtw_n_top as i32,
      dtw_aheads: sys::whisper_aheads {
        n_heads: 0,
        heads: std::ptr::null(),
      },
      dtw_mem_size: 1024 * 1024 * 128,
    }
  }
}
