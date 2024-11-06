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
  pub use_gpu: Option<bool>,
  pub flash_attn: Option<bool>,
  // CUDA device
  pub gpu_device: Option<u32>,
  /// [EXPERIMENTAL] Token-level timestamps with DTW
  pub dtw_token_timestamps: Option<bool>,
  pub dtw_aheads_preset: Option<WhisperAlignmentHeadsPreset>,
  pub dtw_n_top: Option<i32>,
}

impl From<WhisperContextParams> for sys::whisper_context_params {
  fn from(params: WhisperContextParams) -> Self {
    Self {
      use_gpu: params.use_gpu.unwrap_or(true),
      flash_attn: params.flash_attn.unwrap_or(false),
      gpu_device: params.gpu_device.unwrap_or(0) as i32,
      dtw_token_timestamps: params.dtw_token_timestamps.unwrap_or(false),
      dtw_aheads_preset: params
        .dtw_aheads_preset
        .unwrap_or(WhisperAlignmentHeadsPreset::None) as u32,
      dtw_n_top: params.dtw_n_top.unwrap_or(-1),
      dtw_aheads: sys::whisper_aheads {
        n_heads: 0,
        heads: std::ptr::null(),
      },
      dtw_mem_size: 1024 * 1024 * 128,
    }
  }
}
