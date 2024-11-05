use anyhow::Result;
use ffmpeg_next::{codec, format, media, sys, ChannelLayout};
use napi::bindgen_prelude::Float32Array;
use napi_derive::napi;

use crate::WHISPER_SAMPLE_RATE;

#[napi]
pub enum AVLogLevel {
  Quiet = -8,
  Panic = 0,
  Fatal = 8,
  Error = 16,
  Warning = 24,
  Info = 32,
  Verbose = 40,
  Debug = 48,
  Trace = 56,
}

#[napi]
pub fn split_audio_from_video(
  filepath: String,
  log_level: Option<AVLogLevel>,
) -> Result<Float32Array> {
  unsafe { sys::av_log_set_level(log_level.unwrap_or(AVLogLevel::Quiet) as i32) };
  let mut ictx = format::input(&filepath)?;
  let (stream_index, params) = {
    let stream = ictx
      .streams()
      .best(media::Type::Audio)
      .ok_or_else(|| anyhow::format_err!("No audio stream found in the video file"))?;

    (stream.index(), stream.parameters())
  };
  let context = codec::Context::from_parameters(params)?;
  let mut decoder = context.decoder().audio()?;
  let mut resampled = decoder.resampler(
    format::Sample::F32(format::sample::Type::Planar),
    ChannelLayout::MONO,
    WHISPER_SAMPLE_RATE,
  )?;
  let mut samples = Vec::new();

  for (_, packet) in ictx.packets().filter(|(s, _)| s.index() == stream_index) {
    decoder.send_packet(&packet)?;
    let mut decoded = ffmpeg_next::frame::Audio::empty();
    while decoder.receive_frame(&mut decoded).is_ok() {
      let mut resampled_audio = ffmpeg_next::frame::Audio::empty();
      resampled.run(&decoded, &mut resampled_audio)?;
      let data = resampled_audio.plane::<f32>(0);
      samples.extend_from_slice(data);
    }
  }
  decoder.send_eof()?;
  let mut decoded = ffmpeg_next::frame::Audio::empty();
  while decoder.receive_frame(&mut decoded).is_ok() {
    let mut resampled_audio = ffmpeg_next::frame::Audio::empty();
    resampled.run(&decoded, &mut resampled_audio)?;
    let data = resampled_audio.plane::<f32>(0);
    samples.extend_from_slice(data);
  }
  Ok(Float32Array::from(samples))
}
