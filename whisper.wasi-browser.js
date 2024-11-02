import {
  instantiateNapiModuleSync as __emnapiInstantiateNapiModuleSync,
  getDefaultContext as __emnapiGetDefaultContext,
  WASI as __WASI,
  createOnMessage as __wasmCreateOnMessageForFsProxy,
} from '@napi-rs/wasm-runtime'

import __wasmUrl from './whisper.wasm32-wasi.wasm?url'

const __wasi = new __WASI({
  version: 'preview1',
})

const __emnapiContext = __emnapiGetDefaultContext()

const __sharedMemory = new WebAssembly.Memory({
  initial: 4000,
  maximum: 65536,
  shared: true,
})

const __wasmFile = await fetch(__wasmUrl).then((res) => res.arrayBuffer())

const {
  instance: __napiInstance,
  module: __wasiModule,
  napiModule: __napiModule,
} = __emnapiInstantiateNapiModuleSync(__wasmFile, {
  context: __emnapiContext,
  asyncWorkPoolSize: 4,
  wasi: __wasi,
  onCreateWorker() {
    const worker = new Worker(new URL('./wasi-worker-browser.mjs', import.meta.url), {
      type: 'module',
    })

    return worker
  },
  overwriteImports(importObject) {
    importObject.env = {
      ...importObject.env,
      ...importObject.napi,
      ...importObject.emnapi,
      memory: __sharedMemory,
    }
    return importObject
  },
  beforeInit({ instance }) {
    __napi_rs_initialize_modules(instance)
  },
})

function __napi_rs_initialize_modules(__napiInstance) {
  __napiInstance.exports['__napi_register__decode_audio_0']?.()
  __napiInstance.exports['__napi_register__DecodeAudioTask_impl_1']?.()
  __napiInstance.exports['__napi_register__decode_audio_async_2']?.()
  __napiInstance.exports['__napi_register__WhisperAlignmentHeadsPreset_3']?.()
  __napiInstance.exports['__napi_register__WhisperContextParams_struct_4']?.()
  __napiInstance.exports['__napi_register__WhisperSamplingStrategy_5']?.()
  __napiInstance.exports['__napi_register__WhisperGreedyParams_struct_6']?.()
  __napiInstance.exports['__napi_register__WhisperBeamSearchParams_struct_7']?.()
  __napiInstance.exports['__napi_register__Segment_struct_8']?.()
  __napiInstance.exports['__napi_register__WhisperFullParams_struct_9']?.()
  __napiInstance.exports['__napi_register__WhisperFullParams_impl_67']?.()
  __napiInstance.exports['__napi_register__WhisperState_struct_68']?.()
  __napiInstance.exports['__napi_register__WhisperState_impl_71']?.()
  __napiInstance.exports['__napi_register__Whisper_struct_72']?.()
  __napiInstance.exports['__napi_register__Whisper_impl_99']?.()
  __napiInstance.exports['__napi_register__WhisperLogLevel_100']?.()
  __napiInstance.exports['__napi_register__setup_logger_101']?.()
}
export const Whisper = __napiModule.exports.Whisper
export const WhisperFullParams = __napiModule.exports.WhisperFullParams
export const WhisperState = __napiModule.exports.WhisperState
export const decodeAudio = __napiModule.exports.decodeAudio
export const decodeAudioAsync = __napiModule.exports.decodeAudioAsync
export const setupLogger = __napiModule.exports.setupLogger
export const WhisperAlignmentHeadsPreset = __napiModule.exports.WhisperAlignmentHeadsPreset
export const WhisperLogLevel = __napiModule.exports.WhisperLogLevel
export const WhisperSamplingStrategy = __napiModule.exports.WhisperSamplingStrategy
