import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { Whisper, WhisperFullParams, WhisperSamplingStrategy, decodeAudioAsync } from './index.js'

const rootDir = join(fileURLToPath(import.meta.url), '..')

const GGLM_LARGE = await readFile(join(rootDir, 'scripts', 'ggml-large-v3-turbo.bin'))

const audio = await readFile(join(rootDir, '__test__/rolldown.wav'))

const whisper = new Whisper(GGLM_LARGE)

const audioBuffer = await decodeAudioAsync(audio, 'rolldown.wav')

const whisperParams = new WhisperFullParams(WhisperSamplingStrategy.Greedy)
whisperParams.language = 'en'
whisperParams.singleSegment = false
whisperParams.durationMs = 0
whisperParams.onEncoderBegin = (state) => {
  console.info(Whisper.lang(state.fullLangId))
}
whisperParams.onProgress = (progress) => {
  console.info(`Progress: ${progress}`)
}
whisperParams.onNewSegment = (segment) => {
  console.info(segment)
}

const output = whisper.full(whisperParams, audioBuffer)

console.info(output)
