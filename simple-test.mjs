import { readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { Whisper, WhisperFullParams, WhisperSamplingStrategy, decodeAudioAsync, splitAudioFromVideo } from './index.js'

const rootDir = join(fileURLToPath(import.meta.url), '..')

const GGLM_LARGE = await readFile(join(rootDir, 'scripts', 'ggml-large-v3-turbo.bin'))

const audio = await readFile(join(rootDir, '__test__/rolldown.wav'))

const whisper = new Whisper(GGLM_LARGE, {
  useGpu: true,
  gpuDevice: 1
})

const audioBuffer = await decodeAudioAsync(audio, 'rolldown.wav')

const whisperParams = new WhisperFullParams(WhisperSamplingStrategy.Greedy)

whisperParams.debugMode = true
whisperParams.onNewSegment = (segment) => {
  console.info(segment)
}

const output = whisper.full(whisperParams, audioBuffer)

console.info(output)

const audioBufferFromVideo = splitAudioFromVideo(join(rootDir, 'ai.mp4'))
const whisperParamsReact = new WhisperFullParams(WhisperSamplingStrategy.Greedy)

const reactVideoOutput = whisper.full(whisperParamsReact, audioBufferFromVideo)
await writeFile('react.txt', reactVideoOutput)
