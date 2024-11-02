import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

import test from 'ava'

import { Whisper, WhisperFullParams, WhisperSamplingStrategy, decodeAudioAsync } from '../index.js'

const dirname = join(fileURLToPath(import.meta.url), '..')

const GGLM_LARGE = await readFile(join(dirname, '..', 'scripts', 'ggml-tiny.bin'))
const AUDIO = await readFile(join(dirname, 'rolldown.wav'))

test('New Whisper from model', async (t) => {
  const whisper = new Whisper(GGLM_LARGE)
  const params = new WhisperFullParams(WhisperSamplingStrategy.Greedy)
  const audioBuffer = await decodeAudioAsync(AUDIO)

  t.notThrows(() => whisper.full(params, audioBuffer))
})
