# `@napi-rs/whisper`

![https://github.com/Brooooooklyn/whisper-node/actions](https://github.com/Brooooooklyn/whisper-node/workflows/CI/badge.svg)

## Usage

> [!IMPORTANT]
> This package is working in progress, and only support macOS now.
> Download the whisper model before use it.

### Download Whisper Model

```bash
./scripts/download-ggml-model.sh large-v3-turbo
```

### Speech to Text

```js
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { Whisper, WhisperFullParams, WhisperSamplingStrategy, decodeAudioAsync } from './index.js'

const rootDir = join(fileURLToPath(import.meta.url), '..')

const GGLM_LARGE = await readFile(join(rootDir, 'ggml-large-v3-turbo.bin'))

const audio = await readFile(join(rootDir, '__test__/rolldown.wav'))

const whisper = new Whisper(GGLM_LARGE)

const audioBuffer = await decodeAudioAsync(audio, 'rolldown.wav')

const whisperParams = new WhisperFullParams(WhisperSamplingStrategy.Greedy)
whisperParams.language = 'en'
whisperParams.printProgress = true
whisperParams.singleSegment = false
whisperParams.durationMs = 0
whisperParams.printRealtime = true
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
// Rolldown is a JavaScript/TypeScript bundler written in Rust intended to serve as the future bundler used in Vite.
```

### decode audio

> [!TIP]
> This package provide a convenient function to decode various audio format to PCM buffer.

There are many audio formats supports, full list can be found in [Symphonia homepage](https://github.com/pdeljanov/Symphonia?tab=readme-ov-file#status)

```js
import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { decodeAudioAsync } from './index.js'

const rootDir = join(fileURLToPath(import.meta.url), '..')

const audio = await readFile(join(rootDir, '__test__/rolldown.wav'))

// there is also a sync version: `decodeAudio`
const audioBuffer = await decodeAudioAsync(audio, 'rolldown.wav')
```
