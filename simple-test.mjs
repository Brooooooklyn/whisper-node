import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { Whisper } from './index.js'

const GGLM_LARGE = await readFile(join(fileURLToPath(import.meta.url), '..', 'whisper.cpp', 'models', 'for-tests-ggml-base.en.bin'))

const whisper = new Whisper(GGLM_LARGE)