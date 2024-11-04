#!/usr/bin/env node

const fs = require('node:fs')
const path = require('node:path')
const { Readable } = require('node:stream')
const { finished } = require('node:stream/promises')

let src = 'https://huggingface.co/ggerganov/whisper.cpp'
let pfx = 'resolve/main/ggml'

const BOLD = '\x1b[1m'
const RESET = '\x1b[0m'

// get the path of this script
function getScriptPath() {
  return path.dirname(process.argv[1])
}

const modelsPath = process.argv[3] || getScriptPath()

// Whisper models
const models = `tiny
tiny.en
tiny-q5_1
tiny.en-q5_1
base
base.en
base-q5_1
base.en-q5_1
small
small.en
small.en-tdrz
small-q5_1
small.en-q5_1
medium
medium.en
medium-q5_0
medium.en-q5_0
large-v1
large-v2
large-v2-q5_0
large-v3
large-v3-q5_0
large-v3-turbo
large-v3-turbo-q5_0`

// list available models
function listModels() {
  console.log('\nAvailable models:')
  let modelClass = ''
  for (const model of models.split('\n')) {
    const thisModelClass = model.split(/[.-]/)[0]
    if (thisModelClass !== modelClass) {
      process.stdout.write('\n ')
      modelClass = thisModelClass
    }
    process.stdout.write(` ${model}`)
  }
  console.log('\n')
}

if (process.argv.length < 3 || process.argv.length > 4) {
  console.log(`Usage: ${process.argv[1]} <model> [models_path]`)
  listModels()
  console.log('___________________________________________________________')
  console.log(
    `${BOLD}.en${RESET} = english-only ${BOLD}-q5_[01]${RESET} = quantized ${BOLD}-tdrz${RESET} = tinydiarize`,
  )
  process.exit(1)
}

const model = process.argv[2]

if (!models.split('\n').includes(model)) {
  console.log(`Invalid model: ${model}`)
  listModels()
  process.exit(1)
}

// check if model contains `tdrz` and update the src and pfx accordingly
if (model.includes('tdrz')) {
  src = 'https://huggingface.co/akashmjn/tinydiarize-whisper.cpp'
  pfx = 'resolve/main/ggml'
}

// download ggml model
console.log(`Downloading ggml model ${model} from '${src}' ...`)

try {
  process.chdir(modelsPath)
} catch (err) {
  console.error(`Failed to change directory: ${err}`)
  process.exit(1)
}

const outputFile = `ggml-${model}.bin`

if (fs.existsSync(outputFile)) {
  console.log(`Model ${model} already exists. Skipping download.`)
  process.exit(0)
}

const file = fs.createWriteStream(outputFile)
const url = `${src}/${pfx}-${model}.bin`

fetch(url, { method: 'GET', redirect: 'follow' })
  .then((response) => {
    if (response.status !== 200) {
      fs.unlink(outputFile, () => {
        console.log(`Failed to download ggml model ${model}`)
        console.log('Please try again later or download the original Whisper model files and convert them yourself.')
        process.exit(1)
      })
      return
    }

    return finished(Readable.fromWeb(response.body).pipe(file))
  })
  .then(() => {
    console.log(`Done! Model '${model}' saved in '${modelsPath}/ggml-${model}.bin'`)
    console.log('You can now use it like this:\n')
    console.log(`const whisper = new Whisper('${modelsPath}/ggml-${model}.bin')`)
  })
  .catch((err) => {
    fs.unlink(outputFile, () => {
      console.log(`Failed to download ggml model ${model}`)
      console.error(err)
      console.log('Please try again later or download the original Whisper model files and convert them yourself.')
      process.exit(1)
    })
  })
