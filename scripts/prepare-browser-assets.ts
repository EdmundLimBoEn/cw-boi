import { copyFile, mkdir, readFile, stat } from 'node:fs/promises'
import { createHash } from 'node:crypto'
import { createReadStream } from 'node:fs'
import { resolve } from 'node:path'

const release = '8344b39dba22e595ea8170bd4ffdb514aaef1c4b67b7de3ef965b5f66a081374'
const modelRoot = resolve('public/models/cwformer-v6')
const ortRoot = resolve('public/ort')
await mkdir(ortRoot, { recursive: true })
for (const name of ['ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) {
  await copyFile(resolve('node_modules/onnxruntime-web/dist', name), resolve(ortRoot, name))
}

if (!process.argv.includes('--wasm-only')) {
  let manifest
  try { manifest = JSON.parse(await readFile(resolve(modelRoot, 'manifest.json'), 'utf8')) }
  catch { throw new Error('Browser model assets are missing. Run scripts/prepare-browser-model.py with the released CWformer v6 weights before building.') }
  if (manifest.modelSha256 !== release || !Array.isArray(manifest.parts) || !manifest.parts.length) throw new Error('Expected the verified CWformer v6 release manifest.')
  const modelHash = createHash('sha256')
  let total = 0
  for (const part of manifest.parts) {
    const path = resolve(modelRoot, part.file)
    if (!path.startsWith(`${modelRoot}/`) && !path.startsWith(`${modelRoot}\\`)) throw new Error('Invalid model asset path.')
    const size = (await stat(path)).size
    if (size !== part.bytes || size > 25 * 1024 * 1024) throw new Error(`Invalid model part: ${part.file}`)
    const partHash = createHash('sha256')
    for await (const data of createReadStream(path)) { modelHash.update(data); partHash.update(data) }
    if (partHash.digest('hex') !== part.sha256) throw new Error(`Model part failed integrity check: ${part.file}`)
    total += size
  }
  if (modelHash.digest('hex') !== release || total !== manifest.modelBytes) throw new Error('Browser model differs from the verified release.')
  for (const [kind, bytes, sha256] of [
    ['window', 400 * 4, 'c692d19f0b2bd745f0dfc967261c3e3d21a2473e485a677d219e9b4a73e0218e'],
    ['basis', 40 * 201 * 4, '3605837643860610e8505c4818f58f8644df705b3ce61e3f2148cbd78e93db5e'],
  ] as const) {
    const name = `mel_${kind}.f32`
    const asset = manifest[kind]
    const data = await readFile(resolve(modelRoot, name))
    if (asset?.file !== name || asset.bytes !== bytes || asset.sha256 !== sha256 || data.length !== bytes || createHash('sha256').update(data).digest('hex') !== sha256) throw new Error(`Invalid model frontend asset: ${name}`)
  }
}
console.log('Browser assets ready: pinned ONNX Runtime WASM and verified CWformer v6.')
