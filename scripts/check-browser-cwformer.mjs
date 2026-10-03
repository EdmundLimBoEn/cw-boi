// Runs the repository's executable parity page in an installed headless Chrome.
// No audio, browser process, or model inference is run on the development Mac.
import { spawn } from 'node:child_process'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const [url, output = 'benchmarks/browser-cwformer-parity.json'] = process.argv.slice(2)
if (!url || !process.env.CW_CHROME) throw new Error('Usage: CW_CHROME=<installed chrome> node scripts/check-browser-cwformer.mjs <parity-page-url> [report.json]')
const profile = await mkdtemp(join(tmpdir(), 'cw-boi-parity-'))
const browser = spawn(process.env.CW_CHROME, ['--headless=new', '--disable-gpu', '--no-first-run', '--no-default-browser-check',
  '--remote-debugging-pipe', `--user-data-dir=${profile}`], { stdio: ['ignore', 'ignore', 'pipe', 'pipe', 'pipe'] })
let nextId = 1, buffered = ''
const pending = new Map()
const requests = []
let browserError = ''
browser.stderr.on('data', chunk => { browserError = (browserError + chunk).slice(-12000) })
browser.on('error', error => { for (const { reject } of pending.values()) reject(error); pending.clear() })
browser.on('exit', code => {
  for (const { reject } of pending.values()) reject(new Error(`Chrome exited ${code}: ${browserError}`))
  pending.clear()
})
browser.stdio[4].on('data', chunk => {
  buffered += chunk.toString()
  let end
  while ((end = buffered.indexOf('\0')) >= 0) {
    const text = buffered.slice(0, end)
    buffered = buffered.slice(end + 1)
    if (!text) continue
    const message = JSON.parse(text)
    if (message.id && pending.has(message.id)) {
      const { resolve, reject, timer } = pending.get(message.id)
      pending.delete(message.id)
      clearTimeout(timer)
      if (message.error) reject(new Error(JSON.stringify(message.error)))
      else resolve(message.result)
    }
    if (message.method === 'Runtime.consoleAPICalled') console.log(...message.params.args.map(arg => arg.value ?? arg.description))
    if (message.method === 'Runtime.exceptionThrown') console.error(message.params.exceptionDetails)
    if (message.method === 'Network.requestWillBeSent') requests.push({ method: message.params.request.method, url: message.params.request.url })
    if (message.method === 'Target.attachedToTarget') {
      const attached = message.params.sessionId
      command('Network.enable', {}, attached).then(() => command('Runtime.runIfWaitingForDebugger', {}, attached)).catch(console.error)
    }
  }
})

function command(method, params = {}, sessionId) {
  const id = nextId++
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { pending.delete(id); reject(new Error(`Timed out: ${method}`)) }, 600000)
    pending.set(id, { resolve, reject, timer })
    browser.stdio[3].write(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }) + '\0')
  })
}

try {
  const { targetId } = await command('Target.createTarget', { url: 'about:blank' })
  const { sessionId } = await command('Target.attachToTarget', { targetId, flatten: true })
  await command('Runtime.enable', {}, sessionId)
  await command('Page.enable', {}, sessionId)
  await command('Network.enable', {}, sessionId)
  await command('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: true, flatten: true }, sessionId)
  await command('Page.navigate', { url }, sessionId)
  if (process.env.CW_PRODUCTION_WORKER) {
    const fixtureRoot = process.env.CW_REFERENCE_DIR || '.research/browser-reference'
    const reference = JSON.parse(await readFile(join(fixtureRoot, 'reference.json'), 'utf8'))
    const item = reference.cases.find(item => item.id === 'clean-1')
    const audio = (await readFile(join(fixtureRoot, item.file))).toString('base64')
    const html = await readFile('scripts/check-browser-cwformer-production.html', 'utf8')
    const source = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1]
    // The fixture travels only through the local debugging pipe. It is never
    // uploaded to the deployed site or exposed as a public diagnostic route.
    await new Promise(resolve => setTimeout(resolve, 1500))
    await command('Runtime.evaluate', { expression: `(() => { window.__cwReference = ${JSON.stringify(reference)}; window.__cwAudioBase64 = ${JSON.stringify(audio)}; window.__cwWorker = ${JSON.stringify(process.env.CW_PRODUCTION_WORKER)}; ${source} })()` }, sessionId)
  }
  // Wait for the new page's context before asking it for the asynchronous suite.
  let completed
  const deadline = Date.now() + 600000
  while (Date.now() < deadline) {
    try {
      const response = await command('Runtime.evaluate', {
        expression: 'window.parityFinished ? window.parityResult : null', returnByValue: true,
      }, sessionId)
      if (response.result?.value) { completed = response.result.value; break }
    } catch (error) {
      if (!String(error).includes('context')) throw error
    }
    await new Promise(resolve => setTimeout(resolve, 1000))
  }
  if (!completed) throw new Error('Browser parity suite timed out after 10 minutes')
  const unexpected = requests.filter(request => !request.url.startsWith(new URL(url).origin + '/') || request.method !== 'GET' || new URL(request.url).pathname.startsWith('/api/'))
  if (unexpected.length) { completed.passed = false; completed.error = `Unexpected network traffic: ${JSON.stringify(unexpected)}` }
  await writeFile(output, JSON.stringify({ recordedAt: new Date().toISOString(), url, network: { requests, onlySameOriginGets: unexpected.length === 0 }, ...completed }, null, 2) + '\n')
  console.log(JSON.stringify({ passed: completed.passed, checks: completed.results?.length, elapsedMilliseconds: completed.elapsedMilliseconds, output }))
  if (!completed.passed) throw new Error(completed.error)
} finally {
  try { await command('Browser.close') } catch {}
  browser.kill()
  for (const { timer } of pending.values()) clearTimeout(timer)
  pending.clear()
  await rm(profile, { recursive: true, force: true, maxRetries: 10, retryDelay: 250 })
}
