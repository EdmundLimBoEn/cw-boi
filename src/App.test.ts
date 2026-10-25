import { expect, test } from 'bun:test'
import ts from 'typescript'

// Execute the actual app callbacks with controlled audio readiness, without a DOM
// or loading the inference model. AST extraction keeps JSX out of the harness.
const source = await Bun.file(new URL('./App.tsx', import.meta.url)).text()
const ast = ts.createSourceFile('App.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
let stop = '', armKey = '', compose = ''
function visit(node: ts.Node) {
  if (ts.isVariableDeclaration(node) && node.name.getText(ast) === 'stop') stop = node.getText(ast)
  if (ts.isFunctionDeclaration(node) && node.name?.text === 'armKey') armKey = node.getText(ast)
  if (ts.isJsxElement(node) && node.children.some(child => ts.isJsxText(child) && child.text === 'Compose a message')) {
    const click = node.openingElement.attributes.properties.find(attr => ts.isJsxAttribute(attr) && attr.name.getText(ast) === 'onClick') as ts.JsxAttribute
    compose = (click.initializer as ts.JsxExpression).expression!.getText(ast)
  }
  ts.forEachChild(node, visit)
}
visit(ast)
if (!stop || !armKey || !compose) throw new Error('App key callbacks not found')
const callbacks = ts.transpileModule(`const ${stop}; ${armKey}; return { armKey, compose: ${compose} };`, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
}).outputText

function setup(stage: string) {
  let resolve!: () => void
  const pending = new Promise<void>(done => { resolve = done })
  let reached!: () => void
  const waiting = new Promise<void>(done => { reached = done })
  const calls: string[] = []
  const step = async (name: string) => {
    calls.push(name)
    if (stage === name) { reached(); await pending }
    return true
  }
  const audio = {
    ready: () => step('ready'), loopback: () => step('loopback'), prepareKey: () => step('prepareKey'),
    stopReceiver: () => calls.push('stopReceiver'), stopSender() {}, stopKey: () => calls.push('stopKey'),
  }
  const state = { armed: false, busy: false, source: 'idle', mode: 'key', error: '' }
  const scope = {
    ...state,
    operation: { current: 0 }, keySetupPending: { current: false }, keyIsDown: { current: false },
    station: { current: audio }, neuralStream: { current: null }, neuralRequest: { current: null }, worker: { current: null },
    neuralBusy: false, modelState: { phase: 'ready' }, EMPTY: {}, rx: {}, tx: {},
    getAudio: () => audio, startNeural: () => step('neural'), audioError: String,
    setArmed(value: boolean) { scope.armed = value }, setBusy(value: boolean) { scope.busy = value },
    setSource(value: string) { scope.source = value }, setMode(value: string) { scope.mode = value },
    setError(value: string) { scope.error = value },
    setNeuralBusy() {}, setPlaying() {}, setKeyed() {}, setLabRunning() {}, setReading() {}, setNeuralText() {}, setReceiverTab() {}, setSourceName() {},
  }
  // A lexical scope lets the extracted callbacks see state updates between clicks.
  const actions = new Function('scope', `with (scope) { ${callbacks} }`)(scope)
  return { scope, actions, calls, resolve, waiting }
}

for (const stage of ['ready', 'neural', 'loopback', 'prepareKey']) {
  test(`Compose cancels key setup pending at ${stage}`, async () => {
    const { scope, actions, calls, resolve, waiting } = setup(stage)
    const pending = actions.armKey()
    await waiting
    expect(scope.busy).toBe(true)
    expect(scope.armed).toBe(false)
    const ticket = scope.operation.current
    const stops = calls.filter(call => call === 'stopKey').length
    actions.compose()
    expect(scope.mode).toBe('compose')
    expect(scope.operation.current).toBeGreaterThan(ticket)
    expect(calls.filter(call => call === 'stopKey')).toHaveLength(stops + 1)
    expect(scope.busy).toBe(false)
    resolve()
    await pending
    expect(scope.armed).toBe(false)
    expect(scope.source).toBe('idle')
    expect(scope.keySetupPending.current).toBe(false)
    expect(calls.filter(call => ['ready', 'neural', 'loopback', 'prepareKey'].includes(call)).at(-1)).toBe(stage)
  })
}

test('completed key setup still arms normally and Compose disarms it', async () => {
  const { scope, actions } = setup('none')
  await actions.armKey()
  expect(scope.armed).toBe(true)
  expect(scope.source).toBe('key')
  expect(scope.keySetupPending.current).toBe(false)
  actions.compose()
  expect(scope.armed).toBe(false)
  expect(scope.source).toBe('idle')
})

test('Compose leaves unrelated pending reception alone', () => {
  const { scope, actions } = setup('none')
  scope.busy = true
  const ticket = scope.operation.current
  actions.compose()
  expect(scope.operation.current).toBe(ticket)
  expect(scope.busy).toBe(true)
})
