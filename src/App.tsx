import { useEffect, useId, useMemo, useRef, useState, type ReactNode } from 'react'
import { Activity, ArrowDownToLine, ArrowUpRight, AudioLines, BookOpen, Check, ChevronDown, ChevronRight, CircleHelp, Copy, FileAudio, FlaskConical, Headphones, Keyboard, Link2, Mic, Play, Radio, RotateCcw, ShieldCheck, SlidersHorizontal, Square, Upload, Volume2, Waves, X } from 'lucide-react'
import { StationAudio, audioError } from './audio'
import { DEFAULT_RECEIVER, type DecoderReading, type ReceiverSettings } from './decoder'
import { ALPHABET, DEFAULT_SENDER, MORSE, decodeText, synthesize, transmission, wavBytes, type SenderSettings } from './morse'
import { CHALLENGES, characterErrors, makeSignal, type Challenge } from './signals'
import { Spectrum } from './Spectrum'
import { NeuralStream, neuralDecode } from './neural'

const EMPTY: DecoderReading = { text: '', pending: '', frequency: 650, wpm: 20, snr: 0, level: -100, keyed: false, confidence: 0, spectrum: [] }
const DEMO_TEXT = 'CQ CQ DE CWBOI 73'
const LAB_TEXT = 'CQ DE 9V1ABC UR RST 579 QTH SINGAPORE K'
type Source = 'idle' | 'mic' | 'loopback' | 'demo' | 'file' | 'key'
type LabResult = { text: string; cer: number; errors: number; milliseconds: number }

function download(data: BlobPart, type: string, name: string) {
  const url = URL.createObjectURL(new Blob([data], { type }))
  const anchor = document.createElement('a')
  anchor.href = url; anchor.download = name; anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}

function Slider({ label, value, min, max, step = 1, suffix, onChange, disabled = false }: { label: string; value: number; min: number; max: number; step?: number; suffix: string; onChange: (value: number) => void; disabled?: boolean }) {
  const id = useId()
  return <div className={`slider-control ${disabled ? 'disabled' : ''}`}>
    <div className="control-label"><label htmlFor={id}>{label}</label><output htmlFor={id}>{value}<span>{suffix}</span></output></div>
    <input id={id} type="range" min={min} max={max} step={step} value={value} disabled={disabled} onChange={event => onChange(Number(event.target.value))} style={{ '--range-progress': `${(value - min) / (max - min) * 100}%` } as React.CSSProperties} />
  </div>
}

function Toggle({ checked, onChange, children, disabled = false }: { checked: boolean; onChange: (value: boolean) => void; children: ReactNode; disabled?: boolean }) {
  return <label className="toggle-label"><input type="checkbox" role="switch" checked={checked} disabled={disabled} onChange={event => onChange(event.target.checked)} /><span className="toggle-track" /><span>{children}</span></label>
}

function App() {
  const [view, setView] = useState<'station' | 'lab'>('station')
  const [rx, setRx] = useState<ReceiverSettings>(DEFAULT_RECEIVER)
  const [tx, setTx] = useState<SenderSettings>(DEFAULT_SENDER)
  const [message, setMessage] = useState('CQ CQ DE CWBOI K')
  const [reading, setReading] = useState<DecoderReading>(EMPTY)
  const [source, setSource] = useState<Source>('idle')
  const [sourceName, setSourceName] = useState('')
  const [busy, setBusy] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [progress, setProgress] = useState(0)
  const [elapsed, setElapsed] = useState(0)
  const [loopback, setLoopback] = useState(true)
  const [mode, setMode] = useState<'compose' | 'key'>('compose')
  const [armed, setArmed] = useState(false)
  const [keyed, setKeyed] = useState(false)
  const [advanced, setAdvanced] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [help, setHelp] = useState(false)
  const [receiverTab, setReceiverTab] = useState<'audio' | 'text'>('audio')
  const [morseInput, setMorseInput] = useState('')
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [deviceId, setDeviceId] = useState('')
  const [results, setResults] = useState<Record<string, LabResult>>({})
  const [labRunning, setLabRunning] = useState<string | null>(null)
  const [neuralReady, setNeuralReady] = useState(false)
  const [neuralName, setNeuralName] = useState('Local acoustic model')
  const [engine, setEngine] = useState<'adaptive' | 'neural'>('adaptive')
  const [neuralText, setNeuralText] = useState('')
  const [neuralBusy, setNeuralBusy] = useState(false)
  const neuralStream = useRef<NeuralStream | null>(null)
  const neuralRequest = useRef<AbortController | null>(null)
  const station = useRef<StationAudio | null>(null)
  const getAudio = () => station.current ??= new StationAudio()
  const fileInput = useRef<HTMLInputElement>(null)
  const helpDialog = useRef<HTMLDialogElement>(null)
  const worker = useRef<Worker | null>(null)
  const operation = useRef(0)
  const keyIsDown = useRef(false)
  const startTime = useRef(0)
  const currentSource = useRef(source)
  currentSource.current = source
  const currentRx = useRef(rx)
  currentRx.current = rx
  const planResult = useMemo(() => {
    try { return { plan: transmission(message, tx.wpm, tx.spacing), error: '' } }
    catch (error) { return { plan: null, error: audioError(error) } }
  }, [message, tx.wpm, tx.spacing])
  const plan = planResult.plan
  const active = source !== 'idle' && !busy
  const activeToken = playing && plan ? plan.tones.find(tone => elapsed >= tone.start && elapsed < tone.end)?.index : undefined
  const displayedText = receiverTab === 'text' ? decodeText(morseInput) : engine === 'neural' ? neuralText : reading.text
  const changeRx = (settings: Partial<ReceiverSettings>) => setRx(previous => ({ ...previous, ...settings }))
  const changeTx = (settings: Partial<SenderSettings>) => setTx(previous => ({ ...previous, ...settings }))

  useEffect(() => {
    const audio = getAudio()
    audio.onReading = reading => {
      setReading(reading)
      if (neuralStream.current) {
        neuralStream.current.frequency = reading.frequency
        neuralStream.current.bandwidth = currentRx.current.bandwidth
      }
    }
    audio.onReceiverEnded = () => { finishNeural(); setSource('idle'); setNotice('Audio input disconnected.') }
    void fetch('/api/health').then(response => response.ok ? response.json() : null).then(data => { if (data?.ready) { setNeuralReady(true); setNeuralName(data.model) } }).catch(() => {})
    return () => { worker.current?.terminate(); neuralStream.current?.cancel(); neuralRequest.current?.abort(); audio.dispose() }
  }, [])
  useEffect(() => { station.current?.configure(rx) }, [rx])
  useEffect(() => { station.current?.setVolume(tx.volume) }, [tx.volume])
  useEffect(() => {
    if (help) helpDialog.current?.showModal()
    else helpDialog.current?.close()
  }, [help])
  useEffect(() => {
    if (!notice) return
    const timer = setTimeout(() => setNotice(''), 3500)
    return () => clearTimeout(timer)
  }, [notice])
  useEffect(() => {
    if (!playing) return
    const timer = setInterval(() => setElapsed(Math.max(0, (station.current?.context?.currentTime ?? 0) - startTime.current)), 35)
    return () => clearInterval(timer)
  }, [playing])

  const finishNeural = async () => {
    const stream = neuralStream.current
    if (!stream) return
    setNeuralBusy(true)
    await stream.finish()
    if (neuralStream.current === stream) { neuralStream.current = null; setNeuralBusy(false) }
  }

  const finishReception = async () => {
    const ticket = operation.current
    await getAudio().stopReceiver(true)
    if (ticket === operation.current) await finishNeural()
  }

  async function startNeural(settings: ReceiverSettings) {
    getAudio().onSamples = null
    if (engine !== 'neural') return
    const stream = new NeuralStream()
    neuralStream.current = stream
    stream.onText = setNeuralText
    stream.onError = error => { setError(audioError(error)); setNeuralBusy(false) }
    await stream.start(settings)
    if (neuralStream.current !== stream) return
    getAudio().onSamples = samples => stream.feed(samples)
  }

  const stop = (finish = false) => {
    operation.current++
    setNeuralBusy(false)
    if (finish) void finishReception()
    else { neuralStream.current?.cancel(); neuralStream.current = null; setNeuralBusy(false); station.current?.stopReceiver() }
    neuralRequest.current?.abort()
    worker.current?.terminate(); worker.current = null
    station.current?.stopSender(); station.current?.stopKey()
    keyIsDown.current = false
    setSource('idle'); setPlaying(false); setBusy(false); setArmed(false); setKeyed(false); setLabRunning(null)
  }

  useEffect(() => {
    const up = () => { station.current?.releaseKey(); keyIsDown.current = false; setKeyed(false) }
    const down = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { stop(true); return }
      const target = event.target as HTMLElement
      if (['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON'].includes(target.tagName) || target.isContentEditable || help) return
      if (armed && event.code === 'Space') {
        event.preventDefault()
        if (!event.repeat) { getAudio().pressKey(); keyIsDown.current = true; setKeyed(true) }
      }
    }
    const release = (event: KeyboardEvent) => { if (event.code === 'Space') up() }
    const hidden = () => { if (document.hidden) up() }
    window.addEventListener('keydown', down); window.addEventListener('keyup', release); window.addEventListener('blur', up); document.addEventListener('visibilitychange', hidden)
    return () => { window.removeEventListener('keydown', down); window.removeEventListener('keyup', release); window.removeEventListener('blur', up); document.removeEventListener('visibilitychange', hidden) }
  }, [armed, help])

  async function listen() {
    stop(); setError(''); setBusy(true); setReceiverTab('audio'); setReading(EMPTY); setNeuralText('')
    const ticket = operation.current
    try {
      await startNeural(rx)
      if (ticket !== operation.current) return
      const started = await getAudio().listen(rx, deviceId)
      if (!started || ticket !== operation.current) return
      setSource('mic'); setSourceName('Microphone')
      const inputs = await navigator.mediaDevices.enumerateDevices()
      setDevices(inputs.filter(device => device.kind === 'audioinput'))
    } catch (error) { if (ticket === operation.current) setError(audioError(error)) }
    finally { if (ticket === operation.current) setBusy(false) }
  }

  async function send() {
    if (!plan?.tones.length || plan.unsupported.length) return
    setError(''); setElapsed(0)
    const audio = getAudio()
    if (armed || loopback) stop()
    const ticket = operation.current
    setBusy(true)
    try {
      if (loopback) {
        setReading(EMPTY); setNeuralText(''); setReceiverTab('audio')
        await startNeural({ ...rx, frequency: tx.frequency })
        if (ticket !== operation.current) return
        if (!await audio.loopback({ ...rx, frequency: tx.frequency, wpm: tx.wpm, spacing: tx.spacing })) return
        setSource('loopback'); setSourceName('Sender loopback')
      }
      if (ticket !== operation.current) return
      startTime.current = await audio.play(plan, tx, loopback, () => {
        setPlaying(false); setElapsed(plan.duration)
        if (currentSource.current === 'loopback') { void finishReception(); setSource('idle') }
      })
      if (ticket !== operation.current) { audio.stopSender(); return }
      setPlaying(true)
    } catch (error) { stop(); setError(audioError(error)) }
    finally { if (ticket === operation.current) setBusy(false) }
  }

  async function playDemo(challenge = CHALLENGES[0], text = DEMO_TEXT) {
    stop(); setError(''); setBusy(true); setReading(EMPTY); setNeuralText(''); setReceiverTab('audio')
    const ticket = operation.current
    try {
      const audio = getAudio()
      await startNeural({ ...rx, frequency: challenge.frequency })
      if (ticket !== operation.current) return
      if (!await audio.loopback({ ...rx, frequency: challenge.frequency, autoTune: true })) return
      if (ticket !== operation.current) return
      setSource('demo'); setSourceName(challenge.name)
      await audio.playSamples(makeSignal(text, challenge), 8000, tx.volume, () => { void finishReception(); setSource('idle') })
    } catch (error) { if (ticket === operation.current) { stop(); setError(audioError(error)) } }
    finally { if (ticket === operation.current) setBusy(false) }
  }

  function analyze(samples: Float32Array, settings: ReceiverSettings, onDone: (reading: DecoderReading) => void) {
    worker.current?.terminate()
    const decoderWorker = new Worker(new URL('./file.worker.ts', import.meta.url), { type: 'module' })
    worker.current = decoderWorker
    decoderWorker.onmessage = ({ data }) => {
      if (data.type === 'progress') setProgress(data.progress)
      else {
        decoderWorker.terminate(); worker.current = null; setBusy(false); setSource('idle'); setLabRunning(null)
        if (data.type === 'error') setError(data.message)
        else { setReading(data.reading); onDone(data.reading) }
      }
    }
    decoderWorker.onerror = () => { stop(); setError('The audio decoder could not finish. Try a shorter recording.') }
    decoderWorker.postMessage({ samples, settings }, [samples.buffer])
  }

  async function importAudio(file: File) {
    stop(); setError(''); setBusy(true); setProgress(0); setSourceName(file.name); setReading(EMPTY); setNeuralText(''); setReceiverTab('audio')
    const ticket = operation.current
    try {
      const samples = await getAudio().readFile(file)
      if (ticket !== operation.current) return
      setSource('file')
      const neuralSamples = engine === 'neural' ? samples.slice() : null
      analyze(samples, rx, result => {
        if (!neuralSamples) { setNotice(result.text.trim() ? 'Recording decoded. Your audio stayed on this device.' : 'No Morse found. Try tuning the carrier or lowering the signal gate.'); return }
        const controller = new AbortController(); neuralRequest.current = controller; setNeuralBusy(true)
        void neuralDecode(neuralSamples, { ...rx, frequency: rx.autoTune ? result.frequency : rx.frequency }, controller.signal)
          .then(text => { if (ticket === operation.current) { setNeuralText(text); setNotice(text ? 'Recording decoded. Your audio stayed on this device.' : 'No Morse found. Try tuning the carrier.') } })
          .catch(error => { if (!controller.signal.aborted) setError(audioError(error)) })
          .finally(() => { if (ticket === operation.current) setNeuralBusy(false) })
      })
    } catch (error) { if (ticket === operation.current) { setError(audioError(error)); setBusy(false) } }
  }

  async function armKey() {
    if (armed) { stop(true); return }
    stop(); setError(''); setBusy(true); setReading(EMPTY); setNeuralText(''); setReceiverTab('audio')
    const ticket = operation.current
    try {
      const audio = getAudio()
      await startNeural({ ...rx, frequency: tx.frequency })
      if (ticket !== operation.current) return
      if (!await audio.loopback({ ...rx, frequency: tx.frequency, autoTune: false, wpm: tx.wpm, spacing: tx.spacing })) return
      if (ticket !== operation.current) return
      await audio.prepareKey(tx)
      if (ticket !== operation.current) { audio.stopKey(); return }
      setArmed(true); setSource('key'); setSourceName('Straight key')
    } catch (error) { stop(); setError(audioError(error)) }
    finally { if (ticket === operation.current) setBusy(false) }
  }

  async function runChallenge(challenge: Challenge) {
    stop(); setBusy(true); setProgress(0); setLabRunning(challenge.id); setSourceName(challenge.name); setReading(EMPTY)
    const started = performance.now()
    const samples = makeSignal(LAB_TEXT, challenge, 947)
    if (engine === 'neural') {
      const controller = new AbortController(); neuralRequest.current = controller
      try {
        const text = await neuralDecode(samples, { ...rx, frequency: challenge.frequency }, controller.signal)
        if (!controller.signal.aborted) {
          setNeuralText(text)
          setResults(previous => ({ ...previous, [challenge.id]: { text, ...characterErrors(LAB_TEXT, text), milliseconds: Math.round(performance.now() - started) } }))
        }
      } catch (error) { if (!controller.signal.aborted) setError(audioError(error)) }
      finally { if (!controller.signal.aborted) { setBusy(false); setLabRunning(null) } }
      return
    }
    analyze(samples, { ...rx, frequency: challenge.frequency, autoTune: false }, decoded => {
      setResults(previous => ({ ...previous, [challenge.id]: { text: decoded.text.trim(), ...characterErrors(LAB_TEXT, decoded.text), milliseconds: Math.round(performance.now() - started) } }))
    })
  }

  async function copy(text: string) {
    try { await navigator.clipboard.writeText(text); setNotice('Copied to clipboard.') }
    catch { setError('Clipboard access is blocked. Select the transcript and copy it manually.') }
  }

  const clearTranscript = () => { stop(); setReading(EMPTY); setMorseInput(''); setNeuralText('') }
  const buttonBusy = busy || playing || source === 'demo' || armed
  const statusLabel = busy ? source === 'file' || labRunning ? `Decoding ${Math.round(progress * 100)}%` : 'Connecting' : source === 'mic' ? 'Listening' : source === 'demo' ? 'Sample playing' : source === 'loopback' ? 'Loopback' : source === 'key' ? 'Key ready' : 'Standby'

  return <div className="app-shell">
    <header className="site-header">
      <a className="brand" href="#" aria-label="CW boi station" onClick={event => { event.preventDefault(); setView('station') }}><span className="brand-symbol"><i /><b /><b /><i /></span><span>cw<span className="brand-slash">/</span>boi<span className="brand-period">.</span></span></a>
      <nav aria-label="Main navigation"><button className={view === 'station' ? 'nav-button selected' : 'nav-button'} onClick={() => setView('station')}><Radio size={16} /> Station</button><button className={view === 'lab' ? 'nav-button selected' : 'nav-button'} onClick={() => setView('lab')}><FlaskConical size={16} /> Signal lab</button></nav>
      <button className="guide-button" onClick={() => setHelp(true)}><BookOpen size={16} /><span>Field guide</span><ArrowUpRight size={14} /></button>
    </header>

    <main>
      <section className="page-intro"><div><div className="intro-kicker"><span className="status-dot" /> Your personal signal station</div><h1>{view === 'station' ? 'Less noise. More connection.' : 'Good copy is earned.'}</h1><p>{view === 'station' ? 'From a whisper in the static to a perfectly timed reply.' : 'Put the decoder through its paces. See exactly what it gets right.'}</p></div><div className="intro-note"><span className="morse-greeting" aria-hidden="true">−·−·  −−·−</span><span>A little wave goes a long way.</span></div></section>

      {error && <div className="error-banner" role="alert"><CircleHelp size={19} /><span>{error}</span><button aria-label="Dismiss error" onClick={() => setError('')}><X size={16} /></button></div>}

      {view === 'station' ? <div className="station-grid">
        <section className="panel receiver-panel" aria-labelledby="receiver-heading">
          <div className="panel-header"><div className="panel-title"><span className="channel-icon rx-icon"><AudioLines size={19} /></span><h2 id="receiver-heading">Receiver</h2><span className="channel-code">RX</span></div><span className={`status-pill ${active ? 'is-live' : ''}`}><span />{statusLabel}</span></div>

          <div className="receiver-display">
            <div className="display-topline"><span><span className={reading.keyed && active ? 'signal-led lit' : 'signal-led'} /> {active ? sourceName : 'Ready when you are'}</span><span>200–1500 Hz</span></div>
            <div className="signal-readouts"><div className="frequency-readout">{String(active ? reading.frequency : rx.frequency).padStart(3, '0')}<span>Hz</span><small>{rx.autoTune ? 'Auto tune' : 'Manual tune'}</small></div><div className="secondary-readout"><strong>{active || reading.text ? reading.wpm : '—'}<span>WPM</span></strong><small>Detected speed</small></div><div className="secondary-readout"><strong>{active ? reading.snr : '—'}<span>dB</span></strong><small>Bin SNR estimate</small></div></div>
            <Spectrum reading={reading} active={active} frequency={active && rx.autoTune ? reading.frequency : rx.frequency} bandwidth={rx.bandwidth} onTune={frequency => changeRx({ frequency, autoTune: false })} />
            <div className="display-bottomline"><span><span className="tiny-cross">+</span> Click the spectrum to tune</span><span>Live spectrum / waterfall</span></div>
          </div>

          <div className="receiver-controls"><div className="receiver-tuning"><Slider label="Target tone" value={rx.frequency} min={250} max={1400} step={5} suffix="Hz" onChange={frequency => changeRx({ frequency, autoTune: false })} /><Toggle checked={rx.autoTune} onChange={autoTune => changeRx({ autoTune })}>Auto tune</Toggle></div>
            <div className="receive-actions"><button className={`button primary ${source === 'mic' ? 'stop-button' : ''}`} disabled={busy} onClick={() => source === 'mic' ? stop(true) : void listen()}>{source === 'mic' ? <Square size={16} fill="currentColor" /> : <Mic size={17} />}{source === 'mic' ? 'Stop listening' : 'Start listening'}</button><button className="button secondary" disabled={busy} onClick={() => fileInput.current?.click()}><Upload size={16} /> Audio file</button><button className="icon-button advanced-button" aria-label="Receiver settings" aria-expanded={advanced} onClick={() => setAdvanced(!advanced)}><SlidersHorizontal size={18} /></button><input ref={fileInput} type="file" accept="audio/*,.wav,.mp3,.ogg,.m4a,.flac" className="visually-hidden" aria-label="Import audio recording" onChange={event => { const file = event.target.files?.[0]; if (file) void importAudio(file); event.target.value = '' }} /></div>
            {advanced && <div className="advanced-controls"><div className="controls-grid"><Slider label="Filter width" value={rx.bandwidth} min={40} max={300} step={10} suffix="Hz" onChange={bandwidth => changeRx({ bandwidth })} /><Slider disabled={engine === 'neural'} label="Signal gate" value={rx.threshold} min={3} max={18} suffix="dB" onChange={threshold => changeRx({ threshold })} /><Slider disabled={engine === 'neural'} label="Expected speed" value={rx.wpm} min={5} max={60} suffix="WPM" onChange={wpm => changeRx({ wpm, spacing: Math.min(rx.spacing, wpm) })} /><Slider disabled={engine === 'neural'} label="Spacing speed" value={rx.spacing} min={5} max={rx.wpm} suffix="WPM" onChange={spacing => changeRx({ spacing })} /></div><Toggle disabled={engine === 'neural'} checked={rx.autoSpeed} onChange={autoSpeed => changeRx({ autoSpeed })}>Learn the sender’s timing</Toggle><label className="device-picker">Audio input<select value={deviceId} disabled={source === 'mic'} onChange={event => setDeviceId(event.target.value)}><option value="">System default</option>{devices.map(device => <option key={device.deviceId} value={device.deviceId}>{device.label || 'Audio input'}</option>)}</select></label><p>Use a narrow filter for weak signals. Turn off auto tune to stay with one station in a crowded band.{engine === 'neural' && ' The acoustic model learns timing itself; gate and speed controls apply to the adaptive decoder.'}</p></div>}
          </div>

          <div className="transcript-section"><div className="engine-row"><span><Activity size={13} /> Decoder</span><select aria-label="Decoder engine" value={engine} disabled={active || busy || neuralBusy} onChange={event => { setEngine(event.target.value as 'adaptive' | 'neural'); setResults({}) }}><option value="adaptive">Adaptive signal decoder</option>{neuralReady && <option value="neural">{neuralName}</option>}</select>{engine === 'neural' && <small>{neuralBusy ? 'Finishing copy…' : 'Local acoustic decoding'}</small>}</div><div className="section-toolbar"><div className="transcript-tabs"><button className={receiverTab === 'audio' ? 'active' : ''} onClick={() => setReceiverTab('audio')}>Live copy</button><button className={receiverTab === 'text' ? 'active' : ''} onClick={() => setReceiverTab('text')}>Morse text</button></div><div className="toolbar-actions"><button className="icon-button" aria-label="Copy transcript" disabled={!displayedText} onClick={() => void copy(displayedText)}><Copy size={14} /></button><button className="icon-button" aria-label="Save transcript" disabled={!displayedText} onClick={() => download(displayedText, 'text/plain', 'cw-boi-transcript.txt')}><ArrowDownToLine size={15} /></button><button className="icon-button" aria-label="Clear transcript" disabled={!displayedText && !reading.pending} onClick={clearTranscript}><RotateCcw size={14} /></button></div></div>
            {receiverTab === 'text' && <textarea className="morse-input" aria-label="Morse code to decode" value={morseInput} onChange={event => setMorseInput(event.target.value)} placeholder="... --- ... / .- -...   · Separate letters with spaces and words with /" spellCheck={false} maxLength={10000} />}
            <div className={`transcript ${displayedText ? 'has-copy' : ''}`} role="log" aria-live="polite" aria-label="Decoded transcript">{displayedText ? <>{displayedText}<span className={active ? 'text-cursor live' : 'text-cursor'} /></> : <div className="empty-transcript"><Waves size={27} strokeWidth={1.3} /><span>{busy ? 'Finding the signal…' : active ? 'Listening for the first dits and dahs…' : 'There’s a conversation in the static.'}</span><small>{active ? engine === 'neural' ? 'Neural decoding uses a few seconds of context.' : 'Decoded characters will appear here.' : 'Connect your audio, or take a sample for a spin.'}</small>{!active && !busy && receiverTab === 'audio' && <button onClick={() => void playDemo()}><Play size={11} fill="currentColor" /> Try a sample signal <ChevronRight size={13} /></button>}</div>}</div>
            {engine === 'adaptive' && !!reading.alternatives?.length && <details className="copy-alternatives"><summary>Ambiguous timing · other possible copy</summary>{reading.alternatives.map((text, index) => <code key={index}>{text}</code>)}</details>}<div className="transcript-footer"><span className="pending-code">{reading.pending ? reading.pending.replace(/\./g, '·').replace(/-/g, '−') : '· · ·'}<span>{active ? 'Decoding locally' : sourceName ? `Last source: ${sourceName}` : 'Waiting for a signal'}</span></span><span>{displayedText.trim().length} characters</span></div>
          </div>
        </section>

        <section className="panel sender-panel" aria-labelledby="sender-heading">
          <div className="panel-header"><div className="panel-title"><span className="channel-icon tx-icon"><Radio size={19} /></span><h2 id="sender-heading">Transmitter</h2><span className="channel-code">TX</span></div><span className="output-tag"><Headphones size={13} /> Audio out</span></div>
          <div className="sender-body"><div className="segmented"><button aria-pressed={mode === 'compose'} className={mode === 'compose' ? 'active' : ''} onClick={() => { if (armed) stop(); setMode('compose') }}>Compose a message</button><button aria-pressed={mode === 'key'} className={mode === 'key' ? 'active' : ''} onClick={() => { if (playing) stop(); setMode('key') }}><Keyboard size={14} /> Straight key</button></div>
            {mode === 'compose' ? <><div className="message-heading"><label htmlFor="tx-message">Your message</label><span>{message.length}/1000</span></div><div className="message-editor"><textarea id="tx-message" value={message} onChange={event => setMessage(event.target.value)} disabled={playing} spellCheck={false} maxLength={1000} placeholder="Say something in dits and dahs…" /><div className="message-morse" aria-label="Encoded Morse preview">{plan?.tokens.length ? plan.tokens.map((token, index) => <span className={index === activeToken ? 'current-token' : ''} key={index} title={token.text}>{token.code.replace(/\./g, '·').replace(/-/g, '−')}</span>) : <span className="muted">Your Morse appears here.</span>}</div></div>
              <div className="macro-row"><span>Quick calls</span>{[{ label: 'CQ', text: 'CQ CQ CQ DE ' }, { label: '73', text: '73' }, { label: 'TEST', text: 'VVV TEST' }, { label: '<SK>', text: '<SK>' }].map(macro => <button key={macro.label} disabled={playing} title={`Insert ${macro.text}`} onClick={() => setMessage(previous => `${previous.trim()} ${macro.text}`.trimStart().slice(0, 1000))}>{macro.label}</button>)}<button className="clear-message" disabled={playing || !message} onClick={() => setMessage('')}>Clear</button></div>
              {(planResult.error || !!plan?.unsupported.length) && <p className="input-error" role="alert">{planResult.error || `Not in Morse: ${plan?.unsupported.join(' ')}. Remove these characters to send.`}</p>}
            </> : <div className={`straight-key ${keyed ? 'pressed' : ''}`}><span className="key-caption">Your rhythm. Your signal.</span><button className="key-paddle" disabled={!armed} aria-label="Hold to send Morse tone" aria-pressed={keyed} onPointerDown={event => { event.currentTarget.setPointerCapture(event.pointerId); keyIsDown.current = true; getAudio().pressKey(); setKeyed(true) }} onPointerUp={() => { keyIsDown.current = false; getAudio().releaseKey(); setKeyed(false) }} onPointerCancel={() => { keyIsDown.current = false; getAudio().releaseKey(); setKeyed(false) }} onLostPointerCapture={() => { keyIsDown.current = false; getAudio().releaseKey(); setKeyed(false) }} onKeyDown={event => { if (event.key === ' ' || event.key === 'Enter') { event.preventDefault(); if (!event.repeat) { getAudio().pressKey(); setKeyed(true) } } }} onKeyUp={event => { if (event.key === ' ' || event.key === 'Enter') { getAudio().releaseKey(); setKeyed(false) } }}><span className="paddle-top"><span /><i /><span /></span><span className="paddle-stem" /><span className="paddle-base" /></button><p>{armed ? 'Hold the key or press Space. Release for a gap.' : 'Enable the key, then tap out a conversation.'}</p><span className="key-hint"><kbd>Space</kbd> to key <span>·</span> <kbd>Esc</kbd> to stop</span></div>}

            <div className="sender-settings"><div className="controls-grid"><Slider label="Character speed" value={tx.wpm} min={5} max={60} suffix="WPM" disabled={buttonBusy} onChange={wpm => changeTx({ wpm, spacing: tx.spacing === tx.wpm ? wpm : Math.min(tx.spacing, wpm) })} /><Slider label="Farnsworth spacing" value={tx.spacing} min={5} max={tx.wpm} suffix="WPM" disabled={buttonBusy} onChange={spacing => changeTx({ spacing })} /><Slider label="Sidetone" value={tx.frequency} min={250} max={1400} step={5} suffix="Hz" disabled={buttonBusy} onChange={frequency => changeTx({ frequency })} /><Slider label="Volume" value={Math.round(tx.volume * 100)} min={0} max={100} suffix="%" onChange={volume => changeTx({ volume: volume / 100 })} /></div></div>
            <div className="loopback-row"><div><Link2 size={16} /><span>Listen to yourself<small>Route your sender into the decoder</small></span></div><Toggle checked={loopback} disabled={playing || armed} onChange={setLoopback}><span className="visually-hidden">Sender loopback</span></Toggle></div>
            <div className="send-actions">{mode === 'compose' ? <><button className={`button send-button ${playing ? 'sending' : ''}`} disabled={busy || !plan?.tones.length || !!plan.unsupported.length} onClick={() => playing ? stop(true) : void send()}>{playing ? <Square size={15} fill="currentColor" /> : <Play size={16} fill="currentColor" />}{playing ? 'Stop sending' : 'Send message'}<span>{playing ? `${Math.floor(elapsed)}s / ` : ''}{plan ? `${Math.ceil(plan.duration)}s` : '0s'}</span></button><button className="button secondary export-button" disabled={!plan?.tones.length || !!plan.unsupported.length || busy} onClick={() => { if (plan) { try { download(wavBytes(synthesize(plan, tx.frequency, 22050, tx.volume * 0.6)), 'audio/wav', 'cw-boi-message.wav'); setNotice('WAV downloaded.') } catch (error) { setError(audioError(error)) } } }}><ArrowDownToLine size={16} /> WAV</button></> : <button className={`button send-button ${armed ? 'sending' : ''}`} disabled={busy} onClick={() => void armKey()}>{armed ? <Square size={15} /> : <Keyboard size={17} />}{armed ? 'Disable straight key' : 'Enable straight key'}</button>}</div>
            <p className="sender-footnote"><Volume2 size={12} /> Sends audio through your selected system output.</p>
          </div>
        </section>
      </div> : <section className="lab-panel">
        <div className="lab-intro"><div className="lab-icon"><FlaskConical size={26} /></div><div><h2>Less guessing. More testing.</h2><p>Six repeatable synthetic conditions, including deliberately brutal ones. Each goes through the same decoder as your microphone. Character error rate counts substitutions, insertions, and deletions.</p></div></div>
        <div className="lab-engine"><label htmlFor="lab-engine">Test decoder</label><select id="lab-engine" value={engine} disabled={busy || active} onChange={event => { setEngine(event.target.value as 'adaptive' | 'neural'); setResults({}) }}><option value="adaptive">Adaptive signal decoder</option>{neuralReady && <option value="neural">{neuralName}</option>}</select></div><div className="lab-target"><span>Expected copy</span><code>{LAB_TEXT}</code><small>Fixed seed 947 · 8 kHz audio · Known carrier · SNR measured against broadband noise</small></div>
        <div className="challenge-grid">{CHALLENGES.map((challenge, index) => <article className="challenge" key={challenge.id}><div className="challenge-top"><span className="challenge-wave">{index === 0 ? <Waves /> : index === 1 ? <Activity /> : index === 2 ? <AudioLines /> : <Radio />}</span><span className="challenge-tag">{challenge.wpm} WPM / {challenge.snr > 0 ? '+' : ''}{challenge.snr} dB</span></div><h3>{challenge.name}</h3><p>{challenge.description}</p>{results[challenge.id] ? <div className="lab-result"><strong className={results[challenge.id].cer === 0 ? 'perfect' : ''}>{(results[challenge.id].cer * 100).toFixed(1)}% <span>character error</span></strong><code>{results[challenge.id].text || '(No copy)'}</code><small>{results[challenge.id].errors} edits / {LAB_TEXT.length} characters · {results[challenge.id].milliseconds} ms</small></div> : <div className="lab-result untested"><span>Ready for a fair test.</span><small>No result until you run the decoder.</small></div>}<div className="challenge-actions"><button className="button secondary" disabled={busy || active || playing} onClick={() => runChallenge(challenge)}>{labRunning === challenge.id ? <span className="spinner" /> : <FlaskConical size={14} />}{labRunning === challenge.id ? 'Running…' : 'Run test'}</button><button className="icon-button" disabled={busy || active || playing} aria-label={`Listen to ${challenge.name}`} onClick={() => { void playDemo(challenge, LAB_TEXT); setView('station') }}><Play size={15} /></button></div></article>)}</div>
        <div className="lab-caveat"><ShieldCheck size={18} /><p>Synthetic results are a regression check, not a claim of real-world accuracy. Real radio comparisons and failure cases are recorded alongside the benchmarks. Unknown symbols appear as <code>�</code>; callsigns are never autocorrected.</p></div>
      </section>}

      <div className="station-bottom"><div className="privacy-note"><ShieldCheck size={15} /><span>Your audio stays yours. Everything runs on this device.</span></div><button className="text-button" onClick={() => setHelp(true)}>New to CW? <span>Get your bearings</span><ArrowUpRight size={13} /></button></div>
      {(source === 'demo' || busy || source === 'key') && <div className="session-bar"><span><span className="status-dot" />{busy ? `Working${source === 'file' || labRunning ? ` · ${Math.round(progress * 100)}%` : '…'}` : source === 'key' ? 'Straight key is enabled' : `Playing ${sourceName.toLowerCase()}`}</span><button onClick={() => stop(true)}><Square size={12} /> Stop</button></div>}
    </main>

    <footer className="site-footer"><span>Made for the space between the dits.</span><span className="footer-73">73 <span>—</span> good signals, always.</span><span>cw/boi <span className="version">v1.0</span></span></footer>
    {notice && <div className="toast" role="status"><Check size={16} />{notice}</div>}

    <dialog ref={helpDialog} className="guide-dialog" onCancel={() => setHelp(false)} onClick={event => { if (event.target === event.currentTarget) setHelp(false) }}><div className="guide-content"><div className="guide-header"><div><span className="guide-kicker">A pocket companion</span><h2>The field guide.</h2></div><button className="icon-button" aria-label="Close field guide" onClick={() => setHelp(false)}><X size={22} /></button></div><div className="guide-steps"><div><Mic size={18} /><h3>Catch a signal</h3><p>Connect a radio’s audio output or use your microphone. Start listening, then click the carrier in the spectrum. Disable auto tune to stay on it.</p></div><div><Radio size={18} /><h3>Make a little noise</h3><p>Type a message and send it as audio. Slower Farnsworth spacing leaves more room between characters. Loopback lets you hear and decode your own signal.</p></div><div><Keyboard size={18} /><h3>Find your rhythm</h3><p>Enable the straight key and hold Space or the paddle. Release for the gaps. Escape stops the station. Keep a little extra space between words.</p></div></div><div className="guide-reference-title"><h3>The alphabet, at a glance.</h3><span>Dit · Dah −</span></div><div className="alphabet-grid">{ALPHABET.map(char => <div key={char}><strong>{char}</strong><code>{MORSE[char].replace(/\./g, '·').replace(/-/g, '−')}</code></div>)}</div><div className="guide-notes"><p><strong>Prosigns:</strong> Write <code>&lt;AR&gt;</code>, <code>&lt;AS&gt;</code>, <code>&lt;SK&gt;</code>, <code>&lt;BT&gt;</code>, <code>&lt;KN&gt;</code>, or <code>&lt;SOS&gt;</code> to send the letters as a single joined signal.</p><p><strong>Honest copy:</strong> The decoder learns timing from the signal. Badly overlapping marks and gaps can be ambiguous. <code>�</code> means an unrecognized pattern. Use manual tuning and speed when reception is difficult.</p><p><strong>Audio, not RF:</strong> This station produces an audio tone. Radio keying and transmitter control require a separate interface.</p><p><strong>Neural engine:</strong> When the local service is running, select its acoustic model in the Decoder menu. It uses surrounding audio to interpret uncertain timing. Its output is still a hypothesis. Compare it with the adaptive decoder, especially for callsigns.</p><a href="https://github.com/sderhy/morseformer" target="_blank" rel="noreferrer">Morseformer · Sébastien Derhy · Apache 2.0 <ArrowUpRight size={12} /></a><br /><a href="https://www.itu.int/rec/R-REC-M.1677-1-200910-I/" target="_blank" rel="noreferrer">International Morse timing · ITU-R M.1677 <ArrowUpRight size={12} /></a></div></div></dialog>
  </div>
}

export default App
