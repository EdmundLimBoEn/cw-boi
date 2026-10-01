import { CWDecoder, type ReceiverSettings } from './decoder'

self.onmessage = ({ data }: MessageEvent<{ samples: Float32Array; settings: ReceiverSettings }>) => {
  try {
    const decoder = new CWDecoder(8000, data.settings)
    for (let i = 0; i < data.samples.length; i += 256) {
      decoder.process(data.samples.subarray(i, i + 256))
      if (i % 65536 === 0) self.postMessage({ type: 'progress', progress: i / data.samples.length })
    }
    self.postMessage({ type: 'complete', reading: decoder.finish() })
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : 'This recording could not be decoded.' })
  }
}
