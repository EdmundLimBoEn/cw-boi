export type ModelProgress = {
  stage: 'download' | 'initialize' | 'decode'
  loaded: number
  total: number
}

export type WorkerRequest = {
  id: number
  type: 'load' | 'start' | 'feed' | 'finish'
  frequency: number
  bandwidth: number
  samples?: Float32Array
}

export type WorkerResponse =
  | ({ id: number; type: 'progress' } & ModelProgress)
  | { id: number; type: 'result'; text: string } // Newly emitted text for this request, not the whole transcript.
  | { id: number; type: 'error'; message: string }
