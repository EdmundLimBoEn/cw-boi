import { afterAll, beforeAll, expect, test } from 'bun:test'
import { mkdir, writeFile, unlink } from 'node:fs/promises'
import { randomUUID } from 'node:crypto'
import { createDiagnosticServer } from './check-browser-cwformer-server.mjs'

const name = `request-guard-${randomUUID()}.txt`
const roots = ['.cloudflare/output/v0/workers/default/assets', '.research/browser-reference', 'scripts']
const paths = [`/${name}`, `/__fixtures__/${name}`, `/__checks__/${name}`]
const server = createDiagnosticServer()

beforeAll(async () => {
  for (const root of roots) {
    await mkdir(root, { recursive: true })
    await writeFile(`${root}/${name}`, 'diagnostic test content')
  }
})

afterAll(async () => {
  await Promise.all(roots.map(root => unlink(`${root}/${name}`)))
})

function fetchRoute(path: string, headers: Record<string, string> = { Host: 'localhost' }, method = 'GET') {
  // Exercise the real request listener and filesystem without opening a socket.
  return new Promise<{ status: number; body: string; allow?: string }>(resolve => {
    let status = 0
    let allow: string | undefined
    server.emit('request', {
      url: path,
      method,
      headers: Object.fromEntries(Object.entries(headers).map(([key, value]) => [key.toLowerCase(), value])),
    }, {
      writeHead(code: number, headers: Record<string, string> = {}) {
        status = code
        allow = headers.Allow
      },
      end(body?: Buffer | string) { resolve({ status, body: body?.toString() ?? '', allow }) },
    })
  })
}

for (const path of paths) {
  test(`${path}: local GET and HEAD work`, async () => {
    for (const host of ['localhost', 'LOCALHOST:5180', '127.0.0.1:5180', '[::1]:5180']) {
      for (const origin of [undefined, 'http://localhost:5173', 'https://127.0.0.1', 'http://[::1]:5173']) {
        const headers = { Host: host, ...(origin === undefined ? {} : { Origin: origin }) }
        expect(await fetchRoute(path, headers)).toMatchObject({ status: 200, body: 'diagnostic test content' })
        expect(await fetchRoute(path, headers, 'HEAD')).toMatchObject({ status: 200, body: '' })
      }
    }
  })

  test(`${path}: rejects foreign, missing and malformed Hosts`, async () => {
    for (const host of [undefined, '', 'attacker.example', 'localhost.attacker.example', 'localhost@attacker.example', 'attacker.example@localhost', 'localhost/path', 'localhost:99999', '127.0.0.1, attacker.example']) {
      const result = await fetchRoute(path, host === undefined ? {} : { Host: host })
      expect(result.status).toBe(403)
      expect(result.body).not.toContain('diagnostic test content')
    }
  })

  test(`${path}: rejects foreign, opaque and malformed Origins`, async () => {
    for (const origin of ['', 'null', 'https://attacker.example', 'http://localhost.attacker.example', 'file://localhost', 'http://attacker.example@localhost', 'http://localhost/path', 'http://localhost:99999', 'http://localhost http://attacker.example']) {
      expect((await fetchRoute(path, { Host: 'localhost', Origin: origin })).status).toBe(403)
    }
  })

  test(`${path}: rejects unsupported methods with Allow`, async () => {
    for (const method of ['POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS', 'TRACE']) {
      expect(await fetchRoute(path, { Host: 'localhost' }, method)).toEqual({ status: 405, body: 'Method not allowed', allow: 'GET, HEAD' })
    }
  })
}

test('guards apply before file lookup', async () => {
  expect((await fetchRoute('/missing', { Host: 'attacker.example' })).status).toBe(403)
  expect((await fetchRoute('/missing', { Host: 'localhost' }, 'POST')).status).toBe(405)
  expect((await fetchRoute('/missing')).status).toBe(404)
})
