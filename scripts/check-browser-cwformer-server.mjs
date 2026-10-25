import { createServer } from 'node:http'
import { readFile, stat } from 'node:fs/promises'
import { resolve, extname, sep } from 'node:path'
import { pathToFileURL } from 'node:url'

function localAuthority(value) {
  if (typeof value !== 'string' || !/^(localhost|127\.0\.0\.1|\[::1\])(:[0-9]+)?$/i.test(value)) return false
  try { return Boolean(new URL(`http://${value}`).hostname) } catch { return false }
}

function localRequest(request) {
  const origin = request.headers.origin
  return localAuthority(request.headers.host) && (origin === undefined ||
    typeof origin === 'string' && /^https?:\/\//.test(origin) && localAuthority(origin.replace(/^https?:\/\//, '')))
}

const production = resolve('.cloudflare/output/v0/workers/default/assets')
const routes = [
  ['/__fixtures__/', resolve('.research/browser-reference')],
  ['/__checks__/', resolve('scripts')],
  ['/', production],
]
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json', '.css': 'text/css' }
export function createDiagnosticServer() {
  return createServer(async (request, response) => {
    if (!localRequest(request)) {
      response.writeHead(403)
      response.end('Local requests only')
      return
    }
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      response.writeHead(405, { Allow: 'GET, HEAD' })
      response.end('Method not allowed')
      return
    }
    try {
      const path = decodeURIComponent(new URL(request.url, 'http://localhost').pathname)
      const [prefix, root] = routes.find(([prefix]) => path.startsWith(prefix))
      const file = resolve(root, path.slice(prefix.length) || 'index.html')
      if (!file.startsWith(root + sep)) throw new Error('Invalid path')
      if (!(await stat(file)).isFile()) throw new Error('Not a file')
      response.writeHead(200, { 'Content-Type': types[extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-store' })
      response.end(request.method === 'HEAD' ? undefined : await readFile(file))
    } catch {
      response.writeHead(404)
      response.end('Not found')
    }
  })
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  createDiagnosticServer().listen(5180, '127.0.0.1', () => console.log(`Serving exact production assets ${production} on http://127.0.0.1:5180`))
}
