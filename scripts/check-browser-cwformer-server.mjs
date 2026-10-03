import { createServer } from 'node:http'
import { readFile, stat } from 'node:fs/promises'
import { resolve, extname, sep } from 'node:path'

const production = resolve('.cloudflare/output/v0/workers/default/assets')
const routes = [
  ['/__fixtures__/', resolve('.research/browser-reference')],
  ['/__checks__/', resolve('scripts')],
  ['/', production],
]
const types = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json', '.css': 'text/css' }
createServer(async (request, response) => {
  try {
    const path = decodeURIComponent(new URL(request.url, 'http://localhost').pathname)
    const [prefix, root] = routes.find(([prefix]) => path.startsWith(prefix))
    const file = resolve(root, path.slice(prefix.length) || 'index.html')
    if (!file.startsWith(root + sep)) throw new Error('Invalid path')
    if (!(await stat(file)).isFile()) throw new Error('Not a file')
    response.writeHead(200, { 'Content-Type': types[extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-store' })
    response.end(await readFile(file))
  } catch {
    response.writeHead(404)
    response.end('Not found')
  }
}).listen(5180, '127.0.0.1', () => console.log(`Serving exact production assets ${production} on http://127.0.0.1:5180`))
