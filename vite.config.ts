import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { cloudflare } from '@cloudflare/vite-plugin'

export default defineConfig({
  plugins: [react(), cloudflare({ assetsOnly: true, types: { generate: false } })],
  worker: { format: 'es' },
  server: { port: 5173, strictPort: true },
})
