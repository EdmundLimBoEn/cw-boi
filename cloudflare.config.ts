import { defineConfig } from 'cf/config'

export default defineConfig({
  accountId: '2132f9cf03d557cc3be87ca4208e683d',
  worker: {
    name: 'cw-boi',
    compatibilityDate: '2026-10-03',
    domains: ['cw-boi.sillyapps.co'],
    workersDev: false,
    previewUrls: false,
    assets: { notFoundHandling: 'single-page-application' },
  },
})
