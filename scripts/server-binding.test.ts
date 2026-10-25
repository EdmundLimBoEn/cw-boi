import { expect, test } from 'bun:test'
import { parseArgs } from 'node:util'
import { scripts } from '../package.json'

for (const command of ['dev', 'preview'] as const) {
  test(`${command} script explicitly restricts the server to loopback`, () => {
    const { values } = parseArgs({
      args: scripts[command].split(/\s+/),
      options: { host: { type: 'string' } },
      allowPositionals: true,
    })
    // Reject wildcard, absent, and bare --host options that could expose the server.
    expect(['127.0.0.1', '::1', 'localhost']).toContain(values.host)
  })
}
