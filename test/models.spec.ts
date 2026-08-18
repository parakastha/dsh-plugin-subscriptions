/**
 * Login-gated model catalogs and live discovery: `listModels` returns [] when
 * logged out, maps discovered catalogs when logged in (via an injected fetch,
 * no network), and falls back to the static catalog with a warning on
 * discovery failure.
 */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { randomUUID } from 'node:crypto'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { CodexAdapter, fetchCodexModels } from '../src/providers/codex.js'
import { GrokAdapter } from '../src/providers/grok.js'
import { ClaudeAdapter } from '../src/providers/claude.js'
import { TokenManager } from '../src/providers/common.js'
import type { FetchFn } from '../src/providers/common.js'
import type { ClaudeSession, CodexSession, GrokSession } from '../src/auth/store.js'
import type { GenerateOptions, LlmResolvedModelInfo } from '@deepseek-ai/dsh-llm'

type ServiceTierModel = LlmResolvedModelInfo & { serviceTiers?: unknown }
type ServiceTierOptions = GenerateOptions & { serviceTier?: string }

const STATIC_CODEX = [{ id: 'gpt-5.6-terra', name: 'GPT-5.6 Terra' }]
const STATIC_CLAUDE = [
  { id: 'claude-opus-5', name: 'Claude Opus 5' },
  { id: 'claude-fable-5', name: 'Claude Fable 5' },
  { id: 'claude-sonnet-5', name: 'Claude Sonnet 5' },
  { id: 'claude-haiku-4-5', name: 'Claude Haiku 4.5' },
]
const STATIC_GROK = [{ id: 'grok-4', name: 'Grok 4' }]

const codexSession: CodexSession = {
  accessToken: 'at',
  refreshToken: 'rt',
  expiresAt: Date.now() + 3_600_000,
  accountId: 'acct-1',
}
const claudeSession: ClaudeSession = {
  accessToken: 'at',
  refreshToken: 'rt',
  expiresAt: Date.now() + 3_600_000,
  scopes: 'scope',
}
const grokSession: GrokSession = {
  accessToken: 'at',
  refreshToken: 'rt',
  expiresAt: Date.now() + 3_600_000,
  tokenEndpoint: 'https://auth.x.ai/token',
}

/** A TokenManager over an in-memory session; refresh never fires in these tests. */
function memoryTokens<S extends { accessToken: string; refreshToken: string; expiresAt: number }>(
  initial: S | undefined,
): TokenManager<S> {
  let stored = initial
  return new TokenManager<S>({
    displayName: 'Test',
    preemptMs: 0,
    load: () => Promise.resolve(stored),
    save: (session) => {
      stored = session
      return Promise.resolve()
    },
    remove: () => {
      stored = undefined
      return Promise.resolve()
    },
    refresh: session => Promise.resolve(session),
    isPermanent: () => false,
  })
}

/** A fetch implementation answering one JSON payload; records invocation count. */
function fakeFetch(payload: unknown, status = 200): { fetchFn: FetchFn; calls: () => number } {
  let calls = 0
  const fetchFn: FetchFn = (() => {
    calls += 1
    return Promise.resolve(new Response(JSON.stringify(payload), { status }))
  }) as FetchFn
  return { fetchFn, calls: () => calls }
}

function codexAdapter(overrides: {
  session?: CodexSession
  discovery?: boolean
  fetchFn?: FetchFn
  warnings?: string[]
}): CodexAdapter {
  return new CodexAdapter({
    models: STATIC_CODEX,
    streamIdleTimeoutMs: 1000,
    tokens: memoryTokens(overrides.session),
    discovery: overrides.discovery ?? true,
    ...overrides.fetchFn === undefined ? {} : { fetchFn: overrides.fetchFn },
    ...overrides.warnings === undefined
      ? {}
      : { onWarn: (message: string) => { overrides.warnings?.push(message) } },
  })
}

function claudeAdapter(session: ClaudeSession | undefined, models = STATIC_CLAUDE): ClaudeAdapter {
  return new ClaudeAdapter({
    models,
    streamIdleTimeoutMs: 1000,
    tokens: memoryTokens(session),
    onWarn: () => {},
    maxConcurrentRequests: 1,
    maxStepsPerTurn: 8,
    usageWarnPercent: 70,
    usageBlockPercent: 85,
    usageCacheTtlMs: 5000,
    sessionStateTtlMs: 60_000,
    sessionStatePath: join(tmpdir(), `dsh-plugin-subscriptions-models-${randomUUID()}.json`),
    cliMaxTurns: 1,
  })
}

const CODEX_MODELS_PAYLOAD = {
  models: [
    {
      slug: 'gpt-5.6-terra',
      display_name: 'GPT-5.6 Terra',
      description: 'newest',
      context_window: 500_000,
      supported_reasoning_levels: [
        { effort: 'low', description: 'fast' },
        { effort: 'high', description: 'thorough' },
        { effort: 'max' },
        { effort: 'ultra' },
      ],
      default_reasoning_level: 'high',
      visibility: 'list',
      priority: 2,
    },
    {
      slug: 'gpt-5.3-codex-spark',
      display_name: 'GPT-5.3 Codex Spark',
      visibility: 'list',
      priority: 1,
    },
    { slug: 'gpt-5.5', display_name: 'Legacy GPT-5.5', visibility: 'list', priority: 0 },
    { slug: 'gpt-hidden', display_name: 'Hidden', visibility: 'hide', priority: 0 },
    { slug: 'gpt-none', display_name: 'None', visibility: 'none', priority: 0 },
  ],
}

test('listModels returns [] when logged out (codex, claude, grok)', async () => {
  const codex = codexAdapter({})
  assert.deepEqual(await codex.listModels('codex'), [])
  const claude = claudeAdapter(undefined)
  assert.deepEqual(await claude.listModels('claude'), [])
  const grok = new GrokAdapter({
    models: STATIC_GROK,
    streamIdleTimeoutMs: 1000,
    tokens: memoryTokens<GrokSession>(undefined),
    discovery: true,
    fetchFn: fakeFetch({ data: [{ id: 'grok-9' }] }).fetchFn,
  })
  assert.deepEqual(await grok.listModels('grok'), [])
})

test('codex discovery maps, filters hidden entries, and sorts by priority', async () => {
  const { fetchFn, calls } = fakeFetch(CODEX_MODELS_PAYLOAD)
  const adapter = codexAdapter({ session: codexSession, fetchFn })
  const models = await adapter.listModels('codex')
  assert.deepEqual(models.map(model => model.id), ['gpt-5.3-codex-spark', 'gpt-5.6-terra'])
  assert.equal(models[1].name, 'GPT-5.6 Terra')
  assert.equal(models[1].description, 'newest')
  // The TTL cache serves the second call without another fetch.
  await adapter.listModels('codex')
  assert.equal(calls(), 1)
})

test('resolveModel prefers discovered context window and reasoning efforts', async () => {
  const { fetchFn } = fakeFetch(CODEX_MODELS_PAYLOAD)
  const adapter = codexAdapter({ session: codexSession, fetchFn })
  await adapter.listModels('codex')
  const resolved = await adapter.resolveModel('codex', 'gpt-5.6-terra')
  assert.equal(resolved.context?.contextWindow, 500_000)
  assert.deepEqual(
    resolved.reasoning?.efforts.map(effort => effort.id),
    ['low', 'high', 'max', 'ultra'],
  )
  assert.equal(resolved.reasoning?.defaultEffort, 'high')
  // A current model that the catalog did not advertise falls back to static defaults.
  const fallback = await adapter.resolveModel('codex', 'gpt-5.3-codex')
  assert.equal(fallback.context?.contextWindow, 400_000)
  assert.deepEqual(fallback.reasoning?.efforts.map(effort => effort.id), ['low', 'medium', 'high', 'xhigh'])
})

test('codex gpt-5.6 fallback exposes the documented max level and live ultra variants', async () => {
  const adapter = codexAdapter({ session: codexSession, discovery: false })
  assert.deepEqual(
    (await adapter.resolveModel('codex', 'gpt-5.6-terra')).reasoning?.efforts.map(effort => effort.id),
    ['low', 'medium', 'high', 'xhigh', 'max', 'ultra'],
  )
  assert.deepEqual(
    (await adapter.resolveModel('codex', 'gpt-5.6-luna')).reasoning?.efforts.map(effort => effort.id),
    ['low', 'medium', 'high', 'xhigh', 'max'],
  )
})

test('codex rejects retired models before provider I/O', async () => {
  const adapter = codexAdapter({ session: codexSession, discovery: false })
  await assert.rejects(adapter.resolveModel('codex', 'gpt-5.5'), /retired in this profile/)
  await assert.rejects(async () => {
    for await (const _ of adapter.stream({ provider: 'codex', model: 'gpt-5.5', messages: [] })) {
      void _
    }
  }, /retired in this profile/)
})

test('codex exposes supported speeds and sends each selected tier on the wire', async () => {
  const requestBodies: Record<string, unknown>[] = []
  const requestFetch: FetchFn = async (_input, init) => {
    requestBodies.push(JSON.parse(String(init?.body)) as Record<string, unknown>)
    return new Response(JSON.stringify({ error: { message: 'test request' } }), { status: 400 })
  }
  const adapter = new CodexAdapter({
    models: STATIC_CODEX,
    streamIdleTimeoutMs: 1000,
    tokens: memoryTokens(codexSession),
    discovery: false,
    fetchFn: requestFetch,
  })
  const fastModel = await adapter.resolveModel('codex', 'gpt-5.6-terra') as ServiceTierModel
  assert.deepEqual(fastModel.serviceTiers, {
    tiers: [{
      id: 'priority',
      name: 'Fast',
      description: 'Faster responses with higher subscription usage',
    }],
  })
  const sparkModel = await adapter.resolveModel('codex', 'gpt-5.3-codex-spark') as ServiceTierModel
  assert.equal(sparkModel.serviceTiers, undefined)

  for (const request of [
    { model: 'gpt-5.6-terra', serviceTier: 'priority' },
    { model: 'gpt-5.6-terra' },
  ] satisfies { model: string; serviceTier?: string }[]) {
    const options: ServiceTierOptions = { provider: 'codex', ...request, messages: [] }
    await assert.rejects(async () => {
      for await (const _ of adapter.stream(options)) {
        void _
      }
    }, /codex API/)
  }
  assert.equal(requestBodies[0]?.service_tier, 'priority')
  assert.equal(requestBodies[1]?.service_tier, 'default')

  await assert.rejects(async () => {
    const options: ServiceTierOptions = {
      provider: 'codex', model: 'gpt-5.3-codex-spark', serviceTier: 'priority', messages: [],
    }
    for await (const _ of adapter.stream(options)) void _
  }, /does not support service tier/)
})

test('codex retains a listed model\'s efforts after the catalog refresh TTL', async () => {
  const { fetchFn } = fakeFetch(CODEX_MODELS_PAYLOAD)
  const adapter = codexAdapter({ session: codexSession, fetchFn })
  const actualNow = Date.now
  let now = actualNow()
  Date.now = () => now
  try {
    await adapter.listModels('codex')
    now += 5 * 60_000
    const resolved = await adapter.resolveModel('codex', 'gpt-5.6-terra')
    assert.deepEqual(
      resolved.reasoning?.efforts.map(effort => effort.id),
      ['low', 'high', 'max', 'ultra'],
    )
  } finally {
    Date.now = actualNow
  }
})

test('codex discovery failure falls back to the static catalog with a warning', async () => {
  const warnings: string[] = []
  const { fetchFn } = fakeFetch({ error: 'boom' }, 500)
  const adapter = codexAdapter({ session: codexSession, fetchFn, warnings })
  const models = await adapter.listModels('codex')
  assert.deepEqual(models.map(model => model.id), ['gpt-5.6-terra'])
  assert.equal(warnings.length, 1)
  assert.match(warnings[0], /codex model discovery failed/)
})

test('codex config override wins over discovery entirely', async () => {
  const { fetchFn, calls } = fakeFetch(CODEX_MODELS_PAYLOAD)
  const adapter = codexAdapter({ session: codexSession, fetchFn, discovery: false })
  const models = await adapter.listModels('codex')
  assert.deepEqual(models.map(model => model.id), ['gpt-5.6-terra'])
  assert.equal(calls(), 0)
})

test('grok discovery maps the data array', async () => {
  const { fetchFn } = fakeFetch({ data: [{ id: 'grok-4-1' }, { id: 'grok-code-2' }, { nope: true }] })
  const adapter = new GrokAdapter({
    models: STATIC_GROK,
    streamIdleTimeoutMs: 1000,
    tokens: memoryTokens(grokSession),
    discovery: true,
    fetchFn,
  })
  const models = await adapter.listModels('grok')
  assert.deepEqual(models.map(model => model.id), ['grok-4-1', 'grok-code-2'])
})

test('claude serves only current models and only exposes officially supported efforts', async () => {
  const claude = claudeAdapter(claudeSession)
  const models = await claude.listModels('claude')
  assert.deepEqual(models.map(model => model.id), [
    'claude-opus-5', 'claude-fable-5', 'claude-sonnet-5', 'claude-haiku-4-5',
  ])
  const resolved = await claude.resolveModel('claude', 'claude-opus-5')
  assert.deepEqual(resolved.reasoning?.efforts.map(effort => effort.id), ['low', 'medium', 'high', 'xhigh', 'max'])
  assert.equal(resolved.reasoning?.defaultEffort, 'high')
  assert.deepEqual(
    (await claude.resolveModel('claude', 'claude-fable-5')).reasoning?.efforts.map(effort => effort.id),
    ['low', 'medium', 'high', 'xhigh', 'max'],
  )
  assert.equal((await claude.resolveModel('claude', 'claude-haiku-4-5')).reasoning, undefined)
  await assert.rejects(claude.resolveModel('claude', 'claude-opus-4-5'), /retired in this profile/)
})

test('fetchCodexModels tolerates entries without visibility or priority', async () => {
  const models = await fetchCodexModels(codexSession, fakeFetch({
    models: [{ slug: 'gpt-5.6-luna', display_name: 'GPT-5.6 Luna' }],
  }).fetchFn)
  assert.deepEqual(models, [{ id: 'gpt-5.6-luna', name: 'GPT-5.6 Luna' }])
})

test('modalities: codex and claude declare image input; grok gates text-only models', async () => {
  const codex = codexAdapter({ session: codexSession, discovery: false })
  const codexModels = await codex.listModels('codex')
  assert.deepEqual(codexModels[0].inputModalities, ['text', 'image'])
  const codexResolved = await codex.resolveModel('codex', 'gpt-5.6-terra')
  assert.deepEqual(codexResolved.inputModalities, ['text', 'image'])

  const claude = claudeAdapter(claudeSession)
  assert.deepEqual((await claude.listModels('claude'))[0].inputModalities, ['text', 'image'])

  const grok = new GrokAdapter({
    models: [{ id: 'grok-4' }, { id: 'grok-code-fast-1' }, { id: 'grok-embedding-1' }],
    streamIdleTimeoutMs: 1000,
    tokens: memoryTokens(grokSession),
    discovery: false,
  })
  const grokModels = await grok.listModels('grok')
  assert.deepEqual(grokModels[0].inputModalities, ['text', 'image'])
  assert.deepEqual(grokModels[1].inputModalities, ['text'])
  assert.deepEqual(grokModels[2].inputModalities, ['text'])
  assert.deepEqual((await grok.resolveModel('grok', 'grok-4.6')).inputModalities, ['text', 'image'])
  assert.deepEqual((await grok.resolveModel('grok', 'grok-code-fast-1')).inputModalities, ['text'])
})

test('modalities: config entry inputModalities win over the provider default', async () => {
  const adapter = new CodexAdapter({
    models: [{ id: 'gpt-5.6-terra', inputModalities: ['text'] }],
    streamIdleTimeoutMs: 1000,
    tokens: memoryTokens(codexSession),
    discovery: false,
  })
  assert.deepEqual((await adapter.listModels('codex'))[0].inputModalities, ['text'])
  assert.deepEqual((await adapter.resolveModel('codex', 'gpt-5.6-terra')).inputModalities, ['text'])
})

test('grok discovery drops generation and embedding models', async () => {
  const { fetchFn } = fakeFetch({
    data: [
      { id: 'grok-4.6' },
      { id: 'grok-build-0.1' },
      { id: 'grok-imagine-image' },
      { id: 'grok-imagine-image-2.0' },
      { id: 'grok-imagine-video-1.5' },
      { id: 'grok-embedding-1' },
    ],
  })
  const adapter = new GrokAdapter({
    models: STATIC_GROK,
    streamIdleTimeoutMs: 1000,
    tokens: memoryTokens(grokSession),
    discovery: true,
    fetchFn,
  })
  const models = await adapter.listModels('grok')
  assert.deepEqual(models.map(model => model.id), ['grok-4.6', 'grok-build-0.1'])
})

/** The api.x.ai model list: authoritative for which models exist. */
const GROK_API_PAYLOAD = { data: [{ id: 'grok-4.6' }, { id: 'grok-4.5' }, { id: 'grok-build-0.1' }] }
/**
 * The CLI catalog: contributes reasoning/name/context per model. grok-4.6
 * marks two levels `default: true` like the live payload does, so the test
 * proves the top-level `reasoning_effort` field wins.
 */
const GROK_CLI_PAYLOAD = {
  data: [
    {
      id: 'grok-4.6',
      name: 'Grok 4.6',
      description: 'frontier',
      context_window: 500_000,
      supports_reasoning_effort: true,
      reasoning_effort: 'high',
      reasoning_efforts: [
        { value: 'xhigh', label: 'Extra High Effort', default: true },
        { value: 'high', label: 'High Effort', description: 'extensive reasoning', default: true },
        { value: 'medium', label: 'Medium Effort' },
        { value: 'low', label: 'Low Effort' },
      ],
    },
    {
      id: 'grok-4.5',
      name: 'Grok 4.5',
      context_window: 500_000,
      supports_reasoning_effort: true,
      reasoning_effort: 'high',
      reasoning_efforts: [
        { value: 'high', label: 'High Effort', default: true },
        { value: 'medium', label: 'Medium Effort' },
        { value: 'low', label: 'Low Effort' },
      ],
    },
  ],
}

/** A fetch dispatching on URL: the CLI catalog host vs the api.x.ai list. */
function grokDualFetch(cliPayload: unknown = GROK_CLI_PAYLOAD, cliStatus = 200): FetchFn {
  return ((url: unknown) => {
    const isCliCatalog = String(url).includes('cli-chat-proxy')
    return Promise.resolve(new Response(
      JSON.stringify(isCliCatalog ? cliPayload : GROK_API_PAYLOAD),
      { status: isCliCatalog ? cliStatus : 200 },
    ))
  }) as FetchFn
}

test('grok discovery merges CLI-catalog reasoning metadata by model id', async () => {
  const adapter = new GrokAdapter({
    models: STATIC_GROK,
    streamIdleTimeoutMs: 1000,
    tokens: memoryTokens(grokSession),
    discovery: true,
    fetchFn: grokDualFetch(),
  })
  const models = await adapter.listModels('grok')
  assert.deepEqual(models.map(model => model.name), ['Grok 4.6', 'Grok 4.5', 'grok-build-0.1'])
  assert.equal(models[0].description, 'frontier')

  const g46 = await adapter.resolveModel('grok', 'grok-4.6')
  assert.deepEqual(g46.reasoning?.efforts.map(effort => effort.id), ['xhigh', 'high', 'medium', 'low'])
  assert.equal(g46.reasoning?.efforts[1].name, 'High Effort')
  assert.equal(g46.reasoning?.efforts[1].description, 'extensive reasoning')
  // The top-level reasoning_effort wins over the double default flags.
  assert.equal(g46.reasoning?.defaultEffort, 'high')
  assert.equal(g46.context?.contextWindow, 500_000)

  const g45 = await adapter.resolveModel('grok', 'grok-4.5')
  assert.deepEqual(g45.reasoning?.efforts.map(effort => effort.id), ['high', 'medium', 'low'])

  // A model the CLI catalog does not cover exposes no efforts.
  const build = await adapter.resolveModel('grok', 'grok-build-0.1')
  assert.equal(build.reasoning, undefined)
  assert.equal(build.context?.contextWindow, 256_000)
})

test('grok discovery survives a CLI catalog failure with a warning', async () => {
  const warnings: string[] = []
  const adapter = new GrokAdapter({
    models: STATIC_GROK,
    streamIdleTimeoutMs: 1000,
    tokens: memoryTokens(grokSession),
    discovery: true,
    fetchFn: grokDualFetch({ error: 'boom' }, 500),
    onWarn: message => warnings.push(message),
  })
  const models = await adapter.listModels('grok')
  assert.deepEqual(models.map(model => model.id), ['grok-4.6', 'grok-4.5', 'grok-build-0.1'])
  assert.equal(warnings.length, 1)
  assert.match(warnings[0], /CLI catalog fetch failed/)
  assert.equal((await adapter.resolveModel('grok', 'grok-4.6')).reasoning, undefined)
})

test('grok entries without reasoning support expose no efforts', async () => {
  const cliPayload = {
    data: [
      { id: 'grok-4.6', supports_reasoning_effort: false },
      { id: 'grok-4.5', supports_reasoning_effort: true, reasoning_efforts: [] },
    ],
  }
  const adapter = new GrokAdapter({
    models: STATIC_GROK,
    streamIdleTimeoutMs: 1000,
    tokens: memoryTokens(grokSession),
    discovery: true,
    fetchFn: grokDualFetch(cliPayload),
  })
  await adapter.listModels('grok')
  assert.equal((await adapter.resolveModel('grok', 'grok-4.6')).reasoning, undefined)
  assert.equal((await adapter.resolveModel('grok', 'grok-4.5')).reasoning, undefined)
})

test('empty discovery payload falls back to the static catalog with a warning', async () => {
  const warnings: string[] = []
  const { fetchFn } = fakeFetch({ models: [] })
  const adapter = codexAdapter({ session: codexSession, fetchFn, warnings })
  const models = await adapter.listModels('codex')
  assert.deepEqual(models.map(model => model.id), ['gpt-5.6-terra'])
  assert.equal(warnings.length, 1)
  assert.match(warnings[0], /empty catalog/)
})
