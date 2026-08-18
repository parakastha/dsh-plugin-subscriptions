/**
 * Claude Pro/Max subscription provider: OAuth against claude.ai /
 * platform.claude.com with the Claude Code client id, and streaming through
 * the Claude Agent SDK so one native Claude session can survive many DSH turns.
 */

import { CallId, EMPTY_RESPONSE_CODE, LlmAdapter, LlmError } from '@deepseek-ai/dsh-llm'
import type {
  GenerateOptions,
  LlmModelInfo,
  LlmProviderInfo,
  LlmResolvedModelInfo,
  StreamChunk,
  TokenUsage,
} from '@deepseek-ai/dsh-llm'
import {
  createSdkMcpServer,
  query as claudeAgentQuery,
} from '@anthropic-ai/claude-agent-sdk'
import type { SDKUserMessage } from '@anthropic-ai/claude-agent-sdk'
import { createHash } from 'node:crypto'
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'
import { tmpdir } from 'node:os'
import { z } from 'zod'
import type { FlowSpec } from '../auth/oauth-flow.js'
import type { ClaudeSession } from '../auth/store.js'
import type { AttachmentStore } from '@deepseek-ai/dsh-attachment'
import { resolveImages } from '../translate/resolved.js'
import type { TranslatableMessage, TranslatableBlock } from '../translate/resolved.js'
import {
  idleWatchdog,
  mapFetchFailure,
  oauthEndpointError,
  OAuthEndpointError,
  TokenManager,
} from './common.js'
import type { FetchFn, ModelEntry, ProviderUsage, UsageWindow } from './common.js'

export const CLAUDE_CLIENT_ID = '9d1c250a-e61b-44d9-88ed-5944d1962f5e'
export const CLAUDE_AUTHORIZE_URL = 'https://claude.ai/oauth/authorize'
export const CLAUDE_TOKEN_URL = 'https://platform.claude.com/v1/oauth/token'
export const CLAUDE_API_URL = 'https://api.anthropic.com/v1/messages?beta=true'
export const CLAUDE_PROFILE_URL = 'https://api.anthropic.com/api/oauth/profile'
const CLAUDE_SCOPE = 'org:create_api_key user:profile user:inference user:sessions:claude_code user:mcp_servers user:file_upload'
const CLAUDE_CALLBACK_PATH = '/callback'
const CLAUDE_CONTEXT_WINDOW = 200_000
const CLAUDE_DEFAULT_MAX_TOKENS = 32_000
/** Refresh when the access token has less than this much life left. */
export const CLAUDE_PREEMPT_MS = 5 * 60_000

/**
 * The subscription endpoint only serves requests presenting as Claude Code,
 * so these headers impersonate the CLI; the harness attribution user-agent
 * cannot be sent here (one user-agent slot, and the CLI's wins).
 */
const CLAUDE_CLI_USER_AGENT = 'claude-cli/2.1.97 (external, cli)'
const MCP_TOOL_PREFIX = 'mcp__dsh__'
const BRIDGE_SYSTEM = 'You are operating inside DeepSeek Harness. Use only tools whose names start with "mcp__dsh__". Never use Claude Code built-in tools. DeepSeek Harness executes tool calls and returns their results.'

/** Static claude flow facts for the OAuth flow engine. */
export const claudeFlow: FlowSpec = {
  callbackPath: CLAUDE_CALLBACK_PATH,
  // The redirect URI embeds the port, so it must be an ephemeral one.
  listen: { host: 'localhost', ports: [0] },
  buildAuthorizeUrl({ redirectUri, state, pkce }) {
    const params = new URLSearchParams({
      code: 'true',
      client_id: CLAUDE_CLIENT_ID,
      response_type: 'code',
      redirect_uri: redirectUri,
      scope: CLAUDE_SCOPE,
      code_challenge: pkce.challenge,
      code_challenge_method: 'S256',
      state,
    })
    return `${CLAUDE_AUTHORIZE_URL}?${params.toString()}`
  },
}

/** Token endpoint response shape (subset). */
interface ClaudeTokenResponse {
  access_token?: string
  refresh_token?: string
  expires_in?: number
  scope?: string
}

/** Best-effort account profile; login must not fail when this does. */
async function fetchClaudeProfile(accessToken: string): Promise<Pick<ClaudeSession, 'emailAddress' | 'subscriptionType'>> {
  try {
    const response = await fetch(CLAUDE_PROFILE_URL, {
      headers: { authorization: `Bearer ${accessToken}` },
    })
    if (!response.ok) return {}
    const profile = await response.json() as Record<string, unknown>
    const account = typeof profile.account === 'object' && profile.account !== null
      ? profile.account as Record<string, unknown>
      : {}
    const email = profile.emailAddress ?? profile.email ?? account.email_address ?? account.email
    const subscription = profile.subscriptionType ?? profile.subscription_type ?? account.subscription_type
    return {
      ...typeof email === 'string' && email.length > 0 ? { emailAddress: email } : {},
      ...typeof subscription === 'string' && subscription.length > 0 ? { subscriptionType: subscription } : {},
    }
  } catch {
    // Profile lookup is decorative; only the token exchange owns login success.
    return {}
  }
}

/** Build a session from a token response. */
async function claudeSession(
  tokens: ClaudeTokenResponse,
  fallbackRefreshToken: string | undefined,
  withProfile: boolean,
): Promise<ClaudeSession> {
  if (typeof tokens.access_token !== 'string' || tokens.access_token.length === 0) {
    throw new Error('claude token endpoint returned no access token')
  }
  const refreshToken = tokens.refresh_token ?? fallbackRefreshToken
  if (refreshToken === undefined) throw new Error('claude token endpoint returned no refresh token')
  if (typeof tokens.expires_in !== 'number' || tokens.expires_in <= 0) {
    throw new Error('claude token endpoint returned no usable expiry')
  }
  const profile = withProfile ? await fetchClaudeProfile(tokens.access_token) : {}
  return {
    accessToken: tokens.access_token,
    refreshToken,
    expiresAt: Date.now() + tokens.expires_in * 1000,
    scopes: tokens.scope ?? CLAUDE_SCOPE,
    ...profile,
  }
}

/**
 * Exchange an authorization code for a claude session (JSON grant).
 * @param code - the authorization code from the callback.
 * @param verifier - the PKCE verifier minted for the attempt.
 * @param redirectUri - the attempt's redirect URI.
 * @param state - the attempt's state (echoed to the token endpoint).
 * @returns the session to store.
 */
export async function exchangeClaudeCode(
  code: string,
  verifier: string,
  redirectUri: string,
  state: string,
): Promise<ClaudeSession> {
  const response = await fetch(CLAUDE_TOKEN_URL, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      grant_type: 'authorization_code',
      code,
      redirect_uri: redirectUri,
      client_id: CLAUDE_CLIENT_ID,
      code_verifier: verifier,
      state,
    }),
  })
  if (!response.ok) throw await oauthEndpointError(response, 'claude')
  return claudeSession(await response.json() as ClaudeTokenResponse, undefined, true)
}

/**
 * Refresh a claude session (JSON grant echoing the issued scope).
 * @param session - the stored session.
 * @returns the fresh session to store.
 */
export async function refreshClaude(session: ClaudeSession): Promise<ClaudeSession> {
  const response = await fetch(CLAUDE_TOKEN_URL, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      grant_type: 'refresh_token',
      refresh_token: session.refreshToken,
      client_id: CLAUDE_CLIENT_ID,
      scope: session.scopes,
    }),
  })
  if (!response.ok) throw await oauthEndpointError(response, 'claude')
  const next = await claudeSession(await response.json() as ClaudeTokenResponse, session.refreshToken, false)
  return {
    ...next,
    ...session.emailAddress === undefined ? {} : { emailAddress: session.emailAddress },
    ...session.subscriptionType === undefined ? {} : { subscriptionType: session.subscriptionType },
  }
}

/**
 * Whether a claude refresh failure means the login is permanently gone.
 * @param error - the thrown refresh error.
 * @returns true when re-login is the only fix.
 */
export function isClaudePermanentRefreshError(error: unknown): boolean {
  return error instanceof OAuthEndpointError
    && (error.oauthCode === 'invalid_grant' || error.oauthCode === 'invalid_token')
}

export const CLAUDE_USAGE_URL = 'https://api.anthropic.com/api/oauth/usage'

/** RFC3339 `resets_at` value → epoch ms, or undefined when absent/unparsable. */
function claudeResetsAt(value: unknown): number | undefined {
  if (typeof value !== 'string' || value.length === 0) return undefined
  const parsed = Date.parse(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

/** Map one legacy `{utilization, resets_at}` bucket; undefined when null or unusable. */
function claudeLegacyWindow(value: unknown, kind: UsageWindow['kind'], scope?: string): UsageWindow | undefined {
  if (typeof value !== 'object' || value === null) return undefined
  const bucket = value as { utilization?: number; resets_at?: string }
  if (typeof bucket.utilization !== 'number' || !Number.isFinite(bucket.utilization)) return undefined
  const resetsAt = claudeResetsAt(bucket.resets_at)
  return {
    kind,
    ...scope === undefined ? {} : { scope },
    usedPercent: bucket.utilization,
    ...resetsAt === undefined ? {} : { resetsAt },
  }
}

/** One entry of the modern `limits` array (subset). */
interface ClaudeLimitEntry {
  kind?: string
  percent?: number
  resets_at?: string
  scope?: { model?: { display_name?: string } }
}

/** Map the modern `limits` array; empty when absent or carrying nothing usable. */
function claudeLimitsWindows(value: unknown): UsageWindow[] {
  if (!Array.isArray(value)) return []
  const windows: UsageWindow[] = []
  for (const raw of value) {
    if (typeof raw !== 'object' || raw === null) continue
    const entry = raw as ClaudeLimitEntry
    if (typeof entry.percent !== 'number' || !Number.isFinite(entry.percent)) continue
    const kind: UsageWindow['kind'] = entry.kind === 'session'
      ? 'session'
      : entry.kind === 'weekly_all' || entry.kind === 'weekly_scoped' ? 'weekly' : 'other'
    const scope = entry.scope?.model?.display_name
    const resetsAt = claudeResetsAt(entry.resets_at)
    windows.push({
      kind,
      ...typeof scope === 'string' && scope.length > 0 ? { scope } : {},
      usedPercent: entry.percent,
      ...resetsAt === undefined ? {} : { resetsAt },
    })
  }
  return windows
}

/**
 * Fetch the claude subscription usage from the OAuth usage endpoint (the
 * source of Claude Code's `/usage` screen). Newer responses carry a
 * structured `limits` array; older ones the flat `five_hour`/`seven_day*`
 * buckets — both shapes are read, the array winning when it has entries.
 * @param session - the stored session (used as-is; never refreshed here).
 * @param fetchFn - fetch implementation (injectable for tests).
 * @param signal - caller cancellation from the RPC transport.
 * @returns the mapped usage snapshot.
 */
export async function fetchClaudeUsage(
  session: ClaudeSession,
  fetchFn: FetchFn = fetch,
  signal?: AbortSignal,
): Promise<ProviderUsage> {
  const response = await fetchFn(CLAUDE_USAGE_URL, {
    headers: {
      'authorization': `Bearer ${session.accessToken}`,
      'anthropic-beta': 'oauth-2025-04-20',
      // Unrecognized clients are aggressively rate-limited on this endpoint,
      // so it presents as the CLI like every other subscription request.
      'user-agent': CLAUDE_CLI_USER_AGENT,
      'accept': 'application/json',
    },
    ...signal === undefined ? {} : { signal },
  })
  if (!response.ok) throw await oauthEndpointError(response, 'claude usage')
  const payload = await response.json() as Record<string, unknown>
  const modern = claudeLimitsWindows(payload.limits)
  if (modern.length > 0) return { supported: true, windows: modern }
  const windows: UsageWindow[] = []
  const legacy = [
    claudeLegacyWindow(payload.five_hour, 'session'),
    claudeLegacyWindow(payload.seven_day, 'weekly'),
    claudeLegacyWindow(payload.seven_day_opus, 'weekly', 'Opus'),
    claudeLegacyWindow(payload.seven_day_sonnet, 'weekly', 'Sonnet'),
  ]
  for (const window of legacy) {
    if (window !== undefined) windows.push(window)
  }
  return { supported: true, windows }
}

/** Constructor dependencies for {@link ClaudeAdapter}. */
export interface ClaudeAdapterOptions {
  models: readonly ModelEntry[]
  streamIdleTimeoutMs: number
  tokens: TokenManager<ClaudeSession>
  onWarn: (message: string) => void
  maxConcurrentRequests: number
  maxStepsPerTurn: number
  usageWarnPercent: number
  usageBlockPercent: number
  usageCacheTtlMs: number
  sessionStateTtlMs: number
  sessionStatePath: string
  cliMaxTurns: number
  /** Resolve the attachment service per request; absent means image requests fail loudly. */
  resolveAttachments?: () => AttachmentStore | undefined
}

/** The Claude 4.5 family accepts image input. */
const CLAUDE_MODALITIES: readonly ('text' | 'image')[] = ['text', 'image']

/** Claude wire adapter: one instance serves the `claude` provider route. */
export class ClaudeAdapter extends LlmAdapter {
  private readonly gate: RequestGate
  private readonly sessions = new Map<string, ClaudeCliSessionState>()
  private readonly sessionsLoaded: Promise<void>
  private persistQueue = Promise.resolve()
  private usageCache?: { fetchedAt: number; value: ProviderUsage }
  private readonly warnedWindows = new Set<string>()

  constructor(private readonly options: ClaudeAdapterOptions) {
    super()
    this.gate = new RequestGate(options.maxConcurrentRequests)
    this.sessionsLoaded = this.loadSessions()
  }

  override providerInfo(provider: string): LlmProviderInfo {
    return { id: provider, name: 'Claude (Subscription)' }
  }

  override async listModels(provider: string): Promise<readonly LlmModelInfo[]> {
    // Not logged in → empty catalog, so the web picker drops the provider.
    // Claude has no subscription model-list endpoint, so the static catalog
    // is the whole answer when logged in.
    if (!await this.options.tokens.hasSession()) return []
    return this.options.models.map(model => ({
      provider,
      id: model.id,
      name: model.name ?? model.id,
      inputModalities: model.inputModalities ?? CLAUDE_MODALITIES,
    }))
  }

  override resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    const configured = this.options.models.find(entry => entry.id === model)
    return Promise.resolve({
      provider,
      id: model,
      name: configured?.name ?? model,
      inputModalities: configured?.inputModalities ?? CLAUDE_MODALITIES,
      context: { contextWindow: configured?.contextWindow ?? CLAUDE_CONTEXT_WINDOW },
      defaultMaxTokens: configured?.maxTokens ?? CLAUDE_DEFAULT_MAX_TOKENS,
      // No reasoning metadata: the subscription endpoint's thinking support is
      // not exercised, so effort requests reject as unsupported.
    })
  }

  async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    const watchdog = idleWatchdog(options.signal, this.options.streamIdleTimeoutMs)
    const release = await this.gate.acquire(options.signal)
    try {
      const session = await this.guardUsage(watchdog.signal)
      yield* this.streamViaCli(options, session, watchdog)
    } catch (error: unknown) {
      throw mapFetchFailure('claude CLI bridge', error, watchdog, options.signal)
    } finally {
      release()
      watchdog.stop()
    }
  }

  /** Reject requests before they consume quota when an active usage window is too full. */
  private async guardUsage(signal: AbortSignal): Promise<ClaudeSession> {
    const now = Date.now()
    const session = await this.options.tokens.session()
    let usage = this.usageCache?.value
    if (usage === undefined || now - (this.usageCache?.fetchedAt ?? 0) >= this.options.usageCacheTtlMs) {
      try {
        usage = await fetchClaudeUsage(session, fetch, signal)
        this.usageCache = { fetchedAt: now, value: usage }
      } catch (error) {
        throw new LlmError(
          `Claude usage guard could not verify remaining quota; request was not sent: ${error instanceof Error ? error.message : String(error)}`,
          'USAGE_GUARD_UNAVAILABLE',
        )
      }
    }
    if (usage.windows === undefined || usage.windows.length === 0) {
      throw new LlmError(
        'Claude usage guard received no quota windows; request was not sent.',
        'USAGE_GUARD_UNAVAILABLE',
      )
    }
    const active = usage.windows?.filter(window => window.resetsAt === undefined || window.resetsAt > now) ?? []
    const blocking = active.find(window => window.usedPercent >= this.options.usageBlockPercent)
    if (blocking !== undefined) {
      const reset = blocking.resetsAt === undefined ? 'an unspecified reset time' : new Date(blocking.resetsAt).toLocaleString()
      throw new LlmError(
        `Claude usage guard stopped this request at ${blocking.usedPercent}% ${windowLabel(blocking)} usage; it resets at ${reset}. No model request was sent.`,
        'QUOTA_GUARD',
      )
    }
    for (const window of active.filter(item => item.usedPercent >= this.options.usageWarnPercent)) {
      const key = `${window.kind}:${window.scope ?? ''}:${window.resetsAt ?? ''}`
      if (this.warnedWindows.has(key)) continue
      this.warnedWindows.add(key)
      this.options.onWarn(`Claude ${windowLabel(window)} usage is ${window.usedPercent}%; new requests stop at ${this.options.usageBlockPercent}%.`)
    }
    return session
  }

  /** Stream one DSH step through a persistent Claude Agent SDK session. */
  private async *streamViaCli(options: GenerateOptions, auth: ClaudeSession, watchdog: ReturnType<typeof idleWatchdog>): AsyncIterable<StreamChunk> {
    await this.sessionsLoaded
    await this.expireSessions()
    const resolvedMessages = await resolveImages(options.messages, this.options.resolveAttachments?.(), watchdog.signal)
    const sessionKey = options.purpose === undefined && options.sessionId !== undefined
      ? String(options.sessionId)
      : undefined
    const previous = sessionKey === undefined ? undefined : this.sessions.get(sessionKey)
    const ids = options.messages.map(message => String(message.id))
    const systemSignature = signature(options.system ?? '')
    const toolsSignature = signature(JSON.stringify(options.tools ?? []))
    const canResume = previous !== undefined
      && previous.model === options.model
      && previous.systemSignature === systemSignature
      && previous.toolsSignature === toolsSignature
      && previous.inputMessageCount <= ids.length
      && previous.inputMessageSignature === signature(ids.slice(0, previous.inputMessageCount).join('\n'))
    const deltaStart = canResume ? previous.inputMessageCount : 0
    const delta = resolvedMessages.slice(deltaStart)
    const sendableDelta = delta.filter(message => message.role !== 'assistant')
    const resume = canResume && previous !== undefined && sendableDelta.length > 0
    const startsUserTurn = sendableDelta.some(message => message.content.some(block => block.type !== 'tool-result'))
    const stepsThisTurn = resume && !startsUserTurn ? previous.stepsThisTurn + 1 : 1
    if (stepsThisTurn > this.options.maxStepsPerTurn) {
      if (sessionKey !== undefined) {
        this.sessions.delete(sessionKey)
        await this.persistSessions()
      }
      throw new LlmError(
        `Claude step guard stopped this turn after ${this.options.maxStepsPerTurn} model calls. Start a new turn after checking the repeated tool loop.`,
        'STEP_GUARD',
      )
    }

    const controller = new AbortController()
    const onAbort = (): void => { controller.abort() }
    if (options.signal?.aborted === true) controller.abort()
    else options.signal?.addEventListener('abort', onAbort, { once: true })

    const mcpServers: Record<string, ReturnType<typeof createSdkMcpServer>> = {}
    const tools = options.tools ?? []
    if (tools.length > 0) {
      mcpServers.dsh = createSdkMcpServer({
        name: 'dsh',
        version: '1.0.0',
        tools: tools.map(tool => ({
          name: tool.name,
          description: tool.description,
          inputSchema: jsonSchemaToZod(tool.parameters),
          handler: async () => ({ content: [{ type: 'text' as const, text: 'DeepSeek Harness executes this tool out of band.' }], isError: true }),
        })),
      })
    }

    let claudeSessionId = resume ? previous.claudeSessionId : undefined
    let capturedToolCall = false
    let emitted = false
    let usageEmitted = false
    const commitState = async (): Promise<void> => {
      if (sessionKey === undefined || claudeSessionId === undefined) return
      this.sessions.set(sessionKey, {
        claudeSessionId,
        inputMessageCount: ids.length,
        inputMessageSignature: signature(ids.join('\n')),
        model: options.model,
        systemSignature,
        toolsSignature,
        stepsThisTurn,
        lastUsedAt: Date.now(),
      })
      await this.persistSessions()
    }

    try {
      const query = claudeAgentQuery({
        prompt: resume ? sdkUserMessages(sendableDelta) : sdkInitialPrompt(resolvedMessages),
        options: {
          model: options.model,
          cwd: tmpdir(),
          maxTurns: this.options.cliMaxTurns,
          ...resume && claudeSessionId !== undefined ? { resume: claudeSessionId } : {},
          systemPrompt: [options.system, BRIDGE_SYSTEM].filter(Boolean).join('\n\n'),
          mcpServers,
          strictMcpConfig: true,
          allowedTools: [],
          tools: [],
          skills: [],
          agents: {},
          plugins: [],
          settingSources: [],
          abortController: controller,
          env: {
            ...process.env,
            CLAUDE_CODE_OAUTH_TOKEN: auth.accessToken,
            CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC: '1',
            CLAUDE_AGENT_SDK_CLIENT_APP: 'dsh-plugin-subscriptions/0.3.0',
          },
        },
      })
      let index = 0
      for await (const message of query) {
        watchdog.pulse()
        if ('session_id' in message && typeof message.session_id === 'string') claudeSessionId = message.session_id
        if (message.type === 'assistant') {
          const usage = toTokenUsage(message.message.usage)
          if (usage !== undefined) {
            usageEmitted = true
            yield { type: 'usage', usage }
          }
          for (const block of message.message.content) {
            if (block.type === 'text' && block.text.length > 0) {
              emitted = true
              yield { type: 'block-start', index, blockType: 'text' }
              yield { type: 'text-delta', index, text: block.text }
              yield { type: 'block-end', index, block: { type: 'text', text: block.text } }
              index += 1
            } else if (block.type === 'tool_use' && block.name.startsWith(MCP_TOOL_PREFIX)) {
              emitted = true
              capturedToolCall = true
              const name = block.name.slice(MCP_TOOL_PREFIX.length)
              const id = CallId(String(block.id))
              const argumentsJson = JSON.stringify(block.input ?? {})
              yield { type: 'block-start', index, blockType: 'tool-call' }
              yield { type: 'tool-call-delta', index, id, name, argumentsDelta: argumentsJson }
              yield { type: 'block-end', index, block: { type: 'tool-call', id, name, arguments: argumentsJson } }
              index += 1
              controller.abort()
            }
          }
          if (capturedToolCall && message.message.stop_reason === 'tool_use') break
        } else if (message.type === 'result') {
          if (message.subtype !== 'success') {
            throw new LlmError(`Claude CLI error: ${message.errors.join('; ') || message.subtype}`, EMPTY_RESPONSE_CODE)
          }
          if (!usageEmitted) {
            const usage = toTokenUsage(message.usage)
            if (usage !== undefined) yield { type: 'usage', usage }
          }
        }
      }
      if (!emitted) throw new LlmError('Claude CLI returned no assistant content', EMPTY_RESPONSE_CODE)
      await commitState()
      yield { type: 'finish', reason: capturedToolCall ? { kind: 'tool-calls' } : { kind: 'stop' } }
    } catch (error) {
      if (capturedToolCall && controller.signal.aborted) {
        await commitState()
        yield { type: 'finish', reason: { kind: 'tool-calls' } }
      } else {
        if (sessionKey !== undefined) {
          this.sessions.delete(sessionKey)
          await this.persistSessions()
        }
        throw error
      }
    } finally {
      options.signal?.removeEventListener('abort', onAbort)
    }
  }

  private async expireSessions(): Promise<void> {
    const cutoff = Date.now() - this.options.sessionStateTtlMs
    let changed = false
    for (const [key, state] of this.sessions) {
      if (state.lastUsedAt < cutoff) {
        this.sessions.delete(key)
        changed = true
      }
    }
    if (changed) await this.persistSessions()
  }

  private async loadSessions(): Promise<void> {
    try {
      const parsed = JSON.parse(await readFile(this.options.sessionStatePath, 'utf8')) as unknown
      if (!isPersistedSessionFile(parsed)) return
      const cutoff = Date.now() - this.options.sessionStateTtlMs
      for (const [key, state] of Object.entries(parsed.sessions)) {
        if (isClaudeCliSessionState(state) && state.lastUsedAt >= cutoff) this.sessions.set(key, state)
      }
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== 'ENOENT') {
        this.options.onWarn(`Claude resume state could not be loaded; the next call may need one replay: ${error instanceof Error ? error.message : String(error)}`)
      }
    }
  }

  private persistSessions(): Promise<void> {
    const snapshot = JSON.stringify({ version: 1, sessions: Object.fromEntries(this.sessions) })
    this.persistQueue = this.persistQueue.then(async () => {
      const target = this.options.sessionStatePath
      const temporary = `${target}.${process.pid}.tmp`
      await mkdir(dirname(target), { recursive: true })
      await writeFile(temporary, snapshot, { encoding: 'utf8', mode: 0o600 })
      await rename(temporary, target)
    }).catch(error => {
      this.options.onWarn(`Claude resume state could not be saved; restart recovery is degraded: ${error instanceof Error ? error.message : String(error)}`)
    })
    return this.persistQueue
  }
}

interface ClaudeCliSessionState {
  claudeSessionId: string
  inputMessageCount: number
  inputMessageSignature: string
  model: string
  systemSignature: string
  toolsSignature: string
  stepsThisTurn: number
  lastUsedAt: number
}

/** Small FIFO semaphore; the default limit of one prevents parallel quota spikes. */
class RequestGate {
  private active = 0
  private readonly waiting: Array<() => void> = []

  constructor(private readonly limit: number) {}

  async acquire(signal?: AbortSignal): Promise<() => void> {
    if (signal?.aborted === true) throw signal.reason
    if (this.active >= this.limit) {
      await new Promise<void>((resolve, reject) => {
        const wake = (): void => {
          signal?.removeEventListener('abort', abort)
          resolve()
        }
        const abort = (): void => {
          const index = this.waiting.indexOf(wake)
          if (index >= 0) this.waiting.splice(index, 1)
          reject(signal?.reason)
        }
        this.waiting.push(wake)
        signal?.addEventListener('abort', abort, { once: true })
      })
    }
    this.active += 1
    let released = false
    return () => {
      if (released) return
      released = true
      this.active -= 1
      this.waiting.shift()?.()
    }
  }
}

function windowLabel(window: UsageWindow): string {
  return `${window.kind}${window.scope === undefined ? '' : ` ${window.scope}`}`
}

function signature(value: string): string {
  return createHash('sha256').update(value).digest('hex')
}

function isClaudeCliSessionState(value: unknown): value is ClaudeCliSessionState {
  if (typeof value !== 'object' || value === null) return false
  const state = value as Partial<ClaudeCliSessionState>
  return typeof state.claudeSessionId === 'string'
    && typeof state.inputMessageCount === 'number'
    && Number.isSafeInteger(state.inputMessageCount)
    && state.inputMessageCount >= 0
    && typeof state.inputMessageSignature === 'string'
    && typeof state.model === 'string'
    && typeof state.systemSignature === 'string'
    && typeof state.toolsSignature === 'string'
    && typeof state.stepsThisTurn === 'number'
    && Number.isSafeInteger(state.stepsThisTurn)
    && state.stepsThisTurn >= 0
    && typeof state.lastUsedAt === 'number'
    && Number.isFinite(state.lastUsedAt)
}

function isPersistedSessionFile(value: unknown): value is { version: 1; sessions: Record<string, unknown> } {
  if (typeof value !== 'object' || value === null) return false
  const file = value as { version?: unknown; sessions?: unknown }
  return file.version === 1 && typeof file.sessions === 'object' && file.sessions !== null && !Array.isArray(file.sessions)
}

function jsonSchemaToZod(schema: Record<string, unknown>): Record<string, z.ZodTypeAny> {
  const properties = typeof schema.properties === 'object' && schema.properties !== null
    ? schema.properties as Record<string, unknown>
    : {}
  const required = new Set(Array.isArray(schema.required) ? schema.required : [])
  const shape: Record<string, z.ZodTypeAny> = {}
  for (const [key, value] of Object.entries(properties)) {
    let field: z.ZodTypeAny
    try {
      field = z.fromJSONSchema(value as Parameters<typeof z.fromJSONSchema>[0])
    } catch {
      field = z.unknown()
    }
    shape[key] = required.has(key) ? field : field.optional()
  }
  return shape
}

function sdkImage(block: Extract<TranslatableBlock, { type: 'image' }>): Record<string, unknown> | undefined {
  if (!('dataBase64' in block)) return undefined
  return {
    type: 'image',
    source: {
      type: 'base64',
      media_type: block.mediaType,
      data: block.dataBase64,
    },
  }
}

function messageContent(message: TranslatableMessage): Record<string, unknown>[] {
  const content: Record<string, unknown>[] = []
  for (const block of message.content) {
    if (block.type === 'text' && block.text.length > 0) content.push({ type: 'text', text: block.text })
    else if (block.type === 'image') {
      const image = sdkImage(block)
      if (image !== undefined) content.push(image)
    } else if (block.type === 'tool-call') {
      content.push({ type: 'text', text: `[tool call: ${block.name}(${block.arguments})]` })
    }
  }
  return content
}

/** Resume input carries only new user/tool-result messages; Claude retains prior turns. */
async function* sdkUserMessages(messages: readonly TranslatableMessage[]): AsyncIterable<SDKUserMessage> {
  for (const message of messages) {
    const toolResults = message.content.filter(block => block.type === 'tool-result')
    const content = toolResults.length > 0
      ? toolResults.map(block => ({
          type: 'tool_result' as const,
          tool_use_id: String(block.toolCallId),
          content: block.content.map(part => part.type === 'text' ? part.text : '[image]').join(''),
          is_error: block.isError,
        }))
      : messageContent(message)
    if (content.length === 0) continue
    yield {
      type: 'user',
      message: { role: 'user', content } as SDKUserMessage['message'],
      parent_tool_use_id: null,
    }
  }
}

/** Full replay is used only for the first call or when DSH history diverges. */
async function* sdkInitialPrompt(messages: readonly TranslatableMessage[]): AsyncIterable<SDKUserMessage> {
  const content: Record<string, unknown>[] = []
  for (const message of messages) {
    if (message.role === 'system') continue
    content.push({ type: 'text', text: `<${message.role}>\n` })
    for (const block of message.content) {
      if (block.type === 'tool-result') {
        content.push({
          type: 'text',
          text: `[tool result for ${block.toolCallId}: ${block.content.map(part => part.type === 'text' ? part.text : '[image]').join('')}]`,
        })
      } else {
        content.push(...messageContent({ role: message.role, content: [block] }))
      }
    }
    content.push({ type: 'text', text: `\n</${message.role}>` })
  }
  yield {
    type: 'user',
    message: { role: 'user', content } as unknown as SDKUserMessage['message'],
    parent_tool_use_id: null,
  }
}

function toTokenUsage(usage: {
  input_tokens?: number
  output_tokens?: number
  cache_read_input_tokens?: number | null
  cache_creation_input_tokens?: number | null
} | undefined): TokenUsage | undefined {
  if (usage === undefined || typeof usage.input_tokens !== 'number') return undefined
  return {
    inputTokens: usage.input_tokens,
    outputTokens: usage.output_tokens ?? 0,
    cacheReadTokens: usage.cache_read_input_tokens ?? 0,
    cacheWriteTokens: usage.cache_creation_input_tokens ?? 0,
  }
}
