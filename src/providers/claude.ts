/**
 * Claude Pro/Max subscription provider: OAuth against claude.ai /
 * platform.claude.com with the Claude Code client id, and streaming through
 * the Claude Agent SDK so one native Claude session can survive many DSH turns.
 */

import {
  CallId,
  EMPTY_RESPONSE_CODE,
  isQuotaExceededError,
  LlmAdapter,
  LlmError,
  QUOTA_EXCEEDED_CODE,
  ReasoningEffortId,
} from '@deepseek-ai/dsh-llm'
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
import type { EffortLevel, SDKUserMessage } from '@anthropic-ai/claude-agent-sdk'
import { createHash, randomUUID } from 'node:crypto'
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
const CLAUDE_EFFORTS = new Set<EffortLevel>(['low', 'medium', 'high', 'xhigh', 'max'])
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
const PSEUDO_TOOL_CALL_PREFIX = '[tool call: '

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
  sessionStateTtlMs: number
  sessionStatePath: string
  cliMaxTurns: number
  claudeQuery?: typeof claudeAgentQuery
  /** Resolve the attachment service per request; absent means image requests fail loudly. */
  resolveAttachments?: () => AttachmentStore | undefined
}

/** The current Claude subscription models accept image input. */
const CLAUDE_MODALITIES: readonly ('text' | 'image')[] = ['text', 'image']

const CLAUDE_REASONING_EFFORTS = ['low', 'medium', 'high', 'xhigh', 'max'] as const

const CURRENT_CLAUDE_MODELS = new Set([
  'claude-opus-5',
  'claude-fable-5',
  'claude-sonnet-5',
  'claude-haiku-4-5',
])

/** Current Claude subscription families retained by this adapter. */
export function isCurrentClaudeModel(model: string): boolean {
  return CURRENT_CLAUDE_MODELS.has(model)
}

function assertCurrentClaudeModel(model: string): void {
  if (!isCurrentClaudeModel(model)) {
    throw new LlmError(
      `Claude model "${model}" is retired in this profile; select Claude Opus 5, Fable 5, Sonnet 5, or Haiku 4.5.`,
      'UNSUPPORTED_MODEL',
    )
  }
}

/** Anthropic exposes effort for Opus, Fable, and Sonnet; Haiku has no selector. */
function claudeReasoningEfforts(model: string): readonly string[] | undefined {
  return model === 'claude-opus-5' || model === 'claude-fable-5' || model === 'claude-sonnet-5'
    ? CLAUDE_REASONING_EFFORTS
    : undefined
}

/** Claude wire adapter: one instance serves the `claude` provider route. */
export class ClaudeAdapter extends LlmAdapter {
  private readonly gate: RequestGate
  private readonly sessions = new Map<string, ClaudeCliSessionState>()
  private readonly sessionsLoaded: Promise<void>
  private persistQueue = Promise.resolve()

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
    return this.options.models.filter(model => isCurrentClaudeModel(model.id)).map(model => ({
      provider,
      id: model.id,
      name: model.name ?? model.id,
      inputModalities: model.inputModalities ?? CLAUDE_MODALITIES,
    }))
  }

  override resolveModel(provider: string, model: string): Promise<LlmResolvedModelInfo> {
    try {
      assertCurrentClaudeModel(model)
    } catch (error) {
      return Promise.reject(error)
    }
    const configured = this.options.models.find(entry => entry.id === model)
    const efforts = claudeReasoningEfforts(model)?.map(effort => ({
      id: ReasoningEffortId(effort),
      name: effort === 'xhigh' ? 'Extra High' : effort.charAt(0).toUpperCase() + effort.slice(1),
    }))
    return Promise.resolve({
      provider,
      id: model,
      name: configured?.name ?? model,
      inputModalities: configured?.inputModalities ?? CLAUDE_MODALITIES,
      context: { contextWindow: configured?.contextWindow ?? CLAUDE_CONTEXT_WINDOW },
      defaultMaxTokens: configured?.maxTokens ?? CLAUDE_DEFAULT_MAX_TOKENS,
      ...efforts === undefined
        ? {}
        : {
          reasoning: {
            efforts,
            defaultEffort: ReasoningEffortId('high'),
          },
        },
    })
  }

  async *stream(options: GenerateOptions): AsyncIterable<StreamChunk> {
    assertCurrentClaudeModel(options.model)
    const watchdog = idleWatchdog(options.signal, this.options.streamIdleTimeoutMs)
    const release = await this.gate.acquire(options.signal)
    try {
      const session = await this.options.tokens.session()
      yield* this.streamViaCli(options, session, watchdog)
    } catch (error: unknown) {
      throw mapFetchFailure('claude CLI bridge', error, watchdog, options.signal)
    } finally {
      release()
      watchdog.stop()
    }
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
      && previous.reasoningEffort === options.reasoningEffort
      && previous.systemSignature === systemSignature
      && previous.toolsSignature === toolsSignature
      && previous.inputMessageCount <= ids.length
      && previous.inputMessageSignature === signature(ids.slice(0, previous.inputMessageCount).join('\n'))
    const deltaStart = canResume ? previous.inputMessageCount : 0
    const delta = resolvedMessages.slice(deltaStart)
    const sendableDelta = delta.filter(message => message.role !== 'assistant')
    const resume = canResume && previous !== undefined && sendableDelta.length > 0

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
          // The query is aborted as soon as its tool_use frame reaches DSH,
          // which executes the call and feeds its real result into the next
          // resumed message. Do not leave an error result in Claude's durable
          // session before that handoff: it makes a successful DSH call look
          // like a denied tool invocation on the following step.
          handler: async () => ({ content: [{ type: 'text' as const, text: 'DeepSeek Harness will return this tool result in the next message.' }] }),
        })),
      })
    }

    let claudeSessionId = resume ? previous.claudeSessionId : undefined
    let capturedToolCall = false
    let emitted = false
    let usageEmitted = false
    const commitState = async (resumeSessionAt?: string): Promise<void> => {
      if (sessionKey === undefined || claudeSessionId === undefined) return
      this.sessions.set(sessionKey, {
        claudeSessionId,
        inputMessageCount: ids.length,
        inputMessageSignature: signature(ids.join('\n')),
        model: options.model,
        ...options.reasoningEffort === undefined ? {} : { reasoningEffort: options.reasoningEffort },
        systemSignature,
        toolsSignature,
        ...resumeSessionAt === undefined ? {} : { resumeSessionAt },
        lastUsedAt: Date.now(),
      })
      await this.persistSessions()
    }
    let toolResumeAt: string | undefined
    try {
      const query = (this.options.claudeQuery ?? claudeAgentQuery)({
        prompt: resume ? sdkUserMessages(sendableDelta) : sdkInitialPrompt(resolvedMessages),
        options: {
          model: options.model,
          ...options.reasoningEffort === undefined
            ? {}
            : { effort: claudeEffort(options.reasoningEffort) },
          cwd: tmpdir(),
          maxTurns: this.options.cliMaxTurns,
          ...resume && claudeSessionId !== undefined ? { resume: claudeSessionId } : {},
          ...resume && previous?.resumeSessionAt !== undefined
            ? { resumeSessionAt: previous.resumeSessionAt }
            : {},
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
          const failure = claudeAssistantFailure(message.error, assistantText(message.message.content))
          if (failure !== undefined) throw failure
          const usage = toTokenUsage(message.message.usage)
          if (usage !== undefined) {
            usageEmitted = true
            yield { type: 'usage', usage }
          }
          for (const block of message.message.content) {
            if (block.type === 'text' && block.text.length > 0) {
              const pseudoCalls = parsePseudoToolCalls(block.text, tools)
              if (pseudoCalls !== undefined) {
                for (const call of pseudoCalls) {
                  emitted = true
                  capturedToolCall = true
                  const id = CallId(`pseudo-${randomUUID()}`)
                  yield { type: 'block-start', index, blockType: 'tool-call' }
                  yield { type: 'tool-call-delta', index, id, name: call.name, argumentsDelta: call.arguments }
                  yield { type: 'block-end', index, block: { type: 'tool-call', id, name: call.name, arguments: call.arguments } }
                  index += 1
                }
                controller.abort()
                break
              }
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
          if (capturedToolCall) toolResumeAt = String(message.uuid)
          if (capturedToolCall) break
        } else if (message.type === 'result') {
          if (message.subtype !== 'success') {
            throw claudeCliFailure(message.errors, message.subtype)
          }
          if (!usageEmitted) {
            const usage = toTokenUsage(message.usage)
            if (usage !== undefined) yield { type: 'usage', usage }
          }
        }
      }
      if (!emitted) throw new LlmError('Claude CLI returned no assistant content', EMPTY_RESPONSE_CODE)
      // Keep the assistant tool_use, but rewind past the SDK placeholder on
      // the next request so DSH can append the real tool result without
      // replaying the complete conversation.
      await commitState(capturedToolCall ? toolResumeAt : undefined)
      yield { type: 'finish', reason: capturedToolCall ? { kind: 'tool-calls' } : { kind: 'stop' } }
    } catch (error) {
      if (capturedToolCall && controller.signal.aborted) {
        await commitState(toolResumeAt)
        yield { type: 'finish', reason: { kind: 'tool-calls' } }
      } else {
        if (sessionKey !== undefined) {
          this.sessions.delete(sessionKey)
          await this.persistSessions()
        }
        const detail = error instanceof Error ? error.message : String(error)
        if (!(error instanceof LlmError) && isClaudeQuotaExhaustion(detail)) {
          throw new LlmError(`Claude CLI error: ${detail}`, QUOTA_EXCEEDED_CODE, { cause: error })
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
    const snapshot = JSON.stringify({ version: 3, sessions: Object.fromEntries(this.sessions) })
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
  reasoningEffort?: string
  systemSignature: string
  toolsSignature: string
  resumeSessionAt?: string
  lastUsedAt: number
}

/** Narrow one advertised opaque effort id to the Agent SDK vocabulary. */
function claudeEffort(effort: string): EffortLevel {
  if (!CLAUDE_EFFORTS.has(effort as EffortLevel)) {
    throw new LlmError(`Claude Agent SDK does not support reasoning effort "${effort}"`, 'UNSUPPORTED_REASONING_EFFORT')
  }
  return effort as EffortLevel
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

function isClaudeQuotaExhaustion(detail: string): boolean {
  return isQuotaExceededError(detail)
    || /\b(?:hit|reached)\s+(?:your|the)\s+(?:claude\s+)?(?:(?:usage|session)\s+)?limit\b/i.test(detail)
}

/** Classify an SDK assistant error before its human-facing text is emitted. */
export function claudeAssistantFailure(error: string | undefined, detail: string): LlmError | undefined {
  if (error === undefined) return undefined
  if (error === 'rate_limit') {
    return new LlmError(
      `Claude CLI error: ${detail || error}`,
      isClaudeQuotaExhaustion(detail) ? QUOTA_EXCEEDED_CODE : 'RATE_LIMIT',
    )
  }
  return new LlmError(`Claude CLI error: ${detail || error}`, EMPTY_RESPONSE_CODE)
}

/** Map one terminal Claude SDK result without treating transient rate limiting as exhausted quota. */
export function claudeCliFailure(errors: readonly string[], subtype: string): LlmError {
  const detail = errors.join('; ') || subtype
  return new LlmError(
    `Claude CLI error: ${detail}`,
    isClaudeQuotaExhaustion(`${subtype} ${detail}`) ? QUOTA_EXCEEDED_CODE : EMPTY_RESPONSE_CODE,
  )
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
    && (state.reasoningEffort === undefined || typeof state.reasoningEffort === 'string')
    && typeof state.systemSignature === 'string'
    && typeof state.toolsSignature === 'string'
    && (state.resumeSessionAt === undefined || typeof state.resumeSessionAt === 'string')
    && typeof state.lastUsedAt === 'number'
    && Number.isFinite(state.lastUsedAt)
}

function isPersistedSessionFile(value: unknown): value is { version: 3; sessions: Record<string, unknown> } {
  if (typeof value !== 'object' || value === null) return false
  const file = value as { version?: unknown; sessions?: unknown }
  return file.version === 3 && typeof file.sessions === 'object' && file.sessions !== null && !Array.isArray(file.sessions)
}

function assistantText(content: readonly { type: string; text?: string }[]): string {
  return content.filter(block => block.type === 'text').map(block => block.text ?? '').join(' ')
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
    }
  }
  return content
}

interface PseudoToolCall {
  name: string
  arguments: string
}

/**
 * Recover the exact text fallback Claude sometimes emits after seeing an old
 * DSH transcript. Only complete bracketed calls for currently offered tools
 * are accepted, so ordinary prose cannot become executable.
 */
export function parsePseudoToolCalls(
  text: string,
  tools: readonly { name: string }[],
): PseudoToolCall[] | undefined {
  const available = new Set(tools.map(tool => tool.name))
  const calls: PseudoToolCall[] = []
  let cursor = 0
  while (cursor < text.length) {
    while (/\s/.test(text[cursor] ?? '')) cursor += 1
    if (!text.startsWith(PSEUDO_TOOL_CALL_PREFIX, cursor)) return undefined
    cursor += PSEUDO_TOOL_CALL_PREFIX.length
    const opening = text.indexOf('(', cursor)
    if (opening < 0) return undefined
    const name = text.slice(cursor, opening).trim()
    if (!available.has(name)) return undefined
    const argumentStart = opening + 1
    let depth = 1
    let string = false
    let escaped = false
    cursor = argumentStart
    for (; cursor < text.length && depth > 0; cursor += 1) {
      const character = text[cursor]
      if (string) {
        if (escaped) escaped = false
        else if (character === '\\') escaped = true
        else if (character === '"') string = false
      } else if (character === '"') {
        string = true
      } else if (character === '(') {
        depth += 1
      } else if (character === ')') {
        depth -= 1
      }
    }
    if (depth !== 0 || text[cursor] !== ']') return undefined
    const argumentsJson = text.slice(argumentStart, cursor - 1)
    try {
      const parsed = JSON.parse(argumentsJson) as unknown
      if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) return undefined
    } catch {
      return undefined
    }
    calls.push({ name, arguments: argumentsJson })
    cursor += 1
  }
  return calls.length > 0 ? calls : undefined
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
