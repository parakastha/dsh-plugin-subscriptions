/** Claude bridge regression coverage for quota failover and tool-step resume. */

import { test } from 'node:test'
import assert from 'node:assert/strict'
import { randomUUID } from 'node:crypto'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { query as claudeAgentQuery } from '@anthropic-ai/claude-agent-sdk'
import type { SDKMessage, SDKUserMessage } from '@anthropic-ai/claude-agent-sdk'
import type { GenerateOptions } from '@deepseek-ai/dsh-llm'
import { ClaudeAdapter, claudeAssistantFailure, parsePseudoToolCalls } from '../src/providers/claude.js'
import { TokenManager } from '../src/providers/common.js'
import type { ClaudeSession } from '../src/auth/store.js'

const claudeSession: ClaudeSession = {
  accessToken: 'at',
  refreshToken: 'rt',
  expiresAt: Date.now() + 3_600_000,
  scopes: 'scope',
}

function memoryTokens(): TokenManager<ClaudeSession> {
  return new TokenManager({
    displayName: 'Claude',
    preemptMs: 0,
    load: () => Promise.resolve(claudeSession),
    save: () => Promise.resolve(),
    remove: () => Promise.resolve(),
    refresh: session => Promise.resolve(session),
    isPermanent: () => false,
  })
}

type QueryParams = Parameters<typeof claudeAgentQuery>[0]

interface QueryCall {
  params: QueryParams
  input: SDKUserMessage[]
}

function scriptedQuery(scripts: readonly (readonly SDKMessage[])[], calls: QueryCall[]): typeof claudeAgentQuery {
  let index = 0
  return ((params: QueryParams) => {
    const script = scripts[index++]
    if (script === undefined) throw new Error('unexpected Claude query')
    const call: QueryCall = { params, input: [] }
    calls.push(call)
    const stream = async function* (): AsyncGenerator<SDKMessage> {
      if (typeof params.prompt !== 'string') {
        for await (const message of params.prompt) call.input.push(message)
      }
      for (const message of script) yield message
    }
    return stream() as ReturnType<typeof claudeAgentQuery>
  }) as typeof claudeAgentQuery
}

function adapter(claudeQuery: typeof claudeAgentQuery): ClaudeAdapter {
  return new ClaudeAdapter({
    models: [{ id: 'claude-opus-5', name: 'Claude Opus 5' }],
    streamIdleTimeoutMs: 1000,
    tokens: memoryTokens(),
    onWarn: () => {},
    maxConcurrentRequests: 1,
    sessionStateTtlMs: 60_000,
    sessionStatePath: join(tmpdir(), `claude-continuity-${randomUUID()}.json`),
    cliMaxTurns: 1,
    claudeQuery,
  })
}

function assistant(content: readonly Record<string, unknown>[], uuid: string, error?: string): SDKMessage {
  return {
    type: 'assistant',
    message: {
      id: `msg-${uuid}`,
      type: 'message',
      role: 'assistant',
      model: 'claude-opus-5',
      content,
      stop_reason: content.some(block => block.type === 'tool_use') ? 'tool_use' : 'end_turn',
      stop_sequence: null,
      usage: { input_tokens: 1, output_tokens: 1 },
    },
    parent_tool_use_id: null,
    uuid,
    session_id: 'claude-session',
    ...error === undefined ? {} : { error },
  } as unknown as SDKMessage
}

function options(messages: readonly Record<string, unknown>[]): GenerateOptions {
  return {
    provider: 'claude',
    model: 'claude-opus-5',
    sessionId: 'dsh-session',
    system: 'system',
    tools: [{ name: 'inspect', description: 'inspect', parameters: { type: 'object', properties: {} } }],
    messages,
  } as unknown as GenerateOptions
}

async function drain(instance: ClaudeAdapter, request: GenerateOptions): Promise<void> {
  for await (const _chunk of instance.stream(request)) void _chunk
}

test('Claude session-limit assistant errors become quota before text is emitted', () => {
  const failure = claudeAssistantFailure(
    'rate_limit',
    "You've hit your session limit · resets 10:50am (America/Chicago)",
  )
  assert.equal(failure?.code, 'QUOTA')
  assert.equal(claudeAssistantFailure('rate_limit', 'request rate limited')?.code, 'RATE_LIMIT')
})

test('Claude pseudo tool-call recovery accepts only complete calls for offered tools', () => {
  const tools = [{ name: 'inspect' }]
  assert.deepEqual(
    parsePseudoToolCalls('[tool call: inspect({"path":"a(b)"})]', tools),
    [{ name: 'inspect', arguments: '{"path":"a(b)"}' }],
  )
  assert.equal(parsePseudoToolCalls('[tool call: missing({})]', tools), undefined)
  assert.equal(parsePseudoToolCalls('please [tool call: inspect({})]', tools), undefined)
  assert.equal(parsePseudoToolCalls('[tool call: inspect({broken})]', tools), undefined)
})

test('Claude bridge surfaces the live session-limit message as quota', async () => {
  const instance = adapter(scriptedQuery([[
    assistant(
      [{ type: 'text', text: "You've hit your session limit · resets 10:50am (America/Chicago)" }],
      'assistant-quota',
      'rate_limit',
    ),
  ]], []))

  await assert.rejects(
    drain(instance, options([{ id: 'user-1', role: 'user', content: [{ type: 'text', text: 'continue' }] }])),
    (error: unknown) => typeof error === 'object' && error !== null && 'code' in error && error.code === 'QUOTA',
  )
})

test('Claude tool handoff resumes at the tool_use and sends only the real result', async () => {
  const calls: QueryCall[] = []
  const instance = adapter(scriptedQuery([
    [assistant([{ type: 'tool_use', id: 'tool-1', name: 'mcp__dsh__inspect', input: {} }], 'assistant-tool')],
    [assistant([{ type: 'text', text: 'done' }], 'assistant-done')],
  ], calls))

  const first = [{ id: 'user-1', role: 'user', content: [{ type: 'text', text: 'inspect' }] }]
  await drain(instance, options(first))

  const second = [
    ...first,
    { id: 'assistant-1', role: 'assistant', content: [{ type: 'tool-call', id: 'tool-1', name: 'inspect', arguments: '{}' }] },
    { id: 'result-1', role: 'user', content: [{ type: 'tool-result', toolCallId: 'tool-1', content: [{ type: 'text', text: 'real result' }], isError: false }] },
  ]
  await drain(instance, options(second))

  assert.equal(calls.length, 2)
  assert.equal(calls[1].params.options?.resume, 'claude-session')
  assert.equal(calls[1].params.options?.resumeSessionAt, 'assistant-tool')
  assert.equal(calls[1].input.length, 1)
  assert.deepEqual(calls[1].input[0].message.content, [{
    type: 'tool_result',
    tool_use_id: 'tool-1',
    content: 'real result',
    is_error: false,
  }])
})
