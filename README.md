# dsh-plugin-subscriptions

English | [中文](README.zh.md)

Use your **ChatGPT (Codex)**, **Claude**, and **Grok (X Premium)** subscriptions as LLM providers in [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) — no API keys. Login happens in the dsh web UI (Settings → Subscriptions); tokens live at `~/.dsh/plugins/subscriptions/auth.json` (mode 0600) and refresh automatically.

## Demo

Settings → **Subscriptions**: per-provider OAuth login/logout, no API keys (account address masked in the screenshot):

![Subscriptions settings page](https://raw.githubusercontent.com/V1ki/dsh-plugin-subscriptions/main/docs/images/subscriptions.png)

Logged-in providers join the session model picker with their live model catalogs:

![Model picker with subscription models](https://raw.githubusercontent.com/V1ki/dsh-plugin-subscriptions/main/docs/images/model-picker.png)

Models that advertise reasoning levels get an **Effort** selector in the same menu. Codex retains only GPT-5.6 and GPT-5.3-Codex models and uses their live catalog; Claude exposes effort for Opus 5, Fable 5, and Sonnet 5 (Haiku 4.5 has no effort selector).

![Reasoning effort selector](https://raw.githubusercontent.com/V1ki/dsh-plugin-subscriptions/main/docs/images/model-effort.png)

The `image_generate` tool renders its result inline in the conversation:

![image_generate renders the image inline](https://raw.githubusercontent.com/V1ki/dsh-plugin-subscriptions/main/docs/images/image-generate-inline.png)

## Providers

| Route    | Subscription      | Models |
|----------|-------------------|--------|
| `codex`  | ChatGPT Plus/Pro  | live catalog from `chatgpt.com/backend-api/codex/models` |
| `claude` | Claude Pro/Max    | claude-opus-5, claude-fable-5, claude-sonnet-5, claude-haiku-4-5 |
| `grok`   | X Premium (xAI)   | live catalog from `api.x.ai/v1/models` (chat models only); reasoning efforts from the Grok CLI catalog (`cli-chat-proxy.grok.com/v1/models`) |

Only logged-in providers appear in the session model picker; the lists above refresh on login/logout. Vision-capable models declare `['text', 'image']` input modalities, and image content is translated to each provider's wire format. Codex keeps GPT-5.6 Sol/Terra/Luna and GPT-5.3-Codex only; Claude keeps Opus 5, Fable 5, Sonnet 5, and Haiku 4.5 only.

Logged-in cards also show **subscription usage** — per rate-limit window (5-hour session, weekly, and per-model weekly where the plan has one) with the used percentage, a progress bar, and the reset time, plus a Refresh button. Codex usage comes from `chatgpt.com/backend-api/wham/usage` (also reports the plan), Claude usage from `api.anthropic.com/api/oauth/usage`, and Grok usage from the Grok Build CLI proxy's `cli-chat-proxy.grok.com/v1/billing` (the source of the CLI's `/usage` panel; reports the shared weekly pool and the subscription tier).

Also included, registered when the matching provider is enabled:

- **`x_search`** tool (Grok) — xAI's hosted X search, returning `{ answer, citations }`.
- **`image_generate`** tool (ChatGPT) — `gpt-image-2` via the Codex backend; PNGs are saved under `~/.dsh/plugins/subscriptions/images/` and the paths returned.

## Install

With the `dsh` CLI available, install from npm (prebuilt artifacts, no build permission needed):

```sh
dsh plugin --profile web add dsh-plugin-subscriptions
```

Or install the sources from GitHub:

```sh
dsh plugin --profile web add github:V1ki/dsh-plugin-subscriptions
```

pnpm will ask you to allow this package's build script on first install (git installs fetch sources, not built artifacts); add the printed key to the profile's `pnpm-workspace.yaml`:

```yaml
allowBuilds:
  dsh-plugin-subscriptions: true
```

and re-run the `add`. Only grant this to packages you trust — it runs the package's code at install time.

From a local checkout instead:

```sh
git clone https://github.com/V1ki/dsh-plugin-subscriptions.git
cd dsh-plugin-subscriptions && pnpm install && pnpm build
dsh plugin --profile web add ./dsh-plugin-subscriptions
```

Headless-only usage without installing into a profile (log in via the web UI first — the token file is shared):

```sh
cp overlay.example.yml overlay.yml   # then edit the name: to this checkout's absolute lib/index.js path
dsh --profile headless --patch <checkout>/overlay.yml "your task"
```

## Update

Installed from npm:

```sh
dsh plugin --profile web update --latest dsh-plugin-subscriptions
```

Installed from GitHub: re-run the same `add github:V1ki/dsh-plugin-subscriptions` command — it re-fetches the sources and rebuilds. A linked local checkout just needs `git pull && pnpm build` in the checkout.

Either way, restart `dsh web` afterwards so the new version loads.

## Use

1. `dsh web`, open the printed URL.
2. Settings → **Subscriptions**: click **Log in** on a provider and authorize in the opened tab. If the browser flow can't complete (headless host), expand the manual fallback and paste the callback URL or code.
3. In any session, open the model picker (`/model`) and choose a model under **ChatGPT (Codex)** / **Claude (Subscription)** / **Grok (Subscription)**.

Not logged in? The provider stays out of the picker, and requests fail with `MISSING_CREDENTIAL` pointing at the Settings page; nothing else breaks.

## Config

```yaml
- id: llm-subscriptions
  name: dsh-plugin-subscriptions
  config:
    providers: [codex, claude]        # subset; default all three
    streamIdleTimeoutMs: 300000
    claude:
      maxConcurrentRequests: 1        # serialize Claude subscription calls
      sessionStateTtlMs: 21600000      # keep DSH -> Claude resume mappings for 6h
      # sessionStatePath defaults to ~/.dsh/plugins/subscriptions/claude-sessions.json
      cliMaxTurns: 1                    # no hidden second Claude turn
    models:                            # override the discovered/built-in catalogs
      codex:
        - { id: gpt-5.6-sol, name: GPT-5.6 Sol, contextWindow: 272000, inputModalities: [text, image] }
      claude:
        - { id: claude-opus-5, name: Claude Opus 5, contextWindow: 1000000, inputModalities: [text, image], reasoningEfforts: [low, medium, high, xhigh, max], defaultReasoningEffort: high }
```

GPT-5.6 models expose **Standard** and **Fast** in DSH's model picker. The choice is session-scoped; Fast maps only that session's requests to the Codex subscription `service_tier: priority` wire value.

Claude requests use the Claude Agent SDK transport, keep one resumable Claude
session per DSH session, and send only new user/tool-result messages after the
first request. Resume mappings survive DSH restarts; the state file contains
only Claude session ids, message counts, and SHA-256 fingerprints—not prompts,
tool definitions, or tokens. The adapter disables Claude Code's built-in tools,
skills, plugins, agents, MCP settings, and filesystem settings so DSH remains
the sole tool and instruction layer. Model requests are never stopped by a
preemptive usage threshold; the provider's actual exhausted-quota response is
reported as `QUOTA` so a host router can move to another provider.

For a Claude-only profile, copy `examples/efficient-claude` into the profile's
`.agent-presets/efficient-claude` directory. It keeps one model lane, caps the
standing instructions, prunes large tool output, and compacts early with Haiku.
The companion `examples/efficient-claude-profile.patch.yml` disables the hidden
LLM title request and selects this preset by default.

## Develop

```sh
pnpm install   # installs the published dsh development packages
pnpm build     # tsc (lib/) + tsdown (lib/client.js browser bundle)
pnpm test      # node --test over compiled unit specs
```

`prepare` (used by git installs) runs `tsdown.prepare.config.ts`: a self-contained bundle build of both faces with all `@deepseek-ai/*` specifiers external — they resolve from the dsh installation at runtime, so this package never carries a second cordis copy.

After `pnpm build`, restart `dsh web` to pick up changes.

## Layout

- `src/index.ts` — plugin entry: config schema, adapter registration, auth-change re-announce, RPC wiring
- `src/auth/` — PKCE/JWT helpers, token store, OAuth flow engine (temp loopback callback server), `/subscriptions-auth` RPC channel
- `src/providers/` — per-provider OAuth constants/exchange/refresh + `LlmAdapter`s
- `src/translate/` — dsh `Message[]` ⟷ OpenAI Responses / Anthropic Messages wire formats, SSE → `StreamChunk`
- `src/tools/` — `x_search` and `image_generate`
- `src/client/` — the Settings → Subscriptions page (browser half, zh/en, theme-token aware)
