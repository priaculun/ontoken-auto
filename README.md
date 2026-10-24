# ontoken-auto

Pi package: virtual model **`ontoken/auto`**. TypeSafe Jev classifies each user prompt, then this extension picks an [OnToken](https://ontoken.id) model and thinking level.

Short follow-up prompts inherit recent conversation context. During an agent turn, Jev may reassess after tool results and promote the work to a stronger slot; otherwise follow-ups stay on the same model to preserve the prompt cache. Compaction / `direct` uses the `fast` slot.

## Install (any machine)

```bash
pi install https://github.com/priaculun/ontoken-auto
```

Or:

```bash
pi install git:github.com/priaculun/ontoken-auto
```

Then restart Pi and:

```
/model ontoken/auto
```

Requires:

- Provider `ontoken` in `~/.pi/agent/models.json` (the roster ids below).
- TypeSafe Jev key (classifier `typesafe/jev-latest`).

### Jev key (do not commit)

Pick one:

1. `export TYPESAFE_API_KEY=…`
2. File `~/.pi/agent/secrets/typesafe_api_key` (one line, the key)
3. Overlay `jev.keyFile` pointing at a local file whose first non-comment line is the key

```json
{
  "jev": { "keyFile": "/path/to/jev.md" }
}
```

Without a key the router still works: heuristic fallback, thinking `low` if classify fails.

## Commands

```
/auto                       status
/auto explain               last verdict, rule, slot, failover decision
/auto check                 health report: slots, classifier, key
/auto on | off              enable / disable routing
/auto budget cheap|balanced|quality
/auto slot <fast|work|solid|strong|frontier> <model-id>
/auto thinking auto|selection
/auto debug [on|summary|verbose|off]
/auto debug file <path>     mirror the log to a JSONL file
/auto debug nofile          stop writing the file
/auto debug clear           drop the in-memory buffer
/auto log [n]               print the last n debug records
```

## Debug mode

Debug is **off by default** and can be enabled four ways (highest precedence first):

| Source | Value |
|---|---|
| CLI flag | `pi --ontoken-debug verbose …` (or `summary`, `off`) |
| Env | `PI_ONTOKEN_DEBUG=off\|summary\|verbose` |
| Env | `PI_ONTOKEN_DEBUG_FILE=/path/to/log.jsonl` — setting the file implies `summary` |
| Overlay | `{ "debug": { "enabled": true, "level": "verbose" } }` |
| Command | `/auto debug verbose` (runtime, persists to the global overlay) |

`summary` records one line per decision (`route.user`, `route.continuation`, `route.retry`, `route.direct`, `stall.nudge`, `failover`, …). `verbose` also records the classifier request/response, the assembled prompt, and model resolution.

```jsonc
// ~/.pi/agent/ontoken-auto.json
{
  "debug": {
    "enabled": true,
    "level": "verbose",   // off | summary | verbose
    "file": "/tmp/ontoken-debug.jsonl",
    "bufferSize": 200,    // in-memory records kept for /auto log
    "includePrompts": true
  }
}
```

Guarantees:

- The log is **bounded** (`bufferSize`) and never grows without limit.
- Writes are `appendFileSync` in a `try/catch`; a broken path degrades to memory-only instead of failing a turn.
- Secrets are redacted before anything leaves the process: bearer tokens, credential assignments, URL passwords, private keys, and `sk-…` keys. `/auto check` and the `boot` record report *whether* a key was found, never the key.
- Each record is one JSON line: `{ ts, kind, ...fields }`.

Tail it live:

```bash
tail -f /tmp/ontoken-debug.jsonl | jq -c '{ts,kind,slot,modelId,ruleId,appliedThinking,reason}'
```

## Adaptive routing

By default, user prompts up to 800 characters are classified together with recent
user/assistant context, so short replies such as `lanjut`, `fix itu`, or
`implementasikan` retain the original task. Text sent to TypeSafe Jev is capped
by `jev.promptChars` (16,000 by default).

Within the same agent turn, the router reassesses after every two new tool
results, at most twice. It can only promote to a stronger available slot; it
never downgrades mid-turn, and heuristic fallback cannot trigger promotion.
A promotion changes the physical model and therefore accepts one prompt-cache
miss in exchange for stronger handling of newly discovered complexity.

Conversation and tool-result text sent to Jev receives best-effort redaction of
common bearer tokens, credential assignments, URL passwords, and private keys.
Redaction is not a security boundary: disable context or mid-turn classification
for sensitive workloads.

```json
{
  "adaptive": {
    "contextAware": true,
    "shortPromptChars": 800,
    "midTurn": true,
    "maxEvaluationsPerTurn": 2,
    "toolResultsPerEvaluation": 2
  }
}
```

Set `contextAware` or `midTurn` to `false` in a global/project overlay to disable
that behavior.

## Thinking policy

Every rule carries a `thinking` level. The router then clamps it to what the
routed model actually supports (Pi's own `thinkingLevelMap` rule: a level mapped
to `null` is unsupported, and `xhigh`/`max` require an explicit mapping). A slot
can also cap its level with `maxThinking`.

```json
{
  "thinkingPolicy": "auto",
  "maxThinking": { "fast": "low", "frontier": "xhigh" }
}
```

- `auto` (default) — use the rule's level, clamped to the model.
- `selection` — use whatever thinking level the user selected in Pi, still clamped.

## Failover

When a dispatched model fails with a transient error (5xx, 429/rate limit,
overloaded, timeout, network), the next retry moves the turn to another slot
instead of re-hitting the broken model. Auth, quota, billing errors, and user
aborts are **not** failed over. Failover respects the budget ceiling and never
returns to a slot that already failed this turn. Routing state records the
switch (`ruleId` gains `+failover`), and `/auto explain` shows the last
failover decision.

Defaults: enabled, automatic chain = remaining slots ordered strongest-first,
with a UI notification per switch (`"notify": false` to silence).

Explicit chains per slot (overlay, same files as below):

```json
{
  "failover": {
    "enabled": true,
    "notify": true,
    "chain": { "work": ["solid", "fast"], "fast": ["work"] }
  }
}
```

## Stall guard

Small models sometimes answer a tool-using task with a plan instead of acting
("Sekarang saya tulis file-nya.") and then stop. The stall guard detects that
shape — a short, tool-call-free, non-final assistant message whose **final
sentence** is a promise to act — and queues one follow-up telling the model to
execute. It is capped per run and deliberately conservative: lists of results,
completion summaries, questions, and multi-sentence prose never trigger it.

```json
{
  "stallGuard": {
    "enabled": true,
    "maxPerRun": 2,
    "maxChars": 300,
    "message": "Lanjutkan pekerjaanmu: eksekusi langkah berikutnya sekarang dengan memanggil tool yang diperlukan."
  }
}
```

## Change roster / rules without editing code

Merge overlay (only keys you change):

| File | Scope |
|---|---|
| `~/.pi/agent/ontoken-auto.json` | all sessions |
| `<cwd>/.pi/ontoken-auto.json` | this project (wins) |

Package defaults: `extensions/ontoken-auto/defaults.json`.

Every value is re-validated after merging, so a bad overlay (wrong type, unknown
slot name, out-of-range number) falls back to the default instead of breaking
routing.

Swap solid:

```json
{ "slots": { "solid": "qwen3.8-max" } }
```

Cap a repo:

```json
{ "budget": "cheap" }
```

`cheap` never goes above the `solid` slot.

Add a rule (rules are evaluated in order; the strongest match wins, and a later
rule wins ties):

```json
{
  "rules": [
    {
      "id": "docs-only",
      "when": { "taskIn": ["writing", "chat"], "maxDifficulty": 1 },
      "slot": "fast",
      "thinking": "off"
    }
  ]
}
```

`when` supports `taskIn`, `minDifficulty`/`maxDifficulty` (0–4), `needsReasoning`,
`needsVision`, `longHorizon`, `minConfidence`, `budgetPromoteEarly`.

## Default roster

| Slot | Model | Role |
|---|---|---|
| fast | `gpt-6-luna` | chat / lookup, thinking off |
| work | `glm-5.3-flash` | routine–involved coding |
| solid | `muse-spark-1.3` | hard debug / design |
| strong | `claude-sonnet-5.5` | frontier |
| frontier | `gpt-6.1-sol` | frontier + heavy reasoning |

Prices: OnToken catalog `priceOnt*` (1 ONT = Rp1.000). Intelligence: Artificial Analysis, 6 Oct 2026.

## Development

```bash
npm test                 # 60 tests, no network, no Pi runtime needed
npx tsc -p tsconfig.json # typecheck (no emit)
```

`node --test` runs the TypeScript sources directly (Node ≥ 22.6 with type
stripping; verified on Node 26).

Layout:

```
extensions/ontoken-auto/
  index.ts       Pi wiring: virtual model, /auto command, hooks, stall guard
  router.ts      pure routing logic (no runtime Pi imports) — unit tested
  config.ts      defaults + overlays + validation + Jev key discovery
  defaults.json  the shipped roster, rules, budgets, debug/stallGuard blocks
test/
  router.test.ts   config, prompts, redaction, thinking clamp, rules, failover
  harness.test.ts  drives the real extension through a fake Pi harness
```

`router.ts` is deliberately dependency-free so routing decisions can be tested
without a Pi process, a network call, or an API key.
