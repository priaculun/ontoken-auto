# ontoken-auto

Pi package: virtual model **`ontoken/auto`**. TypeSafe Jev classifies each user prompt, then this extension picks an [OnToken](https://ontoken.id) model and thinking level.

Tool follow-ups and retries stay on the same model (prompt cache). Compaction / `direct` uses the `fast` slot.

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
/auto
/auto explain
/auto on
/auto off
/auto budget cheap|balanced|quality
/auto slot <fast|work|solid|strong|frontier> <model-id>
```

## Change roster / rules without editing code

Merge overlay (only keys you change):

| File | Scope |
|---|---|
| `~/.pi/agent/ontoken-auto.json` | all sessions |
| `<cwd>/.pi/ontoken-auto.json` | this project (wins) |

Package defaults: `extensions/ontoken-auto/defaults.json`.

Swap solid:

```json
{ "slots": { "solid": "qwen3.8-max" } }
```

Cap a repo:

```json
{ "budget": "cheap" }
```

`cheap` never goes above the `solid` slot.

## Default roster

| Slot | Model | Role |
|---|---|---|
| fast | `gpt-6-luna` | chat / lookup, thinking off |
| work | `glm-5.3-flash` | routine–involved coding |
| solid | `muse-spark-1.3` | hard debug / design |
| strong | `claude-sonnet-5.5` | frontier |
| frontier | `claude-opus-5.5` | frontier + heavy reasoning |

Prices: OnToken catalog `priceOnt*` (1 ONT = Rp1.000). Intelligence: Artificial Analysis, 6 Oct 2026.

## Layout

```
extensions/ontoken-auto/
  index.ts
  config.ts
  defaults.json
```
