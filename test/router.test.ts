import assert from "node:assert/strict";
import test from "node:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
	THINKING_ORDER,
	type AutoConfig,
	clampSlot,
	ensureJevKey,
	isBudgetName,
	isSlotName,
	loadConfig,
	loadDefaults,
	slotIndex,
	writeGlobalOverlay,
} from "../extensions/ontoken-auto/config.ts";
import {
	type MessageLike,
	type ModelLike,
	adaptiveProgressPrompt,
	clampThinking,
	contextualUserPrompt,
	currentTurnToolResults,
	detectStall,
	failoverTarget,
	heuristic,
	isContextOverflowError,
	isFailoverWorthy,
	isSyntheticUserMessage,
	lastUserText,
	pickRoute,
	registerSyntheticUserText,
	redactClassifierText,
	resolveThinking,
	supportedThinkingLevels,
	userHasImage,
} from "../extensions/ontoken-auto/router.ts";

// --------------------------------------------------------------------- helpers

function userText(text: string): MessageLike {
	return { role: "user", content: [{ type: "text", text }] };
}

function assistantText(text: string): MessageLike {
	return { role: "assistant", content: [{ type: "text", text }] };
}

function toolResult(name: string, text: string, isError = false): MessageLike {
	return { role: "toolResult", toolName: name, content: [{ type: "text", text }], isError };
}

/** Config with a temp agent dir so tests never read the real overlays. */
function isolatedConfig(overlay?: Record<string, unknown>): AutoConfig {
	const dir = mkdtempSync(join(tmpdir(), "ontoken-test-"));
	process.env.PI_AGENT_DIR = dir;
	delete process.env.PI_ONTOKEN_DEBUG;
	delete process.env.PI_ONTOKEN_DEBUG_FILE;
	if (overlay) writeFileSync(join(dir, "ontoken-auto.json"), JSON.stringify(overlay));
	return loadConfig();
}

// ---------------------------------------------------------------------- config

test("defaults.json ships every slot the rules reference", () => {
	process.env.PI_AGENT_DIR = mkdtempSync(join(tmpdir(), "ontoken-defaults-"));
	const cfg = loadDefaults();
	for (const rule of cfg.rules) {
		assert.ok(cfg.slotOrder.includes(rule.slot), `rule ${rule.id} uses unknown slot ${rule.slot}`);
	}
	assert.deepEqual([...cfg.slotOrder], ["fast", "work", "solid", "strong", "frontier"]);
	for (const slot of cfg.slotOrder) {
		assert.equal(typeof cfg.slots[slot], "string");
		assert.ok(cfg.slots[slot].length > 0);
	}
});

test("loadConfig normalizes a hostile overlay instead of trusting it", () => {
	const cfg = isolatedConfig({
		budget: "ultra",
		directSlot: "nope",
		fallbackSlot: "nope",
		slotOrder: ["frontier", "bogus", "frontier"],
		adaptive: { maxEvaluationsPerTurn: 99, toolResultsPerEvaluation: -3, shortPromptChars: 1 },
		debug: { enabled: true, level: "loud", bufferSize: -5 },
		stallGuard: { maxPerRun: 42, maxChars: 1 },
		slots: { work: "  ", solid: "qwen3.8-max" },
		maxThinking: { frontier: "loud", solid: "high" },
	});
	assert.equal(cfg.budget, "balanced");
	assert.equal(cfg.directSlot, "fast");
	assert.equal(cfg.fallbackSlot, "work");
	// Unknown entries dropped, duplicates removed, missing slots appended.
	assert.equal(cfg.slotOrder[0], "frontier");
	assert.equal(new Set(cfg.slotOrder).size, 5);
	assert.equal(cfg.adaptive.maxEvaluationsPerTurn, 5);
	assert.equal(cfg.adaptive.toolResultsPerEvaluation, 1);
	assert.equal(cfg.adaptive.shortPromptChars, 80);
	// Blank model id falls back to the default, a real override is kept.
	assert.equal(cfg.slots.work, "glm-5.3-flash");
	assert.equal(cfg.slots.solid, "qwen3.8-max");
	assert.equal(cfg.maxThinking?.frontier, undefined);
	assert.equal(cfg.maxThinking?.solid, "high");
	// An unknown debug level keeps the default level but honours enabled.
	assert.equal(cfg.debug.enabled, true);
	assert.equal(cfg.debug.level, "summary");
	assert.equal(cfg.debug.bufferSize, 10);
	assert.equal(cfg.stallGuard.maxPerRun, 10);
	assert.equal(cfg.stallGuard.maxChars, 60);
});

test("project overlay wins over the global overlay", () => {
	const dir = mkdtempSync(join(tmpdir(), "ontoken-proj-"));
	process.env.PI_AGENT_DIR = dir;
	writeFileSync(join(dir, "ontoken-auto.json"), JSON.stringify({ budget: "cheap" }));
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	mkdirSync(join(cwd, ".pi"), { recursive: true });
	writeFileSync(join(cwd, ".pi", "ontoken-auto.json"), JSON.stringify({ budget: "quality" }));
	assert.equal(loadConfig(cwd).budget, "quality");
	assert.equal(loadConfig().budget, "cheap");
});

test("writeGlobalOverlay merges instead of replacing", () => {
	const dir = mkdtempSync(join(tmpdir(), "ontoken-write-"));
	process.env.PI_AGENT_DIR = dir;
	writeGlobalOverlay({ budget: "cheap" });
	writeGlobalOverlay({ enabled: false });
	const cfg = loadConfig();
	assert.equal(cfg.budget, "cheap");
	assert.equal(cfg.enabled, false);
});

test("PI_ONTOKEN_DEBUG env overrides the debug block", () => {
	const dir = mkdtempSync(join(tmpdir(), "ontoken-dbg-"));
	process.env.PI_AGENT_DIR = dir;
	process.env.PI_ONTOKEN_DEBUG = "verbose";
	process.env.PI_ONTOKEN_DEBUG_FILE = "/tmp/x.jsonl";
	const cfg = loadConfig();
	assert.equal(cfg.debug.enabled, true);
	assert.equal(cfg.debug.level, "verbose");
	assert.equal(cfg.debug.file, "/tmp/x.jsonl");
	process.env.PI_ONTOKEN_DEBUG = "off";
	assert.equal(loadConfig().debug.enabled, false);
	delete process.env.PI_ONTOKEN_DEBUG;
	delete process.env.PI_ONTOKEN_DEBUG_FILE;
});

test("ensureJevKey reads the secrets file", () => {
	const dir = mkdtempSync(join(tmpdir(), "ontoken-key-"));
	process.env.PI_AGENT_DIR = dir;
	delete process.env.TYPESAFE_API_KEY;
	mkdirSync(join(dir, "secrets"), { recursive: true });
	writeFileSync(join(dir, "secrets", "typesafe_api_key"), "apikey_from_file\n");
	const cfg = loadConfig();
	assert.equal(ensureJevKey(cfg), true);
	assert.equal(process.env.TYPESAFE_API_KEY, "apikey_from_file");
	delete process.env.TYPESAFE_API_KEY;
});

test("slot helpers respect the configured order and budget ceiling", () => {
	const cfg = isolatedConfig();
	assert.equal(slotIndex(cfg, "fast"), 0);
	assert.equal(slotIndex(cfg, "frontier"), 4);
	assert.equal(clampSlot(cfg, "frontier", "cheap"), "solid");
	assert.equal(clampSlot(cfg, "work", "cheap"), "work");
	assert.equal(isSlotName("solid"), true);
	assert.equal(isSlotName("Solid"), false);
	assert.equal(isBudgetName("quality"), true);
	assert.equal(isBudgetName("free"), false);
});

// ------------------------------------------------------------- prompt building

test("contextualUserPrompt includes recent context for short prompts", () => {
	const cfg = isolatedConfig();
	const messages = [
		userText("tolong refactor modul auth jadi modular"),
		assistantText("Baik, saya akan memecah auth.ts menjadi beberapa file."),
		userText("lanjut"),
	];
	const prompt = contextualUserPrompt(messages, cfg);
	assert.match(prompt, /CURRENT USER REQUEST:\nlanjut/);
	assert.match(prompt, /PREVIOUS USER:\n.*refactor modul auth/);
	assert.match(prompt, /PREVIOUS ASSISTANT:/);
	// Current request must be last so the tail slice never drops it.
	assert.ok(prompt.trimEnd().endsWith("lanjut"));
});

test("contextualUserPrompt skips context for long prompts", () => {
	const cfg = isolatedConfig();
	const long = "x".repeat(cfg.adaptive.shortPromptChars + 10);
	const prompt = contextualUserPrompt([userText("previous"), assistantText("prior"), userText(long)], cfg);
	assert.equal(prompt, long);
	assert.doesNotMatch(prompt, /PREVIOUS USER/);
});

test("contextualUserPrompt honours contextAware=false and promptChars", () => {
	// promptChars is clamped to a 500-character floor by config normalization.
	const cfg = isolatedConfig({ adaptive: { contextAware: false }, jev: { promptChars: 100 } });
	assert.equal(cfg.jev.promptChars, 500);
	const prompt = contextualUserPrompt([userText("a"), assistantText("b"), userText("c".repeat(900))], cfg);
	assert.equal(prompt.length, 500);
	assert.doesNotMatch(prompt, /PREVIOUS/);
});

test("classifier text is redacted before leaving the process", () => {
	const secret = "Bearer abc.def.ghi api_key=supersecret sk-abcdefghijklmnop https://user:pw@host/x";
	const red = redactClassifierText(secret);
	assert.doesNotMatch(red, /abc\.def\.ghi/);
	assert.doesNotMatch(red, /supersecret/);
	assert.doesNotMatch(red, /sk-abcdefghijklmnop/);
	assert.doesNotMatch(red, /user:pw@/);
	assert.match(red, /\[REDACTED/);
});

test("adaptiveProgressPrompt keeps the task and appends tool progress", () => {
	const cfg = isolatedConfig();
	const messages = [
		userText("fix the flaky login test"),
		toolResult("read", "test file contents"),
		toolResult("bash", "FAIL auth.spec.ts expected 200 got 500", true),
	];
	const prompt = adaptiveProgressPrompt(messages, cfg);
	assert.match(prompt, /TASK AND RECENT CONTEXT:/);
	assert.match(prompt, /CURRENT-TURN PROGRESS:/);
	assert.match(prompt, /TOOL bash ERROR:/);
	assert.match(prompt, /Reassess the remaining work/);
	assert.ok(prompt.length <= cfg.jev.promptChars);
});

test("currentTurnToolResults counts only results after the last user message", () => {
	const messages = [
		userText("one"),
		toolResult("read", "a"),
		userText("two"),
		toolResult("read", "b"),
		toolResult("bash", "c"),
	];
	assert.equal(currentTurnToolResults(messages), 2);
	assert.equal(currentTurnToolResults([userText("only")]), 0);
});

test("userHasImage detects image blocks", () => {
	assert.equal(userHasImage([{ role: "user", content: [{ type: "image", data: "", mimeType: "image/png" }] }]), true);
	assert.equal(userHasImage([userText("no image")]), false);
	assert.equal(userHasImage([]), false);
});

// --------------------------------------------------------------------- heuristic

test("heuristic never claims to be Jev", () => {
	const v = heuristic("please fix the crash in auth", false);
	assert.equal(v.source, "heuristic");
	assert.ok(v.confidence < 0.5);
	assert.equal(v.task, "debug");
});

test("heuristic treats a tiny greeting as chat difficulty 0", () => {
	const v = heuristic("halo", false);
	assert.equal(v.task, "chat");
	assert.equal(v.difficulty, 0);
	assert.equal(v.needsReasoning, false);
});

// --------------------------------------------------------------- thinking clamp

const claudeLike: ModelLike = {
	id: "claude-sonnet-5.5",
	reasoning: true,
	thinkingLevelMap: { off: null, minimal: null, low: "low", medium: "medium", high: "high", xhigh: "xhigh", max: "max" },
};

test("supportedThinkingLevels mirrors Pi's thinkingLevelMap rules", () => {
	assert.deepEqual(supportedThinkingLevels(claudeLike), ["low", "medium", "high", "xhigh", "max"]);
	// xhigh/max must be explicitly mapped to be offered.
	const noExtended: ModelLike = { id: "x", reasoning: true, thinkingLevelMap: { xhigh: null, max: null } };
	assert.deepEqual(supportedThinkingLevels(noExtended), ["off", "minimal", "low", "medium", "high"]);
	// A non-reasoning model only accepts off.
	assert.deepEqual(supportedThinkingLevels({ id: "y", reasoning: false }), ["off"]);
	// No map at all: the extended levels need an explicit mapping, as in Pi.
	assert.deepEqual(supportedThinkingLevels({ id: "z" }), ["off", "minimal", "low", "medium", "high"]);
});

test("clampThinking moves to the nearest supported level", () => {
	assert.equal(clampThinking(claudeLike, "off"), "low");
	assert.equal(clampThinking(claudeLike, "max"), "max");
	const noMax: ModelLike = { id: "x", reasoning: true, thinkingLevelMap: { max: null } };
	assert.equal(clampThinking(noMax, "max"), "high");
	assert.equal(clampThinking({ id: "y", reasoning: false }, "high"), "off");
});

test("resolveThinking applies the per-slot ceiling before clamping", () => {
	assert.equal(resolveThinking(claudeLike, "max", "high"), "high");
	assert.equal(resolveThinking(claudeLike, "low", "max"), "low");
	// Ceiling that the model does not support still clamps down to a real level.
	assert.equal(resolveThinking(claudeLike, "max", "off"), "low");
});

// -------------------------------------------------------------------- rule table

test("pickRoute: trivial chat goes to the fast slot with thinking off", () => {
	const cfg = isolatedConfig();
	const picked = pickRoute(cfg, heuristic("halo", false), "balanced");
	assert.equal(picked.slot, "fast");
	assert.equal(picked.thinking, "off");
	assert.equal(picked.ruleId, "chat-trivial");
});

test("pickRoute: hard debugging reaches the solid slot with high thinking", () => {
	const cfg = isolatedConfig();
	const verdict = {
		task: "debug",
		difficulty: 3,
		needsReasoning: true,
		needsVision: false,
		longHorizon: false,
		confidence: 0.8,
		source: "jev" as const,
	};
	const picked = pickRoute(cfg, verdict, "balanced");
	assert.equal(picked.slot, "solid");
	assert.equal(picked.thinking, "high");
});

test("pickRoute: frontier verdict + confidence reaches the frontier slot", () => {
	const cfg = isolatedConfig();
	const verdict = {
		task: "design",
		difficulty: 4,
		needsReasoning: true,
		needsVision: false,
		longHorizon: false,
		confidence: 0.9,
		source: "jev" as const,
	};
	const picked = pickRoute(cfg, verdict, "balanced");
	assert.equal(picked.slot, "frontier");
	assert.equal(picked.thinking, "max");
	assert.equal(picked.ruleId, "frontier-opus-max");
});

test("pickRoute: cheap budget caps the slot but keeps the rule's thinking", () => {
	const cfg = isolatedConfig();
	const verdict = {
		task: "design",
		difficulty: 4,
		needsReasoning: true,
		needsVision: false,
		longHorizon: false,
		confidence: 0.9,
		source: "jev" as const,
	};
	const picked = pickRoute(cfg, verdict, "cheap");
	assert.equal(picked.slot, "solid");
	assert.equal(picked.thinking, "max");
});

test("pickRoute: longHorizon raises the floor to solid", () => {
	const cfg = isolatedConfig();
	const verdict = {
		task: "coding",
		difficulty: 1,
		needsReasoning: false,
		needsVision: false,
		longHorizon: true,
		confidence: 0.9,
		source: "jev" as const,
	};
	const picked = pickRoute(cfg, verdict, "balanced");
	assert.equal(picked.slot, "solid");
	assert.match(picked.ruleId, /\+long$/);
	assert.ok(THINKING_ORDER.indexOf(picked.thinking) >= THINKING_ORDER.indexOf("high"));
});

test("pickRoute: a later overlay rule overrides an equal-strength earlier one", () => {
	const cfg = isolatedConfig({
		rules: [
			{ id: "base", when: { minDifficulty: 0 }, slot: "work", thinking: "low" },
			{ id: "override", when: { minDifficulty: 0 }, slot: "work", thinking: "xhigh" },
		],
	});
	const picked = pickRoute(cfg, heuristic("halo", false), "balanced");
	assert.equal(picked.ruleId, "override");
	assert.equal(picked.thinking, "xhigh");
});

// --------------------------------------------------------------------- failover

test("isFailoverWorthy: transient yes, auth/quota/abort no", () => {
	assert.equal(isFailoverWorthy({ message: { stopReason: "error", errorMessage: "503 Service Unavailable" } }), true);
	assert.equal(isFailoverWorthy({ message: { stopReason: "error", errorMessage: "429 rate limit" } }), true);
	assert.equal(isFailoverWorthy({ message: { stopReason: "error", errorMessage: "socket hang up" } }), true);
	assert.equal(isFailoverWorthy({ message: { stopReason: "error", errorMessage: "401 unauthorized" } }), false);
	assert.equal(isFailoverWorthy({ message: { stopReason: "error", errorMessage: "insufficient credit" } }), false);
	assert.equal(isFailoverWorthy({ message: { stopReason: "aborted", errorMessage: "500" } }), false);
	assert.equal(isFailoverWorthy(undefined), false);
});

test("failoverTarget skips tried slots and respects the budget ceiling", () => {
	const cfg = isolatedConfig();
	const target = failoverTarget(
		cfg,
		{ slot: "work", thinking: "high", budget: "balanced" },
		() => true,
	);
	// Automatic chain is strongest-first, so frontier wins.
	assert.equal(target?.slot, "frontier");
	assert.equal(target?.thinking, "high");

	const capped = failoverTarget(
		cfg,
		{ slot: "work", thinking: "high", budget: "cheap" },
		() => true,
	);
	assert.equal(capped?.slot, "solid");

	const exhausted = failoverTarget(
		cfg,
		{ slot: "work", thinking: "low", budget: "cheap", failoverTried: ["solid", "fast"] },
		() => true,
	);
	assert.equal(exhausted, undefined);
});

test("failoverTarget honours an explicit chain and unavailable models", () => {
	const cfg = isolatedConfig({ failover: { chain: { work: ["fast", "solid"] } } });
	assert.equal(failoverTarget(cfg, { slot: "work", thinking: "low", budget: "balanced" }, () => true)?.slot, "fast");
	const skipMissing = failoverTarget(
		cfg,
		{ slot: "work", thinking: "low", budget: "balanced" },
		(_slot, modelId) => modelId !== cfg.slots.fast,
	);
	assert.equal(skipMissing?.slot, "solid");
});

// ----------------------------------------------------------------- stall guard

test("detectStall flags a short announcement without a tool call", () => {
	const message = { role: "assistant", stopReason: "stop", content: [{ type: "text", text: "Sekarang saya tulis file-nya." }] };
	const decision = detectStall(message, { toolResultsSinceUser: 2, maxChars: 300 });
	assert.equal(decision.stall, true);
});

test("detectStall ignores tool calls, questions, and completion summaries", () => {
	const withTool = {
		role: "assistant",
		stopReason: "stop",
		content: [
			{ type: "text", text: "Sekarang saya tulis." },
			{ type: "toolCall", id: "1", name: "write", arguments: {} },
		],
	};
	assert.equal(detectStall(withTool, { toolResultsSinceUser: 2, maxChars: 300 }).stall, false);
	assert.equal(
		detectStall({ role: "assistant", stopReason: "stop", content: [{ type: "text", text: "Lanjut?" }] }, {
			toolResultsSinceUser: 2,
			maxChars: 300,
		}).stall,
		false,
	);
	assert.equal(
		detectStall(
			{ role: "assistant", stopReason: "stop", content: [{ type: "text", text: "Semua sudah selesai, silakan cek." }] },
			{ toolResultsSinceUser: 2, maxChars: 300 },
		).stall,
		false,
	);
	assert.equal(
		detectStall({ role: "assistant", stopReason: "error", content: [{ type: "text", text: "gas lanjut" }] }, {
			toolResultsSinceUser: 2,
			maxChars: 300,
		}).stall,
		false,
	);
	assert.equal(
		detectStall({ role: "assistant", stopReason: "stop", content: [{ type: "text", text: "x".repeat(400) }] }, {
			toolResultsSinceUser: 2,
			maxChars: 300,
		}).stall,
		false,
	);
});

test("detectStall ignores a vague short stop before any tool result", () => {
	const message = { role: "assistant", stopReason: "stop", content: [{ type: "text", text: "ok." }] };
	assert.equal(detectStall(message, { toolResultsSinceUser: 0, maxChars: 300 }).stall, false);
});

test("detectStall ignores a file listing that mentions 'berikutnya'", () => {
	// Real false positive observed in a live run: a list of results was
	// mistaken for a plan announcement because of a word in the middle.
	const text = [
		"File `.md` di cwd:",
		"",
		"- `griphub.md`",
		"- `key_cliproxyhome.md`",
		"- `cliproxyhome.md`",
		"- `inferhub.md`",
		"",
		"Langkah berikutnya belum ditentukan.",
	].join("\n");
	const decision = detectStall(
		{ role: "assistant", stopReason: "stop", content: [{ type: "text", text }] },
		{ toolResultsSinceUser: 2, maxChars: 300 },
	);
	assert.equal(decision.stall, false);
	assert.match(decision.reason, /list of results|completion/);
});

test("detectStall ignores intent words buried in prose", () => {
	const text =
		"Berkas konfigurasi itu memakai format JSON dan berisi daftar provider; nilai berikutnya adalah baseUrl. Semuanya sudah saya periksa.";
	const decision = detectStall(
		{ role: "assistant", stopReason: "stop", content: [{ type: "text", text }] },
		{ toolResultsSinceUser: 2, maxChars: 300 },
	);
	assert.equal(decision.stall, false);
});

test("detectStall still catches a short promise as the final sentence", () => {
	const text = "Saya sudah membaca file itu.\n\nSekarang saya tulis patch-nya.";
	const decision = detectStall(
		{ role: "assistant", stopReason: "stop", content: [{ type: "text", text }] },
		{ toolResultsSinceUser: 2, maxChars: 300 },
	);
	assert.equal(decision.stall, true);
});

// ------------------------------------------------- interplay with self-compact

test("context overflow is not failover-worthy (compaction fixes it, not another model)", () => {
	const overflow = [
		"400 This model's maximum context length is 131072 tokens",
		"prompt is too long: 1048577 tokens > 1048576 maximum",
		"Your input exceeds the context window of this model",
		"context_length_exceeded",
		"413 Request Entity Too Large",
	];
	for (const errorMessage of overflow) {
		assert.equal(isContextOverflowError(errorMessage), true, errorMessage);
		assert.equal(isFailoverWorthy({ message: { stopReason: "error", errorMessage } }), false, errorMessage);
	}
	// Throttling that mentions tokens is still transient.
	assert.equal(isContextOverflowError("429 Too many requests: too many tokens per minute"), false);
	assert.equal(isFailoverWorthy({ message: { stopReason: "error", errorMessage: "429 rate limit, too many tokens" } }), true);
	assert.equal(isFailoverWorthy({ message: { stopReason: "error", errorMessage: "503 UPSTREAM_UNAVAILABLE" } }), true);
});

const COMPACT_NOW = "Compact now: write your note_to_self (max 12000 chars: goal, ...) and call self_compact as your only tool call.";
const RESUME = "Continue from your saved note_to_self above (self-compact cycle 2). Perform only its unfinished NEXT ACTION.";

test("self-compact nudges are not treated as the user's task", () => {
	assert.equal(isSyntheticUserMessage(userText(COMPACT_NOW)), true);
	assert.equal(isSyntheticUserMessage(userText(RESUME)), true);
	assert.equal(isSyntheticUserMessage(userText("fix the flaky login test")), false);
	assert.equal(isSyntheticUserMessage(assistantText(COMPACT_NOW)), false);

	const messages = [userText("fix the flaky login test"), toolResult("read", "a"), userText(COMPACT_NOW), toolResult("bash", "b")];
	assert.equal(lastUserText(messages), "fix the flaky login test");
	assert.equal(currentTurnToolResults(messages), 2);
	const prompt = adaptiveProgressPrompt(messages, isolatedConfig());
	assert.match(prompt, /fix the flaky login test/);
	assert.doesNotMatch(prompt, /Compact now/);
	assert.doesNotMatch(contextualUserPrompt([userText("lanjut"), userText(RESUME)], isolatedConfig()), /saved note_to_self/);
});

test("registered stall text is synthetic too", () => {
	const text = "Kerjakan langkah berikutnya sekarang (custom stall text).";
	assert.equal(isSyntheticUserMessage(userText(text)), false);
	registerSyntheticUserText(text);
	assert.equal(isSyntheticUserMessage(userText(text)), true);
	assert.equal(lastUserText([userText("deploy the service"), userText(text)]), "deploy the service");
});

test("a self-compact handoff note (the real goal) still counts as the objective", () => {
	const note = "GOAL: migrate the billing table. NEXT ACTION: run the migration.";
	assert.equal(isSyntheticUserMessage(userText(note)), false);
	assert.equal(lastUserText([userText("old"), userText(note)]), note);
});

// Regression: short final answers in a subagent were nudged as stalls.
test("detectStall ignores short final answers without an intent word", () => {
	const opts = { toolResultsSinceUser: 3, maxChars: 60 };
	const answers = [
		"50",
		"FILE=src/core/limits.ts HEADROOM=8",
		"src/core/defaults.ts",
		"Yes, HEADROOM is exported.",
	];
	for (const text of answers) {
		const decision = detectStall({ role: "assistant", stopReason: "stop", content: [{ type: "text", text }] }, opts);
		assert.equal(decision.stall, false, `false positive on ${JSON.stringify(text)}`);
	}
});

test("detectStall still flags a short fragment that trails off mid-task", () => {
	const opts = { toolResultsSinceUser: 3, maxChars: 60 };
	for (const text of ["Cek config dulu:", "Membuka file limits.ts…", "Checking the tests...", "Membaca config"]) {
		const decision = detectStall({ role: "assistant", stopReason: "stop", content: [{ type: "text", text }] }, opts);
		assert.equal(decision.stall, true, `missed stall on ${JSON.stringify(text)}`);
	}
	// Same fragment before any tool result is not a stall.
	const early = detectStall(
		{ role: "assistant", stopReason: "stop", content: [{ type: "text", text: "Cek config dulu:" }] },
		{ toolResultsSinceUser: 0, maxChars: 60 },
	);
	assert.equal(early.stall, false);
});
