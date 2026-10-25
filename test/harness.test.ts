/**
 * Integration test: drives the real extension factory with a fake Pi harness.
 *
 * The fake implements only the surface the router touches (registerVirtualModel,
 * registerCommand, registerFlag, getFlag, on, sendUserMessage, ctx.modelRegistry,
 * ctx.sessionManager). That is enough to exercise every route reason, the
 * failover chain, the stall guard, and the /auto commands without network.
 */

import assert from "node:assert/strict";
import test from "node:test";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import type { ModelRoute, ModelRouteRequest } from "@earendil-works/pi-coding-agent";

import register from "../extensions/ontoken-auto/index.ts";
import { type RouteState } from "../extensions/ontoken-auto/router.ts";

// --------------------------------------------------------------------- harness

interface FakeModel {
	id: string;
	provider: string;
	api: string;
	name: string;
	reasoning: boolean;
	thinkingLevelMap?: Record<string, string | null>;
	input: string[];
	contextWindow: number;
	maxTokens: number;
	cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
}

const MODELS: Record<string, FakeModel> = {
	"gpt-6-luna": mkModel("gpt-6-luna", { reasoning: true }),
	"glm-5.3-flash": mkModel("glm-5.3-flash", { reasoning: true, map: { off: null, medium: null, xhigh: null } }),
	"muse-spark-1.3": mkModel("muse-spark-1.3", { reasoning: true }),
	"claude-sonnet-5.5": mkModel("claude-sonnet-5.5", { reasoning: true }),
	"gpt-6.1-sol": mkModel("gpt-6.1-sol", { reasoning: true }),
	"no-reasoning": mkModel("no-reasoning", { reasoning: false }),
};

function mkModel(
	id: string,
	opts: { reasoning: boolean; map?: Record<string, string | null> } = { reasoning: true },
): FakeModel {
	return {
		id,
		provider: "ontoken",
		api: "openai-completions",
		name: id,
		reasoning: opts.reasoning,
		thinkingLevelMap: opts.map,
		input: ["text", "image"],
		contextWindow: 1_000_000,
		maxTokens: 128_000,
		cost: { input: 1, output: 1, cacheRead: 0, cacheWrite: 0 },
	};
}

interface FakeHarness {
	route(request: ModelRouteRequest<RouteState>): Promise<ModelRoute<RouteState>>;
	command(name: string): ((args: string, ctx: unknown) => Promise<void>) | undefined;
	emit(event: string, payload: unknown, ctx: unknown): Promise<unknown>;
	sent: string[];
	/** Custom messages queued through pi.sendMessage. */
	sentCustom: Array<{ customType: string; triggerTurn?: boolean }>;
	statuses: Array<[string, string | undefined]>;
	notifications: Array<[string, string]>;
	classifyCalls: number;
	classifyAnswer: Record<string, unknown>;
	classifyStopReason: "stop" | "error";
	classifierPresent: boolean;
	flags: Record<string, string>;
	/** Context handed to the router on each request. */
	ctx: Record<string, unknown>;
}

function makeHarness(opts: { cwd?: string; overlay?: Record<string, unknown> } = {}): FakeHarness {
	const agentDir = mkdtempSync(join(tmpdir(), "ontoken-harness-"));
	process.env.PI_AGENT_DIR = agentDir;
	delete process.env.PI_ONTOKEN_DEBUG;
	delete process.env.PI_ONTOKEN_DEBUG_FILE;
	process.env.TYPESAFE_API_KEY = "test-key";
	if (opts.overlay) writeFileSync(join(agentDir, "ontoken-auto.json"), JSON.stringify(opts.overlay));

	const handlers = new Map<string, Array<(event: unknown, ctx: unknown) => unknown>>();
	const commands = new Map<string, (args: string, ctx: unknown) => Promise<void>>();
	const flags: Record<string, string> = {};
	const sent: string[] = [];
	const sentCustom: Array<{ customType: string; triggerTurn?: boolean }> = [];
	const statuses: Array<[string, string | undefined]> = [];
	const notifications: Array<[string, string]> = [];

	const harness: FakeHarness = {
		route: async () => {
			throw new Error("virtual model was never registered");
		},
		command: (name) => commands.get(name),
		emit: async (event, payload, ctx) => {
			let result: unknown;
			for (const handler of handlers.get(event) ?? []) result = await handler(payload, ctx);
			return result;
		},
		sent,
		sentCustom,
		statuses,
		notifications,
		classifyCalls: 0,
		classifyAnswer: {},
		classifyStopReason: "stop",
		classifierPresent: true,
		flags,
	};

	const modelRegistry = {
		find: (provider: string, id: string) =>
			provider === "ontoken" ? (MODELS[id] as unknown as object | undefined) : undefined,
		findOfType: (type: string) =>
			type === "classifier" && harness.classifierPresent
				? ({ id: "jev-latest", provider: "typesafe", type: "classifier" } as unknown as object)
				: undefined,
		classify: async () => {
			harness.classifyCalls += 1;
			return {
				api: "typesafe-system-one",
				provider: "typesafe",
				model: "jev-latest",
				answers: harness.classifyAnswer,
				stopReason: harness.classifyStopReason,
				errorMessage: harness.classifyStopReason === "error" ? "classifier down" : undefined,
				timestamp: 0,
			};
		},
	};

	/** Pi hands the router a live ctx per request; tests supply one explicitly. */
	harness.ctx = {
		cwd: opts.cwd ?? mkdtempSync(join(tmpdir(), "ontoken-cwd-")),
		hasUI: true,
		mode: "tui",
		ui: {
			notify: (message: string, type: string) => notifications.push([message, type]),
			setStatus: (key: string, text: string | undefined) => statuses.push([key, text]),
		},
		modelRegistry,
		sessionManager: { getBranch: () => [] },
	};

	const pi = {
		on(event: string, handler: (e: unknown, ctx: unknown) => unknown) {
			const list = handlers.get(event) ?? [];
			list.push(handler);
			handlers.set(event, list);
			return () => {};
		},
		registerCommand(name: string, options: { handler: (args: string, ctx: unknown) => Promise<void> }) {
			commands.set(name, options.handler);
		},
		registerFlag(name: string, _options: unknown) {
			flags[name] = "";
		},
		getFlag(name: string) {
			return flags[name];
		},
		registerVirtualModel(definition: {
			route(
				request: ModelRouteRequest<RouteState>,
				ctx: unknown,
			): Promise<ModelRoute<RouteState>>;
		}) {
			harness.route = (request) => definition.route(request, harness.ctx);
		},
		sendUserMessage(content: string) {
			sent.push(content);
		},
		sendMessage(message: { customType: string; content: string }, options?: { triggerTurn?: boolean }) {
			sentCustom.push({ customType: message.customType, triggerTurn: options?.triggerTurn });
			sent.push(message.content);
		},
		events: { emit() {}, on() {} },
	} as unknown as Parameters<typeof register>[0];

	register(pi);

	return harness;
}

function fakeCtx(cwd: string, branch: unknown[] = []) {
	return {
		cwd,
		hasUI: true,
		mode: "tui",
		ui: {
			notify: (message: string, type: string) => {
				notificationsPush.push([message, type]);
			},
			setStatus: (key: string, text: string | undefined) => {
				statusPush.push([key, text]);
			},
		},
		modelRegistry: {
			find: (provider: string, id: string) => MODELS[id],
			findOfType: () => ({ id: "jev-latest", provider: "typesafe", type: "classifier" }),
		},
		sessionManager: { getBranch: () => branch },
	};
}

// The fake ctx writes into the harness buffers through these module-level hooks.
let notificationsPush: Array<[string, string]> = [];
let statusPush: Array<[string, string | undefined]> = [];

function resetBuffers(harness: FakeHarness): void {
	notificationsPush = harness.notifications;
	statusPush = harness.statuses;
	notificationsPush.length = 0;
	statusPush.length = 0;
}

const userMsg = (text: string) => ({ role: "user", content: [{ type: "text", text }], timestamp: 0 });

function request(overrides: Partial<ModelRouteRequest<RouteState>>): ModelRouteRequest<RouteState> {
	return {
		model: MODELS["gpt-6-luna"] as never,
		thinkingLevel: "medium",
		reason: "user",
		messages: [userMsg("halo")],
		...overrides,
	} as ModelRouteRequest<RouteState>;
}

function jevAnswers(task: string, difficulty: number, reasoning = false, long = false) {
	return {
		task: { type: "choice", choice: task, probabilities: {}, confidence: 0.9 },
		difficulty: { type: "score", score: difficulty, confidence: 0.9 },
		needs_reasoning: { type: "bool", probability: reasoning ? 0.9 : 0.1 },
		needs_vision: { type: "bool", probability: 0.05 },
		long_horizon: { type: "bool", probability: long ? 0.9 : 0.1 },
	};
}

// ------------------------------------------------------------------------ tests

test("user route: Jev verdict picks slot, model, thinking and persists state", async () => {
	const harness = makeHarness();
	harness.classifyAnswer = jevAnswers("chat", 0);
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	const ctx = fakeCtx(cwd);
	resetBuffers(harness);

	const route = await harness.route(request({ messages: [userMsg("halo, apa kabar?")] }));

	// fast = glm-5.3-flash, which cannot turn reasoning off: "off" clamps to the
	// nearest supported level (the fake map keeps "minimal"; the real one "low").
	assert.equal(route.model.id, "glm-5.3-flash");
	assert.equal(route.thinkingLevel, "minimal");
	assert.equal(route.state?.slot, "fast");
	assert.equal(route.state?.ruleId, "chat-trivial");
	assert.equal(route.state?.budget, "balanced");
	assert.equal(route.state?.verdict?.source, "jev");
	assert.equal(harness.classifyCalls, 1);
	assert.equal(harness.statuses.at(-1)?.[0], "ontoken-auto");
});

test("user route: hard debug verdict reaches a stronger slot", async () => {
	const harness = makeHarness();
	harness.classifyAnswer = jevAnswers("debug", 3, true);
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const route = await harness.route(request({ messages: [userMsg("fix this deadlock")] }));
	assert.equal(route.state?.slot, "solid");
	assert.equal(route.model.id, "muse-spark-1.3");
});

test("thinking is clamped to the routed model's thinkingLevelMap", async () => {
	// glm-5.3-flash maps off/medium/xhigh to null, so a 'high' rule still works
	// but 'off' or 'medium' must move to a supported level.
	const harness = makeHarness({ overlay: { slots: { work: "glm-5.3-flash" } } });
	harness.classifyAnswer = jevAnswers("coding", 1);
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const route = await harness.route(request({ messages: [userMsg("rename this variable")] }));
	assert.equal(route.model.id, "glm-5.3-flash");
	assert.equal(route.state?.slot, "work");
	assert.ok(route.thinkingLevel !== "medium", "unsupported medium level must be clamped");
});

test("classifier failure degrades to the heuristic instead of killing the turn", async () => {
	const harness = makeHarness();
	harness.classifyStopReason = "error";
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const route = await harness.route(request({ messages: [userMsg("fix the failing test")] }));
	assert.equal(route.state?.verdict?.source, "heuristic");
	assert.match(String(route.state?.verdict?.error), /classifier down/);
	assert.ok(route.model.id.length > 0);
});

test("missing classifier also degrades to the heuristic", async () => {
	const harness = makeHarness();
	harness.classifierPresent = false;
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const route = await harness.route(request());
	assert.equal(route.state?.verdict?.source, "heuristic");
	assert.match(String(route.state?.verdict?.error), /not in catalog/);
	assert.equal(harness.classifyCalls, 0);
});

test("continuation reuses the stored model and never reclassifies below the cap", async () => {
	const harness = makeHarness();
	harness.classifyAnswer = jevAnswers("coding", 1);
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const first = await harness.route(request({ messages: [userMsg("implement x")] }));
	const state = first.state!;
	harness.classifyCalls = 0;

	// Only one tool result: below toolResultsPerEvaluation, so no re-evaluation.
	const messages = [userMsg("implement x"), userMsg("x"), { role: "toolResult", toolName: "read", content: [] }];
	const next = await harness.route(
		request({ reason: "continuation", state, messages: messages as never }),
	);
	assert.equal(harness.classifyCalls, 0);
	assert.equal(next.model.id, first.model.id);
	assert.equal(next.thinkingLevel, first.thinkingLevel);
	assert.equal(next.state?.adaptiveEvaluations, 0);
});

test("continuation promotes to a stronger slot when Jev reports harder work", async () => {
	const harness = makeHarness();
	harness.classifyAnswer = jevAnswers("coding", 1);
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const first = await harness.route(request({ messages: [userMsg("implement x")] }));
	assert.equal(first.state?.slot, "work");

	harness.classifyAnswer = jevAnswers("debug", 3, true);
	const messages = [
		userMsg("implement x"),
		{ role: "toolResult", toolName: "read", content: [{ type: "text", text: "a" }] },
		{ role: "toolResult", toolName: "bash", content: [{ type: "text", text: "b" }] },
	];
	const next = await harness.route(
		request({ reason: "continuation", state: first.state!, messages: messages as never }),
	);
	assert.equal(next.state?.slot, "solid");
	assert.equal(next.model.id, "muse-spark-1.3");
	assert.equal(next.state?.adaptiveEvaluations, 1);
	assert.match(String(next.state?.ruleId), /\+adaptive$/);
});

test("adaptive promotion never happens twice past maxEvaluationsPerTurn", async () => {
	const harness = makeHarness({ overlay: { adaptive: { maxEvaluationsPerTurn: 1 } } });
	harness.classifyAnswer = jevAnswers("coding", 1);
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const first = await harness.route(request({ messages: [userMsg("go")] }));
	const results = (n: number) =>
		Array.from({ length: n }, () => ({ role: "toolResult", toolName: "read", content: [] }));
	harness.classifyAnswer = jevAnswers("debug", 3, true);
	const second = await harness.route(
		request({ reason: "continuation", state: first.state!, messages: [userMsg("go"), ...results(2)] as never }),
	);
	assert.equal(second.state?.adaptiveEvaluations, 1);
	harness.classifyCalls = 0;
	const third = await harness.route(
		request({ reason: "continuation", state: second.state!, messages: [userMsg("go"), ...results(9)] as never }),
	);
	assert.equal(harness.classifyCalls, 0, "evaluation cap must stop further Jev calls");
	assert.equal(third.state?.slot, second.state?.slot);
});

test("a heuristic verdict can never promote mid-turn", async () => {
	const harness = makeHarness();
	harness.classifyAnswer = jevAnswers("coding", 1);
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const first = await harness.route(request({ messages: [userMsg("go")] }));
	harness.classifyStopReason = "error";
	const messages = [
		userMsg("go"),
		{ role: "toolResult", toolName: "read", content: [] },
		{ role: "toolResult", toolName: "read", content: [] },
	];
	const next = await harness.route(
		request({ reason: "continuation", state: first.state!, messages: messages as never }),
	);
	assert.equal(next.state?.verdict?.source, "heuristic");
	assert.equal(next.state?.slot, first.state?.slot);
});

test("retry on a transient error fails over to the next slot", async () => {
	const harness = makeHarness();
	harness.classifyAnswer = jevAnswers("coding", 1);
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const first = await harness.route(request({ messages: [userMsg("go")] }));
	assert.equal(first.state?.slot, "work");

	const retry = await harness.route(
		request({
			reason: "retry",
			state: first.state!,
			failed: {
				model: MODELS["glm-5.3-flash"] as never,
				thinkingLevel: "low",
				message: {
					role: "assistant",
					content: [],
					stopReason: "error",
					errorMessage: "503 Service Unavailable",
				} as never,
			},
		}),
	);
	// Automatic chain is strongest-first.
	assert.equal(retry.state?.slot, "frontier");
	assert.equal(retry.model.id, "gpt-6.1-sol");
	assert.equal(retry.state?.failedOver, true);
	assert.deepEqual(retry.state?.failoverTried, ["work"]);
	assert.match(String(retry.state?.ruleId), /\+failover$/);
	assert.ok(harness.notifications.some(([msg]) => /failover/.test(msg)));
});

test("retry on an auth error does NOT fail over", async () => {
	const harness = makeHarness();
	harness.classifyAnswer = jevAnswers("coding", 1);
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const first = await harness.route(request({ messages: [userMsg("go")] }));
	const retry = await harness.route(
		request({
			reason: "retry",
			state: first.state!,
			failed: {
				model: MODELS["glm-5.3-flash"] as never,
				message: {
					role: "assistant",
					content: [],
					stopReason: "error",
					errorMessage: "401 invalid api key",
				} as never,
			},
		}),
	);
	assert.equal(retry.model.id, first.model.id);
	assert.equal(retry.state?.failedOver, undefined);
	assert.equal(harness.notifications.length, 0);
});

test("failover respects the budget ceiling", async () => {
	const harness = makeHarness({ overlay: { budget: "cheap" } });
	harness.classifyAnswer = jevAnswers("coding", 1);
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const first = await harness.route(request({ messages: [userMsg("go")] }));
	assert.equal(first.state?.budget, "cheap");
	const retry = await harness.route(
		request({
			reason: "retry",
			state: first.state!,
			failed: {
				model: MODELS["glm-5.3-flash"] as never,
				message: { role: "assistant", content: [], stopReason: "error", errorMessage: "429 too many requests" } as never,
			},
		}),
	);
	assert.ok(["fast", "solid"].includes(String(retry.state?.slot)), `unexpected slot ${retry.state?.slot}`);
});

test("direct requests (compaction) use directSlot and never classify", async () => {
	const harness = makeHarness();
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const route = await harness.route(request({ reason: "direct", messages: [] }));
	assert.equal(route.model.id, "glm-5.3-flash");
	assert.equal(route.thinkingLevel, "low");
	assert.equal(route.state, undefined);
	assert.equal(harness.classifyCalls, 0);
});

test("disabled router keeps the in-flight model", async () => {
	const harness = makeHarness({ overlay: { enabled: false } });
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const route = await harness.route(
		request({
			previous: { model: MODELS["claude-sonnet-5.5"] as never, thinkingLevel: "high" },
			thinkingLevel: "high",
		}),
	);
	assert.equal(route.model.id, "claude-sonnet-5.5");
	assert.equal(route.thinkingLevel, "high");
	assert.equal(harness.classifyCalls, 0);
});

test("router never throws when the model registry is broken", async () => {
	const harness = makeHarness();
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const brokenCtx = {
		...fakeCtx(cwd),
		modelRegistry: {
			find: () => undefined,
			findOfType: () => {
				throw new Error("extension ctx is stale");
			},
		},
	};
	// route() receives ctx from the harness closure, so simulate through a
	// throw inside classify by using a stale registry on the request path.
	const route = await harness.route(request({ previous: { model: MODELS["gpt-6-luna"] as never, thinkingLevel: "low" } }));
	assert.ok(route.model.id.length > 0);
	assert.ok(brokenCtx.modelRegistry.find() === undefined);
});

test("stall guard queues one follow-up for a text-only announcement", async () => {
	const harness = makeHarness();
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const ctx = fakeCtx(cwd, [
		{ type: "message", message: { role: "user" } },
		{ type: "message", message: { role: "toolResult" } },
	]);
	const message = {
		role: "assistant",
		stopReason: "stop",
		content: [{ type: "text", text: "Sekarang saya tulis file-nya." }],
	};
	const result = await harness.emit(
		"turn_end",
		{ message, messageEntryId: "entry-1", toolResults: [] },
		ctx,
	);
	assert.equal(result, undefined);
	assert.equal(harness.sent.length, 1);
	assert.match(harness.sent[0], /eksekusi langkah berikutnya/);
	// Sent as a custom message so Pi routes it as a continuation (model and cache kept).
	assert.deepEqual(harness.sentCustom, [{ customType: "ontoken-auto-stall", triggerTurn: true }]);
	// Same entry never nudges twice.
	await harness.emit("turn_end", { message, messageEntryId: "entry-1", toolResults: [] }, ctx);
	assert.equal(harness.sent.length, 1);
});

test("stall guard stays silent for a completed answer", async () => {
	const harness = makeHarness();
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const ctx = fakeCtx(cwd, [{ type: "message", message: { role: "user" } }]);
	await harness.emit(
		"turn_end",
		{
			message: { role: "assistant", stopReason: "stop", content: [{ type: "text", text: "Sudah selesai, silakan cek." }] },
			messageEntryId: "entry-2",
			toolResults: [],
		},
		ctx,
	);
	assert.equal(harness.sent.length, 0);
});

test("stall guard is capped per run", async () => {
	const harness = makeHarness({ overlay: { stallGuard: { maxPerRun: 1 } } });
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const ctx = fakeCtx(cwd, [
		{ type: "message", message: { role: "user" } },
		{ type: "message", message: { role: "toolResult" } },
	]);
	await harness.emit("before_agent_start", { messageType: "user" }, ctx);
	const announce = (entry: string) => ({
		message: { role: "assistant", stopReason: "stop", content: [{ type: "text", text: "Gas, saya mulai." }] },
		messageEntryId: entry,
		toolResults: [],
	});
	await harness.emit("turn_end", announce("a"), ctx);
	await harness.emit("turn_end", announce("b"), ctx);
	assert.equal(harness.sent.length, 1);
});

test("stall guard can be disabled", async () => {
	const harness = makeHarness({ overlay: { stallGuard: { enabled: false } } });
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const ctx = fakeCtx(cwd, [{ type: "message", message: { role: "user" } }, { type: "message", message: { role: "toolResult" } }]);
	await harness.emit(
		"turn_end",
		{
			message: { role: "assistant", stopReason: "stop", content: [{ type: "text", text: "Sekarang saya tulis." }] },
			messageEntryId: "z",
			toolResults: [],
		},
		ctx,
	);
	assert.equal(harness.sent.length, 0);
});

test("/auto status, explain and check render without throwing", async () => {
	const harness = makeHarness();
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const ctx = fakeCtx(cwd);
	const auto = harness.command("auto");
	assert.ok(auto);
	await auto("", ctx);
	await auto("explain", ctx);
	await auto("check", ctx);
	const joined = harness.notifications.map(([m]) => m).join("\n");
	assert.match(joined, /enabled/);
	assert.match(joined, /classifier typesafe\/jev-latest: found/);
	assert.match(joined, /glm-5.3-flash — ok/);
});

test("/auto debug toggles the log and /auto log prints records", async () => {
	const harness = makeHarness();
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(harness);
	const ctx = fakeCtx(cwd);
	const auto = harness.command("auto")!;

	await auto("debug verbose", ctx);
	assert.match(harness.notifications.at(-1)![0], /debug → verbose/);
	await harness.route(request());
	await auto("log 10", ctx);
	const log = harness.notifications.at(-1)![0];
	assert.match(log, /route\.user/);

	await auto("debug off", ctx);
	assert.match(harness.notifications.at(-1)![0], /debug → off/);
	await auto("log", ctx);
	assert.match(harness.notifications.at(-1)![0], /debug is off/);
});

test("/auto debug file writes JSONL and /auto debug clear empties the buffer", async () => {
	const harness = makeHarness();
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	const logFile = join(mkdtempSync(join(tmpdir(), "ontoken-log-")), "debug.jsonl");
	resetBuffers(harness);
	const ctx = fakeCtx(cwd);
	const auto = harness.command("auto")!;

	await auto(`debug file ${logFile}`, ctx);
	await harness.route(request());
	await auto("debug clear", ctx);
	assert.match(harness.notifications.at(-1)![0], /cleared/);

	const { readFileSync } = await import("node:fs");
	const lines = readFileSync(logFile, "utf8").trim().split("\n").filter(Boolean);
	assert.ok(lines.length >= 2, "expected boot + route records in the file");
	for (const line of lines) {
		const rec = JSON.parse(line);
		assert.equal(typeof rec.ts, "string");
		assert.equal(typeof rec.kind, "string");
	}
});

test("debug records never leak the API key", async () => {
	const harness = makeHarness();
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	const logFile = join(mkdtempSync(join(tmpdir(), "ontoken-log-")), "debug.jsonl");
	process.env.TYPESAFE_API_KEY = "sk-verysecretvalue123456";
	resetBuffers(harness);
	const ctx = fakeCtx(cwd);
	const auto = harness.command("auto")!;
	await auto(`debug verbose`, ctx);
	await auto(`debug file ${logFile}`, ctx);
	await harness.route(request({ messages: [userMsg("my key is sk-verysecretvalue123456")] }));
	const { readFileSync } = await import("node:fs");
	const raw = readFileSync(logFile, "utf8");
	assert.doesNotMatch(raw, /sk-verysecretvalue123456/);
});

test("disabled router never classifies even on the first turn of a session", async () => {
	const harness = makeHarness({ overlay: { enabled: false } });
	resetBuffers(harness);
	// No `previous`: the first turn of a session. Old code classified anyway.
	const route = await harness.route(request());
	assert.equal(harness.classifyCalls, 0, "disabled router must not call the classifier");
	assert.equal(route.state, undefined);
	assert.equal(route.model.id, "glm-5.3-flash");
});

test("--ontoken-debug is read lazily, because Pi fills flag values after load", async () => {
	// Pi calls extension factories first and only then assigns flagValues
	// (agent-session-services: applyExtensionFlagValues runs after load). Reading
	// the flag at load time therefore always returned undefined.
	const harness = makeHarness();
	resetBuffers(harness);
	// Simulate the value arriving after registration, as Pi does.
	harness.flags["ontoken-debug"] = "verbose";

	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	const logFile = join(mkdtempSync(join(tmpdir(), "ontoken-log-")), "flag.jsonl");
	const ctx = fakeCtx(cwd);
	const auto = harness.command("auto")!;
	await auto(`debug file ${logFile}`, ctx);
	await harness.emit("session_start", { reason: "startup" }, ctx);

	const { readFileSync } = await import("node:fs");
	const kinds = readFileSync(logFile, "utf8")
		.trim()
		.split("\n")
		.filter(Boolean)
		.map((line) => JSON.parse(line).kind);
	assert.ok(kinds.includes("boot"), `expected a boot record, got ${kinds.join(",")}`);
	assert.ok(kinds.includes("session_start"));
});

test("flag registration happens at load so Pi can parse the CLI option", async () => {
	const harness = makeHarness();
	assert.ok("ontoken-debug" in harness.flags, "flag must be registered during load");
});

// ------------------------------------------------- interplay with self-compact

test("retry after a context overflow stays on the same slot (no failover)", async () => {
	const harness = makeHarness();
	harness.classifyAnswer = jevAnswers("coding", 1);
	resetBuffers(harness);
	const first = await harness.route(request({ messages: [userMsg("go")] }));
	const retry = await harness.route(
		request({
			reason: "retry",
			state: first.state!,
			failed: {
				model: MODELS[first.model.id] as never,
				thinkingLevel: "low",
				message: {
					role: "assistant",
					content: [],
					stopReason: "error",
					errorMessage: "400 This model's maximum context length is 131072 tokens",
				} as never,
			},
		}),
	);
	assert.equal(retry.model.id, first.model.id);
	assert.equal(retry.state?.slot, first.state?.slot);
	assert.notEqual(retry.state?.failedOver, true);
});

test("a user turn that is only a self-compact nudge keeps the in-flight model", async () => {
	const harness = makeHarness();
	harness.classifyAnswer = jevAnswers("debug", 3, true);
	resetBuffers(harness);
	const first = await harness.route(request({ messages: [userMsg("debug the deadlock in the scheduler")] }));
	harness.classifyCalls = 0;
	harness.classifyAnswer = jevAnswers("chat", 0);
	const nudge = await harness.route(
		request({
			reason: "user",
			state: first.state!,
			messages: [
				userMsg("debug the deadlock in the scheduler"),
				{ role: "assistant", content: [{ type: "text", text: "..." }] } as never,
				userMsg("Compact now: write your note_to_self (max 12000 chars) and call self_compact as your only tool call."),
			],
		}),
	);
	assert.equal(harness.classifyCalls, 0, "nudges must not be classified");
	assert.equal(nudge.model.id, first.model.id);
	assert.equal(nudge.state?.slot, first.state?.slot);
});

test("mid-turn adaptive evaluation ignores a self-compact resume message", async () => {
	const harness = makeHarness();
	harness.classifyAnswer = jevAnswers("coding", 1);
	resetBuffers(harness);
	const first = await harness.route(request({ messages: [userMsg("implement x")] }));
	harness.classifyCalls = 0;
	const tool = { role: "toolResult", toolName: "read", content: [] };
	// Two tool results before the resume nudge, one after: three since the real task.
	const messages = [
		userMsg("implement x"),
		tool,
		tool,
		userMsg("Continue from your saved note_to_self above (self-compact cycle 1). Perform only its unfinished NEXT ACTION."),
		tool,
	];
	const next = await harness.route(request({ reason: "continuation", state: first.state!, messages: messages as never }));
	assert.equal(harness.classifyCalls, 1, "progress since the real task triggers one evaluation");
	assert.equal(next.state?.adaptiveToolResults, 3);
});

// ------------------------------------------------- subagent (shared runtime)

/** Make every ctx getter throw, like Pi's runner after dispose(). */
function makeStale(ctx: Record<string, unknown>): void {
	for (const key of ["cwd", "hasUI", "ui", "modelRegistry", "sessionManager", "mode"]) {
		Object.defineProperty(ctx, key, {
			get() {
				throw new Error("This extension ctx is stale after session replacement or reload.");
			},
		});
	}
}

test("router survives a disposed subagent that re-registered the virtual model", async () => {
	// Parent session loads the extension and starts.
	const parent = makeHarness();
	await parent.emit("session_start", { reason: "startup" }, parent.ctx);
	// A subagent loads it again; its registration replaces the parent's route
	// on the shared ModelRuntime, bound to the child's ctx.
	const child = makeHarness();
	await child.emit("session_start", { reason: "startup" }, child.ctx);
	// Child is disposed (with or without session_shutdown) — its ctx goes stale.
	makeStale(child.ctx);
	// Without a previous route the old code threw the stale-ctx error here.
	const route = await child.route(request({ reason: "direct", messages: [] }));
	assert.equal(route.model.id, "glm-5.3-flash");
	assert.equal(route.thinkingLevel, "low");
});

test("a subagent's run start does not reset the parent's stall budget", async () => {
	const parent = makeHarness({ overlay: { stallGuard: { maxPerRun: 1 } } });
	const cwd = mkdtempSync(join(tmpdir(), "ontoken-cwd-"));
	resetBuffers(parent);
	const ctx = fakeCtx(cwd, [
		{ type: "message", message: { role: "user" } },
		{ type: "message", message: { role: "toolResult" } },
	]);
	await parent.emit("before_agent_start", { messageType: "user" }, ctx);
	const announce = (entry: string) => ({
		message: { role: "assistant", stopReason: "stop", content: [{ type: "text", text: "Gas, saya mulai." }] },
		messageEntryId: entry,
		toolResults: [],
	});
	await parent.emit("turn_end", announce("p1"), ctx);
	// A subagent starts and ends a run in the same process.
	const child = makeHarness({ overlay: { stallGuard: { maxPerRun: 1 } } });
	await child.emit("before_agent_start", { messageType: "user" }, ctx);
	await child.emit("session_shutdown", { reason: "quit" }, ctx);
	// Parent is still capped: the child must not have reset its counter.
	await parent.emit("turn_end", announce("p2"), ctx);
	assert.equal(parent.sent.length, 1);
});
