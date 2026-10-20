/**
 * OnToken Auto — Jev-routed virtual model `ontoken/auto`.
 *
 * Per user prompt, TypeSafe Jev classifies the task; this extension picks an
 * OnToken model + thinking level. Overlay config:
 *   ~/.pi/agent/ontoken-auto.json
 *   <cwd>/.pi/ontoken-auto.json
 *
 * Usage: /model ontoken/auto   then  /auto  /auto debug  /auto explain  /auto log
 *
 * Debug mode:
 *   PI_ONTOKEN_DEBUG=summary|verbose|off   (env)
 *   PI_ONTOKEN_DEBUG_FILE=/tmp/ontoken.jsonl
 *   --ontoken-debug <level>                (CLI flag)
 *   { "debug": { "enabled": true, "level": "verbose", "file": "..." } }   (overlay)
 */

import type { Message } from "@earendil-works/pi-ai";
import type {
	ExtensionAPI,
	ExtensionCommandContext,
	ExtensionContext,
	ModelRoute,
	ModelRouteRequest,
} from "@earendil-works/pi-coding-agent";
import {
	type AutoConfig,
	type BudgetName,
	type DebugConfig,
	type DebugLevel,
	type SlotName,
	type ThinkingLevel,
	ensureJevKey,
	globalConfigPath,
	isBudgetName,
	isSlotName,
	loadConfig,
	slotIndex,
	writeGlobalOverlay,
} from "./config.ts";
import {
	type MessageLike,
	type ModelLike,
	type RouteState,
	type Verdict,
	DebugLog,
	adaptiveProgressPrompt,
	clippedTail,
	contextualUserPrompt,
	currentTurnToolResults,
	detectStall,
	failoverTarget,
	formatVerdict,
	heuristic,
	isFailoverWorthy,
	pickRoute,
	redactClassifierText,
	resolveThinking,
	strongerThinking,
	supportedThinkingLevels,
	userHasImage,
} from "./router.ts";

type AutoRequest = ModelRouteRequest<RouteState>;

/** Last routing decision, rendered by `/auto explain`. */
let lastExplain = "no routing yet this process";
/** Debug log, rebuilt whenever the effective debug config changes. */
let debugLog: DebugLog | undefined;
let debugKey = "";
/** Follow-ups queued by the stall guard in the current run. */
let nudgesThisRun = 0;
/** Assistant entries already nudged, so one message can never nudge twice. */
const nudgedEntries = new Set<string>();

function debugFor(cfg: AutoConfig): DebugLog {
	const key = JSON.stringify(cfg.debug);
	if (!debugLog || key !== debugKey) {
		debugLog = new DebugLog(cfg.debug);
		debugKey = key;
	}
	return debugLog;
}

function safeLoadConfig(cwd: string | undefined): AutoConfig {
	try {
		return loadConfig(cwd);
	} catch {
		// A broken defaults.json must not brick the router; retry without cwd.
		return loadConfig();
	}
}

function describeModel(model: ModelLike | undefined): string {
	if (!model) return "unknown";
	return `${model.id}`;
}

/** Levels the routed model supports, used by `/auto check`. */
function thinkingSummary(model: ModelLike): string {
	return supportedThinkingLevels(model).join("/");
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

async function classify(
	request: AutoRequest,
	ctx: ExtensionContext,
	cfg: AutoConfig,
	dbg: DebugLog,
	promptOverride?: string,
): Promise<Verdict> {
	const prompt = (promptOverride ?? contextualUserPrompt(request.messages as readonly MessageLike[], cfg)).slice(
		0,
		cfg.jev.promptChars,
	);
	const hasImage = userHasImage(request.messages as readonly MessageLike[]);
	const startedAt = Date.now();

	if (!ensureJevKey(cfg)) {
		dbg.record("classify.skip", { reason: "no TypeSafe key", promptChars: prompt.length });
		return { ...heuristic(prompt, hasImage), error: "no TypeSafe key" };
	}
	const jev = ctx.modelRegistry.findOfType(
		"classifier",
		cfg.jev.classifierProvider,
		cfg.jev.classifierId,
	);
	if (!jev) {
		dbg.record("classify.skip", {
			reason: "classifier not in catalog",
			provider: cfg.jev.classifierProvider,
			id: cfg.jev.classifierId,
		});
		return { ...heuristic(prompt, hasImage), error: "classifier not in catalog" };
	}

	if (dbg.allows("verbose")) {
		dbg.record("classify.request", {
			classifier: `${cfg.jev.classifierProvider}/${cfg.jev.classifierId}`,
			hasImage,
			promptChars: prompt.length,
			prompt: dbg.includePrompts ? prompt : "[omitted]",
			mode: promptOverride ? "adaptive" : "user",
		});
	}

	const result = await ctx.modelRegistry.classify(
		jev,
		{
			state: { prompt, has_image: hasImage },
			questions: {
				task: {
					type: "choice",
					instructions:
						"What kind of software-agent work does `prompt` ask for? Pick the single best fit.",
					criteria: {
						chat: "Short question, explanation, or conversation with no code changes",
						lookup: "Look up a fact, file, or API with little synthesis",
						mechanical: "Rename, format, boilerplate, config, or other mechanical edits",
						coding: "Implement or change features, tests, or code in a straightforward way",
						debug: "Find or fix a defect, failing test, or unexpected behavior",
						design: "Architecture, API shape, or cross-cutting design tradeoffs",
						review: "Review, critique, or audit existing code or a plan",
						research: "Investigate options, docs, or the codebase before deciding",
					},
				},
				difficulty: {
					type: "score",
					instructions: "How demanding is the work in `prompt` for a coding agent?",
					criteria: [
						"trivial one-shot answer",
						"routine local change",
						"involved multi-file or careful work",
						"hard debugging or design",
						"frontier: subtle, high-stakes, or very large",
					],
				},
				needs_reasoning: {
					type: "bool",
					instructions: "Does this need extended chain-of-thought / high reasoning effort?",
					criteria: {
						true: "Tradeoffs, subtle bugs, proofs, or ambiguous design",
						false: "Straightforward lookup, chat, or mechanical edit",
					},
				},
				needs_vision: {
					type: "bool",
					instructions: "Must the model read an image/screenshot in this turn?",
					criteria: {
						true: "has_image is true or the prompt is about a screenshot/UI capture",
						false: "Text only",
					},
				},
				long_horizon: {
					type: "bool",
					instructions: "Is this a long, multi-file, or multi-session effort?",
					criteria: {
						true: "Many files, migration, or open-ended exploration",
						false: "Fits in a short turn",
					},
				},
			},
		},
		{ signal: request.signal },
	);

	const elapsedMs = Date.now() - startedAt;
	if (result.stopReason !== "stop") {
		dbg.record("classify.error", {
			elapsedMs,
			stopReason: result.stopReason,
			errorMessage: result.errorMessage,
		});
		return { ...heuristic(prompt, hasImage), error: result.errorMessage || result.stopReason };
	}

	const taskA = result.answers.task;
	const diffA = result.answers.difficulty;
	const reasonA = result.answers.needs_reasoning;
	const visionA = result.answers.needs_vision;
	const longA = result.answers.long_horizon;
	const task = taskA?.type === "choice" ? taskA.choice : "coding";
	const difficulty = diffA?.type === "score" ? Math.round(diffA.score) : 1;
	const needsReasoning = reasonA?.type === "bool" ? reasonA.probability >= 0.5 : false;
	const needsVision = hasImage || (visionA?.type === "bool" ? visionA.probability >= 0.5 : false);
	const longHorizon = longA?.type === "bool" ? longA.probability >= 0.5 : false;
	const confs = [taskA, diffA, reasonA]
		.map((a) => (a && "confidence" in a ? Number(a.confidence) : 0.5))
		.filter((n) => Number.isFinite(n));
	const confidence = confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : 0.5;

	const verdict: Verdict = {
		task,
		difficulty: Math.min(4, Math.max(0, difficulty)),
		needsReasoning,
		needsVision,
		longHorizon,
		confidence,
		source: "jev",
	};

	dbg.record("classify.result", {
		elapsedMs,
		task,
		difficulty: verdict.difficulty,
		needsReasoning,
		needsVision,
		longHorizon,
		confidence: Number(confidence.toFixed(3)),
		usage: result.usage,
	});
	return verdict;
}

/**
 * classify() that never throws. A Jev outage, a stale extension ctx
 * (reload/teardown race), or a catalog hiccup degrades to the heuristic
 * verdict — never kills the turn (which surfaces as the agent going idle
 * with work left). The degradation is visible in `/auto explain`.
 */
async function classifySafe(
	request: AutoRequest,
	ctx: ExtensionContext,
	cfg: AutoConfig,
	dbg: DebugLog,
	promptOverride?: string,
): Promise<Verdict> {
	try {
		return await classify(request, ctx, cfg, dbg, promptOverride);
	} catch (err) {
		const message = err instanceof Error ? err.message : String(err);
		const stale = /stale/i.test(message);
		dbg.record("classify.error", { thrown: message, stale });
		let prompt = "";
		let hasImage = false;
		try {
			prompt = (promptOverride ?? contextualUserPrompt(request.messages as readonly MessageLike[], cfg)).slice(
				0,
				cfg.jev.promptChars,
			);
			hasImage = userHasImage(request.messages as readonly MessageLike[]);
		} catch {
			// request.messages should always be readable; keep the empty defaults.
		}
		return {
			...heuristic(prompt, hasImage),
			error: stale
				? "router ctx stale, heuristic fallback"
				: `classifier error, heuristic fallback: ${message.slice(0, 160)}`,
		};
	}
}

// ---------------------------------------------------------------------------
// Model resolution
// ---------------------------------------------------------------------------

interface Resolved {
	model: NonNullable<ModelRoute<RouteState>["model"]>;
	slot: SlotName;
}

function resolveModel(ctx: ExtensionContext, cfg: AutoConfig, slot: SlotName, dbg: DebugLog): Resolved {
	const id = cfg.slots[slot];
	const model = ctx.modelRegistry.find(cfg.provider, id);
	if (model) return { model, slot };
	for (const fallback of cfg.slotOrder) {
		const candidate = ctx.modelRegistry.find(cfg.provider, cfg.slots[fallback]);
		if (candidate) {
			dbg.record("model.missing", { wanted: `${cfg.provider}/${id}`, used: `${cfg.provider}/${candidate.id}` });
			return { model: candidate, slot: fallback };
		}
	}
	throw new Error(`No OnToken models from roster are in the catalog (provider ${cfg.provider})`);
}

function routeTo(
	ctx: ExtensionContext,
	cfg: AutoConfig,
	dbg: DebugLog,
	slot: SlotName,
	thinking: ThinkingLevel,
	state?: RouteState,
): ModelRoute<RouteState> {
	const resolved = resolveModel(ctx, cfg, slot, dbg);
	const applied = resolveThinking(resolved.model, thinking, cfg.maxThinking?.[resolved.slot]);
	const base: RouteState = state ?? {
		slot: resolved.slot,
		modelId: resolved.model.id,
		thinking: applied,
		budget: cfg.budget,
	};
	// resolveModel can fall back to a different slot when the configured
	// model is missing from the catalog — keep the persisted state truthful
	// so the next continuation/failover reasons about the actual model.
	if (base.slot !== resolved.slot || base.modelId !== resolved.model.id) {
		return {
			model: resolved.model,
			thinkingLevel: applied,
			state: { ...base, slot: resolved.slot, modelId: resolved.model.id, thinking: applied },
		};
	}
	return { model: resolved.model, thinkingLevel: applied, state: base };
}

/** The thinking level the router should target, honouring `thinkingPolicy`. */
function wantedThinking(
	cfg: AutoConfig,
	picked: ThinkingLevel,
	request: AutoRequest,
	state?: RouteState,
): ThinkingLevel {
	if (cfg.thinkingPolicy === "selection") {
		return (state?.thinking ?? request.thinkingLevel ?? picked) as ThinkingLevel;
	}
	return picked;
}

function statusLine(cfg: AutoConfig, v: Verdict | undefined, slot: SlotName, modelId: string, thinking: string): string {
	const src = v ? (v.source === "jev" ? "jev" : "heur") : "auto";
	const detail = v ? `${v.task} d${v.difficulty}` : cfg.budget;
	return `${src} ${detail} → ${modelId} · ${thinking}`;
}

function setStatus(ctx: ExtensionContext, text: string | undefined): void {
	if (!ctx.hasUI) return;
	try {
		ctx.ui.setStatus("ontoken-auto", text);
	} catch {
		// Stale ctx after teardown — status is cosmetic.
	}
}

function notify(ctx: ExtensionContext, message: string, type: "info" | "warning" | "error" = "info"): void {
	if (!ctx.hasUI) return;
	try {
		ctx.ui.notify(message, type);
	} catch {
		// Stale ctx after teardown.
	}
}

// ---------------------------------------------------------------------------
// Extension
// ---------------------------------------------------------------------------

export default function (pi: ExtensionAPI) {
	// CLI flag: --ontoken-debug <off|summary|verbose>
	pi.registerFlag("ontoken-debug", {
		description: "OnToken auto-router debug level: off | summary | verbose",
		type: "string",
	});

	const flagLevel = (() => {
		try {
			const raw = String(pi.getFlag("ontoken-debug") ?? "").toLowerCase();
			return ["off", "summary", "verbose"].includes(raw) ? (raw as DebugLevel) : undefined;
		} catch {
			return undefined;
		}
	})();

	const boot = safeLoadConfig(undefined);
	if (flagLevel) {
		boot.debug.enabled = flagLevel !== "off";
		boot.debug.level = flagLevel;
	}
	ensureJevKey(boot);
	debugFor(boot).record("boot", {
		provider: boot.provider,
		virtual: `${boot.virtual.provider}/${boot.virtual.id}`,
		budget: boot.budget,
		enabled: boot.enabled,
		debugLevel: boot.debug.level,
		slots: boot.slots,
		jevKey: ensureJevKey(boot) ? "present" : "missing",
	});

	/** Effective config for a request/command, with the CLI flag applied. */
	function configFor(cwd: string | undefined): AutoConfig {
		const cfg = safeLoadConfig(cwd);
		if (flagLevel) {
			cfg.debug.enabled = flagLevel !== "off";
			cfg.debug.level = flagLevel;
		}
		return cfg;
	}

	pi.registerVirtualModel<RouteState>({
		provider: boot.virtual.provider,
		id: boot.virtual.id,
		name: boot.virtual.name,
		thinkingLevels: ["off", "minimal", "low", "medium", "high", "xhigh", "max"],
		contextWindow: 1_000_000,
		maxTokens: 128_000,
		async route(request, ctx) {
			const startedAt = Date.now();
			let cfg: AutoConfig;
			try {
				cfg = configFor(ctx.cwd);
			} catch {
				cfg = boot;
			}
			const dbg = debugFor(cfg);
			try {
				return await routeOnce(request, ctx, cfg, dbg);
			} catch (err) {
				// The router must never kill a turn. A Jev outage, a stale
				// extension ctx (reload / newSession / fork / switchSession /
				// teardown race), or a catalog hiccup would otherwise end the
				// turn abruptly — surfacing as the agent going idle with work
				// left. Fall back to the in-flight model so the turn continues;
				// the failure is recorded for /auto explain.
				const message = err instanceof Error ? err.message : String(err);
				lastExplain = `router error, kept in-flight model (${message.slice(0, 200)})`;
				dbg.record("route.error", {
					reason: request.reason,
					elapsedMs: Date.now() - startedAt,
					error: message,
					hadPrevious: Boolean(request.previous),
				});
				if (request.previous) {
					return {
						model: request.previous.model,
						thinkingLevel: request.previous.thinkingLevel ?? "low",
						state: request.state,
					};
				}
				throw err;
			}
		},
	});

	async function routeOnce(
		request: AutoRequest,
		ctx: ExtensionContext,
		cfg: AutoConfig,
		dbg: DebugLog,
	): Promise<ModelRoute<RouteState>> {
		const startedAt = Date.now();

		// Session teardown race: pi can still ask a mid-turn loop for a
		// continuation route after dispose() invalidated the extension runner
		// (quit / newSession / fork / switchSession / reload). Any ctx access
		// then throws the "extension ctx is stale" error into the transcript.
		// Bail without touching ctx; pi discards the result anyway.
		if (request.signal?.aborted && request.previous) {
			dbg.record("route.aborted", { reason: request.reason });
			return {
				model: request.previous.model,
				thinkingLevel: request.previous.thinkingLevel ?? "low",
				state: request.state,
			};
		}

		if (dbg.allows("verbose")) {
			dbg.record("route.request", {
				reason: request.reason,
				selection: `${request.model.provider}/${request.model.id}`,
				selectionThinking: request.thinkingLevel,
				messages: request.messages.length,
				toolResults: currentTurnToolResults(request.messages as readonly MessageLike[]),
				state: request.state,
				previous: request.previous ? `${request.previous.model.provider}/${request.previous.model.id}` : undefined,
			});
		}

		// ---------------------------------------------------------- direct
		// Compaction summaries and extension streamSimple() calls. Pi ignores
		// router state for `direct`, so omit it entirely rather than persisting
		// a state the session branch would otherwise carry.
		if (request.reason === "direct") {
			const resolved = resolveModel(ctx, cfg, cfg.directSlot, dbg);
			const thinking = resolveThinking(resolved.model, "low", cfg.maxThinking?.[resolved.slot]);
			dbg.record("route.direct", {
				slot: resolved.slot,
				modelId: resolved.model.id,
				thinking,
				elapsedMs: Date.now() - startedAt,
			});
			return { model: resolved.model, thinkingLevel: thinking };
		}

		// --------------------------------------------------------- retry
		// Pi retries failed requests with reason "retry". If the failed request
		// belongs to this router and the error is transient, move the turn to
		// the next slot in the failover chain instead of re-hitting it.
		if (request.reason === "retry" && request.state && request.failed) {
			const state = request.state;
			const failedBelongsToRouter = state.modelId === request.failed.model.id;
			const worthy = isFailoverWorthy(request.failed);
			if (dbg.allows("verbose")) {
				dbg.record("route.retry", {
					failedModel: request.failed.model.id,
					stopReason: request.failed.message?.stopReason,
					errorMessage: request.failed.message?.errorMessage,
					belongsToRouter: failedBelongsToRouter,
					failoverWorthy: worthy,
					failoverEnabled: cfg.failover.enabled,
				});
			}
			if (cfg.failover.enabled && failedBelongsToRouter && worthy) {
				const target = failoverTarget(cfg, state, (slot, modelId) =>
					Boolean(ctx.modelRegistry.find(cfg.provider, modelId)),
				);
				if (target) {
					const targetModel = ctx.modelRegistry.find(cfg.provider, target.modelId);
					if (!targetModel) throw new Error(`failover model ${cfg.provider}/${target.modelId} is not in the catalog`);
					const nextState: RouteState = {
						...state,
						slot: target.slot,
						modelId: target.modelId,
						failoverTried: [...(state.failoverTried ?? []), state.slot],
						failedOver: true,
						ruleId: `${state.ruleId ?? "fallback"}+failover`,
					};
					const applied = resolveThinking(
						targetModel as unknown as ModelLike,
						state.thinking,
						cfg.maxThinking?.[target.slot],
					);
					nextState.thinking = applied;
					lastExplain = [
						`failover: ${state.modelId} failed (${request.failed.message?.errorMessage ?? "error"})`,
						`→ ${target.slot}  ${target.modelId}  thinking ${applied}`,
					].join("\n");
					dbg.record("route.failover", {
						from: state.modelId,
						to: target.modelId,
						toSlot: target.slot,
						thinking: applied,
						tried: nextState.failoverTried,
						errorMessage: request.failed.message?.errorMessage,
					});
					setStatus(ctx, `failover ${state.modelId} → ${target.modelId}`);
					if (cfg.failover.notify) {
						notify(
							ctx,
							`ontoken-auto failover: ${state.modelId} → ${target.modelId} (${
								request.failed.message?.errorMessage ?? "transient error"
							})`,
							"warning",
						);
					}
					return { model: targetModel, thinkingLevel: applied, state: nextState };
				}
				lastExplain = `failover: no available slot left for ${state.modelId} (${
					request.failed.message?.errorMessage ?? "error"
				})`;
				dbg.record("route.failover-exhausted", {
					modelId: state.modelId,
					tried: [...(state.failoverTried ?? []), state.slot],
					budget: state.budget,
				});
			}
			// Not transient or nothing to fail over to: retry the same slot.
			return routeTo(ctx, cfg, dbg, state.slot, state.thinking, state);
		}

		// ---------------------------------------------------- continuation
		if (request.reason === "continuation" && request.state) {
			const state = request.state;
			const toolResults = currentTurnToolResults(request.messages as readonly MessageLike[]);
			const evaluations = state.adaptiveEvaluations ?? 0;
			const previousToolResults = state.adaptiveToolResults ?? 0;
			const enoughProgress = toolResults - previousToolResults >= cfg.adaptive.toolResultsPerEvaluation;
			const shouldEvaluate =
				cfg.adaptive.midTurn &&
				!state.failedOver &&
				evaluations < cfg.adaptive.maxEvaluationsPerTurn &&
				enoughProgress;

			if (dbg.allows("verbose")) {
				dbg.record("route.continuation", {
					slot: state.slot,
					modelId: state.modelId,
					thinking: state.thinking,
					toolResults,
					previousToolResults,
					evaluations,
					shouldEvaluate,
				});
			}

			if (shouldEvaluate) {
				const verdict = await classifySafe(
					request,
					ctx,
					cfg,
					dbg,
					adaptiveProgressPrompt(request.messages as readonly MessageLike[], cfg),
				);
				const picked = pickRoute(cfg, verdict, state.budget);
				// Heuristic verdicts are too weak to move the physical model.
				const requestedPromotion =
					verdict.source === "jev" && slotIndex(cfg, picked.slot) > slotIndex(cfg, state.slot);
				const resolved = requestedPromotion ? resolveModel(ctx, cfg, picked.slot, dbg) : undefined;
				const promote = resolved != null && slotIndex(cfg, resolved.slot) > slotIndex(cfg, state.slot);
				const nextThinking = promote ? strongerThinking(state.thinking, picked.thinking) : state.thinking;
				const nextState: RouteState = {
					...state,
					adaptiveEvaluations: evaluations + 1,
					adaptiveToolResults: toolResults,
					verdict,
				};
				dbg.record("route.adaptive", {
					promoted: Boolean(promote),
					fromSlot: state.slot,
					toSlot: resolved?.slot,
					fromThinking: state.thinking,
					toThinking: nextThinking,
					ruleId: picked.ruleId,
					verdict,
				});
				if (promote && resolved) {
					nextState.slot = resolved.slot;
					nextState.modelId = resolved.model.id;
					nextState.thinking = nextThinking;
					nextState.ruleId = `${picked.ruleId}+adaptive`;
					lastExplain = [
						`adaptive promotion after ${toolResults} tool results`,
						formatVerdict(verdict, nextState.ruleId, resolved.slot, resolved.model.id, nextThinking),
					].join("\n");
					setStatus(ctx, `adaptive ${state.modelId} → ${resolved.model.id} · ${nextThinking}`);
					return {
						model: resolved.model,
						thinkingLevel: resolveThinking(resolved.model, nextThinking, cfg.maxThinking?.[resolved.slot]),
						state: nextState,
					};
				}
				return routeTo(ctx, cfg, dbg, state.slot, state.thinking, nextState);
			}
			return routeTo(ctx, cfg, dbg, state.slot, state.thinking, state);
		}

		if (request.reason !== "user" && request.state) {
			const state = request.state;
			return routeTo(ctx, cfg, dbg, state.slot, state.thinking, state);
		}

		// ------------------------------------------------------ disabled
		if (!cfg.enabled && request.previous) {
			dbg.record("route.disabled", {
				modelId: request.previous.model.id,
				thinking: request.thinkingLevel,
			});
			lastExplain = "router disabled (/auto on to enable) — kept in-flight model";
			return {
				model: request.previous.model,
				thinkingLevel: request.thinkingLevel,
				state: request.state,
			};
		}

		// ----------------------------------------------------------- user
		const budget = cfg.budget;
		const verdict = await classifySafe(request, ctx, cfg, dbg);
		const picked = pickRoute(cfg, verdict, budget);
		const resolved = resolveModel(ctx, cfg, picked.slot, dbg);
		const thinking = wantedThinking(cfg, picked.thinking, request);
		const applied = resolveThinking(resolved.model, thinking, cfg.maxThinking?.[resolved.slot]);
		const state: RouteState = {
			slot: resolved.slot,
			modelId: resolved.model.id,
			thinking: applied,
			budget,
			verdict,
			ruleId: picked.ruleId,
			adaptiveEvaluations: 0,
			adaptiveToolResults: 0,
		};
		lastExplain = formatVerdict(verdict, picked.ruleId, resolved.slot, resolved.model.id, applied);
		dbg.record("route.user", {
			verdict,
			ruleId: picked.ruleId,
			slot: resolved.slot,
			modelId: resolved.model.id,
			requestedThinking: picked.thinking,
			appliedThinking: applied,
			thinkingPolicy: cfg.thinkingPolicy,
			budget,
			elapsedMs: Date.now() - startedAt,
		});
		setStatus(ctx, statusLine(cfg, verdict, resolved.slot, resolved.model.id, applied));
		return { model: resolved.model, thinkingLevel: applied, state };
	}

	// ---------------------------------------------------------------- stall guard
	// After a failover downgrade (or any model hiccup) a model sometimes ends its
	// turn with a text-only intent announcement ("Tulis X sekarang", "gas, saya
	// bongkar datanya dulu") instead of the promised tool call — the run goes
	// idle with work left. turn_end is an actionable boundary: queue one follow-up
	// user message so the model gets a valid next request and can act on its own
	// plan. Guarded per assistant message and capped per run so it can never loop.
	pi.on("before_agent_start", () => {
		nudgesThisRun = 0;
	});

	pi.on("turn_end", (event, ctx) => {
		const cfg = configFor(ctx.cwd);
		if (!cfg.stallGuard.enabled) return undefined;
		const dbg = debugFor(cfg);
		const message = event.message as
			| { role?: string; stopReason?: string; content?: unknown }
			| undefined;

		// Mid-task signal: tool results already produced in this user turn.
		let toolResultsSinceUser = 0;
		try {
			const branch = ctx.sessionManager.getBranch() as Array<{ type?: string; message?: { role?: string } }>;
			for (let i = branch.length - 1; i >= 0; i--) {
				const entry = branch[i];
				if (!entry || entry.type !== "message" || !entry.message) continue;
				if (entry.message.role === "user") break;
				if (entry.message.role === "toolResult") toolResultsSinceUser += 1;
			}
		} catch {
			// Branch unreadable (teardown) — treat as mid-task; the caps guard us.
			toolResultsSinceUser = 1;
		}

		const decision = detectStall(message, {
			toolResultsSinceUser,
			maxChars: cfg.stallGuard.maxChars,
		});
		if (!decision.stall) {
			if (dbg.allows("verbose")) dbg.record("stall.skip", { reason: decision.reason });
			return undefined;
		}
		if (nudgedEntries.has(event.messageEntryId) || nudgesThisRun >= cfg.stallGuard.maxPerRun) {
			dbg.record("stall.capped", {
				entry: event.messageEntryId,
				nudgesThisRun,
				maxPerRun: cfg.stallGuard.maxPerRun,
			});
			return undefined;
		}
		nudgedEntries.add(event.messageEntryId);
		nudgesThisRun += 1;
		lastExplain = `stall guard: ${decision.reason} — one follow-up queued`;
		dbg.record("stall.nudge", {
			entry: event.messageEntryId,
			nudgesThisRun,
			reason: decision.reason,
			toolResultsSinceUser,
		});
		// Queue a follow-up instead of returning continue: pi rejects a bare
		// continuation here (the boundary context ends in an assistant message,
		// so there is no runnable model context). A queued user follow-up both
		// satisfies canContinue and gives the model a valid next request.
		try {
			pi.sendUserMessage(cfg.stallGuard.message, { deliverAs: "followUp" });
		} catch {
			return undefined;
		}
		notify(ctx, "ontoken-auto: model announced work but made no tool call; continuing automatically.", "warning");
		return undefined;
	});

	// ------------------------------------------------------------- session hooks
	pi.on("session_start", (event, ctx) => {
		const cfg = configFor(ctx.cwd);
		const dbg = debugFor(cfg);
		dbg.record("session_start", {
			reason: event.reason,
			budget: cfg.budget,
			enabled: cfg.enabled,
			debugLevel: cfg.debug.level,
			debugFile: cfg.debug.file,
		});
		if (dbg.enabled) {
			setStatus(ctx, `debug ${dbg.level}${dbg.file ? " → file" : ""}`);
		}
	});

	pi.on("session_shutdown", () => {
		// Clear the footer slot so a stale status never survives a session.
		nudgesThisRun = 0;
		nudgedEntries.clear();
	});

	// ------------------------------------------------------------------ /auto
	pi.registerCommand("auto", {
		description: "OnToken auto-router: status, explain, log, debug, budget, slot, check",
		handler: async (args, ctx) => {
			const cfg = configFor(ctx.cwd);
			const dbg = debugFor(cfg);
			const parts = (args || "").trim().split(/\s+/).filter(Boolean);
			const cmd = (parts[0] || "status").toLowerCase();
			const rest = parts.slice(1).join(" ");

			if (cmd === "off" || cmd === "on") {
				writeGlobalOverlay({ enabled: cmd === "on" });
				notify(ctx, `ontoken-auto ${cmd}`, "info");
				return;
			}
			if (cmd === "budget") {
				const b = rest.toLowerCase();
				if (!isBudgetName(b)) {
					notify(ctx, "budget: cheap | balanced | quality", "warning");
					return;
				}
				writeGlobalOverlay({ budget: b });
				notify(ctx, `budget → ${b} (reload overlay on next prompt)`, "info");
				return;
			}
			if (cmd === "slot" && parts.length >= 3) {
				const slot = parts[1];
				const model = parts[2];
				if (!isSlotName(slot)) {
					notify(ctx, `unknown slot ${slot}`, "warning");
					return;
				}
				writeGlobalOverlay({ slots: { ...cfg.slots, [slot]: model } });
				notify(ctx, `${slot} → ${model} (written to ${globalConfigPath()})`, "info");
				return;
			}
			if (cmd === "thinking") {
				const policy = rest.toLowerCase();
				if (policy !== "auto" && policy !== "selection") {
					notify(ctx, "thinking: auto | selection", "warning");
					return;
				}
				writeGlobalOverlay({ thinkingPolicy: policy });
				notify(
					ctx,
					policy === "auto"
						? "thinking → auto (rule table decides the level)"
						: "thinking → selection (the /model level wins)",
					"info",
				);
				return;
			}
			if (cmd === "debug") {
				handleDebug(rest, parts.slice(1), cfg, dbg, ctx);
				return;
			}
			if (cmd === "log") {
				const count = Number(parts[1]);
				const n = Number.isFinite(count) && count > 0 ? Math.min(200, Math.round(count)) : 20;
				const records = dbg.tail(n);
				if (!dbg.enabled && records.length === 0) {
					notify(ctx, "debug is off. Enable with: /auto debug on  (or PI_ONTOKEN_DEBUG=1)", "info");
					return;
				}
				if (records.length === 0) {
					notify(ctx, "debug log is empty", "info");
					return;
				}
				const lines = records.map((rec) => `${rec.ts} ${rec.kind} ${compactRecord(rec)}`);
				notify(ctx, lines.join("\n"), "info");
				return;
			}
			if (cmd === "check") {
				notify(ctx, checkReport(ctx, cfg, dbg), "info");
				return;
			}

			const header = [
				`enabled ${cfg.enabled}  budget ${cfg.budget}  provider ${cfg.provider}  thinking ${cfg.thinkingPolicy}`,
				`slots  ${cfg.slotOrder.map((s) => `${s}=${cfg.slots[s]}`).join("  ")}`,
				`debug  ${dbg.level}${dbg.file ? ` → ${dbg.file}` : ""}  (${dbg.size()} records)`,
				`overlay  ${globalConfigPath()}`,
				`project  ${ctx.cwd}/.pi/ontoken-auto.json`,
			].join("\n");

			if (cmd === "explain" || cmd === "status") {
				notify(ctx, `${header}\n\nlast route\n${lastExplain}`, "info");
				return;
			}
			notify(
				ctx,
				[
					"/auto                       status",
					"/auto explain               last routing decision",
					"/auto check                 catalog + key + classifier health",
					"/auto on | off",
					"/auto budget cheap|balanced|quality",
					"/auto slot <fast|work|solid|strong|frontier> <model-id>",
					"/auto thinking auto|selection",
					"/auto debug [on|off|summary|verbose] [file <path>|nofile]",
					"/auto log [n]               last debug records",
					"",
					"JSON overlay (project wins): ~/.pi/agent/ontoken-auto.json  and  .pi/ontoken-auto.json",
				].join("\n"),
				"info",
			);
		},
	});
}

// ---------------------------------------------------------------------------
// Command helpers
// ---------------------------------------------------------------------------

/** One-line rendering of a debug record for `/auto log`. */
function compactRecord(rec: Record<string, unknown>): string {
	const skip = new Set(["ts", "kind"]);
	const parts: string[] = [];
	for (const [key, value] of Object.entries(rec)) {
		if (skip.has(key) || value === undefined) continue;
		if (typeof value === "object" && value !== null) {
			parts.push(`${key}=${JSON.stringify(value)}`);
		} else {
			parts.push(`${key}=${String(value)}`);
		}
	}
	const line = parts.join(" ");
	return line.length > 400 ? `${line.slice(0, 397)}…` : line;
}

function handleDebug(
	rest: string,
	args: string[],
	cfg: AutoConfig,
	dbg: DebugLog,
	ctx: ExtensionCommandContext,
): void {
	const level = (args[0] ?? "").toLowerCase();
	const current = cfg.debug;

	const apply = (patch: Partial<DebugConfig>, note: string): void => {
		const next: DebugConfig = { ...current, ...patch };
		writeGlobalOverlay({ debug: next });
		// Rebuild the in-process log so the new config takes effect immediately.
		debugLog = new DebugLog({ ...next, file: next.file });
		debugKey = JSON.stringify(next);
		debugLog.record("debug.enabled", { level: next.level, file: next.file, note });
		notify(
			ctx,
			`debug → ${next.enabled ? next.level : "off"}${next.file ? ` → ${next.file}` : " (memory only)"}`,
			"info",
		);
	};

	if (level === "on" || level === "summary") {
		apply({ enabled: true, level: level === "on" ? "verbose" : "summary" }, level);
		return;
	}
	if (level === "verbose") {
		apply({ enabled: true, level: "verbose" }, level);
		return;
	}
	if (level === "off") {
		apply({ enabled: false, level: "off" }, level);
		return;
	}
	if (level === "file") {
		const path = args.slice(1).join(" ").trim();
		if (!path) {
			notify(ctx, "usage: /auto debug file <path>", "warning");
			return;
		}
		apply({ enabled: true, level: current.level === "off" ? "summary" : current.level, file: path }, "file");
		return;
	}
	if (level === "nofile") {
		apply({ file: null }, "nofile");
		return;
	}
	if (level === "clear") {
		dbg.clear();
		notify(ctx, "debug log cleared", "info");
		return;
	}

	notify(
		ctx,
		[
			`debug: ${cfg.debug.enabled ? cfg.debug.level : "off"}  buffer ${dbg.size()}  file ${cfg.debug.file ?? "none"}`,
			"",
			"/auto debug on          verbose, memory only",
			"/auto debug summary     one record per decision",
			"/auto debug verbose     full classifier + route payloads",
			"/auto debug off",
			"/auto debug file <path> also append JSONL to <path>",
			"/auto debug nofile      stop writing to a file",
			"/auto debug clear       empty the in-memory buffer",
			"",
			"Env: PI_ONTOKEN_DEBUG=summary|verbose  PI_ONTOKEN_DEBUG_FILE=/tmp/ontoken.jsonl",
			"Flag: --ontoken-debug verbose",
		].join("\n"),
		"info",
	);
}

/** Health check for the routed roster, the Jev classifier, and the key. */
function checkReport(ctx: ExtensionCommandContext, cfg: AutoConfig, dbg: DebugLog): string {
	const lines: string[] = [`provider ${cfg.provider}  budget ${cfg.budget}  debug ${dbg.level}`, "", "slots:"];
	for (const slot of cfg.slotOrder) {
		const id = cfg.slots[slot];
		const model = ctx.modelRegistry.find(cfg.provider, id);
		if (!model) {
			lines.push(`  ${slot.padEnd(9)} ${id} — MISSING from catalog`);
			continue;
		}
		const levels = thinkingSummary(model as unknown as ModelLike);
		const image = Array.isArray((model as { input?: string[] }).input) && (model as { input: string[] }).input.includes("image");
		lines.push(`  ${slot.padEnd(9)} ${id} — ok  thinking ${levels}${image ? "  image" : ""}`);
	}
	const jev = ctx.modelRegistry.findOfType("classifier", cfg.jev.classifierProvider, cfg.jev.classifierId);
	lines.push(
		"",
		`classifier ${cfg.jev.classifierProvider}/${cfg.jev.classifierId}: ${jev ? "found" : "MISSING"}`,
		`TypeSafe key: ${process.env.TYPESAFE_API_KEY ? "present" : "missing"} (env or ${"~/.pi/agent/secrets/typesafe_api_key"})`,
		`virtual model: ${cfg.virtual.provider}/${cfg.virtual.id}`,
	);
	return lines.join("\n");
}
