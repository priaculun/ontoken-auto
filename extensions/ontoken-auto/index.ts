/**
 * OnToken Auto — Jev-routed virtual model `ontoken/auto`.
 *
 * Per user prompt, TypeSafe Jev classifies the task; this extension picks an
 * OnToken model + thinking level. Overlay config:
 *   ~/.pi/agent/ontoken-auto.json
 *   <cwd>/.pi/ontoken-auto.json
 *
 * Usage: /model ontoken/auto   then  /auto  /auto budget  /auto explain
 */

import type { Message } from "@earendil-works/pi-ai";
import type {
	ExtensionAPI,
	ExtensionContext,
	ModelRoute,
	ModelRouteRequest,
} from "@earendil-works/pi-coding-agent";
import {
	type AutoConfig,
	type BudgetName,
	type SlotName,
	type ThinkingLevel,
	clampSlot,
	ensureJevKey,
	loadConfig,
	slotIndex,
	writeGlobalOverlay,
} from "./config.ts";

const THINKING: ThinkingLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

/**
 * Downgrade a thinking level to one the target model actually supports.
 *
 * A model declares unsupported levels via `thinkingLevelMap[level] === null`.
 * If the map is absent we assume all levels pass through, but an optional
 * per-slot `maxThinking` clamp in the overlay config can still cap them
 * (useful for servers whose effort enum is narrower than pi's levels).
 */
function supportedThinking(
	model: { thinkingLevelMap?: Partial<Record<ThinkingLevel, string | null>>; reasoning?: boolean },
	wanted: ThinkingLevel,
	maxThinking?: ThinkingLevel,
): ThinkingLevel {
	let cap = THINKING.indexOf(wanted);
	if (cap < 0) cap = THINKING.indexOf("high");
	const maxIdx = maxThinking ? THINKING.indexOf(maxThinking) : -1;
	if (maxIdx >= 0 && maxIdx < cap) cap = maxIdx;
	const map = model.thinkingLevelMap;
	if (model.reasoning === false) return "off";
	if (!map) return THINKING[cap] as ThinkingLevel;
	for (let i = cap; i >= 0; i--) {
		const level = THINKING[i] as ThinkingLevel;
		if (map[level] !== null) return level;
	}
	return "off";
}

let lastExplain = "no routing yet this process";

interface Verdict {
	task: string;
	difficulty: number;
	needsReasoning: boolean;
	needsVision: boolean;
	longHorizon: boolean;
	confidence: number;
	source: "jev" | "fallback";
	error?: string;
}

interface AutoState {
	slot: SlotName;
	modelId: string;
	thinking: ThinkingLevel;
	budget: BudgetName;
	verdict?: Verdict;
	ruleId?: string;
	/** Slot already failed this turn and excluded from failover. */
	failoverTried?: SlotName[];
	/** True when the current slot was reached via failover. */
	failedOver?: boolean;
	/** Number of adaptive Jev evaluations already performed in the current user turn. */
	adaptiveEvaluations?: number;
	/** Tool-result count at the previous adaptive evaluation. */
	adaptiveToolResults?: number;
}

type AutoRequest = ModelRouteRequest<AutoState>;

function lastUserText(messages: readonly Message[]): string {
	const content = messages.filter((m) => m.role === "user").at(-1)?.content ?? "";
	if (typeof content === "string") return content;
	return content.flatMap((b) => (b.type === "text" ? [b.text] : [])).join("\n");
}

function userHasImage(messages: readonly Message[]): boolean {
	const last = messages.filter((m) => m.role === "user").at(-1);
	if (!last || typeof last.content === "string") return false;
	return last.content.some((b) => b.type === "image");
}

function contentText(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content.flatMap((block) => {
		if (!block || typeof block !== "object") return [];
		const b = block as Record<string, unknown>;
		if (b.type === "text" && typeof b.text === "string") return [b.text];
		return [];
	}).join("\n");
}

function clippedTail(text: string, max: number): string {
	if (max <= 1) return "…";
	if (text.length <= max) return text;
	return `…${text.slice(-(max - 1))}`;
}

/** Best-effort redaction before conversation/tool text leaves the Pi process. */
function redactClassifierText(text: string): string {
	return text
		.replace(/-----BEGIN [^-\n]*PRIVATE KEY-----[\s\S]*?-----END [^-\n]*PRIVATE KEY-----/gi, "[REDACTED PRIVATE KEY]")
		.replace(/\b(Bearer)\s+[A-Za-z0-9._~+\/-]+=*/gi, "$1 [REDACTED]")
		.replace(/\b((?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret)\s*[:=]\s*)[^\s,;]+/gi, "$1[REDACTED]")
		.replace(/([a-z][a-z0-9+.-]*:\/\/[^\s:@/]+:)[^\s@/]+@/gi, "$1[REDACTED]@");
}

/**
 * Classification input for a new user turn. Short/referential prompts receive
 * recent user/assistant context so users can naturally say "lanjut" or
 * "kerjakan itu" without causing the router to forget the actual task.
 */
function contextualUserPrompt(messages: readonly Message[], cfg: AutoConfig): string {
	const latest = redactClassifierText(lastUserText(messages).trim());
	if (!cfg.adaptive.contextAware || latest.length > cfg.adaptive.shortPromptChars) {
		return latest.slice(0, cfg.jev.promptChars);
	}
	const prior = messages.filter((m) => m.role === "user" || m.role === "assistant").slice(0, -1);
	const sections: string[] = [`CURRENT USER REQUEST:\n${latest}`];
	let remaining = Math.max(0, cfg.jev.promptChars - sections[0].length - 2);
	for (let i = prior.length - 1; i >= 0 && remaining > 80; i--) {
		const message = prior[i] as Message & { role: string; content?: unknown };
		const text = redactClassifierText(contentText(message.content).trim());
		if (!text) continue;
		const label = message.role === "user" ? "PREVIOUS USER" : "PREVIOUS ASSISTANT";
		const section = `${label}:\n${clippedTail(text, Math.min(4000, remaining - label.length - 3))}`;
		sections.unshift(section);
		remaining -= section.length + 2;
	}
	return sections.join("\n\n").slice(-cfg.jev.promptChars);
}

function currentTurnToolResults(messages: readonly Message[]): number {
	const lastUser = messages.findLastIndex((m) => m.role === "user");
	return messages.slice(lastUser + 1).filter((m) => m.role === "toolResult").length;
}

/** Classification input used after tool progress within the current turn. */
function adaptiveProgressPrompt(messages: readonly Message[], cfg: AutoConfig): string {
	const lastUser = messages.findLastIndex((m) => m.role === "user");
	const before = messages.slice(0, lastUser + 1);
	const progress = messages.slice(lastUser + 1);
	const maxChars = Math.max(1000, cfg.jev.promptChars);
	const instruction = "Reassess the remaining work using the original task plus evidence discovered so far.";
	// Reserve roughly half the classifier budget for the task. A final tail slice
	// would otherwise let long tool output erase the objective entirely.
	const objective = clippedTail(contextualUserPrompt(before, cfg), Math.floor(maxChars * 0.55));
	const heading = `TASK AND RECENT CONTEXT:\n${objective}\n\nCURRENT-TURN PROGRESS:`;
	let remaining = Math.max(0, maxChars - heading.length - instruction.length - 4);
	const progressSections: string[] = [];
	for (let i = progress.length - 1; i >= 0 && remaining > 80; i--) {
		const message = progress[i] as Message & {
			role: string;
			content?: unknown;
			toolName?: string;
			isError?: boolean;
		};
		const text = redactClassifierText(contentText(message.content).trim());
		if (!text) continue;
		const label = message.role === "toolResult"
			? `TOOL ${message.toolName ?? "unknown"}${message.isError ? " ERROR" : ""}`
			: message.role === "assistant" ? "ASSISTANT PROGRESS" : undefined;
		if (!label) continue;
		const section = `${label}:\n${clippedTail(text, Math.min(1800, remaining - label.length - 3))}`;
		progressSections.unshift(section);
		remaining -= section.length + 2;
	}
	return [heading, ...progressSections, instruction].join("\n\n").slice(0, maxChars);
}

function heuristic(prompt: string, hasImage: boolean): Verdict {
	const p = prompt.toLowerCase();
	const hard =
		/\b(architect|refactor|deadlock|race|subtle|security|migrate|design doc)\b/.test(p) ||
		prompt.length > 4000;
	const debug = /\b(bug|crash|failing test|stack trace|hang|reproduc)/.test(p);
	const coding = /\b(implement|fix|add|edit|write|refactor|test|pr\b)/.test(p);
	const chat = prompt.length < 240 && !coding && !debug;
	let task = "coding";
	if (chat) task = "chat";
	else if (debug) task = "debug";
	else if (/\breview\b/.test(p)) task = "review";
	else if (/\b(research|compare|benchmark)\b/.test(p)) task = "research";
	else if (/\bdesign\b/.test(p)) task = "design";
	const difficulty = hard ? 3 : debug ? 3 : chat ? 0 : coding ? 1 : 2;
	return {
		task,
		difficulty,
		needsReasoning: hard || debug,
		needsVision: hasImage,
		longHorizon: hard || prompt.length > 2500,
		confidence: 0.35,
		source: "fallback",
	};
}

async function classify(
	request: AutoRequest,
	ctx: ExtensionContext,
	cfg: AutoConfig,
	promptOverride?: string,
): Promise<Verdict> {
	const prompt = (promptOverride ?? contextualUserPrompt(request.messages, cfg)).slice(0, cfg.jev.promptChars);
	const hasImage = userHasImage(request.messages);
	if (!ensureJevKey(cfg)) {
		return { ...heuristic(prompt, hasImage), error: "no TypeSafe key" };
	}
	const jev = ctx.modelRegistry.findOfType(
		"classifier",
		cfg.jev.classifierProvider,
		cfg.jev.classifierId,
	);
	if (!jev) {
		return { ...heuristic(prompt, hasImage), error: "classifier not in catalog" };
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
	if (result.stopReason !== "stop") {
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
	const needsVision =
		hasImage || (visionA?.type === "bool" ? visionA.probability >= 0.5 : false);
	const longHorizon = longA?.type === "bool" ? longA.probability >= 0.5 : false;
	const confs = [taskA, diffA, reasonA]
		.map((a) => (a && "confidence" in a ? Number(a.confidence) : 0.5))
		.filter((n) => Number.isFinite(n));
	const confidence = confs.length ? confs.reduce((a, b) => a + b, 0) / confs.length : 0.5;
	return {
		task,
		difficulty: Math.min(4, Math.max(0, difficulty)),
		needsReasoning,
		needsVision,
		longHorizon,
		confidence,
		source: "jev",
	};
}

function ruleMatches(
	rule: AutoConfig["rules"][number],
	v: Verdict,
	cfg: AutoConfig,
	budget: BudgetName,
): boolean {
	const w = rule.when;
	if (w.taskIn && !w.taskIn.includes(v.task)) return false;
	if (w.minDifficulty != null && v.difficulty < w.minDifficulty) return false;
	if (w.maxDifficulty != null && v.difficulty > w.maxDifficulty) return false;
	if (w.needsReasoning != null && v.needsReasoning !== w.needsReasoning) return false;
	if (w.needsVision != null && v.needsVision !== w.needsVision) return false;
	if (w.longHorizon != null && v.longHorizon !== w.longHorizon) return false;
	if (w.budgetPromoteEarly != null && Boolean(cfg.budgets[budget]?.promoteEarly) !== w.budgetPromoteEarly) {
		return false;
	}
	if (w.minConfidence != null && v.confidence < w.minConfidence) return false;
	return true;
}

function pickRoute(
	cfg: AutoConfig,
	v: Verdict,
	budget: BudgetName,
): { slot: SlotName; thinking: ThinkingLevel; ruleId: string } {
	let slot: SlotName = cfg.fallbackSlot;
	let thinking: ThinkingLevel = "low";
	let ruleId = "fallback";
	for (const rule of cfg.rules) {
		if (!ruleMatches(rule, v, cfg, budget)) continue;
		if (slotIndex(cfg, rule.slot) >= slotIndex(cfg, slot) || ruleId === "fallback") {
			slot = rule.slot;
			thinking = rule.thinking;
			ruleId = rule.id;
		}
	}
	if (v.longHorizon && slotIndex(cfg, slot) < slotIndex(cfg, "solid")) {
		slot = "solid";
		if (THINKING.indexOf(thinking) < THINKING.indexOf("high")) thinking = "high";
		ruleId += "+long";
	}
	if (v.needsReasoning && thinking === "off") thinking = "low";
	slot = clampSlot(cfg, slot, budget);
	return { slot, thinking, ruleId };
}

function resolveModel(ctx: ExtensionContext, cfg: AutoConfig, slot: SlotName) {
	const id = cfg.slots[slot];
	const model = ctx.modelRegistry.find(cfg.provider, id);
	if (!model) {
		for (const s of cfg.slotOrder) {
			const m = ctx.modelRegistry.find(cfg.provider, cfg.slots[s]);
			if (m) return { model: m, slot: s as SlotName };
		}
		throw new Error(`No OnToken models from roster are in the catalog`);
	}
	return { model, slot };
}

function routeTo(
	request: AutoRequest,
	ctx: ExtensionContext,
	cfg: AutoConfig,
	slot: SlotName,
	thinking: ThinkingLevel,
	state?: AutoState,
): ModelRoute<AutoState> {
	const resolved = resolveModel(ctx, cfg, slot);
	const applied = supportedThinking(resolved.model, thinking, cfg.maxThinking?.[resolved.slot]);
	return {
		model: resolved.model,
		thinkingLevel: applied,
		state: state ?? {
			slot: resolved.slot,
			modelId: resolved.model.id,
			thinking: applied,
			budget: cfg.budget,
		},
	};
}

function strongerThinking(a: ThinkingLevel, b: ThinkingLevel): ThinkingLevel {
	return THINKING.indexOf(a) >= THINKING.indexOf(b) ? a : b;
}

function formatVerdict(v: Verdict, ruleId: string, slot: SlotName, modelId: string, thinking: string): string {
	return [
		`source ${v.source}${v.error ? ` (${v.error})` : ""}`,
		`task ${v.task}  difficulty ${v.difficulty}/4  reasoning ${v.needsReasoning}  vision ${v.needsVision}`,
		`confidence ${v.confidence.toFixed(2)}  rule ${ruleId}`,
		`→ ${slot}  ${modelId}  thinking ${thinking}`,
	].join("\n");
}

/**
 * Transient provider failures worth failing over for. Auth/quota/billing errors
 * and aborts are NOT transient — failing over would just burn another slot.
 */
function isFailoverWorthy(message: ModelRouteRequest<AutoState>["failed"]): boolean {
	if (!message) return false;
	const stop = message.message?.stopReason;
	if (stop === "aborted") return false;
	const err = (message.message?.errorMessage ?? "").toLowerCase();
	if (!err) return stop === "error";
	if (/\b(401|403|invalid api key|authentication|unauthorized|api key|quota|billing|payment|credit)\b/.test(err)) {
		return false;
	}
	// 5xx, 429, overloaded, timeout, network errors — the transient family.
	return /\b(429|5[0-9][0-9]|overloaded|rate limit|too many requests|timeout|timed out|econn|enotfound|econnreset|econnrefused|etimedout|socket|network|temporarily|unavailable|bad gateway|service unavailable|internal server error)\b/.test(err)
		|| stop === "error";
}

/**
 * Pick the failover target for a failed slot: the explicit chain if configured,
 * otherwise every other slot ordered strongest-first. Already-tried slots,
 * models missing from the catalog, and the budget ceiling are respected.
 */
function failoverTarget(
	request: AutoRequest,
	ctx: ExtensionContext,
	cfg: AutoConfig,
	state: AutoState,
): { model: NonNullable<ModelRoute<AutoState>["model"]>; slot: SlotName; thinking: ThinkingLevel } | undefined {
	const failedSlot = state.slot;
	const tried = new Set([...(state.failoverTried ?? []), failedSlot]);
	const chain = cfg.failover.chain?.[failedSlot]
		?? [...cfg.slotOrder].reverse().filter((s) => s !== failedSlot);
	const budgetCeiling = cfg.budgets[state.budget]?.ceiling ?? "frontier";
	for (const candidate of chain) {
		if (tried.has(candidate)) continue;
		if (slotIndex(cfg, candidate) > slotIndex(cfg, budgetCeiling)) continue;
		const model = ctx.modelRegistry.find(cfg.provider, cfg.slots[candidate]);
		if (!model) continue;
		return { model, slot: candidate, thinking: state.thinking };
	}
	return undefined;
}

export default function (pi: ExtensionAPI) {
	const boot = loadConfig();
	ensureJevKey(boot);

	pi.registerVirtualModel<AutoState>({
		provider: boot.virtual.provider,
		id: boot.virtual.id,
		name: boot.virtual.name,
		thinkingLevels: ["off", "minimal", "low", "medium", "high", "xhigh", "max"],
		contextWindow: 1_000_000,
		maxTokens: 128_000,
		async route(request, ctx) {
			try {
				return await this.routeOnce(request, ctx);
			} catch (err) {
				// Same teardown race, but the runner was invalidated mid-route
				// (e.g. during the awaited classify call). If the turn is being
				// aborted anyway, fall back to the previous model instead of
				// surfacing the stale-ctx error in the transcript.
				if (request.signal?.aborted && request.previous) {
					return {
						model: request.previous.model,
						thinkingLevel: request.previous.thinkingLevel ?? "low",
						state: request.state,
					};
				}
				throw err;
			}
		},
		async routeOnce(request, ctx) {
			// Session teardown race: pi can still ask a mid-turn loop for a
			// continuation route after dispose() invalidated the extension runner
			// (quit / newSession / fork / switchSession / reload). Any ctx access
			// then throws the "extension ctx is stale" error into the transcript.
			// Bail without touching ctx; pi discards the result anyway.
			if (request.signal?.aborted && request.previous) {
				return {
					model: request.previous.model,
					thinkingLevel: request.previous.thinkingLevel ?? "low",
					state: request.state,
				};
			}
			const cfg = loadConfig(ctx.cwd);
			if (request.reason === "direct") {
				return routeTo(request, ctx, cfg, cfg.directSlot, "low");
			}
			// Failover: pi retries failed requests with reason "retry". If the failed
			// request belongs to this router and the error is transient, move the turn
			// to the next slot in the failover chain instead of re-hitting it.
			if (request.reason === "retry" && request.state && request.failed) {
				const state = request.state;
				const failedBelongsToRouter = state.modelId === request.failed.model.id;
				if (cfg.failover.enabled && failedBelongsToRouter && isFailoverWorthy(request.failed)) {
					const target = failoverTarget(request, ctx, cfg, state);
					if (target) {
						const nextState: AutoState = {
							...state,
							slot: target.slot,
							modelId: target.model.id,
							failoverTried: [...(state.failoverTried ?? []), state.slot],
							failedOver: true,
							ruleId: `${state.ruleId ?? "fallback"}+failover`,
						};
						lastExplain = [
							`failover: ${state.modelId} failed (${request.failed.message?.errorMessage ?? "error"})`,
							`→ ${target.slot}  ${target.model.id}  thinking ${state.thinking}`,
						].join("\n");
						if (ctx.hasUI) {
							ctx.ui.setStatus(
								"ontoken-auto",
								`failover ${state.modelId} → ${target.model.id}`,
							);
						}
						return {
							model: target.model,
							thinkingLevel: supportedThinking(target.model, target.thinking, cfg.maxThinking?.[target.slot]),
							state: nextState,
						};
					}
					lastExplain = `failover: no available slot left for ${state.modelId} (${request.failed.message?.errorMessage ?? "error"})`;
				}
				// Not transient or nothing to fail over to: retry the same slot.
				return routeTo(request, ctx, cfg, state.slot, state.thinking, state);
			}
			if (request.reason === "continuation" && request.state) {
				const state = request.state;
				const toolResults = currentTurnToolResults(request.messages);
				const evaluations = state.adaptiveEvaluations ?? 0;
				const previousToolResults = state.adaptiveToolResults ?? 0;
				const enoughProgress = toolResults - previousToolResults >= cfg.adaptive.toolResultsPerEvaluation;
				if (
					cfg.adaptive.midTurn &&
					!state.failedOver &&
					evaluations < cfg.adaptive.maxEvaluationsPerTurn &&
					enoughProgress
				) {
					const verdict = await classify(request, ctx, cfg, adaptiveProgressPrompt(request.messages, cfg));
					const picked = pickRoute(cfg, verdict, state.budget);
					const requestedPromotion = verdict.source === "jev"
						&& slotIndex(cfg, picked.slot) > slotIndex(cfg, state.slot);
					const resolved = requestedPromotion ? resolveModel(ctx, cfg, picked.slot) : undefined;
					const promote = resolved != null
						&& slotIndex(cfg, resolved.slot) > slotIndex(cfg, state.slot);
					const nextThinking = promote
						? strongerThinking(state.thinking, picked.thinking)
						: state.thinking;
					const nextState: AutoState = {
						...state,
						adaptiveEvaluations: evaluations + 1,
						adaptiveToolResults: toolResults,
						verdict,
					};
					if (promote && resolved) {
						nextState.slot = resolved.slot;
						nextState.modelId = resolved.model.id;
						nextState.thinking = nextThinking;
						nextState.ruleId = `${picked.ruleId}+adaptive`;
						lastExplain = [
							`adaptive promotion after ${toolResults} tool results`,
							formatVerdict(verdict, nextState.ruleId, resolved.slot, resolved.model.id, nextThinking),
						].join("\n");
						if (ctx.hasUI) {
							ctx.ui.setStatus(
								"ontoken-auto",
								`adaptive ${state.modelId} → ${resolved.model.id} · ${nextThinking}`,
							);
						}
						return {
							model: resolved.model,
							thinkingLevel: supportedThinking(resolved.model, nextThinking, cfg.maxThinking?.[resolved.slot]),
							state: nextState,
						};
					}
					return routeTo(request, ctx, cfg, state.slot, state.thinking, nextState);
				}
				return routeTo(request, ctx, cfg, state.slot, state.thinking, state);
			}
			if (request.reason !== "user" && request.state) {
				return routeTo(request, ctx, cfg, request.state.slot, request.state.thinking, request.state);
			}
			if (!cfg.enabled && request.previous) {
				return {
					model: request.previous.model,
					thinkingLevel: request.thinkingLevel,
					state: request.state,
				};
			}
			const budget = cfg.budget;
			const verdict = await classify(request, ctx, cfg);
			const picked = pickRoute(cfg, verdict, budget);
			const resolved = resolveModel(ctx, cfg, picked.slot);
			const state: AutoState = {
				slot: resolved.slot,
				modelId: resolved.model.id,
				thinking: picked.thinking,
				budget,
				verdict,
				ruleId: picked.ruleId,
				adaptiveEvaluations: 0,
				adaptiveToolResults: 0,
			};
			lastExplain = formatVerdict(verdict, picked.ruleId, resolved.slot, resolved.model.id, picked.thinking);
			if (ctx.hasUI) {
				const src = verdict.source === "jev" ? "jev" : "heur";
				ctx.ui.setStatus(
					"ontoken-auto",
					`${src} ${verdict.task} d${verdict.difficulty} → ${resolved.model.id} · ${picked.thinking}`,
				);
			}
			return {
				model: resolved.model,
				thinkingLevel: supportedThinking(resolved.model, picked.thinking, cfg.maxThinking?.[resolved.slot]),
				state,
			};
		},
	});

	pi.registerCommand("auto", {
		description: "OnToken auto-router: status, budget, explain, on/off",
		handler: async (args, ctx) => {
			const cfg = loadConfig(ctx.cwd);
			const parts = (args || "").trim().split(/\s+/).filter(Boolean);
			const cmd = (parts[0] || "status").toLowerCase();
			const rest = parts.slice(1).join(" ");

			if (cmd === "off" || cmd === "on") {
				writeGlobalOverlay({ enabled: cmd === "on" });
				ctx.ui.notify(`ontoken-auto ${cmd}`, "info");
				return;
			}
			if (cmd === "budget") {
				const b = rest.toLowerCase() as BudgetName;
				if (!["cheap", "balanced", "quality"].includes(b)) {
					ctx.ui.notify("budget: cheap | balanced | quality", "warning");
					return;
				}
				writeGlobalOverlay({ budget: b });
				ctx.ui.notify(`budget → ${b} (reload overlay on next prompt)`, "info");
				return;
			}
			if (cmd === "slot" && parts.length >= 3) {
				const slot = parts[1] as SlotName;
				const model = parts[2];
				if (!cfg.slotOrder.includes(slot)) {
					ctx.ui.notify(`unknown slot ${slot}`, "warning");
					return;
				}
				writeGlobalOverlay({ slots: { ...cfg.slots, [slot]: model } });
				ctx.ui.notify(`${slot} → ${model} (written to ~/.pi/agent/ontoken-auto.json)`, "info");
				return;
			}

			const header = [
				`enabled ${cfg.enabled}  budget ${cfg.budget}  provider ${cfg.provider}`,
				`slots  fast=${cfg.slots.fast}  work=${cfg.slots.work}  solid=${cfg.slots.solid}`,
				`       strong=${cfg.slots.strong}  frontier=${cfg.slots.frontier}`,
				`overlay  ${globalHint()}`,
				`project  ${ctx.cwd}/.pi/ontoken-auto.json`,
			].join("\n");

			if (cmd === "explain") {
				ctx.ui.notify(`${header}\n\nlast route\n${lastExplain}`, "info");
				return;
			}
			if (cmd === "status") {
				ctx.ui.notify(`${header}\n\nlast route\n${lastExplain}`, "info");
				return;
			}
			ctx.ui.notify(
				[
					"/auto              status",
					"/auto on|off",
					"/auto budget cheap|balanced|quality",
					"/auto slot <fast|work|solid|strong|frontier> <model-id>",
					"/auto explain",
					"",
					"JSON overlay (project wins): ~/.pi/agent/ontoken-auto.json  and  .pi/ontoken-auto.json",
				].join("\n"),
				"info",
			);
		},
	});
}

function globalHint(): string {
	return "~/.pi/agent/ontoken-auto.json";
}
