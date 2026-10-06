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
): Promise<Verdict> {
	const prompt = lastUserText(request.messages).slice(0, cfg.jev.promptChars);
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
	return {
		model: resolved.model,
		thinkingLevel: thinking,
		state: state ?? {
			slot: resolved.slot,
			modelId: resolved.model.id,
			thinking,
			budget: cfg.budget,
		},
	};
}

function formatVerdict(v: Verdict, ruleId: string, slot: SlotName, modelId: string, thinking: string): string {
	return [
		`source ${v.source}${v.error ? ` (${v.error})` : ""}`,
		`task ${v.task}  difficulty ${v.difficulty}/4  reasoning ${v.needsReasoning}  vision ${v.needsVision}`,
		`confidence ${v.confidence.toFixed(2)}  rule ${ruleId}`,
		`→ ${slot}  ${modelId}  thinking ${thinking}`,
	].join("\n");
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
			const cfg = loadConfig(ctx.cwd);
			if (request.reason === "direct") {
				return routeTo(request, ctx, cfg, cfg.directSlot, "low");
			}
			if (request.reason !== "user" && request.state) {
				return routeTo(
					request,
					ctx,
					cfg,
					request.state.slot,
					request.state.thinking,
					request.state,
				);
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
			};
			lastExplain = formatVerdict(verdict, picked.ruleId, resolved.slot, resolved.model.id, picked.thinking);
			if (ctx.hasUI) {
				const src = verdict.source === "jev" ? "jev" : "heur";
				ctx.ui.setStatus(
					"ontoken-auto",
					`${src} ${verdict.task} d${verdict.difficulty} → ${resolved.model.id} · ${picked.thinking}`,
				);
			}
			return { model: resolved.model, thinkingLevel: picked.thinking, state };
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
