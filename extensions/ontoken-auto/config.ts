import { existsSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";
import { fileURLToPath } from "node:url";

const THINKING_ORDER: ThinkingLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];

/** Optional per-slot ceiling, e.g. { "frontier": "high" } — clamps any rule level above it. */
export type MaxThinkingMap = Partial<Record<SlotName, ThinkingLevel>>;

function normalizeMaxThinking(value: unknown): MaxThinkingMap | undefined {
	if (!value || typeof value !== "object") return undefined;
	const out: MaxThinkingMap = {};
	for (const [slot, level] of Object.entries(value as Record<string, unknown>)) {
		if (typeof level === "string" && (THINKING_ORDER as string[]).includes(level)) {
			out[slot as SlotName] = level as ThinkingLevel;
		}
	}
	return Object.keys(out).length ? out : undefined;
}

export type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
export type SlotName = "fast" | "work" | "solid" | "strong" | "frontier";
export type BudgetName = "cheap" | "balanced" | "quality";

export interface RuleWhen {
	taskIn?: string[];
	minDifficulty?: number;
	maxDifficulty?: number;
	needsReasoning?: boolean;
	needsVision?: boolean;
	longHorizon?: boolean;
	budgetPromoteEarly?: boolean;
	minConfidence?: number;
}

export interface RouteRule {
	id: string;
	when: RuleWhen;
	slot: SlotName;
	thinking: ThinkingLevel;
}

/**
 * Failover on transient provider errors: on `reason: "retry"`, move the turn to
 * the next slot in the chain instead of re-hitting the same broken model.
 */
export interface FailoverConfig {
	enabled: boolean;
	/** Explicit per-slot chain. Missing slots use the automatic chain (other slots, strongest first). */
	chain?: Partial<Record<SlotName, SlotName[]>>;
}

function normalizeFailover(value: unknown): FailoverConfig {
	const base: FailoverConfig = { enabled: true };
	if (!value || typeof value !== "object" || Array.isArray(value)) return base;
	const v = value as Record<string, unknown>;
	const out: FailoverConfig = { enabled: v.enabled !== false };
	if (v.chain && typeof v.chain === "object" && !Array.isArray(v.chain)) {
		const chain: Partial<Record<SlotName, SlotName[]>> = {};
		for (const [slot, list] of Object.entries(v.chain as Record<string, unknown>)) {
			if (Array.isArray(list)) {
				const slots = list.filter((s): s is SlotName =>
					["fast", "work", "solid", "strong", "frontier"].includes(s as string),
				);
				if (slots.length) chain[slot as SlotName] = slots;
			}
		}
		if (Object.keys(chain).length) out.chain = chain;
	}
	return out;
}

export interface AdaptiveConfig {
	/** Include recent conversation text when the latest user prompt is short or referential. */
	contextAware: boolean;
	/** Latest user prompts at or below this length receive recent conversation context. */
	shortPromptChars: number;
	/** Reclassify continuations after tool results and only promote to a stronger slot. */
	midTurn: boolean;
	/** Maximum Jev continuation evaluations within one user turn. */
	maxEvaluationsPerTurn: number;
	/** Require this many new tool results since the previous adaptive evaluation. */
	toolResultsPerEvaluation: number;
}

function normalizeAdaptive(value: unknown): AdaptiveConfig {
	const base: AdaptiveConfig = {
		contextAware: true,
		shortPromptChars: 800,
		midTurn: true,
		maxEvaluationsPerTurn: 2,
		toolResultsPerEvaluation: 2,
	};
	if (!value || typeof value !== "object" || Array.isArray(value)) return base;
	const v = value as Record<string, unknown>;
	return {
		contextAware: v.contextAware !== false,
		shortPromptChars: Math.max(
			80,
			Math.round(Number.isFinite(Number(v.shortPromptChars)) ? Number(v.shortPromptChars) : base.shortPromptChars),
		),
		midTurn: v.midTurn !== false,
		maxEvaluationsPerTurn: Math.max(
			0,
			Math.min(
				5,
				Math.round(Number.isFinite(Number(v.maxEvaluationsPerTurn)) ? Number(v.maxEvaluationsPerTurn) : base.maxEvaluationsPerTurn),
			),
		),
		toolResultsPerEvaluation: Math.max(
			1,
			Math.min(
				10,
				Math.round(Number.isFinite(Number(v.toolResultsPerEvaluation)) ? Number(v.toolResultsPerEvaluation) : base.toolResultsPerEvaluation),
			),
		),
	};
}

export interface AutoConfig {
	enabled: boolean;
	budget: BudgetName;
	provider: string;
	virtual: { provider: string; id: string; name: string };
	jev: {
		keyFile?: string;
		classifierProvider: string;
		classifierId: string;
		promptChars: number;
	};
	slots: Record<SlotName, string>;
	slotOrder: SlotName[];
	budgets: Record<BudgetName, { ceiling: SlotName; promoteEarly: boolean }>;
	directSlot: SlotName;
	fallbackSlot: SlotName;
	/** Optional per-slot thinking ceiling applied after rule selection. */
	maxThinking?: MaxThinkingMap;
	adaptive: AdaptiveConfig;
	failover: FailoverConfig;
	rules: RouteRule[];
}

export function agentDir(): string {
	return process.env.PI_AGENT_DIR || join(homedir(), ".pi", "agent");
}

export function globalConfigPath(): string {
	return join(agentDir(), "ontoken-auto.json");
}

export function projectConfigPath(cwd: string): string {
	return join(cwd, ".pi", "ontoken-auto.json");
}

function deepMerge<T>(base: T, overlay: unknown): T {
	if (!overlay || typeof overlay !== "object" || Array.isArray(overlay)) return base;
	const out: any = Array.isArray(base) ? [...(base as any)] : { ...(base as any) };
	for (const [k, v] of Object.entries(overlay as Record<string, unknown>)) {
		if (v === undefined) continue;
		if (Array.isArray(v)) out[k] = v;
		else if (v && typeof v === "object" && typeof out[k] === "object" && !Array.isArray(out[k])) {
			out[k] = deepMerge(out[k], v);
		} else out[k] = v;
	}
	return out;
}

function readJson(path: string): unknown {
	if (!existsSync(path)) return undefined;
	try {
		return JSON.parse(readFileSync(path, "utf8"));
	} catch {
		return undefined;
	}
}

export function loadDefaults(): AutoConfig {
	const path = join(dirname(fileURLToPath(import.meta.url)), "defaults.json");
	return JSON.parse(readFileSync(path, "utf8")) as AutoConfig;
}

export function loadConfig(cwd?: string): AutoConfig {
	let cfg = loadDefaults();
	const global = readJson(globalConfigPath());
	if (global) cfg = deepMerge(cfg, global);
	if (cwd) {
		const project = readJson(projectConfigPath(cwd));
		if (project) cfg = deepMerge(cfg, project);
	}
	cfg.maxThinking = normalizeMaxThinking(cfg.maxThinking);
	cfg.adaptive = normalizeAdaptive(cfg.adaptive);
	cfg.failover = normalizeFailover(cfg.failover);
	return cfg;
}

export function writeGlobalOverlay(partial: Record<string, unknown>): void {
	const path = globalConfigPath();
	mkdirSync(dirname(path), { recursive: true });
	const existing = (readJson(path) as Record<string, unknown>) || {};
	writeFileSync(path, JSON.stringify({ ...existing, ...partial }, null, 2) + "\n");
}

export function readJevKey(cfg: AutoConfig): string | undefined {
	const env = process.env.TYPESAFE_API_KEY?.trim();
	if (env) return env;
	const secret = join(agentDir(), "secrets", "typesafe_api_key");
	if (existsSync(secret)) {
		const v = readFileSync(secret, "utf8").trim();
		if (v) return v;
	}
	const file = cfg.jev.keyFile;
	if (file && existsSync(file)) {
		const raw = readFileSync(file, "utf8").trim();
		const line = raw.split(/\r?\n/).map((l) => l.trim()).find((l) => l && !l.startsWith("#"));
		return line || undefined;
	}
	return undefined;
}

export function ensureJevKey(cfg: AutoConfig): boolean {
	if (process.env.TYPESAFE_API_KEY?.trim()) return true;
	const key = readJevKey(cfg);
	if (!key) return false;
	process.env.TYPESAFE_API_KEY = key;
	return true;
}

export function slotIndex(cfg: AutoConfig, slot: SlotName): number {
	const i = cfg.slotOrder.indexOf(slot);
	return i < 0 ? cfg.slotOrder.length : i;
}

export function clampSlot(cfg: AutoConfig, slot: SlotName, budget: BudgetName): SlotName {
	const ceiling = cfg.budgets[budget]?.ceiling ?? "frontier";
	return slotIndex(cfg, slot) > slotIndex(cfg, ceiling) ? ceiling : slot;
}
