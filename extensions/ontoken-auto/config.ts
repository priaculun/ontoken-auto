/**
 * Configuration for the OnToken Auto router.
 *
 * Layering (later wins):
 *   defaults.json  →  ~/.pi/agent/ontoken-auto.json  →  <cwd>/.pi/ontoken-auto.json
 *
 * Every block is normalized on load, so a malformed overlay degrades to the
 * built-in default instead of crashing the router (which would kill the turn).
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

export type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
export type SlotName = "fast" | "work" | "solid" | "strong" | "frontier";
export type BudgetName = "cheap" | "balanced" | "quality";

export const THINKING_ORDER: readonly ThinkingLevel[] = [
	"off",
	"minimal",
	"low",
	"medium",
	"high",
	"xhigh",
	"max",
];

export const SLOT_NAMES: readonly SlotName[] = ["fast", "work", "solid", "strong", "frontier"];
export const BUDGET_NAMES: readonly BudgetName[] = ["cheap", "balanced", "quality"];

/** Optional per-slot ceiling, e.g. { "frontier": "high" } — clamps any rule level above it. */
export type MaxThinkingMap = Partial<Record<SlotName, ThinkingLevel>>;

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
	/** Show a UI notification when a failover happens. */
	notify: boolean;
	/** Explicit per-slot chain. Missing slots use the automatic chain (other slots, strongest first). */
	chain?: Partial<Record<SlotName, SlotName[]>>;
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

export type DebugLevel = "off" | "summary" | "verbose";

export interface DebugConfig {
	enabled: boolean;
	level: DebugLevel;
	/** Append JSONL records to this file. Created on demand; `null` keeps logs in memory only. */
	file: string | null;
	/** Number of records kept in memory for `/auto log`. */
	bufferSize: number;
	/** Include a redacted prompt excerpt in verbose records. */
	includePrompts: boolean;
}

export interface StallGuardConfig {
	enabled: boolean;
	/** Maximum automatic follow-ups per agent run. */
	maxPerRun: number;
	/** Assistant text longer than this is treated as a final answer, not an announcement. */
	maxChars: number;
	/** Text of the queued follow-up. */
	message: string;
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
	/**
	 * `auto` (default): the rule table decides the thinking level.
	 * `selection`: the level picked in `/model` wins; the router only picks the model.
	 */
	thinkingPolicy: "auto" | "selection";
	/** Optional per-slot thinking ceiling applied after rule selection. */
	maxThinking?: MaxThinkingMap;
	adaptive: AdaptiveConfig;
	failover: FailoverConfig;
	debug: DebugConfig;
	stallGuard: StallGuardConfig;
	rules: RouteRule[];
}

const DEBUG_LEVELS: readonly DebugLevel[] = ["off", "summary", "verbose"];

function asRecord(value: unknown): Record<string, unknown> | undefined {
	if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
	return value as Record<string, unknown>;
}

function num(value: unknown, fallback: number, min: number, max: number): number {
	const n = Number(value);
	if (!Number.isFinite(n)) return fallback;
	return Math.max(min, Math.min(max, Math.round(n)));
}

function normalizeMaxThinking(value: unknown): MaxThinkingMap | undefined {
	const rec = asRecord(value);
	if (!rec) return undefined;
	const out: MaxThinkingMap = {};
	for (const [slot, level] of Object.entries(rec)) {
		if (typeof level === "string" && (THINKING_ORDER as readonly string[]).includes(level)) {
			out[slot as SlotName] = level as ThinkingLevel;
		}
	}
	return Object.keys(out).length ? out : undefined;
}

function normalizeFailover(value: unknown): FailoverConfig {
	const base: FailoverConfig = { enabled: true, notify: true };
	const v = asRecord(value);
	if (!v) return base;
	const out: FailoverConfig = { enabled: v.enabled !== false, notify: v.notify !== false };
	const chainRec = asRecord(v.chain);
	if (chainRec) {
		const chain: Partial<Record<SlotName, SlotName[]>> = {};
		for (const [slot, list] of Object.entries(chainRec)) {
			if (!Array.isArray(list)) continue;
			const slots = list.filter((s): s is SlotName => SLOT_NAMES.includes(s as SlotName));
			if (slots.length) chain[slot as SlotName] = slots;
		}
		if (Object.keys(chain).length) out.chain = chain;
	}
	return out;
}

function normalizeAdaptive(value: unknown): AdaptiveConfig {
	const base: AdaptiveConfig = {
		contextAware: true,
		shortPromptChars: 800,
		midTurn: true,
		maxEvaluationsPerTurn: 2,
		toolResultsPerEvaluation: 2,
	};
	const v = asRecord(value);
	if (!v) return base;
	return {
		contextAware: v.contextAware !== false,
		shortPromptChars: num(v.shortPromptChars, base.shortPromptChars, 80, 200_000),
		midTurn: v.midTurn !== false,
		maxEvaluationsPerTurn: num(v.maxEvaluationsPerTurn, base.maxEvaluationsPerTurn, 0, 5),
		toolResultsPerEvaluation: num(v.toolResultsPerEvaluation, base.toolResultsPerEvaluation, 1, 10),
	};
}

/** `PI_ONTOKEN_DEBUG=1|summary|verbose` overrides `debug.enabled` / `debug.level`. */
function debugEnv(): { enabled?: boolean; level?: DebugLevel; file?: string } {
	const out: { enabled?: boolean; level?: DebugLevel; file?: string } = {};
	const raw = (process.env.PI_ONTOKEN_DEBUG ?? "").trim().toLowerCase();
	if (raw) {
		if (raw === "0" || raw === "false" || raw === "off" || raw === "no") {
			out.enabled = false;
			out.level = "off";
		} else if ((DEBUG_LEVELS as readonly string[]).includes(raw)) {
			out.enabled = raw !== "off";
			out.level = raw as DebugLevel;
		} else {
			out.enabled = true;
		}
	}
	const file = (process.env.PI_ONTOKEN_DEBUG_FILE ?? "").trim();
	if (file) out.file = file;
	return out;
}

function normalizeDebug(value: unknown): DebugConfig {
	const base: DebugConfig = {
		enabled: false,
		level: "summary",
		file: null,
		bufferSize: 200,
		includePrompts: true,
	};
	const v = asRecord(value) ?? {};
	const levelRaw = typeof v.level === "string" ? v.level.toLowerCase() : "";
	const level: DebugLevel = (DEBUG_LEVELS as readonly string[]).includes(levelRaw)
		? (levelRaw as DebugLevel)
		: base.level;
	const fileRaw = typeof v.file === "string" ? v.file.trim() : "";
	const out: DebugConfig = {
		enabled: v.enabled === true || level !== "off" ? v.enabled !== false : false,
		level,
		file: fileRaw ? fileRaw : null,
		bufferSize: num(v.bufferSize, base.bufferSize, 10, 5000),
		includePrompts: v.includePrompts !== false,
	};
	if (v.enabled === false) {
		out.enabled = false;
		out.level = "off";
	}
	const env = debugEnv();
	if (env.enabled !== undefined) out.enabled = env.enabled;
	if (env.level) out.level = env.level;
	if (env.file) out.file = env.file;
	// Pointing PI_ONTOKEN_DEBUG_FILE at a path is an explicit request for logs.
	if (env.file && env.enabled === undefined) {
		out.enabled = true;
		if (out.level === "off") out.level = "summary";
	}
	if (!out.enabled) out.level = "off";
	return out;
}

function normalizeStallGuard(value: unknown): StallGuardConfig {
	const base: StallGuardConfig = {
		enabled: true,
		maxPerRun: 2,
		maxChars: 300,
		message:
			"Lanjutkan pekerjaanmu: eksekusi langkah berikutnya sekarang dengan memanggil tool yang diperlukan. Jangan hanya mengumumkan rencana.",
	};
	const v = asRecord(value);
	if (!v) return base;
	const message = typeof v.message === "string" && v.message.trim() ? v.message.trim() : base.message;
	return {
		enabled: v.enabled !== false,
		maxPerRun: num(v.maxPerRun, base.maxPerRun, 0, 10),
		maxChars: num(v.maxChars, base.maxChars, 60, 4000),
		message,
	};
}

function normalizeSlots(value: unknown, fallback: Record<SlotName, string>): Record<SlotName, string> {
	const v = asRecord(value);
	const out: Record<SlotName, string> = { ...fallback };
	if (!v) return out;
	for (const slot of SLOT_NAMES) {
		const id = v[slot];
		if (typeof id === "string" && id.trim()) out[slot] = id.trim();
	}
	return out;
}

function normalizeSlotOrder(value: unknown, fallback: SlotName[]): SlotName[] {
	if (!Array.isArray(value)) return [...fallback];
	const seen = new Set<SlotName>();
	const out: SlotName[] = [];
	for (const entry of value) {
		if (SLOT_NAMES.includes(entry as SlotName) && !seen.has(entry as SlotName)) {
			seen.add(entry as SlotName);
			out.push(entry as SlotName);
		}
	}
	for (const slot of SLOT_NAMES) if (!seen.has(slot)) out.push(slot);
	return out;
}

function normalizeBudgets(
	value: unknown,
	fallback: Record<BudgetName, { ceiling: SlotName; promoteEarly: boolean }>,
): Record<BudgetName, { ceiling: SlotName; promoteEarly: boolean }> {
	const v = asRecord(value);
	const out: Record<BudgetName, { ceiling: SlotName; promoteEarly: boolean }> = {
		cheap: { ...fallback.cheap },
		balanced: { ...fallback.balanced },
		quality: { ...fallback.quality },
	};
	if (!v) return out;
	for (const budget of BUDGET_NAMES) {
		const entry = asRecord(v[budget]);
		if (!entry) continue;
		const ceiling = entry.ceiling;
		if (SLOT_NAMES.includes(ceiling as SlotName)) out[budget].ceiling = ceiling as SlotName;
		if (typeof entry.promoteEarly === "boolean") out[budget].promoteEarly = entry.promoteEarly;
	}
	return out;
}

function normalizeRules(value: unknown, fallback: RouteRule[]): RouteRule[] {
	if (!Array.isArray(value)) return fallback;
	const out: RouteRule[] = [];
	for (const [index, entry] of value.entries()) {
		const rule = asRecord(entry);
		if (!rule) continue;
		const when = asRecord(rule.when) ?? {};
		const slot = SLOT_NAMES.includes(rule.slot as SlotName) ? (rule.slot as SlotName) : "work";
		const thinking = (THINKING_ORDER as readonly string[]).includes(rule.thinking as string)
			? (rule.thinking as ThinkingLevel)
			: "low";
		const normalizedWhen: RuleWhen = {};
		if (Array.isArray(when.taskIn)) {
			normalizedWhen.taskIn = when.taskIn.filter((t): t is string => typeof t === "string");
		}
		if (Number.isFinite(Number(when.minDifficulty))) normalizedWhen.minDifficulty = Number(when.minDifficulty);
		if (Number.isFinite(Number(when.maxDifficulty))) normalizedWhen.maxDifficulty = Number(when.maxDifficulty);
		if (typeof when.needsReasoning === "boolean") normalizedWhen.needsReasoning = when.needsReasoning;
		if (typeof when.needsVision === "boolean") normalizedWhen.needsVision = when.needsVision;
		if (typeof when.longHorizon === "boolean") normalizedWhen.longHorizon = when.longHorizon;
		if (typeof when.budgetPromoteEarly === "boolean") {
			normalizedWhen.budgetPromoteEarly = when.budgetPromoteEarly;
		}
		if (Number.isFinite(Number(when.minConfidence))) normalizedWhen.minConfidence = Number(when.minConfidence);
		out.push({
			id: typeof rule.id === "string" && rule.id.trim() ? rule.id.trim() : `rule-${index + 1}`,
			when: normalizedWhen,
			slot,
			thinking,
		});
	}
	return out.length ? out : fallback;
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

export function defaultsPath(): string {
	return join(dirname(fileURLToPath(import.meta.url)), "defaults.json");
}

function deepMerge<T>(base: T, overlay: unknown): T {
	if (!overlay || typeof overlay !== "object" || Array.isArray(overlay)) return base;
	const out: Record<string, unknown> = { ...(base as unknown as Record<string, unknown>) };
	for (const [k, v] of Object.entries(overlay as Record<string, unknown>)) {
		if (v === undefined) continue;
		if (Array.isArray(v)) out[k] = v;
		else if (v && typeof v === "object" && out[k] && typeof out[k] === "object" && !Array.isArray(out[k])) {
			out[k] = deepMerge(out[k], v);
		} else out[k] = v;
	}
	return out as T;
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
	const raw = JSON.parse(readFileSync(defaultsPath(), "utf8")) as Record<string, unknown>;
	return normalizeConfig(raw);
}

function normalizeConfig(raw: Record<string, unknown>): AutoConfig {
	const virtual = asRecord(raw.virtual) ?? {};
	const jev = asRecord(raw.jev) ?? {};
	const provider = typeof raw.provider === "string" && raw.provider.trim() ? raw.provider.trim() : "ontoken";
	const slots = normalizeSlots(raw.slots, {
		fast: "glm-5.3-flash",
		work: "glm-5.3-flash",
		solid: "kimi-k3",
		strong: "claude-sonnet-5.5",
		frontier: "gpt-6.1-sol",
	});
	const slotOrder = normalizeSlotOrder(raw.slotOrder, [...SLOT_NAMES]);
	const budgets = normalizeBudgets(raw.budgets, {
		cheap: { ceiling: "solid", promoteEarly: false },
		balanced: { ceiling: "frontier", promoteEarly: false },
		quality: { ceiling: "frontier", promoteEarly: true },
	});
	const budget = BUDGET_NAMES.includes(raw.budget as BudgetName) ? (raw.budget as BudgetName) : "balanced";
	const directSlot = SLOT_NAMES.includes(raw.directSlot as SlotName) ? (raw.directSlot as SlotName) : "fast";
	const fallbackSlot = SLOT_NAMES.includes(raw.fallbackSlot as SlotName) ? (raw.fallbackSlot as SlotName) : "work";
	const thinkingPolicy = raw.thinkingPolicy === "selection" ? "selection" : "auto";
	return {
		enabled: raw.enabled !== false,
		budget,
		provider,
		virtual: {
			provider: typeof virtual.provider === "string" && virtual.provider.trim() ? virtual.provider.trim() : provider,
			id: typeof virtual.id === "string" && virtual.id.trim() ? virtual.id.trim() : "auto",
			name: typeof virtual.name === "string" && virtual.name.trim() ? virtual.name.trim() : "Auto (Jev · OnToken)",
		},
		jev: {
			keyFile: typeof jev.keyFile === "string" && jev.keyFile.trim() ? jev.keyFile.trim() : undefined,
			classifierProvider:
				typeof jev.classifierProvider === "string" && jev.classifierProvider.trim()
					? jev.classifierProvider.trim()
					: "typesafe",
			classifierId:
				typeof jev.classifierId === "string" && jev.classifierId.trim() ? jev.classifierId.trim() : "jev-latest",
			promptChars: num(jev.promptChars, 16_000, 500, 200_000),
		},
		slots,
		slotOrder,
		budgets,
		directSlot,
		fallbackSlot,
		thinkingPolicy,
		maxThinking: normalizeMaxThinking(raw.maxThinking),
		adaptive: normalizeAdaptive(raw.adaptive),
		failover: normalizeFailover(raw.failover),
		debug: normalizeDebug(raw.debug),
		stallGuard: normalizeStallGuard(raw.stallGuard),
		rules: normalizeRules(raw.rules, []),
	};
}

export function loadConfig(cwd?: string): AutoConfig {
	let cfg = loadDefaults();
	const global = readJson(globalConfigPath());
	if (global) cfg = deepMerge(cfg, global);
	if (cwd) {
		const project = readJson(projectConfigPath(cwd));
		if (project) cfg = deepMerge(cfg, project);
	}
	// Re-normalize after merging so a hand-written overlay cannot inject
	// out-of-range numbers or unknown slot/model names.
	return normalizeConfig(cfg as unknown as Record<string, unknown>);
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
		const line = raw
			.split(/\r?\n/)
			.map((l) => l.trim())
			.find((l) => l && !l.startsWith("#"));
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

export function isSlotName(value: unknown): value is SlotName {
	return SLOT_NAMES.includes(value as SlotName);
}

export function isBudgetName(value: unknown): value is BudgetName {
	return BUDGET_NAMES.includes(value as BudgetName);
}
