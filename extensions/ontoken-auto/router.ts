/**
 * Pure routing logic for OnToken Auto.
 *
 * Nothing in this file imports a Pi package at runtime: every Pi type is a
 * type-only import, so the module can be unit-tested with plain `node --test`
 * and loaded by Pi without resolving peer dependencies.
 */

import { appendFileSync } from "node:fs";
import {
	type AutoConfig,
	type BudgetName,
	type DebugConfig,
	type SlotName,
	type ThinkingLevel,
	THINKING_ORDER,
	clampSlot,
	slotIndex,
} from "./config.ts";

// ---------------------------------------------------------------- loose shapes

/** The parts of a Pi chat model the router reads. */
export interface ModelLike {
	id: string;
	provider?: string;
	reasoning?: boolean;
	thinkingLevelMap?: Partial<Record<ThinkingLevel, string | null>>;
}

/** The parts of a conversation message the router reads. */
export interface MessageLike {
	role?: string;
	content?: unknown;
	toolName?: string;
	isError?: boolean;
}

export interface Verdict {
	task: string;
	difficulty: number;
	needsReasoning: boolean;
	needsVision: boolean;
	longHorizon: boolean;
	confidence: number;
	source: "jev" | "heuristic";
	/** Why the heuristic was used instead of Jev. */
	error?: string;
}

export interface RouteState {
	slot: SlotName;
	modelId: string;
	thinking: ThinkingLevel;
	budget: BudgetName;
	verdict?: Verdict;
	ruleId?: string;
	/** Slots already failed this turn and excluded from failover. */
	failoverTried?: SlotName[];
	/** True when the current slot was reached via failover. */
	failedOver?: boolean;
	/** Adaptive Jev evaluations already performed in the current user turn. */
	adaptiveEvaluations?: number;
	/** Tool-result count at the previous adaptive evaluation. */
	adaptiveToolResults?: number;
}

// ------------------------------------------------------------------- text utils

/**
 * User-role messages that are not a task: nudges that extensions inject into
 * the conversation (Pi converts custom messages to role "user" before routing).
 * Treating them as the objective would make the classifier route on
 * "Compact now…" or "Lanjutkan pekerjaanmu…" instead of the real task.
 * Compaction summaries and self-compact handoff notes are NOT listed: they
 * carry the goal after the original request was compacted away.
 */
const SYNTHETIC_USER_RE: readonly RegExp[] = [
	/^Compact now: write your note_to_self\b/, // self-compact idle/forced nudge
	/^Continue from your saved note_to_self above\b/, // self-compact crash resume
];

/** Exact texts registered at runtime, e.g. the configured stall-guard follow-up. */
const syntheticUserTexts = new Set<string>();

export function registerSyntheticUserText(text: string): void {
	const t = text.trim();
	if (t) syntheticUserTexts.add(t);
}

export function isSyntheticUserMessage(message: MessageLike | undefined): boolean {
	if (!message || message.role !== "user") return false;
	const text = contentText(message.content).trim();
	if (!text) return false;
	return syntheticUserTexts.has(text) || SYNTHETIC_USER_RE.some((re) => re.test(text));
}

/** Index of the latest real user message (synthetic nudges skipped), or -1. */
export function lastObjectiveIndex(messages: readonly MessageLike[]): number {
	for (let i = messages.length - 1; i >= 0; i--) {
		const m = messages[i];
		if (m?.role === "user" && !isSyntheticUserMessage(m)) return i;
	}
	return -1;
}

export function lastUserText(messages: readonly MessageLike[]): string {
	const idx = lastObjectiveIndex(messages);
	const content = (idx >= 0 ? messages[idx]?.content : undefined) ?? "";
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.flatMap((block) => {
			if (!block || typeof block !== "object") return [];
			const b = block as Record<string, unknown>;
			return b.type === "text" && typeof b.text === "string" ? [b.text] : [];
		})
		.join("\n");
}

export function userHasImage(messages: readonly MessageLike[]): boolean {
	const last = messages[lastObjectiveIndex(messages)];
	if (!last || !Array.isArray(last.content)) return false;
	return last.content.some((block) => {
		if (!block || typeof block !== "object") return false;
		return (block as Record<string, unknown>).type === "image";
	});
}

export function contentText(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content
		.flatMap((block) => {
			if (!block || typeof block !== "object") return [];
			const b = block as Record<string, unknown>;
			return b.type === "text" && typeof b.text === "string" ? [b.text] : [];
		})
		.join("\n");
}

export function clippedTail(text: string, max: number): string {
	if (max <= 1) return "…";
	if (text.length <= max) return text;
	return `…${text.slice(-(max - 1))}`;
}

/** Best-effort redaction before conversation/tool text leaves the Pi process. */
export function redactClassifierText(text: string): string {
	return text
		.replace(/-----BEGIN [^-\n]*PRIVATE KEY-----[\s\S]*?-----END [^-\n]*PRIVATE KEY-----/gi, "[REDACTED PRIVATE KEY]")
		.replace(/\b(Bearer)\s+[A-Za-z0-9._~+/-]+=*/gi, "$1 [REDACTED]")
		.replace(
			/\b((?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret)\s*[:=]\s*)[^\s,;]+/gi,
			"$1[REDACTED]",
		)
		.replace(/([a-z][a-z0-9+.-]*:\/\/[^\s:@/]+:)[^\s@/]+@/gi, "$1[REDACTED]@")
		.replace(/\b(sk-[A-Za-z0-9_-]{12,})\b/g, "[REDACTED KEY]");
}

// -------------------------------------------------------------- prompt building

/**
 * Classification input for a new user turn. Short/referential prompts receive
 * recent user/assistant context so users can naturally say "lanjut" or
 * "kerjakan itu" without causing the router to forget the actual task.
 */
export function contextualUserPrompt(messages: readonly MessageLike[], cfg: AutoConfig): string {
	const latest = redactClassifierText(lastUserText(messages).trim());
	if (!latest) return "";
	if (!cfg.adaptive.contextAware || latest.length > cfg.adaptive.shortPromptChars) {
		return latest.slice(0, cfg.jev.promptChars);
	}
	// Context before the objective; synthetic nudges carry no task information.
	const objective = lastObjectiveIndex(messages);
	const prior = messages
		.slice(0, Math.max(0, objective))
		.filter((m) => (m.role === "user" && !isSyntheticUserMessage(m)) || m.role === "assistant");
	const sections: string[] = [`CURRENT USER REQUEST:\n${latest}`];
	let remaining = Math.max(0, cfg.jev.promptChars - sections[0].length - 2);
	for (let i = prior.length - 1; i >= 0 && remaining > 80; i--) {
		const message = prior[i];
		const text = redactClassifierText(contentText(message?.content).trim());
		if (!text) continue;
		const label = message?.role === "user" ? "PREVIOUS USER" : "PREVIOUS ASSISTANT";
		const section = `${label}:\n${clippedTail(text, Math.min(4000, remaining - label.length - 3))}`;
		sections.unshift(section);
		remaining -= section.length + 2;
	}
	return sections.join("\n\n").slice(-cfg.jev.promptChars);
}

/** Number of tool results produced since the latest real user message. */
export function currentTurnToolResults(messages: readonly MessageLike[]): number {
	const lastUser = lastObjectiveIndex(messages);
	return messages.slice(lastUser + 1).filter((m) => m.role === "toolResult").length;
}

/** Classification input used after tool progress within the current turn. */
export function adaptiveProgressPrompt(messages: readonly MessageLike[], cfg: AutoConfig): string {
	const lastUser = lastObjectiveIndex(messages);
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
		const message = progress[i];
		const text = redactClassifierText(contentText(message?.content).trim());
		if (!text) continue;
		const label =
			message?.role === "toolResult"
				? `TOOL ${message.toolName ?? "unknown"}${message.isError ? " ERROR" : ""}`
				: message?.role === "assistant"
					? "ASSISTANT PROGRESS"
					: undefined;
		if (!label) continue;
		const section = `${label}:\n${clippedTail(text, Math.min(1800, remaining - label.length - 3))}`;
		progressSections.unshift(section);
		remaining -= section.length + 2;
	}
	return [heading, ...progressSections, instruction].join("\n\n").slice(0, maxChars);
}

// -------------------------------------------------------------------- heuristic

/**
 * Offline fallback verdict used when Jev is unavailable (no key, no catalog
 * entry, network error, stale extension context). Deliberately conservative:
 * it never promotes mid-turn and reports low confidence.
 */
export function heuristic(prompt: string, hasImage: boolean): Verdict {
	const p = prompt.toLowerCase();
	const hard =
		/\b(architect|refactor|deadlock|race|subtle|security|migrate|design doc)\b/.test(p) || prompt.length > 4000;
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
		source: "heuristic",
	};
}

// -------------------------------------------------------------- thinking clamp

/**
 * Thinking levels a model actually accepts, mirroring Pi's own rule:
 * `thinkingLevelMap[level] === null` marks a level unsupported, and the
 * extended levels `xhigh` / `max` must be mapped explicitly to count.
 */
export function supportedThinkingLevels(model: ModelLike): ThinkingLevel[] {
	if (model.reasoning === false) return ["off"];
	return THINKING_ORDER.filter((level) => {
		const mapped = model.thinkingLevelMap?.[level];
		if (mapped === null) return false;
		if (level === "xhigh" || level === "max") return mapped !== undefined;
		return true;
	});
}

/**
 * Clamp a wanted thinking level to what the model supports. Prefers the wanted
 * level, then the next stronger supported level, then the next weaker one — the
 * same order Pi uses, so a route never asks for a level Pi would silently drop.
 */
export function clampThinking(model: ModelLike, wanted: ThinkingLevel): ThinkingLevel {
	const available = supportedThinkingLevels(model);
	if (available.includes(wanted)) return wanted;
	const requested = THINKING_ORDER.indexOf(wanted);
	if (requested < 0) return available[0] ?? "off";
	for (let i = requested; i < THINKING_ORDER.length; i++) {
		const candidate = THINKING_ORDER[i];
		if (available.includes(candidate)) return candidate;
	}
	for (let i = requested - 1; i >= 0; i--) {
		const candidate = THINKING_ORDER[i];
		if (available.includes(candidate)) return candidate;
	}
	return available[0] ?? "off";
}

/**
 * Final thinking level for a route: apply the optional per-slot `maxThinking`
 * ceiling, then clamp to the model's supported levels.
 */
export function resolveThinking(
	model: ModelLike,
	wanted: ThinkingLevel,
	maxThinking?: ThinkingLevel,
): ThinkingLevel {
	let capped = wanted;
	const wantedIdx = THINKING_ORDER.indexOf(wanted);
	const maxIdx = maxThinking ? THINKING_ORDER.indexOf(maxThinking) : -1;
	if (wantedIdx < 0) capped = "high";
	else if (maxIdx >= 0 && maxIdx < wantedIdx) capped = maxThinking as ThinkingLevel;
	return clampThinking(model, capped);
}

// ------------------------------------------------------------------- rule table

export function ruleMatches(rule: AutoConfig["rules"][number], v: Verdict, cfg: AutoConfig, budget: BudgetName): boolean {
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

/**
 * Pick a slot + thinking level for a verdict.
 *
 * Among all matching rules the strongest slot wins; a tie is broken by the later
 * rule, so an overlay can append an override instead of editing the table.
 * `longHorizon` then raises the floor to the `solid` slot, and the budget ceiling
 * is applied last so `cheap` can never escape its cap.
 */
export function pickRoute(
	cfg: AutoConfig,
	v: Verdict,
	budget: BudgetName,
): { slot: SlotName; thinking: ThinkingLevel; ruleId: string } {
	let slot: SlotName = cfg.fallbackSlot;
	let thinking: ThinkingLevel = "low";
	let ruleId = "fallback";
	let best = -1;
	for (const rule of cfg.rules) {
		if (!ruleMatches(rule, v, cfg, budget)) continue;
		const idx = slotIndex(cfg, rule.slot);
		if (idx >= best) {
			best = idx;
			slot = rule.slot;
			thinking = rule.thinking;
			ruleId = rule.id;
		}
	}
	if (v.longHorizon && slotIndex(cfg, slot) < slotIndex(cfg, "solid")) {
		slot = "solid";
		if (THINKING_ORDER.indexOf(thinking) < THINKING_ORDER.indexOf("high")) thinking = "high";
		ruleId += "+long";
	}
	if (v.needsReasoning && thinking === "off") thinking = "low";
	slot = clampSlot(cfg, slot, budget);
	return { slot, thinking, ruleId };
}

export function strongerThinking(a: ThinkingLevel, b: ThinkingLevel): ThinkingLevel {
	return THINKING_ORDER.indexOf(a) >= THINKING_ORDER.indexOf(b) ? a : b;
}

// --------------------------------------------------------------------- failover

export type FailedRequest = { message?: { stopReason?: string; errorMessage?: string } } | undefined;

/**
 * Transient provider failures worth failing over for. Auth/quota/billing errors
 * and aborts are NOT transient — failing over would just burn another slot.
 */
export function isFailoverWorthy(failed: FailedRequest): boolean {
	if (!failed) return false;
	const stop = failed.message?.stopReason;
	if (stop === "aborted") return false;
	const err = (failed.message?.errorMessage ?? "").toLowerCase();
	if (!err) return stop === "error";
	if (/\b(401|403|invalid api key|authentication|unauthorized|api key|quota|billing|payment|credit)\b/.test(err)) {
		return false;
	}
	// Context overflow is fixed by compaction, not by another model: Pi compacts
	// and retries. Switching slots here would only drop the prompt cache (and
	// usually land on a pricier model) for a request that now fits anyway.
	if (isContextOverflowError(err)) return false;
	// 5xx, 429, overloaded, timeout, network errors — the transient family.
	return (
		/\b(429|5[0-9][0-9]|overloaded|rate limit|too many requests|timeout|timed out|econn|enotfound|econnreset|econnrefused|etimedout|socket|network|temporarily|unavailable|bad gateway|service unavailable|internal server error)\b/.test(
			err,
		) || stop === "error"
	);
}

/**
 * Mirror of pi-ai's OVERFLOW_PATTERNS (utils/overflow.ts), kept local because
 * this module must not import Pi at runtime. Rate-limit texts are excluded the
 * same way Pi does ("Too many tokens, please wait" is throttling).
 */
const OVERFLOW_RE =
	/prompt (?:is )?too long|prompt exceeds max length|request_too_large|input is too long|exceeds the context window|maximum context length|exceeds (?:the )?maximum allowed input length|context length|context window exceeds|exceeded model token limit|maximum prompt length|reduce the length of the messages|input token count.*exceeds|exceeds the (?:limit|available context)|context[_ ]length[_ ]exceeded|model_context_window_exceeded|range of input length|too many tokens|token limit exceeded|^(?:[\w -]+ api error \(413\):|413(?:\s|:|$))/i;
const NON_OVERFLOW_RE = /^(throttling error|service unavailable):|rate limit|too many requests/i;

export function isContextOverflowError(errorMessage: string): boolean {
	if (!errorMessage || NON_OVERFLOW_RE.test(errorMessage)) return false;
	return OVERFLOW_RE.test(errorMessage);
}

export interface FailoverTarget {
	slot: SlotName;
	modelId: string;
	thinking: ThinkingLevel;
}

/**
 * Pick the failover target for a failed slot: the explicit chain if configured,
 * otherwise every other slot ordered strongest-first. Already-tried slots and
 * the budget ceiling are respected; `isAvailable` filters slots whose model is
 * missing from the catalog.
 */
export function failoverTarget(
	cfg: AutoConfig,
	state: Pick<RouteState, "slot" | "thinking" | "budget" | "failoverTried">,
	isAvailable: (slot: SlotName, modelId: string) => boolean,
): FailoverTarget | undefined {
	const failedSlot = state.slot;
	const tried = new Set<SlotName>([...(state.failoverTried ?? []), failedSlot]);
	const chain =
		cfg.failover.chain?.[failedSlot] ?? [...cfg.slotOrder].reverse().filter((s) => s !== failedSlot);
	const ceiling = cfg.budgets[state.budget]?.ceiling ?? "frontier";
	for (const candidate of chain) {
		if (tried.has(candidate)) continue;
		if (slotIndex(cfg, candidate) > slotIndex(cfg, ceiling)) continue;
		const modelId = cfg.slots[candidate];
		if (!isAvailable(candidate, modelId)) continue;
		return { slot: candidate, modelId, thinking: state.thinking };
	}
	return undefined;
}

// ---------------------------------------------------------------------- reports

export function formatVerdict(
	v: Verdict,
	ruleId: string,
	slot: SlotName,
	modelId: string,
	thinking: ThinkingLevel,
): string {
	return [
		`source ${v.source}${v.error ? ` (${v.error})` : ""}`,
		`task ${v.task}  difficulty ${v.difficulty}/4  reasoning ${v.needsReasoning}  vision ${v.needsVision}  long ${v.longHorizon}`,
		`confidence ${v.confidence.toFixed(2)}  rule ${ruleId}`,
		`→ ${slot}  ${modelId}  thinking ${thinking}`,
	].join("\n");
}

// ----------------------------------------------------------------- stall guard

export interface StallDecision {
	stall: boolean;
	reason: string;
}

const FINISHED_RE =
	/(sudah selesai|udah (selesai|beres)|semua selesai|tuntas|beres[.!]?$|\bdone\b|selesai,?\s+silakan|silakan (cek|coba|dicoba)|ringkasan|kesimpulan|semoga|terima kasih|belum ada langkah)/i;
const INTENT_RE =
	/\b(sekarang|lanjut|selanjutnya|berikutnya|akan saya|saya akan|saya (tulis|buat|ubah|hapus|jalankan|buka|baca)|gas|gaskeun|yuk|ayo|mulai|eksekusi|now|i'll|i will|let me|next i)\b/i;
const BULLET_RE = /^\s*(?:[-*•‣]|\d+[.)])\s+/;

/** A list-heavy message is a result, not a promise to act. */
function looksLikeList(text: string): boolean {
	const lines = text
		.split("\n")
		.map((line) => line.trim())
		.filter(Boolean);
	if (lines.length < 3) return false;
	const bullets = lines.filter((line) => BULLET_RE.test(line)).length;
	return bullets >= Math.max(2, Math.ceil(lines.length * 0.5));
}

/** Last sentence of the message: where an announcement puts its promise. */
function finalSentence(text: string): string {
	const lines = text
		.split("\n")
		.map((line) => line.trim())
		.filter(Boolean);
	const last = lines.at(-1) ?? "";
	const parts = last.split(/(?<=[.!;:])\s+/);
	return (parts.at(-1) ?? last).trim();
}

/**
 * Decide whether an assistant message is a text-only intent announcement that
 * left the run idle with work remaining. Only short, tool-free, non-final
 * messages qualify, and the caller caps the number of nudges per run.
 *
 * Guards against the two ways this misfires in practice:
 *   - a result that happens to mention "next" (a file list, a summary)
 *   - a vague short stop that was a real answer
 * so the intent cue must sit in the final, short sentence of the message.
 */
export function detectStall(
	message: { role?: string; stopReason?: string; content?: unknown } | undefined,
	opts: { toolResultsSinceUser: number; maxChars: number },
): StallDecision {
	if (!message || message.role !== "assistant" || message.stopReason !== "stop") {
		return { stall: false, reason: "not a completed assistant turn" };
	}
	const blocks = Array.isArray(message.content) ? (message.content as Array<Record<string, unknown>>) : [];
	if (blocks.some((block) => block?.type === "toolCall")) {
		return { stall: false, reason: "message contains a tool call" };
	}
	const text = blocks
		.flatMap((block) => (block?.type === "text" && typeof block.text === "string" ? [block.text] : []))
		.join("\n")
		.trim();
	if (!text) return { stall: false, reason: "no text" };
	if (text.length > opts.maxChars) return { stall: false, reason: "text too long for an announcement" };
	if (text.includes("?")) return { stall: false, reason: "text asks the user a question" };
	if (FINISHED_RE.test(text)) return { stall: false, reason: "text signals completion" };
	if (looksLikeList(text)) return { stall: false, reason: "text is a list of results" };

	const tail = finalSentence(text);
	const tailHasIntent = INTENT_RE.test(tail);
	if (!tailHasIntent) {
		// A vague short stop only counts as a stall mid-task, while really short,
		// and only when it is one fragment rather than multi-sentence prose.
		const isFragment = !/[.!;:]\s/.test(text.slice(0, -1));
		if (!isFragment || text.length > Math.min(140, opts.maxChars) || opts.toolResultsSinceUser === 0) {
			return { stall: false, reason: "no intent signal" };
		}
	}
	// An announcement is a short promise; a long final sentence is prose.
	if (tailHasIntent && tail.length > 200) {
		return { stall: false, reason: "final sentence too long for a promise" };
	}
	return { stall: true, reason: `text-only intent "${clippedTail(text, 80)}" without a tool call` };
}

// ------------------------------------------------------------------------ debug

export interface DebugRecord {
	ts: string;
	kind: string;
	[key: string]: unknown;
}

/**
 * Bounded, redacting debug log. Records stay in memory for `/auto log` and are
 * appended as JSONL when `debug.file` is set. Logging never throws into the
 * routing path: every write is guarded.
 */
export class DebugLog {
	private readonly records: DebugRecord[] = [];
	private readonly config: DebugConfig;
	private fileWarned = false;
	private readonly now: () => Date;

	constructor(config: DebugConfig, now: () => Date = () => new Date()) {
		this.config = config;
		this.now = now;
	}

	get enabled(): boolean {
		return this.config.enabled && this.config.level !== "off";
	}

	get level(): DebugConfig["level"] {
		return this.config.level;
	}

	get file(): string | null {
		return this.config.file;
	}

	get includePrompts(): boolean {
		return this.config.includePrompts;
	}

	/** True when records at this level should be produced. */
	allows(level: "summary" | "verbose"): boolean {
		if (!this.enabled) return false;
		return this.config.level === "verbose" || level === "summary";
	}

	record(kind: string, data: Record<string, unknown> = {}): DebugRecord | undefined {
		if (!this.enabled) return undefined;
		const rec: DebugRecord = { ts: this.now().toISOString(), kind, ...redactRecord(data) };
		this.records.push(rec);
		while (this.records.length > this.config.bufferSize) this.records.shift();
		this.appendToFile(rec);
		return rec;
	}

	tail(count = 20): DebugRecord[] {
		return this.records.slice(-Math.max(1, count));
	}

	clear(): void {
		this.records.length = 0;
	}

	size(): number {
		return this.records.length;
	}

	private appendToFile(rec: DebugRecord): void {
		const file = this.config.file;
		if (!file) return;
		try {
			appendFileSync(file, `${JSON.stringify(rec)}\n`);
		} catch (err) {
			if (this.fileWarned) return;
			this.fileWarned = true;
			this.records.push({
				ts: this.now().toISOString(),
				kind: "debug-file-error",
				error: err instanceof Error ? err.message : String(err),
			});
		}
	}
}

/** Redact obvious secrets from any string in a debug record. */
export function redactRecord(data: Record<string, unknown>): Record<string, unknown> {
	const out: Record<string, unknown> = {};
	for (const [key, value] of Object.entries(data)) {
		if (typeof value === "string") out[key] = redactClassifierText(value);
		else if (Array.isArray(value)) out[key] = value.map((v) => (typeof v === "string" ? redactClassifierText(v) : v));
		else out[key] = value;
	}
	return out;
}
