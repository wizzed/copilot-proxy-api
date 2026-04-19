/**
 * Context management for Copilot API payloads.
 *
 * The actual constraint we care about is the Copilot gateway's hard 2.5 MiB
 * (2,621,440 byte) ceiling — not the model's claimed token window. Tokens
 * are only used when we genuinely overflow bytes; otherwise we skip the
 * expensive tokenizer pass entirely.
 *
 * Fast path (~99% of Claude Code requests): byte size check → return.
 * Slow path: prune tool outputs → strip images → drop oldest messages →
 *           byte-based final backstop.
 */

import consola from "consola"

import type {
  ChatCompletionsPayload,
  Message,
} from "~/services/copilot/create-chat-completions"
import type { Model } from "~/services/copilot/get-models"

// ── Configuration ───────────────────────────────────────────────────────────

/**
 * Hard byte ceiling for Copilot. Empirically the cliff is at ~5.4 MB for
 * Azure Front Door; we set the proxy ceiling at 5,000,000 to leave ~400 KB
 * headroom for headers, tool definitions, and downstream JSON framing.
 *
 * Why 5 MB and not 2.5 MB (v0.9.1-v0.10.2 value):
 *   The 2.5 MB ceiling fired aggressively on normal Claude Code sessions
 *   (typical 6 MB → 0.9 MB prune per turn). The pruned payload causes
 *   Copilot to report a small prompt_tokens count, which Claude Code uses
 *   to compute its local context estimate via tokenCountWithEstimation.
 *   Because Claude Code's local message array still holds the full unpruned
 *   conversation, its local count climbs by the rough delta each turn until
 *   it hits the blocking limit (window - 3K) and Claude Code emits
 *   "Context limit reached" via the preempt path in query.ts:641 — without
 *   ever calling our proxy, so auto-compact never fires.
 *
 *   Raising the ceiling to 5 MB lets normal payloads pass through unmodified.
 *   Copilot then returns honest prompt_tokens, and Claude Code's auto-compact
 *   threshold (window - 13K) trips on schedule. Pruning still kicks in as a
 *   last-resort safety net for genuinely oversized payloads (≥5 MB), which
 *   is rare in normal Claude Code use.
 *
 *   This restores the v0.8.0 behavior (no proxy-side context manager) which
 *   the user reports worked reliably for auto-compact.
 */
const MAX_PAYLOAD_BYTES = 5_000_000

/** Minimum protected byte budget for recent tool outputs before pruning. */
const PRUNE_PROTECT_BYTES = 200_000

/** Don't prune unless we'd reclaim at least this many bytes. */
const PRUNE_MINIMUM_BYTES = 50_000

/** Per-message hard cap for tool output content even when "recent". A single
 *  giant Read of a 1MB file shouldn't blow the budget by itself. */
const TOOL_OUTPUT_HARD_CAP = 100_000

const PRUNED_PLACEHOLDER = "[content pruned — tool output was too old]"
const TOOL_TRUNCATED_SUFFIX = "\n\n[…tool output truncated to fit context]"
const IMAGE_STRIPPED_PLACEHOLDER = "[image removed to save context]"

// ── Public API ──────────────────────────────────────────────────────────────

/**
 * Fit a payload within the Copilot gateway's byte ceiling. Returns the input
 * payload unmodified in the common case (cheap byte size check).
 */
export function fitContext(
  payload: ChatCompletionsPayload,
  _model: Model,
): ChatCompletionsPayload {
  const initialBytes = estimatePayloadBytes(payload)

  // Fast path: well under the gateway ceiling — just send.
  if (initialBytes <= MAX_PAYLOAD_BYTES) {
    return payload
  }

  consola.info(
    `Context fit: payload ${formatBytes(initialBytes)} exceeds ${formatBytes(MAX_PAYLOAD_BYTES)} ceiling — reducing`,
  )

  // Slow path: progressive byte-based reduction.
  let current = pruneToolOutputs(payload)
  let bytes = estimatePayloadBytes(current)
  if (bytes <= MAX_PAYLOAD_BYTES) {
    consola.info(
      `Context fit: ${formatBytes(initialBytes)} → ${formatBytes(bytes)} after tool pruning`,
    )
    return current
  }

  current = stripOldImages(current)
  bytes = estimatePayloadBytes(current)
  if (bytes <= MAX_PAYLOAD_BYTES) {
    consola.info(
      `Context fit: ${formatBytes(initialBytes)} → ${formatBytes(bytes)} after stripping old images`,
    )
    return current
  }

  // Aggressive: strip ALL images regardless of age.
  current = stripAllImages(current)
  bytes = estimatePayloadBytes(current)
  if (bytes <= MAX_PAYLOAD_BYTES) {
    consola.warn(
      `Context fit: ${formatBytes(initialBytes)} → ${formatBytes(bytes)} after stripping ALL images`,
    )
    return current
  }

  // Truncate any oversized tool outputs that survived pruning (e.g. recent ones).
  current = truncateOversizedToolOutputs(current)
  bytes = estimatePayloadBytes(current)
  if (bytes <= MAX_PAYLOAD_BYTES) {
    consola.warn(
      `Context fit: ${formatBytes(initialBytes)} → ${formatBytes(bytes)} after truncating oversized tool outputs`,
    )
    return current
  }

  current = dropOldMessages(current)
  bytes = estimatePayloadBytes(current)
  consola.warn(
    `Context fit: ${formatBytes(initialBytes)} → ${formatBytes(bytes)} after dropping old messages`,
  )
  return current
}

function formatBytes(b: number): string {
  if (b < 1024) return `${b}B`
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)}KB`
  return `${(b / 1024 / 1024).toFixed(2)}MB`
}

// ── Byte estimation ─────────────────────────────────────────────────────────

/**
 * Cheap byte-size estimate for a payload. Uses content lengths directly
 * instead of JSON.stringify, which is ~10x faster on large payloads. The
 * estimate is a lower bound (ignores JSON framing overhead) but accurate
 * to within a few percent for realistic Claude Code payloads.
 */
function estimatePayloadBytes(payload: ChatCompletionsPayload): number {
  let bytes = 200 // baseline for top-level JSON framing
  for (const msg of payload.messages) {
    bytes += estimateMessageBytes(msg)
  }
  if (payload.tools) {
    for (const tool of payload.tools) {
      // Tool definitions are dense JSON; conservatively double-count via stringify.
      bytes += JSON.stringify(tool).length + 20
    }
  }
  return bytes
}

function estimateMessageBytes(msg: Message): number {
  let bytes = 40 // role + framing
  if (typeof msg.content === "string") {
    bytes += msg.content.length
  } else if (Array.isArray(msg.content)) {
    for (const part of msg.content) {
      if ("text" in part && typeof part.text === "string") {
        bytes += part.text.length + 20
      } else if ("type" in part && part.type === "image_url") {
        bytes += part.image_url.url.length + 40
      }
    }
  }
  if (msg.tool_calls) {
    for (const tc of msg.tool_calls) {
      bytes += tc.function.name.length + tc.function.arguments.length + 80
    }
  }
  if (msg.tool_call_id) bytes += msg.tool_call_id.length + 20
  return bytes
}

// ── Reduction step 1: prune old tool outputs ────────────────────────────────

function pruneToolOutputs(
  payload: ChatCompletionsPayload,
): ChatCompletionsPayload {
  const messages = payload.messages
  let protectedBytes = 0
  let prunedBytes = 0
  const prunedIndices: Array<number> = []

  // Walk newest → oldest, protecting recent tool outputs.
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]
    if (msg.role !== "tool") continue

    const msgBytes = estimateMessageBytes(msg)
    if (protectedBytes < PRUNE_PROTECT_BYTES) {
      protectedBytes += msgBytes
    } else {
      prunedBytes += msgBytes
      prunedIndices.push(i)
    }
  }

  if (prunedBytes < PRUNE_MINIMUM_BYTES) {
    return payload
  }

  const newMessages = [...messages]
  for (const idx of prunedIndices) {
    newMessages[idx] = { ...newMessages[idx], content: PRUNED_PLACEHOLDER }
  }

  consola.info(
    `Pruned ${prunedIndices.length} old tool outputs (~${prunedBytes} bytes reclaimed)`,
  )
  return { ...payload, messages: newMessages }
}

// ── Reduction step 2: strip base64 images from older messages ───────────────

function stripOldImages(
  payload: ChatCompletionsPayload,
): ChatCompletionsPayload {
  const messages = payload.messages
  const newMessages = [...messages]
  let strippedCount = 0
  let strippedBytes = 0

  // Keep images only in the last 4 messages (recent visual context).
  const protectFromEnd = 4
  const protectBoundary = Math.max(0, messages.length - protectFromEnd)

  for (let i = 0; i < protectBoundary; i++) {
    const msg = messages[i]
    if (!Array.isArray(msg.content)) continue

    const beforeStripped = strippedCount
    const newContent = msg.content.map((part) => {
      if ("type" in part && part.type === "image_url") {
        strippedCount++
        strippedBytes += part.image_url.url.length
        return { type: "text" as const, text: IMAGE_STRIPPED_PLACEHOLDER }
      }
      return part
    })

    if (strippedCount > beforeStripped) {
      newMessages[i] = { ...msg, content: newContent }
    }
  }

  if (strippedCount > 0) {
    consola.info(
      `Stripped ${strippedCount} base64 images (~${strippedBytes} bytes reclaimed)`,
    )
  }
  return { ...payload, messages: newMessages }
}

// ── Reduction step 2b: strip ALL images regardless of recency ───────────────

function stripAllImages(
  payload: ChatCompletionsPayload,
): ChatCompletionsPayload {
  let strippedCount = 0
  let strippedBytes = 0
  const newMessages = payload.messages.map((msg) => {
    if (!Array.isArray(msg.content)) return msg
    const before = strippedCount
    const newContent = msg.content.map((part) => {
      if ("type" in part && part.type === "image_url") {
        strippedCount++
        strippedBytes += part.image_url.url.length
        return { type: "text" as const, text: IMAGE_STRIPPED_PLACEHOLDER }
      }
      return part
    })
    return strippedCount > before ? { ...msg, content: newContent } : msg
  })
  if (strippedCount > 0) {
    consola.warn(
      `Stripped ${strippedCount} ALL images (~${strippedBytes} bytes reclaimed)`,
    )
  }
  return { ...payload, messages: newMessages }
}

// ── Reduction step 2c: truncate oversized tool outputs ──────────────────────

function truncateOversizedToolOutputs(
  payload: ChatCompletionsPayload,
): ChatCompletionsPayload {
  let truncatedCount = 0
  let reclaimedBytes = 0
  const newMessages = payload.messages.map((msg) => {
    if (msg.role !== "tool") return msg
    if (typeof msg.content !== "string") return msg
    if (msg.content.length <= TOOL_OUTPUT_HARD_CAP) return msg
    const before = msg.content.length
    const truncated =
      msg.content.slice(0, TOOL_OUTPUT_HARD_CAP) + TOOL_TRUNCATED_SUFFIX
    truncatedCount++
    reclaimedBytes += before - truncated.length
    return { ...msg, content: truncated }
  })
  if (truncatedCount > 0) {
    consola.warn(
      `Truncated ${truncatedCount} oversized tool outputs (~${reclaimedBytes} bytes reclaimed)`,
    )
  }
  return { ...payload, messages: newMessages }
}

// ── Reduction step 3: drop oldest conversation messages ─────────────────────

function dropOldMessages(
  payload: ChatCompletionsPayload,
): ChatCompletionsPayload {
  const systemMessages = payload.messages.filter(
    (m) => m.role === "system" || m.role === "developer",
  )
  const conversationMessages = payload.messages.filter(
    (m) => m.role !== "system" && m.role !== "developer",
  )

  // Compute reusable system byte total.
  let baseBytes = 200
  for (const m of systemMessages) baseBytes += estimateMessageBytes(m)
  if (payload.tools) {
    for (const tool of payload.tools) {
      baseBytes += JSON.stringify(tool).length + 20
    }
  }

  // Compute per-message byte sizes once for the conversation tail.
  const convBytes = conversationMessages.map((m) => estimateMessageBytes(m))
  let totalBytes = baseBytes + convBytes.reduce((a, b) => a + b, 0)

  let dropped = 0
  while (conversationMessages.length > 2 && totalBytes > MAX_PAYLOAD_BYTES) {
    const removedBytes = convBytes.shift() ?? 0
    conversationMessages.shift()
    totalBytes -= removedBytes
    dropped++
  }

  if (dropped > 0) {
    consola.warn(
      `Dropped ${dropped} oldest conversation messages (~${totalBytes} bytes remaining, limit ${MAX_PAYLOAD_BYTES})`,
    )
  }

  if (totalBytes > MAX_PAYLOAD_BYTES) {
    consola.error(
      `Payload still over ${MAX_PAYLOAD_BYTES}B after all reductions (${totalBytes}B). Forwarding anyway — Copilot will likely return 413 → 400 prompt-too-long for reactive compaction.`,
    )
  }

  return {
    ...payload,
    messages: [...systemMessages, ...conversationMessages],
  }
}
