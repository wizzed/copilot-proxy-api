/**
 * OpenCode-style context management for Copilot API payloads.
 *
 * Strategy (applied in order, only when over the token limit):
 *  1. Prune old tool outputs — walk backwards, keep the most recent PRUNE_PROTECT
 *     tokens of tool results, replace older ones with a short placeholder.
 *  2. Strip base64 images from older messages (huge byte cost, low info value).
 *  3. Drop oldest conversation messages until the payload fits within the model's
 *     context window minus a safety buffer.
 *
 * Performance: token counting is done once upfront. If within limits, all steps
 * are skipped (zero overhead in the common case). When truncation is needed,
 * per-message token estimates avoid re-counting the entire payload in a loop.
 */

import consola from "consola"

import type {
  ChatCompletionsPayload,
  Message,
} from "~/services/copilot/create-chat-completions"
import type { Model } from "~/services/copilot/get-models"

import { getTokenCount } from "./tokenizer"

// ── Tunables ────────────────────────────────────────────────────────────────

/** Tokens of recent tool output to protect from pruning. */
const PRUNE_PROTECT_TOKENS = 40_000

/** Minimum tokens that must be pruned to bother (avoids tiny savings). */
const PRUNE_MINIMUM_TOKENS = 20_000

/** Safety buffer subtracted from the context window to leave room for the
 *  response and overhead. */
const CONTEXT_BUFFER_TOKENS = 20_000

/** Placeholder that replaces pruned tool output. */
const PRUNED_PLACEHOLDER = "[content pruned — tool output was too old]"

/** Placeholder that replaces stripped images. */
const IMAGE_STRIPPED_PLACEHOLDER = "[image removed to save context]"

/** Absolute byte-size backstop. Empirically determined via probe scripts:
 *  the Copilot gateway/backend has a hard cutoff at exactly 2.5 MiB
 *  (2,621,440 bytes). Payloads at or above that cliff hang indefinitely
 *  (no response within 90s) instead of being cleanly rejected. Payloads
 *  ≥ ~5.4 MB get a fast HTTP 413 from the Azure Front Door gateway.
 *
 *  We target 2,500,000 bytes (decimal MB) — leaves ~120 KB of margin below
 *  the 2.5 MiB cliff for JSON framing, headers, and tool definitions added
 *  downstream of fitContext(). */
const MAX_PAYLOAD_BYTES = 2_500_000

// ── Public API ──────────────────────────────────────────────────────────────

/**
 * Fit a payload within the model's context window using progressive strategies.
 * Returns a new payload (the original is not mutated).
 *
 * Fast path: byte-size check first (microseconds). Only invokes the expensive
 * tokenizer if the payload could plausibly be over the token limit.
 */
export async function fitContext(
  payload: ChatCompletionsPayload,
  model: Model,
): Promise<ChatCompletionsPayload> {
  const contextLimit = model.capabilities.limits.max_context_window_tokens ?? 0
  if (contextLimit === 0) {
    // Unknown model limits — fall back to byte-based backstop only
    return byteBackstop(payload)
  }

  const maxTokens = contextLimit - CONTEXT_BUFFER_TOKENS

  // Cheap fast-path: if the JSON byte size implies we can't possibly exceed
  // the token limit (very conservative: 2 chars per token), skip everything.
  // This avoids invoking the tokenizer on the common case (well under limit).
  const bytes = JSON.stringify(payload).length
  const minPossibleTokens = Math.ceil(bytes / 8) // very loose lower bound
  if (minPossibleTokens <= maxTokens && bytes <= MAX_PAYLOAD_BYTES) {
    return payload
  }

  // Borderline or over — do the precise token count
  const { input: currentTokens } = await getTokenCount(payload, model)
  if (currentTokens <= maxTokens) {
    return byteBackstop(payload) // still check byte limit (images can be huge)
  }

  consola.info(
    `Context overflow: ${currentTokens} tokens > ${maxTokens} limit. Applying context management.`,
  )

  // Clone once — all steps mutate this copy
  let current = structuredClone(payload)
  let tokens = currentTokens

  // Step 1: Prune old tool outputs
  const pruneResult = pruneToolOutputs(current, tokens)
  current = pruneResult.payload
  tokens = pruneResult.tokens

  if (tokens <= maxTokens) return byteBackstop(current)

  // Step 2: Strip base64 images from older messages
  const stripResult = stripOldImages(current, tokens)
  current = stripResult.payload
  tokens = stripResult.tokens

  if (tokens <= maxTokens) return byteBackstop(current)

  // Step 3: Drop oldest conversation messages
  current = dropOldMessages(current, tokens, maxTokens)

  // Step 4: Byte-size backstop for Copilot gateway limit
  return byteBackstop(current)
}

// ── Step 1: Prune old tool outputs ──────────────────────────────────────────

function pruneToolOutputs(
  payload: ChatCompletionsPayload,
  currentTokens: number,
): { payload: ChatCompletionsPayload; tokens: number } {
  const messages = payload.messages
  let protectedTokens = 0
  let prunedTokens = 0
  const prunedIndices: Array<number> = []

  // Walk backwards — protect recent tool outputs, prune older ones
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]
    if (msg.role !== "tool") continue

    const estimatedTokens = estimateMessageTokens(msg)

    if (protectedTokens < PRUNE_PROTECT_TOKENS) {
      protectedTokens += estimatedTokens
    } else {
      prunedTokens += estimatedTokens
      prunedIndices.push(i)
    }
  }

  if (prunedTokens < PRUNE_MINIMUM_TOKENS) {
    return { payload, tokens: currentTokens }
  }

  // Apply pruning
  const newMessages = [...messages]
  const placeholderTokens = estimateStringTokens(PRUNED_PLACEHOLDER)
  let savedTokens = 0

  for (const idx of prunedIndices) {
    const originalTokens = estimateMessageTokens(newMessages[idx])
    newMessages[idx] = { ...newMessages[idx], content: PRUNED_PLACEHOLDER }
    savedTokens += originalTokens - placeholderTokens
  }

  consola.info(
    `Pruned ${prunedIndices.length} old tool outputs (~${savedTokens} tokens saved)`,
  )

  return {
    payload: { ...payload, messages: newMessages },
    tokens: currentTokens - savedTokens,
  }
}

// ── Step 2: Strip base64 images from older messages ─────────────────────────

function stripOldImages(
  payload: ChatCompletionsPayload,
  currentTokens: number,
): { payload: ChatCompletionsPayload; tokens: number } {
  const messages = payload.messages
  const newMessages = [...messages]
  let savedTokens = 0
  let strippedCount = 0

  // Keep images only in the last 4 messages (recent context)
  const protectFromEnd = 4
  const protectBoundary = Math.max(0, messages.length - protectFromEnd)
  const placeholderTokens = estimateStringTokens(IMAGE_STRIPPED_PLACEHOLDER)

  for (let i = 0; i < protectBoundary; i++) {
    const msg = messages[i]
    if (!Array.isArray(msg.content)) continue

    const beforeStripped = strippedCount
    const newContent = msg.content.map((part) => {
      if ("type" in part && part.type === "image_url") {
        strippedCount++
        // Base64 images are typically huge — estimate conservatively
        const imageTokens = estimateStringTokens(part.image_url.url) + 85
        savedTokens += imageTokens - placeholderTokens
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
      `Stripped ${strippedCount} base64 images (~${savedTokens} tokens saved)`,
    )
  }

  return {
    payload: { ...payload, messages: newMessages },
    tokens: currentTokens - savedTokens,
  }
}

// ── Step 3: Drop oldest conversation messages ───────────────────────────────

function dropOldMessages(
  payload: ChatCompletionsPayload,
  currentTokens: number,
  maxTokens: number,
): ChatCompletionsPayload {
  const messages = payload.messages

  // Separate system/developer messages from conversation
  const systemMessages = messages.filter(
    (m) => m.role === "system" || m.role === "developer",
  )
  const conversationMessages = messages.filter(
    (m) => m.role !== "system" && m.role !== "developer",
  )

  let tokens = currentTokens
  let dropped = 0

  // Drop from the front (oldest) — subtract estimated tokens per message
  while (conversationMessages.length > 2 && tokens > maxTokens) {
    const removed = conversationMessages.shift()
    if (!removed) break
    tokens -= estimateMessageTokens(removed)
    dropped++
  }

  if (dropped > 0) {
    consola.info(
      `Dropped ${dropped} old messages (~${currentTokens - tokens} tokens saved, ~${tokens} remaining, limit ${maxTokens})`,
    )
  }

  return { ...payload, messages: [...systemMessages, ...conversationMessages] }
}

// ── Step 4: Byte backstop ───────────────────────────────────────────────────

function byteBackstop(payload: ChatCompletionsPayload): ChatCompletionsPayload {
  const size = JSON.stringify(payload).length
  if (size <= MAX_PAYLOAD_BYTES) return payload

  consola.warn(
    `Payload ${size} bytes exceeds ${MAX_PAYLOAD_BYTES} byte limit. Applying byte-level truncation.`,
  )

  const systemMessages = payload.messages.filter(
    (m) => m.role === "system" || m.role === "developer",
  )
  const conversationMessages = payload.messages.filter(
    (m) => m.role !== "system" && m.role !== "developer",
  )

  const trimmed = [...conversationMessages]
  while (trimmed.length > 2) {
    trimmed.shift()
    const candidate = { ...payload, messages: [...systemMessages, ...trimmed] }
    if (JSON.stringify(candidate).length <= MAX_PAYLOAD_BYTES) {
      consola.info(`Byte backstop: trimmed to ${trimmed.length} messages`)
      return candidate
    }
  }

  return { ...payload, messages: [...systemMessages, ...trimmed] }
}

// ── Token estimation helpers ────────────────────────────────────────────────

/** Rough token estimate for a string (~4 chars per token). */
function estimateStringTokens(str: string): number {
  return Math.ceil(str.length / 4)
}

/** Estimate tokens for a single message without loading the tokenizer. */
function estimateMessageTokens(msg: Message): number {
  const overhead = 4 // role + framing tokens
  if (typeof msg.content === "string") {
    return overhead + estimateStringTokens(msg.content)
  }
  if (Array.isArray(msg.content)) {
    let tokens = overhead
    for (const part of msg.content) {
      if ("text" in part && typeof part.text === "string") {
        tokens += estimateStringTokens(part.text)
      } else if ("type" in part && part.type === "image_url") {
        tokens +=
          estimateStringTokens(
            (part as { image_url: { url: string } }).image_url.url,
          ) + 85
      }
    }
    return tokens
  }
  if (msg.tool_calls) {
    return overhead + estimateStringTokens(JSON.stringify(msg.tool_calls))
  }
  return overhead
}
