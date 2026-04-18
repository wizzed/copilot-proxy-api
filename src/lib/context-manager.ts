/**
 * OpenCode-style context management for Copilot API payloads.
 *
 * Strategy (applied in order):
 *  1. Prune old tool outputs — walk backwards, keep the most recent PRUNE_PROTECT
 *     tokens of tool results, replace older ones with a short placeholder.
 *  2. Strip base64 images from older messages (huge byte cost, low info value).
 *  3. Drop oldest conversation messages until the payload fits within the model's
 *     context window minus a safety buffer.
 *
 * All token counting uses the existing gpt-tokenizer infrastructure.
 */

import consola from "consola"

import type { ChatCompletionsPayload } from "~/services/copilot/create-chat-completions"
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

/** Absolute byte-size backstop (Copilot gateway rejects ~5 MB). */
const MAX_PAYLOAD_BYTES = 4_000_000

// ── Public API ──────────────────────────────────────────────────────────────

/**
 * Fit a payload within the model's context window using progressive strategies.
 * Returns a new payload (the original is not mutated).
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
  let current = structuredClone(payload)

  // Step 1: Prune old tool outputs
  current = await pruneToolOutputs(current, model, maxTokens)

  // Step 2: Strip base64 images from older messages
  current = await stripOldImages(current, model, maxTokens)

  // Step 3: Drop oldest conversation messages
  current = await dropOldMessages(current, model, maxTokens)

  // Step 4: Byte-size backstop for Copilot gateway limit
  current = byteBackstop(current)

  return current
}

// ── Step 1: Prune old tool outputs ──────────────────────────────────────────

async function pruneToolOutputs(
  payload: ChatCompletionsPayload,
  model: Model,
  maxTokens: number,
): Promise<ChatCompletionsPayload> {
  const { input } = await getTokenCount(payload, model)
  if (input <= maxTokens) return payload

  const messages = payload.messages
  let protectedTokens = 0
  let prunedTokens = 0
  const prunedIndices: Array<number> = []

  // Walk backwards — protect recent tool outputs, prune older ones
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]
    if (msg.role !== "tool") continue

    const contentStr =
      typeof msg.content === "string" ?
        msg.content
      : JSON.stringify(msg.content)
    // Rough token estimate: ~4 chars per token
    const estimatedTokens = Math.ceil(contentStr.length / 4)

    if (protectedTokens < PRUNE_PROTECT_TOKENS) {
      protectedTokens += estimatedTokens
    } else {
      // This tool output is old enough to prune
      prunedTokens += estimatedTokens
      prunedIndices.push(i)
    }
  }

  if (prunedTokens < PRUNE_MINIMUM_TOKENS) {
    return payload // Not worth pruning
  }

  // Apply pruning
  const newMessages = [...messages]
  for (const idx of prunedIndices) {
    newMessages[idx] = {
      ...newMessages[idx],
      content: PRUNED_PLACEHOLDER,
    }
  }

  consola.info(
    `Pruned ${prunedIndices.length} old tool outputs (~${prunedTokens} tokens)`,
  )
  return { ...payload, messages: newMessages }
}

// ── Step 2: Strip base64 images from older messages ─────────────────────────

async function stripOldImages(
  payload: ChatCompletionsPayload,
  model: Model,
  maxTokens: number,
): Promise<ChatCompletionsPayload> {
  const { input } = await getTokenCount(payload, model)
  if (input <= maxTokens) return payload

  const messages = payload.messages
  const newMessages = [...messages]
  let strippedCount = 0

  // Keep images only in the last 4 messages (recent context)
  const protectFromEnd = 4
  const protectBoundary = Math.max(0, messages.length - protectFromEnd)

  for (let i = 0; i < protectBoundary; i++) {
    const msg = messages[i]
    if (!Array.isArray(msg.content)) continue

    const hasImages = msg.content.some(
      (part) => "type" in part && part.type === "image_url",
    )
    if (!hasImages) continue

    // Replace image parts with placeholder text
    const newContent = msg.content.map((part) => {
      if ("type" in part && part.type === "image_url") {
        strippedCount++
        return { type: "text" as const, text: IMAGE_STRIPPED_PLACEHOLDER }
      }
      return part
    })

    newMessages[i] = { ...msg, content: newContent }
  }

  if (strippedCount > 0) {
    consola.info(`Stripped ${strippedCount} base64 images from older messages`)
  }
  return { ...payload, messages: newMessages }
}

// ── Step 3: Drop oldest conversation messages ───────────────────────────────

async function dropOldMessages(
  payload: ChatCompletionsPayload,
  model: Model,
  maxTokens: number,
): Promise<ChatCompletionsPayload> {
  const { input } = await getTokenCount(payload, model)
  if (input <= maxTokens) return payload

  const messages = payload.messages

  // Separate system/developer messages from conversation
  const systemMessages = messages.filter(
    (m) => m.role === "system" || m.role === "developer",
  )
  const conversationMessages = messages.filter(
    (m) => m.role !== "system" && m.role !== "developer",
  )

  const trimmed = [...conversationMessages]
  const originalCount = trimmed.length

  while (trimmed.length > 2) {
    trimmed.shift()

    const candidate = { ...payload, messages: [...systemMessages, ...trimmed] }
    const { input: candidateTokens } = await getTokenCount(candidate, model)
    if (candidateTokens <= maxTokens) {
      consola.info(
        `Dropped ${originalCount - trimmed.length} old messages (${candidateTokens} tokens now, limit ${maxTokens})`,
      )
      return candidate
    }
  }

  const minimal = { ...payload, messages: [...systemMessages, ...trimmed] }
  consola.warn(
    `Context still over limit after dropping ${originalCount - trimmed.length} messages. Sending minimal payload.`,
  )
  return minimal
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
