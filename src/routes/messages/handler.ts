import type { Context } from "hono"

import consola from "consola"
import { streamSSE } from "hono/streaming"

import { awaitApproval } from "~/lib/approval"
import { fitContext } from "~/lib/context-manager"
import { checkRateLimit } from "~/lib/rate-limit"
import { state } from "~/lib/state"
import {
  createChatCompletions,
  type ChatCompletionChunk,
  type ChatCompletionResponse,
} from "~/services/copilot/create-chat-completions"

import {
  type AnthropicMessagesPayload,
  type AnthropicStreamState,
} from "./anthropic-types"
import {
  preprocessAnthropicPayload,
  translateToAnthropic,
  translateToOpenAI,
} from "./non-stream-translation"
import {
  translateChunkToAnthropicEvents,
  translateErrorToAnthropicErrorEvent,
} from "./stream-translation"

/** Heartbeat interval for SSE keepalive. Claude Code's idle timeout is 90s
 *  (CLAUDE_STREAM_IDLE_TIMEOUT_MS); 15s gives a 6× safety margin. */
const PING_INTERVAL_MS = 15_000

function generateRequestId(): string {
  // RFC 4122 v4-ish; sufficient for response correlation.
  return `req_${crypto.randomUUID().replaceAll("-", "")}`
}

// eslint-disable-next-line max-lines-per-function
export async function handleCompletion(c: Context) {
  await checkRateLimit(state)

  const requestId = c.req.header("x-client-request-id") ?? generateRequestId()
  c.header("request-id", requestId)
  c.header("x-request-id", requestId)

  const anthropicPayload = await c.req.json<AnthropicMessagesPayload>()
  // Lazy debug logging — JSON.stringify on a large payload is 5-30ms even
  // when debug output is suppressed. Guard with level check (debug = 4).
  if (consola.level >= 4) {
    consola.debug(
      `[${requestId}] Anthropic request payload:`,
      JSON.stringify(anthropicPayload).slice(0, 2000),
    )
  }

  // Preserve the client-requested model name so we can echo it back in the
  // response (Claude Code reads `response.model` for display/telemetry).
  const clientModel = anthropicPayload.model

  // Async preprocessing: PDF document block extraction, etc.
  const preprocessed = await preprocessAnthropicPayload(anthropicPayload)

  const openAIPayload = translateToOpenAI(preprocessed)
  if (consola.level >= 4) {
    consola.debug(
      `[${requestId}] Translated OpenAI request payload:`,
      JSON.stringify(openAIPayload),
    )
  }

  // Byte-based context management — fast path returns input unchanged.
  const model = state.models?.data.find((m) => m.id === openAIPayload.model)
  const fittedPayload = model ? fitContext(openAIPayload, model) : openAIPayload

  if (fittedPayload.messages.length !== openAIPayload.messages.length) {
    consola.info(
      `[${requestId}] Context management: ${openAIPayload.messages.length} → ${fittedPayload.messages.length} messages`,
    )
  }

  if (state.manualApprove) {
    await awaitApproval()
  }

  const response = await createChatCompletions(fittedPayload)

  if (isNonStreaming(response)) {
    if (consola.level >= 4) {
      consola.debug(
        `[${requestId}] Non-streaming response from Copilot:`,
        JSON.stringify(response).slice(-400),
      )
    }
    // Wrap translation in try/catch — translateToAnthropic invokes
    // JSON.parse on tool_call.function.arguments which can throw on
    // malformed Copilot output and would otherwise surface as an
    // unhandled 500 with no Anthropic-shaped body.
    try {
      const anthropicResponse = translateToAnthropic(response, clientModel)
      if (consola.level >= 4) {
        consola.debug(
          `[${requestId}] Translated Anthropic response:`,
          JSON.stringify(anthropicResponse),
        )
      }
      return c.json(anthropicResponse)
    } catch (error) {
      consola.error(
        `[${requestId}] Failed to translate non-streaming response:`,
        error,
      )
      return c.json(
        {
          type: "error" as const,
          error: {
            type: "api_error",
            message:
              error instanceof Error ?
                `Translation failed: ${error.message}`
              : "Translation failed",
          },
        },
        500,
      )
    }
  }

  consola.debug(`[${requestId}] Streaming response from Copilot`)
  return streamSSE(c, async (stream) => {
    const streamState: AnthropicStreamState = {
      messageStartSent: false,
      contentBlockIndex: 0,
      contentBlockOpen: false,
      toolCalls: {},
    }

    // Heartbeat — write SSE comment lines every PING_INTERVAL_MS so
    // Claude Code's 90s idle watchdog doesn't tear down slow streams.
    // Anthropic uses `event: ping` for this; we emit both the structured
    // event and a plain comment for maximum compatibility.
    let pingTimer: ReturnType<typeof setInterval> | undefined
    const startPings = () => {
      pingTimer = setInterval(() => {
        // Best-effort — failures here are non-fatal; the next event will surface
        // any closed-stream condition.
        void stream
          .writeSSE({ event: "ping", data: JSON.stringify({ type: "ping" }) })
          .catch(() => {})
      }, PING_INTERVAL_MS)
    }
    const stopPings = () => {
      if (pingTimer) clearInterval(pingTimer)
      pingTimer = undefined
    }

    startPings()

    try {
      for await (const rawEvent of response) {
        if (rawEvent.data === "[DONE]") {
          break
        }

        if (!rawEvent.data) {
          continue
        }

        let chunk: ChatCompletionChunk
        try {
          chunk = JSON.parse(rawEvent.data) as ChatCompletionChunk
        } catch (parseError) {
          consola.warn(
            `[${requestId}] Skipping unparseable Copilot chunk:`,
            parseError,
          )
          continue
        }

        // Echo client-requested model in the message_start event.
        if (!streamState.messageStartSent) {
          chunk.model = clientModel
        }

        const events = translateChunkToAnthropicEvents(chunk, streamState)

        for (const event of events) {
          await stream.writeSSE({
            event: event.type,
            data: JSON.stringify(event),
          })
        }
      }
    } catch (error) {
      consola.error(`[${requestId}] Streaming error:`, error)

      // Clean up open content blocks so Claude Code's content_block index
      // tracker doesn't throw "Content block not found".
      try {
        if (streamState.contentBlockOpen) {
          await stream.writeSSE({
            event: "content_block_stop",
            data: JSON.stringify({
              type: "content_block_stop",
              index: streamState.contentBlockIndex,
            }),
          })
          streamState.contentBlockOpen = false
        }

        if (streamState.messageStartSent) {
          await stream.writeSSE({
            event: "message_delta",
            data: JSON.stringify({
              type: "message_delta",
              delta: { stop_reason: "end_turn", stop_sequence: null },
              usage: { output_tokens: 0 },
            }),
          })
          await stream.writeSSE({
            event: "message_stop",
            data: JSON.stringify({ type: "message_stop" }),
          })
        }

        const errorEvent = translateErrorToAnthropicErrorEvent(
          error instanceof Error ? error.message : undefined,
        )
        await stream.writeSSE({
          event: "error",
          data: JSON.stringify(errorEvent),
        })
      } catch (cleanupError) {
        consola.error(
          `[${requestId}] Failed to emit stream cleanup events:`,
          cleanupError,
        )
      }
    } finally {
      stopPings()
    }
  })
}

const isNonStreaming = (
  response: Awaited<ReturnType<typeof createChatCompletions>>,
): response is ChatCompletionResponse => Object.hasOwn(response, "choices")
