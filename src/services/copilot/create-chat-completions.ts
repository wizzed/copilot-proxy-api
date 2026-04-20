import consola from "consola"
import { events } from "fetch-event-stream"

import { copilotHeaders, copilotBaseUrl } from "~/lib/api-config"
import { HTTPError } from "~/lib/error"
import { state } from "~/lib/state"

/* eslint-disable complexity */
export const createChatCompletions = async (
  payload: ChatCompletionsPayload,
) => {
  if (!state.copilotToken) throw new Error("Copilot token not found")

  const enableVision = payload.messages.some(
    (x) =>
      typeof x.content !== "string"
      && x.content?.some((x) => x.type === "image_url"),
  )

  // Agent/user check for X-Initiator header
  const isAgentCall = payload.messages.some((msg) =>
    ["assistant", "tool"].includes(msg.role),
  )

  const headers: Record<string, string> = {
    ...copilotHeaders(state, enableVision),
    "X-Initiator": isAgentCall ? "agent" : "user",
  }

  const body = JSON.stringify(payload)
  consola.info(
    `Sending payload: ${body.length} bytes, ${payload.messages.length} messages, model: ${payload.model}`,
  )

  const response = await fetch(`${copilotBaseUrl(state)}/chat/completions`, {
    method: "POST",
    headers,
    body,
  })

  if (!response.ok) {
    const errorBody = await response.text()
    consola.error(
      `Failed to create chat completions - Status: ${response.status} ${response.statusText}`,
    )
    consola.error(`Response body: ${errorBody}`)
    consola.error(`Request payload size: ${body.length} bytes`)

    // Detect context overflow errors. Includes:
    //  - HTTP 413 from Azure Front Door (>~5.4 MB)
    //  - Backend "operation timed out" / 500 in the 2.5 MiB - 5.3 MB dead
    //    zone (Copilot backend hangs >90s instead of cleanly rejecting;
    //    Bun fetch eventually surfaces this as a 500 or upstream timeout)
    //  - Various worded variants Copilot has emitted in the past
    const isContextOverflow =
      response.status === 413
      || /request entity too large/i.test(errorBody)
      || /exceeds the limit of \d+/i.test(errorBody)
      || /context_length_exceeded/i.test(errorBody)
      || /operation timed out/i.test(errorBody)
      || /payload too large/i.test(errorBody)
      || /maximum context length/i.test(errorBody)
      || (response.status >= 500
        && response.status < 600
        && body.length > 2_000_000)

    if (isContextOverflow) {
      // Return HTTP 400 with "prompt is too long" so Claude Code's reactive
      // compaction kicks in (verified from Claude Code source: services/api/
      // errors.ts triggers tryReactiveCompact on this exact phrase). Returning
      // 529/overloaded_error would only trigger retry-with-backoff (3 attempts
      // then fatal "Repeated 529 Overloaded errors").
      //
      // Token estimate: bytes/4 is a rough but conservative approximation.
      // Claude Code uses the gap (estimatedTokens - modelLimit) to decide how
      // many message groups to peel in a single compaction pass.
      //
      // Use max_prompt_tokens (the field Copilot actually enforces — e.g.
      // 168K for opus-4.7) rather than max_context_window_tokens (the
      // total window incl. output, ≥200K). Falling back to the larger field
      // and finally 200K preserves behavior for models lacking metadata.
      const estimatedTokens = Math.ceil(body.length / 4)
      const modelCaps = state.models?.data.find((m) => m.id === payload.model)
        ?.capabilities.limits
      const modelLimit =
        modelCaps?.max_prompt_tokens
        ?? modelCaps?.max_context_window_tokens
        ?? 200_000
      const maxOutputTokens = payload.max_tokens ?? 0

      consola.warn(
        `Context overflow → returning 400 prompt-too-long (~${estimatedTokens} + ${maxOutputTokens} > ${modelLimit}) to trigger Claude Code reactive compaction`,
      )

      // Message format satisfies BOTH Claude Code recovery paths:
      //   1. substring "prompt is too long" → tryReactiveCompact (compact.ts)
      //   2. regex "input length and `max_tokens` exceed context limit:
      //      (\d+) \+ (\d+) > (\d+)" → parseMaxTokensContextOverflowError
      //      (withRetry.ts), which auto-shrinks max_tokens and retries
      //      cleanly without throwing away history.
      throw new HTTPError(
        "Prompt too long",
        new Response(
          JSON.stringify({
            type: "error",
            error: {
              type: "invalid_request_error",
              message: `prompt is too long: input length and \`max_tokens\` exceed context limit: ${estimatedTokens} + ${maxOutputTokens} > ${modelLimit} tokens`,
            },
          }),
          {
            status: 400,
            statusText: "Bad Request",
            headers: { "content-type": "application/json" },
          },
        ),
      )
    }

    throw new HTTPError(
      "Failed to create chat completions",
      new Response(errorBody, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      }),
    )
  }

  if (payload.stream) {
    return events(response)
  }

  return (await response.json()) as ChatCompletionResponse
}

// Streaming types

export interface ChatCompletionChunk {
  id: string
  object: "chat.completion.chunk"
  created: number
  model: string
  choices: Array<Choice>
  system_fingerprint?: string
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    prompt_tokens_details?: {
      cached_tokens: number
    }
    completion_tokens_details?: {
      accepted_prediction_tokens: number
      rejected_prediction_tokens: number
    }
  }
}

interface Delta {
  content?: string | null
  role?: "user" | "assistant" | "system" | "tool"
  tool_calls?: Array<{
    index: number
    id?: string
    type?: "function"
    function?: {
      name?: string
      arguments?: string
    }
  }>
}

interface Choice {
  index: number
  delta: Delta
  finish_reason: "stop" | "length" | "tool_calls" | "content_filter" | null
  logprobs: object | null
}

// Non-streaming types

export interface ChatCompletionResponse {
  id: string
  object: "chat.completion"
  created: number
  model: string
  choices: Array<ChoiceNonStreaming>
  system_fingerprint?: string
  usage?: {
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    prompt_tokens_details?: {
      cached_tokens: number
    }
  }
}

interface ResponseMessage {
  role: "assistant"
  content: string | null
  tool_calls?: Array<ToolCall>
}

interface ChoiceNonStreaming {
  index: number
  message: ResponseMessage
  logprobs: object | null
  finish_reason: "stop" | "length" | "tool_calls" | "content_filter"
}

// Payload types

export interface ChatCompletionsPayload {
  messages: Array<Message>
  model: string
  temperature?: number | null
  top_p?: number | null
  max_tokens?: number | null
  stop?: string | Array<string> | null
  n?: number | null
  stream?: boolean | null

  frequency_penalty?: number | null
  presence_penalty?: number | null
  logit_bias?: Record<string, number> | null
  logprobs?: boolean | null
  response_format?: { type: "json_object" } | null
  seed?: number | null
  tools?: Array<Tool> | null
  tool_choice?:
    | "none"
    | "auto"
    | "required"
    | { type: "function"; function: { name: string } }
    | null
  user?: string | null
}

export interface Tool {
  type: "function"
  function: {
    name: string
    description?: string
    parameters: Record<string, unknown>
  }
}

export interface Message {
  role: "user" | "assistant" | "system" | "tool" | "developer"
  content: string | Array<ContentPart> | null

  name?: string
  tool_calls?: Array<ToolCall>
  tool_call_id?: string
}

export interface ToolCall {
  id: string
  type: "function"
  function: {
    name: string
    arguments: string
  }
}

export type ContentPart = TextPart | ImagePart

export interface TextPart {
  type: "text"
  text: string
}

export interface ImagePart {
  type: "image_url"
  image_url: {
    url: string
    detail?: "low" | "high" | "auto"
  }
}
