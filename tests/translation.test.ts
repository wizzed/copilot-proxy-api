/**
 * Fixture-based regression tests for the Anthropic ↔ Copilot translation
 * layer. Pure unit-style: no network, no Copilot, no Anthropic.
 *
 * Run with: bun test
 */

/* eslint-disable @typescript-eslint/no-non-null-assertion */

import { describe, expect, test } from "bun:test"
import consola from "consola"
import { Hono } from "hono"

// Silence the consola.error noise that forwardError emits — it's expected
// behavior, not a test failure signal.
consola.level = -999

import type {
  AnthropicMessagesPayload,
  AnthropicStreamState,
} from "~/routes/messages/anthropic-types"
import type {
  ChatCompletionChunk,
  ChatCompletionResponse,
} from "~/services/copilot/create-chat-completions"

import { forwardError, HTTPError } from "~/lib/error"
import { extractPdfText } from "~/lib/pdf"
import {
  preprocessAnthropicPayload,
  translateToAnthropic,
  translateToOpenAI,
} from "~/routes/messages/non-stream-translation"
import {
  translateChunkToAnthropicEvents,
  translateErrorToAnthropicErrorEvent,
} from "~/routes/messages/stream-translation"
import { mapOpenAIStopReasonToAnthropic } from "~/routes/messages/utils"

// ── Helpers ─────────────────────────────────────────────────────────────────

async function callForwardError(error: unknown): Promise<{
  status: number
  body: unknown
}> {
  const app = new Hono()
  app.get("/", async (c) => forwardError(c, error))
  const res = await app.request("/")
  return { status: res.status, body: await res.json() }
}

// ── Error envelope ──────────────────────────────────────────────────────────

describe("forwardError", () => {
  test("maps 400 → invalid_request_error with sanitized message", async () => {
    const upstream = new Response("upstream raw error text", { status: 400 })
    const { status, body } = await callForwardError(
      new HTTPError("boom", upstream),
    )
    expect(status).toBe(400)
    expect(body).toEqual({
      type: "error",
      error: {
        type: "invalid_request_error",
        message: "upstream raw error text",
      },
    })
  })

  test("maps 401 → authentication_error", async () => {
    const { body } = await callForwardError(
      new HTTPError("auth", new Response("nope", { status: 401 })),
    )
    expect((body as { error: { type: string } }).error.type).toBe(
      "authentication_error",
    )
  })

  test("maps 403 → permission_error", async () => {
    const { body } = await callForwardError(
      new HTTPError("p", new Response("nope", { status: 403 })),
    )
    expect((body as { error: { type: string } }).error.type).toBe(
      "permission_error",
    )
  })

  test("maps 404 → not_found_error", async () => {
    const { body } = await callForwardError(
      new HTTPError("nf", new Response("nope", { status: 404 })),
    )
    expect((body as { error: { type: string } }).error.type).toBe(
      "not_found_error",
    )
  })

  test("maps 413 → request_too_large", async () => {
    const { body } = await callForwardError(
      new HTTPError("big", new Response("nope", { status: 413 })),
    )
    expect((body as { error: { type: string } }).error.type).toBe(
      "request_too_large",
    )
  })

  test("maps 429 → rate_limit_error", async () => {
    const { body } = await callForwardError(
      new HTTPError("rl", new Response("nope", { status: 429 })),
    )
    expect((body as { error: { type: string } }).error.type).toBe(
      "rate_limit_error",
    )
  })

  test("maps 500 → api_error", async () => {
    const { body } = await callForwardError(
      new HTTPError("oof", new Response("nope", { status: 500 })),
    )
    expect((body as { error: { type: string } }).error.type).toBe("api_error")
  })

  test("maps 529 → overloaded_error", async () => {
    const { body } = await callForwardError(
      new HTTPError("o", new Response("nope", { status: 529 })),
    )
    expect((body as { error: { type: string } }).error.type).toBe(
      "overloaded_error",
    )
  })

  test("does NOT double-wrap an already Anthropic-shaped body", async () => {
    const anthropicBody = {
      type: "error",
      error: {
        type: "invalid_request_error",
        message: "prompt is too long: 300000 tokens > 200000 maximum",
      },
    }
    const upstream = new Response(JSON.stringify(anthropicBody), {
      status: 400,
    })
    const { status, body } = await callForwardError(
      new HTTPError("ptl", upstream),
    )
    expect(status).toBe(400)
    expect(body).toEqual(anthropicBody)
    // Crucially, the message is NOT a stringified JSON blob
    expect(
      (body as { error: { message: string } }).error.message,
    ).not.toContain('{"type"')
  })

  test("extracts message from upstream {error: {message}} shape", async () => {
    const upstream = new Response(
      JSON.stringify({ error: { message: "upstream said no" } }),
      { status: 400 },
    )
    const { body } = await callForwardError(new HTTPError("x", upstream))
    expect((body as { error: { message: string } }).error.message).toBe(
      "upstream said no",
    )
  })

  test("sanitizes 'Copilot token not found' message", async () => {
    const { body } = await callForwardError(
      new Error("Copilot token not found"),
    )
    expect((body as { error: { message: string } }).error.message).toBe(
      "Upstream authentication unavailable",
    )
  })

  test("non-HTTPError generic Error → 500 api_error", async () => {
    const { status, body } = await callForwardError(new Error("kaboom"))
    expect(status).toBe(500)
    expect((body as { error: { type: string } }).error.type).toBe("api_error")
  })

  test("envelope has top-level type=error (matches Anthropic spec)", async () => {
    const { body } = await callForwardError(
      new HTTPError("x", new Response("y", { status: 400 })),
    )
    expect((body as { type: string }).type).toBe("error")
  })
})

// ── Stop reason mapping ─────────────────────────────────────────────────────

describe("mapOpenAIStopReasonToAnthropic", () => {
  test("stop → end_turn", () => {
    expect(mapOpenAIStopReasonToAnthropic("stop")).toBe("end_turn")
  })
  test("length → max_tokens", () => {
    expect(mapOpenAIStopReasonToAnthropic("length")).toBe("max_tokens")
  })
  test("tool_calls → tool_use", () => {
    expect(mapOpenAIStopReasonToAnthropic("tool_calls")).toBe("tool_use")
  })
  test("content_filter → refusal (NOT end_turn)", () => {
    expect(mapOpenAIStopReasonToAnthropic("content_filter")).toBe("refusal")
  })
  test("null passes through", () => {
    expect(mapOpenAIStopReasonToAnthropic(null)).toBeNull()
  })
})

// ── Non-stream translation ──────────────────────────────────────────────────

describe("translateToAnthropic (response side)", () => {
  test("echoes clientModel when provided, ignoring response.model", () => {
    const response: ChatCompletionResponse = {
      id: "cmpl_abc",
      object: "chat.completion",
      created: 0,
      model: "gpt-4o-2024-08-06",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "hi" },
          finish_reason: "stop",
          logprobs: null,
        },
      ],
    }
    const result = translateToAnthropic(response, "claude-opus-4-5-20250929")
    expect(result.model).toBe("claude-opus-4-5-20250929")
    expect(result.type).toBe("message")
    expect(result.role).toBe("assistant")
  })

  test("falls back to response.model when clientModel undefined", () => {
    const response: ChatCompletionResponse = {
      id: "cmpl_abc",
      object: "chat.completion",
      created: 0,
      model: "claude-opus-4.6-1m",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "hi" },
          finish_reason: "stop",
          logprobs: null,
        },
      ],
    }
    const result = translateToAnthropic(response)
    expect(result.model).toBe("claude-opus-4.6-1m")
  })

  test("safely handles malformed tool_call.function.arguments (no throw)", () => {
    const response: ChatCompletionResponse = {
      id: "cmpl_abc",
      object: "chat.completion",
      created: 0,
      model: "x",
      choices: [
        {
          index: 0,
          message: {
            role: "assistant",
            content: null,
            tool_calls: [
              {
                id: "toolu_1",
                type: "function",
                function: {
                  name: "Bash",
                  arguments: '{"command": "echo hi", "broken json',
                },
              },
            ],
          },
          finish_reason: "tool_calls",
          logprobs: null,
        },
      ],
    }
    expect(() => translateToAnthropic(response, "claude")).not.toThrow()
    const result = translateToAnthropic(response, "claude")
    const toolUse = result.content.find((b) => b.type === "tool_use") as
      | { input: Record<string, unknown> }
      | undefined
    expect(toolUse).toBeDefined()
    expect(toolUse!.input).toEqual({})
  })

  test("usage with cached_tokens emits cache_read_input_tokens", () => {
    const response: ChatCompletionResponse = {
      id: "cmpl_abc",
      object: "chat.completion",
      created: 0,
      model: "x",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "hi" },
          finish_reason: "stop",
          logprobs: null,
        },
      ],
      usage: {
        prompt_tokens: 1000,
        completion_tokens: 50,
        total_tokens: 1050,
        prompt_tokens_details: { cached_tokens: 400 },
      },
    }
    const result = translateToAnthropic(response, "claude")
    expect(result.usage.input_tokens).toBe(600) // 1000 - 400
    expect(result.usage.cache_read_input_tokens).toBe(400)
    expect(result.usage.output_tokens).toBe(50)
  })
})

describe("translateToOpenAI (request side)", () => {
  test("drops thinking blocks from assistant turn (does not promote to text)", () => {
    const payload: AnthropicMessagesPayload = {
      model: "claude-opus-4-5",
      max_tokens: 100,
      messages: [
        {
          role: "user",
          content: "what is 2+2",
        },
        {
          role: "assistant",
          content: [
            {
              type: "thinking",
              thinking: "secret reasoning that shouldn't leak",
            },
            { type: "text", text: "4" },
          ],
        },
      ],
    }
    const result = translateToOpenAI(payload)
    const assistantMsg = result.messages.find((m) => m.role === "assistant")!
    const stringified = JSON.stringify(assistantMsg)
    expect(stringified).toContain("4")
    expect(stringified).not.toContain("secret reasoning")
  })

  test("translates Anthropic tool to OpenAI function tool", () => {
    const payload: AnthropicMessagesPayload = {
      model: "claude-opus",
      max_tokens: 1,
      messages: [{ role: "user", content: "x" }],
      tools: [
        {
          name: "Bash",
          description: "Run a shell command",
          input_schema: { type: "object", properties: {} },
        },
      ],
    }
    const result = translateToOpenAI(payload)
    expect(result.tools).toEqual([
      {
        type: "function",
        function: {
          name: "Bash",
          description: "Run a shell command",
          parameters: { type: "object", properties: {} },
        },
      },
    ])
  })
})

// ── Stream translation ──────────────────────────────────────────────────────

function freshState(): AnthropicStreamState {
  return {
    messageStartSent: false,
    contentBlockIndex: 0,
    contentBlockOpen: false,
    toolCalls: {},
  }
}

describe("translateChunkToAnthropicEvents", () => {
  test("first chunk emits message_start with model & content_block_start+delta", () => {
    const state = freshState()
    const chunk: ChatCompletionChunk = {
      id: "cmpl_x",
      object: "chat.completion.chunk",
      created: 0,
      model: "claude-opus-4.6-1m",
      choices: [
        {
          index: 0,
          delta: { role: "assistant", content: "Hello" },
          finish_reason: null,
          logprobs: null,
        },
      ],
    }
    const events = translateChunkToAnthropicEvents(chunk, state)
    expect(events[0].type).toBe("message_start")
    expect(events[1].type).toBe("content_block_start")
    expect(events[2].type).toBe("content_block_delta")
    expect(state.messageStartSent).toBe(true)
    expect(state.contentBlockOpen).toBe(true)
  })

  test("finish_reason emits content_block_stop, message_delta, message_stop", () => {
    const state: AnthropicStreamState = {
      messageStartSent: true,
      contentBlockIndex: 0,
      contentBlockOpen: true,
      toolCalls: {},
    }
    const chunk: ChatCompletionChunk = {
      id: "cmpl_x",
      object: "chat.completion.chunk",
      created: 0,
      model: "x",
      choices: [{ index: 0, delta: {}, finish_reason: "stop", logprobs: null }],
    }
    const events = translateChunkToAnthropicEvents(chunk, state)
    expect(events.map((e) => e.type)).toEqual([
      "content_block_stop",
      "message_delta",
      "message_stop",
    ])
  })

  test("tool_call delta emits content_block_start with tool_use then input_json_delta", () => {
    const state = freshState()
    state.messageStartSent = true
    const chunk: ChatCompletionChunk = {
      id: "cmpl_x",
      object: "chat.completion.chunk",
      created: 0,
      model: "x",
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: 0,
                id: "toolu_1",
                type: "function",
                function: { name: "Bash", arguments: '{"command":"ls"}' },
              },
            ],
          },
          finish_reason: null,
          logprobs: null,
        },
      ],
    }
    const events = translateChunkToAnthropicEvents(chunk, state)
    const startEvent = events.find((e) => e.type === "content_block_start")!
    expect(startEvent).toMatchObject({
      type: "content_block_start",
      content_block: { type: "tool_use", id: "toolu_1", name: "Bash" },
    })
    const deltaEvent = events.find((e) => e.type === "content_block_delta")!
    expect(deltaEvent).toMatchObject({
      delta: { type: "input_json_delta", partial_json: '{"command":"ls"}' },
    })
  })
})

describe("translateErrorToAnthropicErrorEvent", () => {
  test("default message", () => {
    const event = translateErrorToAnthropicErrorEvent()
    expect(event).toEqual({
      type: "error",
      error: {
        type: "api_error",
        message: "An unexpected error occurred during streaming.",
      },
    })
  })

  test("custom message threaded through", () => {
    const event = translateErrorToAnthropicErrorEvent("backend exploded")
    expect(event.type).toBe("error")
    if (event.type === "error") {
      expect(event.error.message).toBe("backend exploded")
    }
  })
})

// ── PDF preprocessing ───────────────────────────────────────────────────────

describe("preprocessAnthropicPayload (no documents)", () => {
  test("returns same payload reference when no document blocks present", async () => {
    const payload: AnthropicMessagesPayload = {
      model: "claude",
      max_tokens: 1,
      messages: [{ role: "user", content: "hello" }],
    }
    const out = await preprocessAnthropicPayload(payload)
    expect(out).toBe(payload)
  })

  test("inlines text-source document block as text", async () => {
    const payload: AnthropicMessagesPayload = {
      model: "claude",
      max_tokens: 1,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "document",
              source: {
                type: "text",
                media_type: "text/plain",
                data: "the quick brown fox",
              },
              title: "fox.txt",
            },
            { type: "text", text: "summarize this" },
          ],
        },
      ],
    }
    const out = await preprocessAnthropicPayload(payload)
    const userMsg = out.messages[0]
    expect(Array.isArray(userMsg.content)).toBe(true)
    const blocks = userMsg.content as Array<{ type: string; text?: string }>
    expect(blocks).toHaveLength(2)
    expect(blocks[0].type).toBe("text")
    expect(blocks[0].text).toContain("fox.txt")
    expect(blocks[0].text).toContain("the quick brown fox")
  })

  test("URL-source document block emits placeholder text (proxy doesn't fetch)", async () => {
    const payload: AnthropicMessagesPayload = {
      model: "claude",
      max_tokens: 1,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "document",
              source: { type: "url", url: "https://example.com/x.pdf" },
            },
          ],
        },
      ],
    }
    const out = await preprocessAnthropicPayload(payload)
    const blocks = out.messages[0].content as Array<{
      type: string
      text?: string
    }>
    expect(blocks[0].text).toContain("not inlined")
    expect(blocks[0].text).toContain("https://example.com/x.pdf")
  })
})

describe("extractPdfText (PDF lib)", () => {
  test("rejects non-PDF base64 with invalid_request_error", async () => {
    const notAPdf = btoa("this is not a pdf at all")
    let caught: unknown
    try {
      await extractPdfText(notAPdf)
    } catch (e) {
      caught = e
    }
    expect(caught).toBeInstanceOf(HTTPError)
    if (caught instanceof HTTPError) {
      expect(caught.response.status).toBe(400)
      const body = (await caught.response.json()) as {
        error: { type: string; message: string }
      }
      expect(body.error.type).toBe("invalid_request_error")
      expect(body.error.message).toMatch(/invalid|corrupt/i)
    }
  })

  test("rejects truncated PDF magic bytes", async () => {
    const truncated = btoa("%PD")
    let caught: unknown
    try {
      await extractPdfText(truncated)
    } catch (e) {
      caught = e
    }
    expect(caught).toBeInstanceOf(HTTPError)
  })
})
