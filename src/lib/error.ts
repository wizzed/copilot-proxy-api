import type { Context } from "hono"
import type { ContentfulStatusCode } from "hono/utils/http-status"

import consola from "consola"

export class HTTPError extends Error {
  response: Response

  constructor(message: string, response: Response) {
    super(message)
    this.response = response
  }
}

/**
 * Anthropic API error envelope shape (per docs.anthropic.com/api/errors):
 *   { "type": "error", "error": { "type": "<category>", "message": "..." } }
 *
 * Error categories Claude Code recognizes (services/api/errors.ts):
 *   invalid_request_error  — 400 (incl. "prompt is too long" → reactive compact)
 *   authentication_error   — 401
 *   permission_error       — 403
 *   not_found_error        — 404
 *   request_too_large      — 413
 *   rate_limit_error       — 429
 *   api_error              — 500 / generic upstream
 *   overloaded_error       — 529
 */

interface AnthropicErrorBody {
  type: "error"
  error: { type: string; message: string }
}

function statusToErrorType(status: number): string {
  switch (status) {
    case 400: {
      return "invalid_request_error"
    }
    case 401: {
      return "authentication_error"
    }
    case 403: {
      return "permission_error"
    }
    case 404: {
      return "not_found_error"
    }
    case 413: {
      return "request_too_large"
    }
    case 429: {
      return "rate_limit_error"
    }
    case 529: {
      return "overloaded_error"
    }
    default: {
      return status >= 500 ? "api_error" : "invalid_request_error"
    }
  }
}

/** Detect bodies already shaped as Anthropic errors (don't double-wrap). */
function isAnthropicErrorShape(value: unknown): value is AnthropicErrorBody {
  if (typeof value !== "object" || value === null) return false
  const obj = value as Record<string, unknown>
  if (obj.type !== "error") return false
  const err = obj.error
  if (typeof err !== "object" || err === null) return false
  const e = err as Record<string, unknown>
  return typeof e.type === "string" && typeof e.message === "string"
}

/** Strip internal implementation details from leaked messages. */
function sanitizeMessage(msg: string): string {
  // Common internal-only strings we don't want to leak to clients.
  if (/copilot token not found/i.test(msg)) {
    return "Upstream authentication unavailable"
  }
  if (/copilot/i.test(msg) && /token/i.test(msg)) {
    return "Upstream authentication error"
  }
  // Trim verbose stack-traces; keep only first line.
  const firstLine = msg.split("\n", 1)[0]
  return firstLine.length > 500 ? firstLine.slice(0, 500) + "…" : firstLine
}

export async function forwardError(c: Context, error: unknown) {
  consola.error("Error occurred:", error)

  if (error instanceof HTTPError) {
    const errorText = await error.response.text()
    const status = error.response.status as ContentfulStatusCode

    let parsed: unknown
    try {
      parsed = JSON.parse(errorText)
    } catch {
      parsed = errorText
    }
    consola.error("HTTP error body:", parsed)

    // If upstream already shaped it as an Anthropic error, pass through verbatim.
    if (isAnthropicErrorShape(parsed)) {
      return c.json(parsed, status)
    }

    // Try to extract a useful message from upstream JSON ({error: {message}}, etc.).
    let message: string
    if (typeof parsed === "string") {
      message = parsed
    } else if (parsed && typeof parsed === "object") {
      const obj = parsed as Record<string, unknown>
      const err = obj.error as Record<string, unknown> | string | undefined
      if (typeof err === "string") {
        message = err
      } else if (err && typeof err.message === "string") {
        message = err.message
      } else if (typeof obj.message === "string") {
        message = obj.message
      } else {
        message = errorText.slice(0, 500)
      }
    } else {
      message = errorText.slice(0, 500)
    }

    const body: AnthropicErrorBody = {
      type: "error",
      error: {
        type: statusToErrorType(status),
        message: sanitizeMessage(message),
      },
    }
    return c.json(body, status)
  }

  const message =
    error instanceof Error ? error.message : "An unknown error occurred"
  const body: AnthropicErrorBody = {
    type: "error",
    error: {
      type: "api_error",
      message: sanitizeMessage(message),
    },
  }
  return c.json(body, 500)
}
