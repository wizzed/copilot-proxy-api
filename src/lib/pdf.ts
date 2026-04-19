/**
 * PDF text extraction for Anthropic `document` blocks.
 *
 * Why this exists: GitHub Copilot has no native PDF understanding. Claude Code
 * emits raw `document` blocks (with base64-encoded PDF bytes) for attachments
 * <=3 MB; for larger PDFs it pre-extracts to images on its side. Since Copilot
 * also can't reliably understand images of PDF pages without significant
 * cost, we take the pragmatic approach: extract text in the proxy and inject
 * as a text block.
 *
 * Trade-offs (documented in README):
 *   - Visual content (diagrams, scans, image-only PDFs) is lost.
 *   - Tables/columns may flow incorrectly.
 *   - Password-protected PDFs are rejected with a Claude-Code-recognizable
 *     error message so the user gets actionable feedback.
 */

import consola from "consola"
import { PDFParse } from "pdf-parse"

import { HTTPError } from "./error"

/** Claude Code's `errors.ts` matches "password" + "protected" / "encrypted"
 *  to surface a friendly UX. Use a phrase that hits that pattern. */
const PASSWORD_PROTECTED_MESSAGE =
  "Cannot read password-protected or encrypted PDF document"

const INVALID_PDF_MESSAGE = "Invalid or corrupted PDF document"

const EMPTY_PDF_MESSAGE = "PDF document contains no extractable text"

/** Trim text block to keep payloads sane. Most PDFs we see are <100 pages. */
const MAX_EXTRACTED_CHARS = 500_000

export interface ExtractedPdf {
  text: string
  pageCount: number
  truncated: boolean
}

function decodeBase64(data: string): Uint8Array {
  // Bun.atob returns a binary string; convert to Uint8Array.
  const binary = atob(data)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.codePointAt(i) ?? 0
  }
  return bytes
}

function isLikelyPasswordProtected(error: unknown): boolean {
  const msg = (
    error instanceof Error ?
      error.message
    : String(error)).toLowerCase()
  return (
    msg.includes("password")
    || msg.includes("encrypted")
    || (msg.includes("invalid pdf") && msg.includes("encrypt"))
  )
}

function pdfErrorResponse(message: string): HTTPError {
  return new HTTPError(
    message,
    new Response(
      JSON.stringify({
        type: "error",
        error: {
          type: "invalid_request_error",
          message,
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

/**
 * Extract text from a base64-encoded PDF. Throws HTTPError (Anthropic-shaped)
 * for invalid / password-protected / empty PDFs.
 */
export async function extractPdfText(
  base64Data: string,
): Promise<ExtractedPdf> {
  let bytes: Uint8Array
  try {
    bytes = decodeBase64(base64Data)
  } catch {
    throw pdfErrorResponse(INVALID_PDF_MESSAGE)
  }

  // Magic byte check: PDFs start with "%PDF-"
  if (
    bytes.length < 5
    || bytes[0] !== 0x25
    || bytes[1] !== 0x50
    || bytes[2] !== 0x44
    || bytes[3] !== 0x46
    || bytes[4] !== 0x2d
  ) {
    throw pdfErrorResponse(INVALID_PDF_MESSAGE)
  }

  let parsed: { text: string; total: number }
  try {
    const parser = new PDFParse({ data: bytes })
    const result = await parser.getText()
    parsed = { text: result.text, total: result.total }
  } catch (error) {
    consola.warn("PDF parse failed:", error)
    if (isLikelyPasswordProtected(error)) {
      throw pdfErrorResponse(PASSWORD_PROTECTED_MESSAGE)
    }
    throw pdfErrorResponse(INVALID_PDF_MESSAGE)
  }

  const text = parsed.text.trim()
  if (text.length === 0) {
    throw pdfErrorResponse(EMPTY_PDF_MESSAGE)
  }

  const truncated = text.length > MAX_EXTRACTED_CHARS
  return {
    text:
      truncated ?
        text.slice(0, MAX_EXTRACTED_CHARS)
        + "\n\n[…document truncated for length…]"
      : text,
    pageCount: parsed.total,
    truncated,
  }
}
