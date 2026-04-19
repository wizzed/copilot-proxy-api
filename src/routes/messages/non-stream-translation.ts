import {
  type ChatCompletionResponse,
  type ChatCompletionsPayload,
  type ContentPart,
  type Message,
  type TextPart,
  type Tool,
  type ToolCall,
} from "~/services/copilot/create-chat-completions"

import {
  type AnthropicAssistantContentBlock,
  type AnthropicAssistantMessage,
  type AnthropicDocumentBlock,
  type AnthropicMessage,
  type AnthropicMessagesPayload,
  type AnthropicResponse,
  type AnthropicTextBlock,
  type AnthropicTool,
  type AnthropicToolResultBlock,
  type AnthropicToolUseBlock,
  type AnthropicUserContentBlock,
  type AnthropicUserMessage,
} from "./anthropic-types"
import { mapOpenAIStopReasonToAnthropic } from "./utils"

// Payload translation

export function translateToOpenAI(
  payload: AnthropicMessagesPayload,
): ChatCompletionsPayload {
  return {
    model: translateModelName(payload.model),
    messages: translateAnthropicMessagesToOpenAI(
      payload.messages,
      payload.system,
    ),
    max_tokens: payload.max_tokens,
    stop: payload.stop_sequences,
    stream: payload.stream,
    temperature: payload.temperature,
    top_p: payload.top_p,
    user: payload.metadata?.user_id,
    tools: translateAnthropicToolsToOpenAI(payload.tools),
    tool_choice: translateAnthropicToolChoiceToOpenAI(payload.tool_choice),
  }
}

/**
 * Pre-process the Anthropic payload to handle async / proxy-side concerns
 * (currently: PDF document block extraction). Returns a new payload with
 * `document` blocks replaced by extracted-text blocks.
 *
 * Throws HTTPError (Anthropic-shaped) on PDF extraction failure.
 */
export async function preprocessAnthropicPayload(
  payload: AnthropicMessagesPayload,
): Promise<AnthropicMessagesPayload> {
  // Fast path: no document blocks anywhere.
  const hasDocuments = payload.messages.some(
    (m) =>
      Array.isArray(m.content) && m.content.some((b) => b.type === "document"),
  )
  if (!hasDocuments) return payload

  const newMessages: Array<AnthropicMessage> = []
  for (const msg of payload.messages) {
    if (msg.role !== "user" || !Array.isArray(msg.content)) {
      newMessages.push(msg)
      continue
    }
    const newContent: Array<AnthropicUserContentBlock> = []
    for (const block of msg.content) {
      if (block.type === "document") {
        newContent.push(...(await extractDocumentBlock(block)))
      } else {
        newContent.push(block)
      }
    }
    newMessages.push({ role: "user", content: newContent })
  }
  return { ...payload, messages: newMessages }
}

async function extractDocumentBlock(
  block: AnthropicDocumentBlock,
): Promise<Array<AnthropicTextBlock>> {
  // Plain-text document source — just inline.
  if (block.source.type === "text") {
    return [
      {
        type: "text",
        text: documentHeader(block) + block.source.data,
      },
    ]
  }

  // URL document source — Copilot can't fetch URLs; skip with note.
  if (block.source.type === "url") {
    return [
      {
        type: "text",
        text:
          documentHeader(block)
          + `[document at ${block.source.url} not inlined — proxy does not fetch external URLs]`,
      },
    ]
  }

  // base64 PDF — extract via pdf-parse.
  const { extractPdfText } = await import("~/lib/pdf")
  const extracted = await extractPdfText(block.source.data)
  return [
    {
      type: "text",
      text:
        documentHeader(block)
        + `[Extracted text from PDF (${extracted.pageCount} page${extracted.pageCount === 1 ? "" : "s"}${extracted.truncated ? ", truncated" : ""}):]\n\n`
        + extracted.text,
    },
  ]
}

function documentHeader(block: AnthropicDocumentBlock): string {
  const parts: Array<string> = []
  if (block.title) parts.push(`Document: ${block.title}`)
  if (block.context) parts.push(`Context: ${block.context}`)
  return parts.length > 0 ? parts.join("\n") + "\n\n" : ""
}

// Exact model name mappings
const MODEL_NAME_MAP: Record<string, string> = {
  haiku: "claude-haiku-4.5",
  sonnet: "claude-sonnet-4",
  opus: "claude-opus-4.6-1m",
  "claude-opus-4": "claude-opus-4.6-1m",
  "claude-opus-4-0": "claude-opus-4.6-1m",
  "claude-opus-4-1": "claude-opus-4.6-1m",
  "claude-opus-4-5": "claude-opus-4.6-1m",
  "claude-opus-4.5": "claude-opus-4.6-1m",
  "claude-opus-4.6-1m": "claude-opus-4.6-1m",
  "claude-sonnet-4-0": "claude-sonnet-4",
  "claude-sonnet-4-5": "claude-sonnet-4",
  "claude-sonnet-4.5": "claude-sonnet-4",
  "claude-sonnet-4-6": "claude-sonnet-4",
  "claude-haiku-4": "claude-haiku-4.5",
  "claude-haiku-4-5": "claude-haiku-4.5",
}

// Pattern-based model mappings: [pattern, target]
const MODEL_PATTERN_MAP: Array<[string, string]> = [
  // Claude 3.5 models (older naming convention)
  ["claude-3-5-sonnet", "claude-sonnet-4"],
  ["claude-3.5-sonnet", "claude-sonnet-4"],
  ["claude-3-5-haiku", "claude-haiku-4.5"],
  ["claude-3.5-haiku", "claude-haiku-4.5"],
  ["claude-3-opus", "claude-opus-4.6-1m"],
  ["claude-3.0-opus", "claude-opus-4.6-1m"],
  // Claude 4.x models with version suffixes (e.g., claude-sonnet-4-20241022)
  ["claude-sonnet-4-", "claude-sonnet-4"],
  ["claude-sonnet-4.", "claude-sonnet-4"],
  ["claude-opus-4-", "claude-opus-4.6-1m"],
  ["claude-opus-4.", "claude-opus-4.6-1m"],
  ["claude-haiku-4-", "claude-haiku-4.5"],
  ["claude-haiku-4.", "claude-haiku-4.5"],
  // Claude 4.6 models (including 1M context variants)
  ["claude-opus-4.6", "claude-opus-4.6-1m"],
  ["claude-sonnet-4.6", "claude-sonnet-4"],
]

function translateModelName(model: string): string {
  // Claude Code uses Anthropic model IDs, but GitHub Copilot uses different naming
  // Map common Claude Code model names to GitHub Copilot equivalents

  // Check exact matches first
  if (MODEL_NAME_MAP[model]) {
    return MODEL_NAME_MAP[model]
  }

  // Check pattern-based matches
  for (const [pattern, target] of MODEL_PATTERN_MAP) {
    if (model.includes(pattern)) {
      return target
    }
  }

  return model
}

function translateAnthropicMessagesToOpenAI(
  anthropicMessages: Array<AnthropicMessage>,
  system: string | Array<AnthropicTextBlock> | undefined,
): Array<Message> {
  const systemMessages = handleSystemPrompt(system)

  const otherMessages = anthropicMessages.flatMap((message) =>
    message.role === "user" ?
      handleUserMessage(message)
    : handleAssistantMessage(message),
  )

  return [...systemMessages, ...otherMessages]
}

// Reserved keywords that GitHub Copilot API blocks in system prompts
// These will be completely removed from system prompts (case-insensitive)
const RESERVED_KEYWORD_PATTERNS = [
  /x-anthropic-billing-header/gi,
  /anthropic-billing-header/gi,
  /x-anthropic-/gi,
]

function sanitizeSystemPrompt(text: string): string {
  let sanitized = text
  for (const pattern of RESERVED_KEYWORD_PATTERNS) {
    sanitized = sanitized.replaceAll(pattern, "")
  }
  return sanitized
}

function handleSystemPrompt(
  system: string | Array<AnthropicTextBlock> | undefined,
): Array<Message> {
  if (!system) {
    return []
  }

  if (typeof system === "string") {
    return [{ role: "system", content: sanitizeSystemPrompt(system) }]
  } else {
    const systemText = system.map((block) => block.text).join("\n\n")
    return [{ role: "system", content: sanitizeSystemPrompt(systemText) }]
  }
}

function handleUserMessage(message: AnthropicUserMessage): Array<Message> {
  const newMessages: Array<Message> = []

  if (Array.isArray(message.content)) {
    const toolResultBlocks = message.content.filter(
      (block): block is AnthropicToolResultBlock =>
        block.type === "tool_result",
    )
    const otherBlocks = message.content.filter(
      (block) => block.type !== "tool_result",
    )

    // Tool results must come first to maintain protocol: tool_use -> tool_result -> user
    for (const block of toolResultBlocks) {
      newMessages.push({
        role: "tool",
        tool_call_id: block.tool_use_id,
        content: mapContent(block.content),
      })
    }

    if (otherBlocks.length > 0) {
      newMessages.push({
        role: "user",
        content: mapContent(otherBlocks),
      })
    }
  } else {
    newMessages.push({
      role: "user",
      content: mapContent(message.content),
    })
  }

  return newMessages
}

function handleAssistantMessage(
  message: AnthropicAssistantMessage,
): Array<Message> {
  if (!Array.isArray(message.content)) {
    return [
      {
        role: "assistant",
        content: mapContent(message.content),
      },
    ]
  }

  const toolUseBlocks = message.content.filter(
    (block): block is AnthropicToolUseBlock => block.type === "tool_use",
  )

  const textBlocks = message.content.filter(
    (block): block is AnthropicTextBlock => block.type === "text",
  )

  // Thinking blocks have signed-by-Anthropic semantics that Copilot can't
  // verify or replay. Promoting their content to plain text would destroy
  // the assistant turn semantics (the model would treat its own private
  // reasoning as visible text). Drop them cleanly on the request side.
  const allTextContent = textBlocks.map((b) => b.text).join("\n\n")

  return toolUseBlocks.length > 0 ?
      [
        {
          role: "assistant",
          content: allTextContent || null,
          tool_calls: toolUseBlocks.map((toolUse) => ({
            id: toolUse.id,
            type: "function",
            function: {
              name: toolUse.name,
              arguments: JSON.stringify(toolUse.input),
            },
          })),
        },
      ]
    : [
        {
          role: "assistant",
          content: mapContent(message.content),
        },
      ]
}

function mapContent(
  content:
    | string
    | Array<AnthropicUserContentBlock | AnthropicAssistantContentBlock>,
): string | Array<ContentPart> | null {
  if (typeof content === "string") {
    return content
  }
  if (!Array.isArray(content)) {
    return null
  }

  const hasImage = content.some((block) => block.type === "image")
  if (!hasImage) {
    return content
      .filter((block): block is AnthropicTextBlock => block.type === "text")
      .map((block) => block.text)
      .join("\n\n")
  }

  const contentParts: Array<ContentPart> = []
  for (const block of content) {
    switch (block.type) {
      case "text": {
        contentParts.push({ type: "text", text: block.text })

        break
      }
      // Thinking blocks dropped — see handleAssistantMessage rationale.
      case "image": {
        contentParts.push({
          type: "image_url",
          image_url: {
            url: `data:${block.source.media_type};base64,${block.source.data}`,
          },
        })

        break
      }
      // No default
    }
  }
  return contentParts
}

function translateAnthropicToolsToOpenAI(
  anthropicTools: Array<AnthropicTool> | undefined,
): Array<Tool> | undefined {
  if (!anthropicTools) {
    return undefined
  }
  return anthropicTools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.input_schema,
    },
  }))
}

function translateAnthropicToolChoiceToOpenAI(
  anthropicToolChoice: AnthropicMessagesPayload["tool_choice"],
): ChatCompletionsPayload["tool_choice"] {
  if (!anthropicToolChoice) {
    return undefined
  }

  switch (anthropicToolChoice.type) {
    case "auto": {
      return "auto"
    }
    case "any": {
      return "required"
    }
    case "tool": {
      if (anthropicToolChoice.name) {
        return {
          type: "function",
          function: { name: anthropicToolChoice.name },
        }
      }
      return undefined
    }
    case "none": {
      return "none"
    }
    default: {
      return undefined
    }
  }
}

// Response translation

// eslint-disable-next-line complexity
export function translateToAnthropic(
  response: ChatCompletionResponse,
  /** Original client-requested model id, echoed back so Claude Code displays
   *  the model the user asked for rather than Copilot's internal id. */
  clientModel?: string,
): AnthropicResponse {
  // Merge content from all choices
  const allTextBlocks: Array<AnthropicTextBlock> = []
  const allToolUseBlocks: Array<AnthropicToolUseBlock> = []
  let stopReason: "stop" | "length" | "tool_calls" | "content_filter" | null =
    null // default
  stopReason = response.choices[0]?.finish_reason ?? stopReason

  // Process all choices to extract text and tool use blocks
  for (const choice of response.choices) {
    const textBlocks = getAnthropicTextBlocks(choice.message.content)
    const toolUseBlocks = getAnthropicToolUseBlocks(choice.message.tool_calls)

    allTextBlocks.push(...textBlocks)
    allToolUseBlocks.push(...toolUseBlocks)

    // Use the finish_reason from the first choice, or prioritize tool_calls
    if (choice.finish_reason === "tool_calls" || stopReason === "stop") {
      stopReason = choice.finish_reason
    }
  }

  // Note: GitHub Copilot doesn't generate thinking blocks, so we don't include them in responses

  return {
    id: response.id,
    type: "message",
    role: "assistant",
    model: clientModel ?? response.model,
    content: [...allTextBlocks, ...allToolUseBlocks],
    stop_reason: mapOpenAIStopReasonToAnthropic(stopReason),
    stop_sequence: null,
    usage: {
      input_tokens:
        (response.usage?.prompt_tokens ?? 0)
        - (response.usage?.prompt_tokens_details?.cached_tokens ?? 0),
      output_tokens: response.usage?.completion_tokens ?? 0,
      ...(response.usage?.prompt_tokens_details?.cached_tokens
        !== undefined && {
        cache_read_input_tokens:
          response.usage.prompt_tokens_details.cached_tokens,
      }),
    },
  }
}

function getAnthropicTextBlocks(
  messageContent: Message["content"],
): Array<AnthropicTextBlock> {
  if (typeof messageContent === "string") {
    return [{ type: "text", text: messageContent }]
  }

  if (Array.isArray(messageContent)) {
    return messageContent
      .filter((part): part is TextPart => part.type === "text")
      .map((part) => ({ type: "text", text: part.text }))
  }

  return []
}

function getAnthropicToolUseBlocks(
  toolCalls: Array<ToolCall> | undefined,
): Array<AnthropicToolUseBlock> {
  if (!toolCalls) {
    return []
  }
  return toolCalls.map((toolCall) => {
    let input: Record<string, unknown> = {}
    try {
      const parsed = JSON.parse(toolCall.function.arguments) as unknown
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        input = parsed as Record<string, unknown>
      }
    } catch {
      // Copilot can emit malformed JSON for tool args (especially mid-stream
      // truncation or partial outputs). Fall back to an empty object so the
      // assistant turn is at least well-formed; Claude Code will surface the
      // missing args via tool execution failure rather than a hard crash.
    }
    return {
      type: "tool_use",
      id: toolCall.id,
      name: toolCall.function.name,
      input,
    }
  })
}
