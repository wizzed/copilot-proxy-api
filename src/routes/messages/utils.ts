import { type AnthropicResponse } from "./anthropic-types"

export function mapOpenAIStopReasonToAnthropic(
  finishReason: "stop" | "length" | "tool_calls" | "content_filter" | null,
): AnthropicResponse["stop_reason"] {
  if (finishReason === null) {
    return null
  }
  const stopReasonMap = {
    stop: "end_turn",
    length: "max_tokens",
    tool_calls: "tool_use",
    // Anthropic's spec uses "refusal" when the model declines to respond
    // due to safety/content policy. Map Copilot's "content_filter" to that
    // so Claude Code can surface a meaningful UX (it has a special render
    // path for refusal blocks).
    content_filter: "refusal",
  } as const
  return stopReasonMap[finishReason]
}
