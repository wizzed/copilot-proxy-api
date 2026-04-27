import type { Context } from "hono"

import consola from "consola"

import { awaitApproval } from "~/lib/approval"
import { checkRateLimit } from "~/lib/rate-limit"
import { state } from "~/lib/state"
import { createResponses } from "~/services/copilot/create-responses"

import type { ResponsesApiRequest } from "./types"

export async function handleResponses(c: Context): Promise<Response> {
  await checkRateLimit(state)

  const request = await c.req.json<ResponsesApiRequest>()
  consola.debug("Responses API request:", JSON.stringify(request).slice(-400))

  if (state.manualApprove) await awaitApproval()

  return createResponses(request)
}
