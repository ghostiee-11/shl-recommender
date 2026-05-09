// Tiny typed client for the SHL Recommender HTTP API.
//
// Why a hand-rolled fetch instead of axios/swr/tanstack-query:
// the surface is two endpoints with one happy path and a graceful
// fallback. Pulling in a query library would be more code than it
// saves, and the agent's "always 200, never throw on backend bugs"
// contract removes most of the reasons we'd want one.

import type { ChatRequest, ChatResponse } from "./types";

// In dev, Vite proxies /chat → http://localhost:8000.
// In prod, set VITE_API_BASE_URL to the deployed Render URL.
const BASE = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

export class ApiError extends Error {
  status?: number;
  detail?: unknown;

  constructor(message: string, status?: number, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal,
): Promise<T> {
  const url = `${BASE}${path}`;
  let resp: Response;
  try {
    resp = await fetch(url, {
      headers: { "content-type": "application/json", ...(init.headers ?? {}) },
      signal,
      ...init,
    });
  } catch (err) {
    // Network errors land here (DNS, CORS preflight, offline).
    throw new ApiError(
      err instanceof Error ? err.message : "Network error",
      undefined,
      err,
    );
  }
  // The agent's contract is to return spec-compliant 200 even on
  // internal failure. Anything else is an unexpected condition we
  // surface as a typed error.
  if (!resp.ok) {
    let body: unknown;
    try {
      body = await resp.json();
    } catch {
      body = await resp.text();
    }
    throw new ApiError(
      `HTTP ${resp.status} from ${path}`,
      resp.status,
      body,
    );
  }
  return (await resp.json()) as T;
}

export function postChat(
  messages: ChatRequest["messages"],
  signal?: AbortSignal,
): Promise<ChatResponse> {
  return request<ChatResponse>(
    "/chat",
    { method: "POST", body: JSON.stringify({ messages }) },
    signal,
  );
}

export function getHealth(signal?: AbortSignal): Promise<{ status: string }> {
  return request<{ status: string }>("/health", undefined, signal);
}
