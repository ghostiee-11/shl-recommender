// Wire-format types, mirror the FastAPI backend exactly.
//
// Keeping these in one tiny file means a backend schema change shows
// up as a TypeScript error in every consumer (the chat panel, the
// recommendation card, the prompt buttons), not as a silent runtime
// surprise.

export type Role = "user" | "assistant";

/** Exactly the shape the backend's `Message` Pydantic model accepts. */
export interface Message {
  role: Role;
  content: string;
}

/** Request body for POST /chat, stateless, full history each call. */
export interface ChatRequest {
  messages: Message[];
}

/** SHL test-type single-letter code. */
export type TestType = "A" | "B" | "C" | "D" | "E" | "K" | "P" | "S";

export const TEST_TYPE_LABEL: Record<TestType, string> = {
  A: "Ability & Aptitude",
  B: "Biodata & Situational Judgment",
  C: "Competencies",
  D: "Development & 360",
  E: "Assessment Exercises",
  K: "Knowledge & Skills",
  P: "Personality & Behavior",
  S: "Simulations",
};

export interface Recommendation {
  name: string;
  url: string;
  test_type: TestType;
}

export interface ChatResponse {
  reply: string;
  recommendations: Recommendation[];
  end_of_conversation: boolean;
}
