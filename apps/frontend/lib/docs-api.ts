// Typed client for the Document Service (apps/backend/services/docs_api).
// Errors follow the same schema as the Platform API, so ApiError is shared.

import { ApiError } from "@/lib/api";

export const DOCS_API_URL = process.env.NEXT_PUBLIC_DOCS_API_URL ?? "http://localhost:8200";
const V1 = `${DOCS_API_URL}/v1`;

export type Scope = "flagged" | "flagged_uncertain" | "all";
export type JobStatus =
  | "queued"
  | "analyzing"
  | "rewriting"
  | "exporting"
  | "rescoring"
  | "done"
  | "failed";

export type ModeInfo = {
  id: string;
  label: string;
  audience: "student" | "professional";
  phase: number;
  default_view: "coach" | "rewrite";
};

export type VoiceInfo = {
  id: string;
  label: string;
  description: string;
  source: string;
  license: string;
};

export type Band = "low" | "uncertain" | "elevated" | "abstain";

export type SentenceView = {
  index: number;
  text: string;
  band: Band | null;
  score: number | null;
  reasons: string[];
};

export type BlockView = {
  id: string;
  kind: "heading" | "paragraph" | "list_item" | "quote" | "caption";
  section: string | null;
  heading_level: number | null;
  editable: boolean;
  text: string;
  sentences: SentenceView[];
};

export type Check = { gate: string; passed: boolean; message: string; blocking: boolean };
export type SentenceCheck = { passed: boolean; score: number; checks: Check[] };
export type Suggestion = { approach: string; text: string; check: SentenceCheck };
export type Edit = {
  block_id: string;
  sentence_index: number;
  text: string;
  source: "suggestion" | "manual";
};

export type CoachSession = {
  id: string;
  filename: string;
  mode: string;
  voice: string | null;
  document: { blocks: BlockView[]; summary: ScoreSummary; flagged: number };
};

export type ExportResult = {
  after: ScoreSummary;
  warnings: string[];
  applied: number;
  download_url: string;
};

export type ScoreSummary = {
  score: number | null; // 0 (human-like) .. 1 (AI-like), word-weighted
  document_band: "insufficient_text" | "*" | "low" | "uncertain" | "elevated";
  elevated_share: number | null;
  eligible_words: number;
  bands: Partial<Record<"low" | "uncertain" | "elevated" | "abstain", number>>;
  providers: string[];
};

export type ParagraphChange = {
  block_id: string;
  section: string | null;
  status: "rewritten" | "kept_original" | "skipped";
  reason: string | null;
  original: string;
  rewrite: string | null;
  score_before: number | null;
  score_after: number | null;
  flagged_sentences: string[];
  reasons: string[];
};

export type Job = {
  id: string;
  filename: string;
  mode: string;
  scope: Scope;
  status: JobStatus;
  progress_done: number;
  progress_total: number;
  error: string | null;
  created_at: string;
  finished_at: string | null;
  result: { before: ScoreSummary; after: ScoreSummary; paragraphs: ParagraphChange[] } | null;
};

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch {
    throw new ApiError(
      0,
      "network_error",
      `Cannot reach the document service at ${DOCS_API_URL}. Is it running (make dev-docs)?`,
    );
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const err = body?.error;
    throw new ApiError(
      response.status,
      err?.code ?? "unknown_error",
      err?.message ?? response.statusText,
      err?.request_id,
      err?.details,
    );
  }
  return body as T;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

function upload(file: File, fields: Record<string, string | null | undefined>): FormData {
  const form = new FormData();
  form.set("file", file);
  for (const [key, value] of Object.entries(fields)) if (value) form.set(key, value);
  return form;
}

export const docsApi = {
  modes: () => call<ModeInfo[]>(`${V1}/modes`),
  voices: () => call<VoiceInfo[]>(`${V1}/voices`),

  // Automatic rewrite jobs
  createJob: (file: File, mode: string, scope: Scope, voice?: string | null) =>
    call<Job>(`${V1}/jobs`, { method: "POST", body: upload(file, { mode, scope, voice }) }),
  job: (id: string) => call<Job>(`${V1}/jobs/${id}`),
  downloadUrl: (id: string) => `${V1}/jobs/${id}/download`,

  // Coach
  createDocument: (file: File, mode: string, voice?: string | null) =>
    call<CoachSession>(`${V1}/documents`, { method: "POST", body: upload(file, { mode, voice }) }),
  suggestions: (docId: string, blockId: string, sentenceIndex: number) =>
    call<Suggestion[]>(
      `${V1}/documents/${docId}/suggestions`,
      json({ block_id: blockId, sentence_index: sentenceIndex }),
    ),
  check: (docId: string, blockId: string, sentenceIndex: number, text: string) =>
    call<SentenceCheck>(
      `${V1}/documents/${docId}/check`,
      json({ block_id: blockId, sentence_index: sentenceIndex, text }),
    ),
  exportDocument: (docId: string, edits: Edit[]) =>
    call<ExportResult>(`${V1}/documents/${docId}/export`, json({ edits })),
  documentDownloadUrl: (path: string) => `${DOCS_API_URL}${path}`,
};
