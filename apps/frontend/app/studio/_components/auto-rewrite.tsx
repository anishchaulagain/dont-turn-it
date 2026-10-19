"use client";

import { useEffect, useMemo, useState } from "react";
import { ArrowRight, Download, Loader2, ShieldCheck, Sparkles } from "lucide-react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ApiError } from "@/lib/api";
import { docsApi, type Job, type ModeInfo, type ParagraphChange, type Scope, type VoiceInfo } from "@/lib/docs-api";
import { wordDiff } from "@/lib/word-diff";

import {
  FilePicker,
  INDICATOR_NOTE,
  Meter,
  ModeSelect,
  NO_VOICE,
  pct,
  ScoreTile,
  VoiceNote,
  VoiceSelect,
} from "./shared";

const SCOPES: { value: Scope; label: string }[] = [
  { value: "flagged_uncertain", label: "Flagged + uncertain" },
  { value: "flagged", label: "Flagged only" },
  { value: "all", label: "All paragraphs" },
];

const STAGES: Record<Job["status"], string> = {
  queued: "Queued",
  analyzing: "Analysing writing patterns",
  rewriting: "Rewriting and checking meaning",
  exporting: "Writing the document",
  rescoring: "Scoring the rewritten document",
  done: "Done",
  failed: "Failed",
};

const KEPT_REASON: Record<string, string> = {
  failed_verification: "Every rewrite changed the meaning, so the original was kept",
  no_change_needed: "No change was needed",
  llm_unavailable: "The language model was unavailable, so the original was kept",
  section_locked: "This section is locked in this mode",
};

function highlightFlagged(text: string, flagged: string[]) {
  const parts: { text: string; flagged: boolean }[] = [];
  let rest = text;
  for (const sentence of flagged) {
    const at = rest.indexOf(sentence);
    if (at < 0) continue;
    if (at > 0) parts.push({ text: rest.slice(0, at), flagged: false });
    parts.push({ text: sentence, flagged: true });
    rest = rest.slice(at + sentence.length);
  }
  if (rest) parts.push({ text: rest, flagged: false });
  return parts;
}

function ParagraphCard({ p }: { p: ParagraphChange }) {
  const diff = useMemo(() => (p.rewrite ? wordDiff(p.original, p.rewrite) : null), [p]);
  const original = useMemo(
    () => highlightFlagged(p.original, p.flagged_sentences),
    [p.original, p.flagged_sentences],
  );
  return (
    <Card>
      <CardHeader className="gap-2">
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle className="text-base">{p.section ?? "Untitled section"}</CardTitle>
          {p.status === "rewritten" ? <Badge>Rewritten</Badge> : <Badge variant="outline">Original kept</Badge>}
          <span className="ml-auto flex items-center gap-2 text-sm tabular-nums">
            <span className="text-muted-foreground">AI-likeness</span>
            {pct(p.score_before)}
            <ArrowRight className="size-3.5 text-muted-foreground" />
            <span className="font-medium">{pct(p.score_after)}</span>
          </span>
        </div>
        {p.status !== "rewritten" && p.reason && (
          <CardDescription>{KEPT_REASON[p.reason] ?? p.reason}</CardDescription>
        )}
      </CardHeader>
      <CardContent className="grid gap-4 md:grid-cols-2">
        <div className="space-y-1.5">
          <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            Original · flagged sentences highlighted
          </div>
          <p className="text-sm leading-relaxed">
            {original.map((part, i) =>
              part.flagged ? (
                <mark key={i} className="rounded-sm bg-amber-200/70 px-0.5 dark:bg-amber-500/30">
                  {part.text}
                </mark>
              ) : (
                <span key={i}>{part.text}</span>
              ),
            )}
          </p>
        </div>
        <div className="space-y-1.5">
          <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            {diff ? "Rewrite · changed words highlighted" : "No rewrite"}
          </div>
          <p className="text-sm leading-relaxed">
            {diff
              ? diff.right.map((part, i) =>
                  part.kind === "added" ? (
                    <span key={i} className="rounded-sm bg-emerald-200/70 px-0.5 dark:bg-emerald-500/30">
                      {part.text}
                    </span>
                  ) : (
                    <span key={i}>{part.text}</span>
                  ),
                )
              : <span className="text-muted-foreground">{p.original}</span>}
          </p>
        </div>
        {p.reasons.length > 0 && (
          <div className="flex flex-wrap gap-1.5 md:col-span-2">
            {p.reasons.slice(0, 6).map((r) => (
              <Badge key={r} variant="outline" className="font-normal">
                {r}
              </Badge>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export function AutoRewrite({ modes, voices }: { modes: ModeInfo[]; voices: VoiceInfo[] }) {
  const [mode, setMode] = useState("student.essay");
  const [voice, setVoice] = useState(NO_VOICE);
  const [scope, setScope] = useState<Scope>("flagged_uncertain");
  const [file, setFile] = useState<File | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const jobId = job?.id;
  const running = job != null && job.status !== "done" && job.status !== "failed";
  useEffect(() => {
    if (!jobId || !running) return;
    const timer = setTimeout(() => {
      docsApi
        .job(jobId)
        .then(setJob)
        .catch((err) => setError(err instanceof ApiError ? err.message : "Lost the job"));
    }, 1500);
    return () => clearTimeout(timer);
  }, [jobId, running, job]);

  async function start() {
    if (!file) return;
    setError(null);
    setSubmitting(true);
    try {
      setJob(await docsApi.createJob(file, mode, scope, voice === NO_VOICE ? null : voice));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Upload failed");
    } finally {
      setSubmitting(false);
    }
  }

  const result = job?.result;
  const changed = result?.paragraphs.filter((p) => p.status === "rewritten").length ?? 0;

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <CardHeader>
          <CardTitle>Automatic rewrite</CardTitle>
          <CardDescription>
            Flagged paragraphs are rewritten for you and checked for meaning, citations and numbers.
            Everything else in the file stays exactly as it was.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-[2fr_1fr_1fr_1fr_auto] md:items-end">
          <FilePicker file={file} onChange={setFile} />
          <ModeSelect modes={modes} value={mode} onChange={setMode} />
          <VoiceSelect voices={voices} value={voice} onChange={setVoice} />
          <div className="space-y-1.5">
            <Label>What to rewrite</Label>
            <Select value={scope} onValueChange={(v) => setScope(v as Scope)}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {SCOPES.map((s) => (
                  <SelectItem key={s.value} value={s.value}>
                    {s.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <Button onClick={start} disabled={!file || submitting || running}>
            {submitting || running ? <Loader2 className="animate-spin" /> : <Sparkles />}
            Analyse &amp; rewrite
          </Button>
          <div className="md:col-span-5">
            <VoiceNote voice={voices.find((v) => v.id === voice)} />
          </div>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {job && running && (
        <Card>
          <CardContent className="flex flex-col gap-3 pt-6">
            <div className="flex items-center gap-2 text-sm">
              <Loader2 className="size-4 animate-spin" />
              {STAGES[job.status]}
              {job.status === "rewriting" && job.progress_total > 0 && (
                <span className="text-muted-foreground">
                  · paragraph {job.progress_done} of {job.progress_total}
                </span>
              )}
            </div>
            {job.status === "rewriting" && job.progress_total > 0 && (
              <Meter value={job.progress_done / job.progress_total} />
            )}
            <p className="text-xs text-muted-foreground">
              Free-tier models are rate-limited; a long document can take several minutes.
            </p>
          </CardContent>
        </Card>
      )}

      {job?.status === "failed" && (
        <Alert variant="destructive">
          <AlertDescription>{job.error ?? "The job failed."}</AlertDescription>
        </Alert>
      )}

      {result && job && (
        <>
          <Card>
            <CardHeader>
              <CardTitle>AI-likeness score</CardTitle>
              <CardDescription>
                {result.before.eligible_words} words of prose ({result.before.providers.join(", ")}).{" "}
                {INDICATOR_NOTE}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <div className="flex flex-col gap-4 md:flex-row md:items-stretch">
                <ScoreTile title="Before" summary={result.before} />
                <div className="flex items-center justify-center text-muted-foreground">
                  <ArrowRight className="size-5 rotate-90 md:rotate-0" />
                </div>
                <ScoreTile title="After" summary={result.after} />
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <Button asChild>
                  <a href={docsApi.downloadUrl(job.id)}>
                    <Download /> Download rewritten {job.filename.split(".").pop()}
                  </a>
                </Button>
                <span className="flex items-center gap-1.5 text-sm text-muted-foreground">
                  <ShieldCheck className="size-4" />
                  {changed} of {result.paragraphs.length} selected paragraphs rewritten · formatting,
                  tables and references untouched
                </span>
              </div>
            </CardContent>
          </Card>

          {result.paragraphs.length === 0 ? (
            <Alert>
              <AlertDescription>
                Nothing was flagged for this scope. Try “All paragraphs” to rewrite anyway.
              </AlertDescription>
            </Alert>
          ) : (
            <section className="flex flex-col gap-4">
              <h2 className="text-lg font-semibold">Flagged paragraphs</h2>
              {result.paragraphs.map((p) => (
                <ParagraphCard key={p.block_id} p={p} />
              ))}
            </section>
          )}
        </>
      )}
    </div>
  );
}
