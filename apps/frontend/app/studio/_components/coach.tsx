"use client";

import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  Check as CheckIcon,
  Download,
  GraduationCap,
  Lightbulb,
  Loader2,
  RotateCcw,
} from "lucide-react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { ApiError } from "@/lib/api";
import {
  docsApi,
  type BlockView,
  type CoachSession,
  type Edit,
  type ExportResult,
  type ModeInfo,
  type SentenceCheck,
  type SentenceView,
  type Suggestion,
  type VoiceInfo,
} from "@/lib/docs-api";
import { cn } from "@/lib/utils";

import {
  FilePicker,
  INDICATOR_NOTE,
  ModeSelect,
  NO_VOICE,
  pct,
  ScoreTile,
  VoiceNote,
  VoiceSelect,
} from "./shared";

type Key = string; // `${blockId}:${sentenceIndex}`
const keyOf = (blockId: string, index: number): Key => `${blockId}:${index}`;

const APPROACH: Record<string, string> = {
  minimal: "Light touch",
  restructured: "Restructured",
  plain: "Plainest",
};

function without<T>(record: Record<Key, T>, key: Key): Record<Key, T> {
  const next = { ...record };
  delete next[key];
  return next;
}

const errorText = (err: unknown, fallback: string) =>
  err instanceof ApiError ? err.message : fallback;

function CheckList({ check }: { check: SentenceCheck }) {
  const problems = check.checks.filter((c) => !c.passed);
  if (problems.length === 0) {
    return (
      <div className="flex items-center gap-1.5 text-xs text-emerald-700 dark:text-emerald-400">
        <CheckIcon className="size-3.5" /> Meaning checks passed
      </div>
    );
  }
  return (
    <ul className="space-y-1 text-xs">
      {problems.map((c) => (
        <li
          key={c.gate}
          className={cn(
            "flex gap-1.5",
            c.blocking ? "text-red-700 dark:text-red-400" : "text-muted-foreground",
          )}
        >
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
          {c.message}
        </li>
      ))}
    </ul>
  );
}

function sentenceClass(s: SentenceView, edited: boolean, selected: boolean) {
  return cn(
    "cursor-pointer rounded-sm px-0.5 transition-colors",
    edited && "bg-emerald-200/70 dark:bg-emerald-500/30",
    !edited && s.band === "elevated" && "bg-red-200/70 dark:bg-red-500/30",
    !edited && s.band === "uncertain" && "bg-amber-200/70 dark:bg-amber-500/30",
    !edited && !s.band && "hover:bg-muted",
    !edited && s.band === "low" && "hover:bg-muted",
    selected && "ring-2 ring-primary",
  );
}

function DocumentPane({
  blocks,
  edits,
  selected,
  onSelect,
}: {
  blocks: BlockView[];
  edits: Record<Key, Edit>;
  selected: Key | null;
  onSelect: (block: BlockView, s: SentenceView) => void;
}) {
  return (
    <div className="space-y-4 text-[15px] leading-7">
      {blocks.map((b) => {
        if (b.kind === "heading") {
          return (
            <h3
              key={b.id}
              className={cn("font-semibold", (b.heading_level ?? 2) <= 1 ? "pt-4 text-lg" : "pt-2")}
            >
              {b.text.trim()}
            </h3>
          );
        }
        if (!b.editable) {
          return (
            <p key={b.id} className="text-muted-foreground">
              {b.text}
            </p>
          );
        }
        return (
          <p key={b.id} className={cn(b.kind === "list_item" && "pl-5")}>
            {b.kind === "list_item" && <span className="-ml-4 mr-2">•</span>}
            {b.sentences.map((s, i) => {
              const k = keyOf(b.id, s.index);
              const edit = edits[k];
              return (
                <span key={k}>
                  <span
                    role="button"
                    tabIndex={0}
                    className={sentenceClass(s, !!edit, selected === k)}
                    onClick={() => onSelect(b, s)}
                    onKeyDown={(e) => e.key === "Enter" && onSelect(b, s)}
                  >
                    {edit ? edit.text : s.text}
                  </span>
                  {i < b.sentences.length - 1 ? " " : ""}
                </span>
              );
            })}
          </p>
        );
      })}
    </div>
  );
}

export function Coach({ modes, voices }: { modes: ModeInfo[]; voices: VoiceInfo[] }) {
  const [file, setFile] = useState<File | null>(null);
  const [mode, setMode] = useState("student.essay");
  const [voice, setVoice] = useState(NO_VOICE);
  const [session, setSession] = useState<CoachSession | null>(null);
  const [opening, setOpening] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [selected, setSelected] = useState<{ block: BlockView; sentence: SentenceView } | null>(null);
  const [edits, setEdits] = useState<Record<Key, Edit>>({});
  const [drafts, setDrafts] = useState<Record<Key, string>>({});
  const [suggestions, setSuggestions] = useState<Record<Key, Suggestion[]>>({});
  const [suggesting, setSuggesting] = useState<Key | null>(null);
  const [checks, setChecks] = useState<Record<Key, SentenceCheck>>({});
  const [exported, setExported] = useState<ExportResult | null>(null);
  const [exporting, setExporting] = useState(false);

  const selKey = selected ? keyOf(selected.block.id, selected.sentence.index) : null;
  const draft = selKey ? (drafts[selKey] ?? edits[selKey]?.text ?? selected?.sentence.text ?? "") : "";
  const changedDraft = !!selected && draft.trim() !== selected.sentence.text.trim();

  // Live meaning check of the writer's own wording (debounced).
  useEffect(() => {
    if (!session || !selected || !selKey || !changedDraft || !draft.trim()) return;
    const timer = setTimeout(() => {
      docsApi
        .check(session.id, selected.block.id, selected.sentence.index, draft)
        .then((c) => setChecks((prev) => ({ ...prev, [selKey]: c })))
        .catch(() => undefined);
    }, 500);
    return () => clearTimeout(timer);
  }, [session, selected, selKey, draft, changedDraft]);

  const flaggedLeft = useMemo(() => {
    if (!session) return 0;
    return session.document.blocks
      .flatMap((b) => b.sentences.map((s) => ({ b, s })))
      .filter(({ b, s }) => b.editable && (s.band === "elevated" || s.band === "uncertain"))
      .filter(({ b, s }) => !edits[keyOf(b.id, s.index)]).length;
  }, [session, edits]);

  async function open() {
    if (!file) return;
    setError(null);
    setOpening(true);
    try {
      const s = await docsApi.createDocument(file, mode, voice === NO_VOICE ? null : voice);
      setSession(s);
      setSelected(null);
      setEdits({});
      setDrafts({});
      setSuggestions({});
      setChecks({});
      setExported(null);
    } catch (err) {
      setError(errorText(err, "Could not open the document"));
    } finally {
      setOpening(false);
    }
  }

  async function loadSuggestions() {
    if (!session || !selected || !selKey) return;
    setSuggesting(selKey);
    setError(null);
    try {
      const list = await docsApi.suggestions(session.id, selected.block.id, selected.sentence.index);
      setSuggestions((prev) => ({ ...prev, [selKey]: list }));
    } catch (err) {
      setError(errorText(err, "Could not get suggestions"));
    } finally {
      setSuggesting(null);
    }
  }

  function accept(source: Edit["source"]) {
    if (!selected || !selKey || !changedDraft) return;
    setEdits((prev) => ({
      ...prev,
      [selKey]: {
        block_id: selected.block.id,
        sentence_index: selected.sentence.index,
        text: draft.trim(),
        source,
      },
    }));
    setExported(null);
  }

  function undo() {
    if (!selKey) return;
    setEdits((prev) => without(prev, selKey));
    setDrafts((prev) => without(prev, selKey));
    setExported(null);
  }

  async function applyEdits() {
    if (!session) return;
    setExporting(true);
    setError(null);
    try {
      setExported(await docsApi.exportDocument(session.id, Object.values(edits)));
    } catch (err) {
      setError(errorText(err, "Export failed"));
    } finally {
      setExporting(false);
    }
  }

  const fromSuggestion =
    !!selKey && (suggestions[selKey] ?? []).some((s) => s.text === draft.trim());
  const liveCheck = selKey && changedDraft ? checks[selKey] : undefined;
  const editCount = Object.keys(edits).length;

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <GraduationCap className="size-5" /> Coach
          </CardTitle>
          <CardDescription>
            You stay the author. Sentences that read as generic are highlighted with the reason; ask
            for alternatives, or write your own wording and get live meaning checks. Only the
            sentences you accept change in the downloaded file.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-[2fr_1fr_1fr_auto] md:items-end">
          <FilePicker file={file} onChange={setFile} />
          <ModeSelect modes={modes} value={mode} onChange={setMode} />
          <VoiceSelect voices={voices} value={voice} onChange={setVoice} />
          <Button onClick={open} disabled={!file || opening}>
            {opening ? <Loader2 className="animate-spin" /> : <GraduationCap />}
            Open in coach
          </Button>
          <div className="md:col-span-4">
            <VoiceNote voice={voices.find((v) => v.id === voice)} />
          </div>
        </CardContent>
      </Card>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {session && (
        <>
          <Card>
            <CardContent className="flex flex-col gap-4 pt-6">
              <div className="flex flex-col gap-4 md:flex-row md:items-stretch">
                <ScoreTile title="Your document now" summary={session.document.summary} />
                <div className="flex items-center justify-center text-muted-foreground">
                  <ArrowRight className="size-5 rotate-90 md:rotate-0" />
                </div>
                {exported ? (
                  <ScoreTile title="With your accepted edits" summary={exported.after} />
                ) : (
                  <div className="flex flex-1 flex-col justify-center gap-2 rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
                    <span>
                      {editCount} edit{editCount === 1 ? "" : "s"} accepted · {flaggedLeft} highlighted
                      sentence{flaggedLeft === 1 ? "" : "s"} left
                    </span>
                    <span>Apply your edits to see the new score and download the file.</span>
                  </div>
                )}
              </div>
              <div className="flex flex-wrap items-center gap-3">
                <Button onClick={applyEdits} disabled={editCount === 0 || exporting}>
                  {exporting ? <Loader2 className="animate-spin" /> : <CheckIcon />}
                  Apply {editCount} edit{editCount === 1 ? "" : "s"} &amp; rescore
                </Button>
                {exported && (
                  <Button asChild variant="outline">
                    <a href={docsApi.documentDownloadUrl(exported.download_url)}>
                      <Download /> Download edited {session.filename.split(".").pop()}
                    </a>
                  </Button>
                )}
                <span className="text-xs text-muted-foreground">{INDICATOR_NOTE}</span>
              </div>
              {exported?.warnings.map((w) => (
                <Alert key={w}>
                  <AlertDescription>{w}</AlertDescription>
                </Alert>
              ))}
            </CardContent>
          </Card>

          <div className="grid gap-6 lg:grid-cols-[minmax(0,3fr)_minmax(0,2fr)]">
            <Card>
              <CardHeader>
                <CardTitle className="text-base">{session.filename}</CardTitle>
                <CardDescription className="flex flex-wrap gap-2">
                  <span className="rounded-sm bg-red-200/70 px-1 dark:bg-red-500/30">elevated</span>
                  <span className="rounded-sm bg-amber-200/70 px-1 dark:bg-amber-500/30">uncertain</span>
                  <span className="rounded-sm bg-emerald-200/70 px-1 dark:bg-emerald-500/30">your edit</span>
                  <span>· click any sentence to work on it</span>
                </CardDescription>
              </CardHeader>
              <CardContent>
                <DocumentPane
                  blocks={session.document.blocks}
                  edits={edits}
                  selected={selKey}
                  onSelect={(block, sentence) => setSelected({ block, sentence })}
                />
              </CardContent>
            </Card>

            <div className="lg:sticky lg:top-4 lg:self-start">
              <Card>
                {!selected || !selKey ? (
                  <CardContent className="flex flex-col items-center gap-2 py-10 text-center text-sm text-muted-foreground">
                    <Lightbulb className="size-6" />
                    Click a highlighted sentence to see why it was flagged and how you could rewrite it.
                  </CardContent>
                ) : (
                  <>
                    <CardHeader className="gap-2">
                      <div className="flex items-center gap-2">
                        <CardTitle className="text-base">Sentence</CardTitle>
                        {selected.sentence.band && (
                          <Badge
                            variant={selected.sentence.band === "elevated" ? "destructive" : "outline"}
                          >
                            {selected.sentence.band} · {pct(selected.sentence.score)}
                          </Badge>
                        )}
                      </div>
                      <CardDescription className="text-foreground">{selected.sentence.text}</CardDescription>
                      {selected.sentence.reasons.length > 0 && (
                        <ul className="list-disc space-y-0.5 pl-5 text-xs text-muted-foreground">
                          {selected.sentence.reasons.slice(0, 5).map((r) => (
                            <li key={r}>{r}</li>
                          ))}
                        </ul>
                      )}
                    </CardHeader>
                    <CardContent className="flex flex-col gap-4">
                      <div className="space-y-2">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={loadSuggestions}
                          disabled={suggesting === selKey}
                        >
                          {suggesting === selKey ? <Loader2 className="animate-spin" /> : <Lightbulb />}
                          {suggestions[selKey] ? "Suggest again" : "Suggest alternatives"}
                        </Button>
                        {suggesting === selKey && (
                          <p className="text-xs text-muted-foreground">
                            Free-tier models can take up to a minute.
                          </p>
                        )}
                        {(suggestions[selKey] ?? []).map((sg) => (
                          <div key={sg.text} className="space-y-2 rounded-md border p-3">
                            <div className="flex items-center justify-between gap-2">
                              <span className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                                {APPROACH[sg.approach] ?? sg.approach} · {pct(sg.check.score)}
                              </span>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => setDrafts((prev) => ({ ...prev, [selKey]: sg.text }))}
                              >
                                Use this
                              </Button>
                            </div>
                            <p className="text-sm">{sg.text}</p>
                            <CheckList check={sg.check} />
                          </div>
                        ))}
                        {suggestions[selKey]?.length === 0 && (
                          <p className="text-xs text-muted-foreground">No usable suggestions this time.</p>
                        )}
                      </div>

                      <div className="space-y-2">
                        <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                          Your version
                        </div>
                        <Textarea
                          value={draft}
                          rows={4}
                          onChange={(e) => setDrafts((prev) => ({ ...prev, [selKey]: e.target.value }))}
                        />
                        {liveCheck && (
                          <div className="flex flex-col gap-1.5">
                            <span className="text-xs text-muted-foreground">
                              AI-likeness of your version: {pct(liveCheck.score)} (was{" "}
                              {pct(selected.sentence.score)})
                            </span>
                            <CheckList check={liveCheck} />
                          </div>
                        )}
                        <div className="flex flex-wrap gap-2">
                          <Button
                            size="sm"
                            onClick={() => accept(fromSuggestion ? "suggestion" : "manual")}
                            disabled={!changedDraft}
                          >
                            <CheckIcon />
                            {liveCheck && !liveCheck.passed ? "Accept anyway" : "Accept"}
                          </Button>
                          {edits[selKey] && (
                            <Button size="sm" variant="ghost" onClick={undo}>
                              <RotateCcw /> Revert to original
                            </Button>
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </>
                )}
              </Card>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
