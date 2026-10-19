"use client";

import { FileUp } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { ModeInfo, ScoreSummary, VoiceInfo } from "@/lib/docs-api";
import { cn } from "@/lib/utils";

export const BAND_LABEL: Record<ScoreSummary["document_band"], string> = {
  insufficient_text: "Too short to score (< 300 words)",
  "*": "Low (under 20% flagged)",
  low: "Low",
  uncertain: "Uncertain",
  elevated: "Elevated",
};

export const pct = (v: number | null | undefined) =>
  v == null ? "–" : `${Math.round(v * 100)}%`;

export function scoreTone(v: number | null | undefined) {
  if (v == null) return "bg-muted-foreground/40";
  if (v >= 0.65) return "bg-red-500";
  if (v >= 0.35) return "bg-amber-500";
  return "bg-emerald-500";
}

export function Meter({ value }: { value: number | null | undefined }) {
  return (
    <div className="h-2 w-full overflow-hidden rounded-full bg-muted">
      <div
        className={cn("h-full rounded-full transition-all", scoreTone(value))}
        style={{ width: `${Math.round((value ?? 0) * 100)}%` }}
      />
    </div>
  );
}

export function ScoreTile({ title, summary }: { title: string; summary: ScoreSummary }) {
  return (
    <div className="flex flex-1 flex-col gap-3 rounded-lg border p-4">
      <div className="text-sm text-muted-foreground">{title}</div>
      <div className="text-4xl font-semibold tabular-nums">{pct(summary.score)}</div>
      <Meter value={summary.score} />
      <div className="text-sm">{BAND_LABEL[summary.document_band]}</div>
      <div className="flex flex-wrap gap-1.5">
        <Badge variant="destructive">{summary.bands.elevated ?? 0} elevated</Badge>
        <Badge variant="outline">{summary.bands.uncertain ?? 0} uncertain</Badge>
        <Badge variant="secondary">{summary.bands.low ?? 0} low</Badge>
      </div>
    </div>
  );
}

export const INDICATOR_NOTE =
  "Writing-pattern indicator: it estimates how generic or AI-like the style reads. It is not " +
  "evidence of who wrote the text, and other detectors will give different numbers.";

export function FilePicker({ file, onChange }: { file: File | null; onChange: (f: File | null) => void }) {
  return (
    <div className="space-y-1.5">
      <Label htmlFor="file">File</Label>
      <label
        htmlFor="file"
        className="flex h-9 cursor-pointer items-center gap-2 rounded-md border px-3 text-sm hover:bg-muted"
      >
        <FileUp className="size-4 text-muted-foreground" />
        <span className="truncate">{file?.name ?? "Choose a .docx, .txt or .md file…"}</span>
      </label>
      <input
        id="file"
        type="file"
        accept=".docx,.txt,.md"
        className="sr-only"
        onChange={(e) => onChange(e.target.files?.[0] ?? null)}
      />
    </div>
  );
}

export function ModeSelect({
  modes,
  value,
  onChange,
}: {
  modes: ModeInfo[];
  value: string;
  onChange: (v: string) => void;
}) {
  return (
    <div className="space-y-1.5">
      <Label>Writing mode</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {(["student", "professional"] as const).map((audience) => (
            <SelectGroup key={audience}>
              <SelectLabel className="capitalize">{audience}</SelectLabel>
              {modes
                .filter((m) => m.audience === audience)
                .map((m) => (
                  <SelectItem key={m.id} value={m.id}>
                    {m.label}
                    {m.phase > 1 ? " (beta)" : ""}
                  </SelectItem>
                ))}
            </SelectGroup>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

export const NO_VOICE = "none";

export function VoiceSelect({
  voices,
  value,
  onChange,
}: {
  voices: VoiceInfo[];
  value: string;
  onChange: (v: string) => void;
}) {
  const current = voices.find((v) => v.id === value);
  return (
    <div className="space-y-1.5">
      <Label>Voice</Label>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="w-full" title={current?.description}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={NO_VOICE}>Keep the document&apos;s own voice</SelectItem>
          {voices.map((v) => (
            <SelectItem key={v.id} value={v.id}>
              {v.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}

export function VoiceNote({ voice }: { voice: VoiceInfo | undefined }) {
  if (!voice) return null;
  return (
    <p className="text-xs text-muted-foreground">
      <span className="font-medium text-foreground">{voice.label}:</span> {voice.description}{" "}
      Learned from human writing: {voice.source.split(":")[0]} ({voice.license}).
    </p>
  );
}
