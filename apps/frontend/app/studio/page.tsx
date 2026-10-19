"use client";

import { useEffect, useState } from "react";
import { GraduationCap, Sparkles } from "lucide-react";

import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/api";
import { docsApi, type ModeInfo, type VoiceInfo } from "@/lib/docs-api";

import { AutoRewrite } from "./_components/auto-rewrite";
import { Coach } from "./_components/coach";

type View = "coach" | "auto";

export default function StudioPage() {
  const [view, setView] = useState<View>("coach");
  const [modes, setModes] = useState<ModeInfo[]>([]);
  const [voices, setVoices] = useState<VoiceInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([docsApi.modes(), docsApi.voices()])
      .then(([m, v]) => {
        setModes(m);
        setVoices(v);
      })
      .catch((err) => setError(err instanceof ApiError ? err.message : "Failed to load"));
  }, []);

  return (
    <main className="mx-auto flex w-full max-w-7xl flex-1 flex-col gap-6 px-4 py-8">
      <header className="flex flex-wrap items-end justify-between gap-4">
        <div className="space-y-1">
          <h1 className="flex items-center gap-2 text-2xl font-semibold">
            <Sparkles className="size-6" /> Writing Studio
          </h1>
          <p className="max-w-2xl text-sm text-muted-foreground">
            Find sentences that read as generic or AI-like, see why, and improve them without
            changing what they say. Citations, numbers and formatting are protected.
          </p>
        </div>
        <div className="flex rounded-lg border p-1" role="tablist">
          <Button
            role="tab"
            aria-selected={view === "coach"}
            size="sm"
            variant={view === "coach" ? "default" : "ghost"}
            onClick={() => setView("coach")}
          >
            <GraduationCap /> Coach
          </Button>
          <Button
            role="tab"
            aria-selected={view === "auto"}
            size="sm"
            variant={view === "auto" ? "default" : "ghost"}
            onClick={() => setView("auto")}
          >
            <Sparkles /> Auto rewrite
          </Button>
        </div>
      </header>

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {view === "coach" ? (
        <Coach modes={modes} voices={voices} />
      ) : (
        <AutoRewrite modes={modes} voices={voices} />
      )}
    </main>
  );
}
