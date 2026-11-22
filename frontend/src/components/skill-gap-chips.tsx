"use client";

import { AlertCircle, ArrowLeftRight, BookOpen, CheckCircle2, Circle } from "lucide-react";
import type { SkillGapSummary } from "@/lib/types";

type Props = {
  gap?: SkillGapSummary | null;
  /** Card view: counts + the first few missing skills. Full view: every group. */
  compact?: boolean;
};

const GROUPS: {
  key: keyof Pick<SkillGapSummary, "matched" | "learning" | "transferable" | "missing_required" | "missing_preferred">;
  label: string;
  hint: string;
  icon: typeof CheckCircle2;
  chip: string;
  head: string;
}[] = [
  {
    key: "matched",
    label: "You have",
    hint: "Skills on your profile that this job asks for",
    icon: CheckCircle2,
    chip: "bg-emerald-500/10 border-emerald-500/20 text-emerald-300",
    head: "text-emerald-400",
  },
  {
    key: "learning",
    label: "Learning",
    hint: "Skills you marked as currently learning",
    icon: BookOpen,
    chip: "bg-sky-500/10 border-sky-500/20 text-sky-300",
    head: "text-sky-400",
  },
  {
    key: "transferable",
    label: "Transferable",
    hint: "Missing, but closely related to skills you already have",
    icon: ArrowLeftRight,
    chip: "bg-violet-500/10 border-violet-500/20 text-violet-300",
    head: "text-violet-400",
  },
  {
    key: "missing_required",
    label: "Missing (required)",
    hint: "Required by the job and not on your profile",
    icon: AlertCircle,
    chip: "bg-amber-500/10 border-amber-500/20 text-amber-300",
    head: "text-amber-400",
  },
  {
    key: "missing_preferred",
    label: "Nice to have",
    hint: "Optional skills mentioned by the job",
    icon: Circle,
    chip: "bg-zinc-800 border-zinc-700 text-zinc-300",
    head: "text-zinc-400",
  },
];

/** Seeker-vs-job skill breakdown, shown on every search/AI-search/recommended job. */
export default function SkillGapChips({ gap, compact = false }: Props) {
  if (!gap) return null;
  const total = GROUPS.reduce((n, g) => n + gap[g.key].length, 0);

  if (total === 0) {
    return (
      <p className="text-xs text-zinc-500">
        This posting doesn&apos;t list specific skills — open it to read the full requirements.
      </p>
    );
  }

  const coverage =
    gap.required_total > 0 ? `You cover ${gap.required_covered} of ${gap.required_total} required skills` : null;

  if (compact) {
    const missing = gap.missing_required;
    return (
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 text-xs">
        <span className="font-semibold uppercase tracking-wider text-[10px] text-zinc-500">Your skill gap</span>
        {gap.matched.length > 0 && (
          <span className="inline-flex items-center gap-1 text-emerald-400" title={gap.matched.join(", ")}>
            <CheckCircle2 className="h-3.5 w-3.5" /> {gap.matched.length} you have
          </span>
        )}
        {gap.learning.length > 0 && (
          <span className="inline-flex items-center gap-1 text-sky-400" title={gap.learning.join(", ")}>
            <BookOpen className="h-3.5 w-3.5" /> {gap.learning.length} learning
          </span>
        )}
        {gap.transferable.length > 0 && (
          <span className="inline-flex items-center gap-1 text-violet-400" title={gap.transferable.join(", ")}>
            <ArrowLeftRight className="h-3.5 w-3.5" /> {gap.transferable.length} transferable
          </span>
        )}
        {missing.length > 0 ? (
          <span className="inline-flex flex-wrap items-center gap-1 text-amber-400">
            <AlertCircle className="h-3.5 w-3.5" /> Missing:
            {missing.slice(0, 3).map((s) => (
              <span key={s} className="rounded-md border border-amber-500/20 bg-amber-500/10 px-1.5 py-0.5 text-amber-300">
                {s}
              </span>
            ))}
            {missing.length > 3 && <span className="text-zinc-500">+{missing.length - 3} more</span>}
          </span>
        ) : (
          gap.required_total > 0 && <span className="text-emerald-400">No required skills missing</span>
        )}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {coverage && <p className="text-xs text-zinc-400">{coverage}.</p>}
      {GROUPS.map((g) =>
        gap[g.key].length > 0 ? (
          <div key={g.key} className="flex flex-wrap items-center gap-1.5">
            <span className={`mr-1 flex items-center gap-1 text-xs font-medium ${g.head}`} title={g.hint}>
              <g.icon className="h-3.5 w-3.5" /> {g.label}:
            </span>
            {gap[g.key].map((s) => (
              <span key={s} className={`rounded-md border px-2 py-0.5 text-xs font-medium ${g.chip}`}>
                {s}
              </span>
            ))}
          </div>
        ) : null,
      )}
    </div>
  );
}
