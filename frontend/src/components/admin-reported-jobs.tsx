"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { toast } from "sonner";
import {
  CheckCircle2,
  ChevronDown,
  ExternalLink,
  Flag,
  Loader2,
  RotateCcw,
  ShieldAlert,
  Trash2,
  Users,
} from "lucide-react";
import { actOnReportedJob, getReportedJobs } from "@/lib/api";
import type { AdminReportedJob, AdminReportedJobsResponse } from "@/lib/types";

type View = "multi" | "open" | "history";

/** Backend MULTI_REPORT_THRESHOLD (the response echoes the live value). */
const DEFAULT_MULTI_THRESHOLD = 2;

function ago(iso: string | null): string {
  if (!iso) return "—";
  const mins = Math.max(0, Math.round((Date.now() - new Date(iso).getTime()) / 60000));
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.round(mins / 60);
  if (hours < 48) return `${hours}h ago`;
  return `${Math.round(hours / 24)}d ago`;
}

const SEVERITY_TONE: Record<string, string> = {
  high: "bg-rose-500/10 text-rose-300 ring-rose-500/30",
  medium: "bg-amber-500/10 text-amber-300 ring-amber-500/30",
  low: "bg-zinc-500/10 text-zinc-300 ring-zinc-500/30",
};

/**
 * Admin queue of listings reported by job seekers. Listings reported by
 * several different users come first; the admin removes (archive + lock,
 * reports resolved), dismisses, or restores a removed listing.
 */
export default function AdminReportedJobs({ token, onChanged }: { token: string; onChanged?: () => void }) {
  const [view, setView] = useState<View>("multi");
  const [data, setData] = useState<AdminReportedJobsResponse | null>(null);
  // which (view, reload) the shown data belongs to — loading is derived from it
  const [loadedKey, setLoadedKey] = useState<string | null>(null);
  const [reload, setReload] = useState(0);
  const [busy, setBusy] = useState<string | null>(null);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [notes, setNotes] = useState<Record<string, string>>({});

  const key = `${view}:${reload}`;
  const loading = loadedKey !== key;

  useEffect(() => {
    let cancelled = false;
    getReportedJobs(token, {
      min_reporters: view === "multi" ? DEFAULT_MULTI_THRESHOLD : 1,
      include_closed: view === "history",
      limit: 100,
    })
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((err) => {
        if (!cancelled) toast.error(err instanceof Error ? err.message : "Failed to load reported listings");
      })
      .finally(() => {
        if (!cancelled) setLoadedKey(`${view}:${reload}`);
      });
    return () => {
      cancelled = true;
    };
  }, [token, view, reload]);

  const load = useCallback(() => setReload((n) => n + 1), []);

  const act = async (job: AdminReportedJob, action: "remove" | "dismiss" | "restore") => {
    const verb = { remove: "Remove", dismiss: "Dismiss reports on", restore: "Restore" }[action];
    if (
      action !== "dismiss" &&
      !window.confirm(
        action === "remove"
          ? `Remove "${job.title}"? It will be hidden from job seekers and the employer cannot republish it.`
          : `Restore "${job.title}" and publish it again?`,
      )
    ) {
      return;
    }
    setBusy(job.job_id);
    try {
      await actOnReportedJob(job.job_id, action, token, notes[job.job_id]);
      toast.success(`${verb} "${job.title}" — done.`);
      load();
      onChanged?.();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Action failed");
    } finally {
      setBusy(null);
    }
  };

  const items = data?.items ?? [];
  const threshold = data?.multi_report_threshold ?? 2;

  return (
    <section className="rounded-2xl border border-rose-500/20 bg-zinc-900/40 p-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-zinc-300">
            <Users className="h-4 w-4 text-rose-400" />
            Reported by users
          </h2>
          <p className="mt-1 max-w-2xl text-xs text-zinc-500">
            Listings job seekers reported as fake, scam or misleading. Counts are distinct people (one open
            report per user). Listings reported by {threshold}+ users are prioritised — remove them, or dismiss
            the reports if the listing is genuine.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-1.5">
          {(
            [
              ["multi", `${threshold}+ reporters`],
              ["open", "All open"],
              ["history", "History"],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              type="button"
              onClick={() => setView(key)}
              className={`rounded-lg border px-2.5 py-1 text-[11px] font-semibold ${
                view === key
                  ? "border-rose-500/40 bg-rose-500/10 text-rose-300"
                  : "border-zinc-800 text-zinc-400 hover:text-zinc-200"
              }`}
            >
              {label}
              {key === "multi" && data && view === "multi" && <span className="ml-1 text-zinc-500">{data.total}</span>}
            </button>
          ))}
        </div>
      </div>

      <div className="mt-4 space-y-2">
        {loading && !data ? (
          <div className="flex items-center justify-center py-10 text-zinc-500">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
        ) : items.length === 0 ? (
          <div className="rounded-xl border border-dashed border-zinc-800 p-8 text-center text-xs text-zinc-500">
            {view === "multi"
              ? `No listing has been reported by ${threshold} or more users.`
              : view === "open"
                ? "No open user reports."
                : "No reported listings yet."}
          </div>
        ) : (
          items.map((job) => {
            const isOpen = expanded === job.job_id;
            const removed = !!job.removed_at;
            return (
              <div
                key={job.job_id}
                className={`rounded-xl border p-3 text-xs ${
                  job.multi_reported ? "border-rose-500/30 bg-rose-500/5" : "border-zinc-800/70 bg-zinc-950/50"
                }`}
              >
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[10px] font-semibold ring-1 ring-inset ${
                      job.multi_reported ? SEVERITY_TONE.high : SEVERITY_TONE.medium
                    }`}
                  >
                    <Flag className="h-3 w-3" />
                    {job.reporters} {job.reporters === 1 ? "user" : "users"}
                  </span>
                  {job.exists ? (
                    <Link
                      href={`/jobs/${job.job_id}`}
                      className="inline-flex items-center gap-1 font-semibold text-zinc-100 hover:text-teal-300"
                    >
                      {job.title}
                      <ExternalLink className="h-3 w-3 opacity-60" />
                    </Link>
                  ) : (
                    <span className="font-semibold text-zinc-400 line-through">{job.title}</span>
                  )}
                  {removed && (
                    <span className="rounded bg-rose-500/15 px-1.5 py-0.5 text-[10px] font-semibold text-rose-300">
                      removed
                    </span>
                  )}
                  {job.risk_score > 0 && (
                    <span
                      className="inline-flex items-center gap-1 text-[10px] text-amber-300"
                      title={job.risk_reasons.join(" · ")}
                    >
                      <ShieldAlert className="h-3 w-3" /> risk {job.risk_score}
                    </span>
                  )}
                  <span className="ml-auto text-[10px] text-zinc-500">
                    {job.status ?? "deleted"} · {job.applicants} applicants · last report {ago(job.last_reported_at)}
                  </span>
                </div>

                <p className="mt-1 text-[11px] text-zinc-500">
                  {job.company ? `${job.company.name}${job.company.is_verified ? " · verified" : " · unverified"}` : "No company"}
                  {job.owner ? ` · posted by ${job.owner.email}${job.owner.is_active ? "" : " (deactivated)"}` : ""}
                  {job.location ? ` · ${job.location}` : ""}
                </p>

                <ul className="mt-1.5 flex flex-wrap gap-1.5">
                  {job.reasons.map((r) => (
                    <li key={r.reason} className="rounded bg-zinc-800/70 px-1.5 py-0.5 text-[10px] text-zinc-300">
                      {r.reason} × {r.count}
                    </li>
                  ))}
                </ul>
                {removed && job.removal_note && (
                  <p className="mt-1.5 text-[11px] text-rose-300/80">Removal note: {job.removal_note}</p>
                )}

                <div className="mt-2 flex flex-wrap items-center gap-1.5">
                  <button
                    type="button"
                    onClick={() => setExpanded(isOpen ? null : job.job_id)}
                    className="inline-flex items-center gap-1 rounded border border-zinc-800 px-2 py-1 text-[10px] font-semibold text-zinc-300 hover:text-zinc-100"
                  >
                    <ChevronDown className={`h-3 w-3 transition-transform ${isOpen ? "rotate-180" : ""}`} />
                    {isOpen ? "Hide" : "Show"} {job.total_reports} report{job.total_reports === 1 ? "" : "s"}
                  </button>
                  {job.exists && !removed && job.open_reports > 0 && (
                    <>
                      <input
                        value={notes[job.job_id] ?? ""}
                        onChange={(e) => setNotes((n) => ({ ...n, [job.job_id]: e.target.value.slice(0, 500) }))}
                        placeholder="Note for the record / employer (optional)"
                        className="min-w-[12rem] flex-1 rounded border border-zinc-800 bg-zinc-950 px-2 py-1 text-[11px] text-zinc-200 placeholder:text-zinc-600 focus:border-rose-500/40 focus:outline-none"
                      />
                      <button
                        type="button"
                        disabled={busy === job.job_id}
                        onClick={() => act(job, "remove")}
                        className="inline-flex items-center gap-1 rounded bg-rose-500/90 px-2 py-1 text-[10px] font-semibold text-white hover:bg-rose-500 disabled:opacity-40"
                      >
                        {busy === job.job_id ? <Loader2 className="h-3 w-3 animate-spin" /> : <Trash2 className="h-3 w-3" />}
                        Remove listing
                      </button>
                      <button
                        type="button"
                        disabled={busy === job.job_id}
                        onClick={() => act(job, "dismiss")}
                        className="inline-flex items-center gap-1 rounded border border-zinc-700 px-2 py-1 text-[10px] font-semibold text-zinc-300 hover:border-emerald-500/40 hover:text-emerald-300 disabled:opacity-40"
                      >
                        <CheckCircle2 className="h-3 w-3" />
                        Dismiss reports
                      </button>
                    </>
                  )}
                  {job.exists && removed && (
                    <button
                      type="button"
                      disabled={busy === job.job_id}
                      onClick={() => act(job, "restore")}
                      className="inline-flex items-center gap-1 rounded border border-zinc-700 px-2 py-1 text-[10px] font-semibold text-zinc-300 hover:border-teal-500/40 hover:text-teal-300 disabled:opacity-40"
                    >
                      <RotateCcw className="h-3 w-3" />
                      Restore & publish
                    </button>
                  )}
                </div>

                {isOpen && (
                  <ul className="mt-2 space-y-1.5 border-t border-zinc-800/60 pt-2">
                    {job.reports.map((r) => (
                      <li key={r.id} className="rounded-lg bg-zinc-900/60 px-2.5 py-1.5">
                        <div className="flex flex-wrap items-center gap-2 text-[10px]">
                          <span className={`rounded-full px-1.5 py-0.5 font-semibold ring-1 ring-inset ${SEVERITY_TONE[r.severity]}`}>
                            {r.severity}
                          </span>
                          <span className="font-semibold text-zinc-200">{r.reason}</span>
                          <span className="text-zinc-500">{r.created_by ?? "deleted user"}</span>
                          <span className="text-zinc-600">{ago(r.created_at)}</span>
                          <span className="ml-auto capitalize text-zinc-500">{r.status}</span>
                        </div>
                        {r.details && <p className="mt-1 whitespace-pre-wrap text-[11px] text-zinc-400">{r.details}</p>}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            );
          })
        )}
      </div>
    </section>
  );
}
