"use client";

import { useEffect, useState } from "react";
import { Flag, Loader2, ShieldCheck, X } from "lucide-react";
import { toast } from "sonner";
import { useAuth } from "@/lib/auth-context";
import { getMyJobReport, reportJob } from "@/lib/api";
import type { JobReportReason } from "@/lib/types";

const REASONS: { key: JobReportReason; label: string; hint: string }[] = [
  { key: "fake_or_scam", label: "Fake job or scam", hint: "The company or role does not seem real" },
  { key: "asks_for_payment", label: "Asks me to pay", hint: "Registration, training or security-deposit fees" },
  { key: "misleading", label: "Misleading details", hint: "Pay, location or role differ from reality" },
  { key: "already_filled", label: "No longer available", hint: "Position is filled or closed" },
  { key: "duplicate_or_spam", label: "Duplicate or spam", hint: "Same posting repeated, or not a job" },
  { key: "offensive", label: "Offensive or discriminatory", hint: "Inappropriate language or unfair criteria" },
  { key: "other", label: "Something else", hint: "Tell us below" },
];

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

type Props = {
  jobId: string;
  jobTitle: string;
  /** Employer who posted it (hidden for them). */
  postedByUserId?: string | null;
  className?: string;
};

/**
 * "Report listing" for platform postings. Reports go to the admin moderation
 * queue; listings reported by several different users are prioritised there.
 */
export default function ReportJobButton({ jobId, jobTitle, postedByUserId, className = "" }: Props) {
  const { user, token } = useAuth();
  const [open, setOpen] = useState(false);
  const [reason, setReason] = useState<JobReportReason | null>(null);
  const [details, setDetails] = useState("");
  const [busy, setBusy] = useState(false);
  const [reported, setReported] = useState(false);

  const eligible = !!token && !!user && UUID_RE.test(jobId) && user.id !== postedByUserId && user.role !== "admin";

  useEffect(() => {
    if (!eligible || !token) return;
    let cancelled = false;
    getMyJobReport(jobId, token)
      .then((r) => !cancelled && setReported(r.reported))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [eligible, jobId, token]);

  if (!eligible) return null;

  if (reported) {
    return (
      <span className={`inline-flex items-center gap-1.5 text-xs text-zinc-500 ${className}`}>
        <ShieldCheck className="h-3.5 w-3.5 text-teal-400" />
        You reported this listing — our team will review it.
      </span>
    );
  }

  const submit = async () => {
    if (!reason || !token) return;
    if (reason === "other" && details.trim().length < 5) {
      toast.error("Please tell us briefly what is wrong.");
      return;
    }
    setBusy(true);
    try {
      await reportJob(jobId, { reason, details: details.trim() || undefined }, token);
      setReported(true);
      setOpen(false);
      toast.success("Thanks — the listing was reported to our moderation team.");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Could not send the report");
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className={`inline-flex items-center gap-1.5 text-xs font-medium text-zinc-500 transition-colors hover:text-rose-300 ${className}`}
      >
        <Flag className="h-3.5 w-3.5" />
        Report listing
      </button>

      {open && (
        <div
          className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
          role="dialog"
          aria-modal="true"
          aria-labelledby="report-job-title"
          onClick={() => !busy && setOpen(false)}
        >
          <div
            className="w-full max-w-md rounded-2xl border border-zinc-800 bg-zinc-950 p-5 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-start justify-between gap-3">
              <div>
                <h2 id="report-job-title" className="flex items-center gap-2 text-base font-semibold text-zinc-100">
                  <Flag className="h-4 w-4 text-rose-400" /> Report this listing
                </h2>
                <p className="mt-0.5 line-clamp-1 text-xs text-zinc-500">{jobTitle}</p>
              </div>
              <button
                type="button"
                aria-label="Close"
                onClick={() => setOpen(false)}
                className="rounded-md p-1 text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <fieldset className="mt-4 space-y-1.5">
              <legend className="sr-only">Reason</legend>
              {REASONS.map((r) => (
                <label
                  key={r.key}
                  className={`flex cursor-pointer items-start gap-2.5 rounded-lg border px-3 py-2 transition-colors ${
                    reason === r.key
                      ? "border-rose-500/40 bg-rose-500/10"
                      : "border-zinc-800 hover:border-zinc-700"
                  }`}
                >
                  <input
                    type="radio"
                    name="report-reason"
                    value={r.key}
                    checked={reason === r.key}
                    onChange={() => setReason(r.key)}
                    className="mt-0.5 accent-rose-500"
                  />
                  <span>
                    <span className="block text-sm font-medium text-zinc-200">{r.label}</span>
                    <span className="block text-[11px] text-zinc-500">{r.hint}</span>
                  </span>
                </label>
              ))}
            </fieldset>

            <textarea
              value={details}
              onChange={(e) => setDetails(e.target.value.slice(0, 2000))}
              rows={3}
              placeholder="Anything that helps us check (optional)"
              className="mt-3 w-full rounded-lg border border-zinc-800 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-rose-500/50 focus:outline-none"
            />
            <p className="mt-1 text-[11px] text-zinc-600">
              Never share passwords or pay anyone to get a job. Reports are anonymous to the employer.
            </p>

            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setOpen(false)}
                disabled={busy}
                className="rounded-lg border border-zinc-800 px-3 py-2 text-xs font-medium text-zinc-300 hover:bg-zinc-900"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={submit}
                disabled={!reason || busy}
                className="inline-flex items-center gap-1.5 rounded-lg bg-rose-500 px-3 py-2 text-xs font-semibold text-white hover:bg-rose-400 disabled:opacity-40"
              >
                {busy && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
                Submit report
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
