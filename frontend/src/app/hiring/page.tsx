"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useAuth } from "@/lib/auth-context";
import { getEmployerJobs, getEmployerOverview, getMyCompany } from "@/lib/api";
import { CompanyDetail, EmployerOverview, JobDetail } from "@/lib/types";
import { formatSalary } from "@/lib/format";
import { EmployerLoading, useEmployerAuth } from "@/lib/use-employer-auth";
import { StatusPill } from "@/components/status-pill";
import {
  AlertTriangle,
  ArrowRight,
  Award,
  Building2,
  Clock,
  FileText,
  Flag,
  MapPin,
  PlusCircle,
  ShieldAlert,
  Users,
} from "lucide-react";

const FUNNEL: { key: "applied" | "shortlisted" | "interview" | "offer" | "hired"; label: string; bar: string }[] = [
  { key: "applied", label: "Applied", bar: "bg-zinc-500" },
  { key: "shortlisted", label: "Shortlisted", bar: "bg-amber-400" },
  { key: "interview", label: "Interview", bar: "bg-indigo-400" },
  { key: "offer", label: "Offer", bar: "bg-emerald-400" },
  { key: "hired", label: "Hired", bar: "bg-teal-400" },
];

/** Applied -> hired funnel across every posting ("reached at least this stage"). */
function HiringFunnel({ overview }: { overview: EmployerOverview }) {
  const total = overview.total_applicants;
  const values: Record<string, number> = {
    applied: total,
    shortlisted: Math.round((overview.conversion.shortlisted / 100) * total),
    interview: Math.round((overview.conversion.interview / 100) * total),
    offer: Math.round((overview.conversion.offer / 100) * total),
    hired: overview.counts.hired ?? 0,
  };
  return (
    <div className="rounded-xl border border-zinc-800/80 bg-zinc-900/40 p-5">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-semibold text-zinc-200">Hiring funnel · all postings</h2>
        <span className="text-xs text-zinc-500">
          {overview.new_7d} new in the last 7 days · {overview.counts.rejected} rejected
        </span>
      </div>
      {total === 0 ? (
        <p className="mt-4 text-xs text-zinc-500">No applications yet — they will show up here as candidates apply.</p>
      ) : (
        <div className="mt-4 space-y-2.5">
          {FUNNEL.map((stage) => {
            const n = values[stage.key];
            const pct = total ? Math.round((100 * n) / total) : 0;
            return (
              <div key={stage.key} className="grid grid-cols-[6.5rem_1fr_4.5rem] items-center gap-3 text-xs">
                <span className="text-zinc-400">{stage.label}</span>
                <div className="h-2.5 overflow-hidden rounded-full bg-zinc-800">
                  <div className={`h-full rounded-full ${stage.bar}`} style={{ width: `${Math.max(pct, n > 0 ? 3 : 0)}%` }} />
                </div>
                <span className="text-right font-semibold text-zinc-200">
                  {n} <span className="font-normal text-zinc-500">({pct}%)</span>
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function KpiCard({
  label,
  value,
  hint,
  icon,
  tone = "teal",
}: {
  label: string;
  value: string | number;
  hint?: string;
  icon: React.ReactNode;
  tone?: "teal" | "amber" | "zinc";
}) {
  const toneCls =
    tone === "teal"
      ? "text-teal-300"
      : tone === "amber"
        ? "text-amber-300"
        : "text-zinc-300";
  return (
    <div className="rounded-xl border border-zinc-800/80 bg-zinc-900/40 p-5">
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold uppercase tracking-wider text-zinc-500">
          {label}
        </span>
        <span className={toneCls}>{icon}</span>
      </div>
      <div className="mt-3 text-3xl font-bold text-zinc-100">{value}</div>
      {hint && <div className="mt-1 text-xs text-zinc-500">{hint}</div>}
    </div>
  );
}

export default function HiringOverviewPage() {
  const { token, ready, isLoading } = useEmployerAuth();
  const { user } = useAuth();
  const [jobs, setJobs] = useState<JobDetail[]>([]);
  const [company, setCompany] = useState<CompanyDetail | null>(null);
  const [overview, setOverview] = useState<EmployerOverview | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!ready || !token) return;
    let cancelled = false;
    Promise.all([
      getEmployerJobs(token),
      getMyCompany(token).catch(() => null), // employer may have no company yet
      getEmployerOverview(token).catch(() => null),
    ])
      .then(([jobList, comp, ov]) => {
        if (cancelled) return;
        setJobs(jobList);
        setCompany(comp);
        setOverview(ov);
      })
      .catch((err) => !cancelled && setError(err instanceof Error ? err.message : "Failed to load"))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [ready, token]);

  if (isLoading || !ready) return <EmployerLoading />;

  const published = jobs.filter((j) => j.status === "published");
  const drafts = jobs.filter((j) => j.status === "draft");
  const closed = jobs.filter((j) => j.status === "closed" || j.status === "archived");
  const totalApplicants = jobs.reduce((sum, j) => sum + (j.applicant_count ?? 0), 0);
  const noApplicants = published.filter((j) => (j.applicant_count ?? 0) === 0);
  const removed = jobs.filter((j) => !!j.moderation_removed_at);
  const reported = jobs.filter((j) => !j.moderation_removed_at && (j.open_reports ?? 0) > 0);

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-teal-400">
            <Building2 className="h-4 w-4" />
            Hiring workspace
          </div>
          <h1 className="mt-1 text-3xl font-bold tracking-tight text-zinc-100">
            {company?.name ?? "Your company"}
          </h1>
          <p className="mt-1 text-sm text-zinc-400">
            Welcome back{user?.profile?.first_name ? `, ${user.profile.first_name}` : ""} —
            here&apos;s how your postings are doing.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Link
            href="/hiring/postings"
            className="inline-flex items-center gap-2 rounded-xl border border-zinc-700 bg-zinc-900/60 px-4 py-2.5 text-sm font-semibold text-zinc-200 hover:bg-zinc-800 transition-colors"
          >
            <FileText className="h-4 w-4" />
            Manage postings
          </Link>
          {user?.role !== "admin" && (
            <Link
              href="/jobs/new"
              className="inline-flex items-center gap-2 rounded-xl bg-teal-500 px-4 py-2.5 text-sm font-semibold text-zinc-950 shadow-lg shadow-teal-500/25 hover:bg-teal-400 active:scale-[0.98] transition-all"
            >
              <PlusCircle className="h-4 w-4" />
              Post a job
            </Link>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-6 rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
          {error}
        </div>
      )}

      {/* KPI grid */}
      <div className="mt-8 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <KpiCard
          label="Open postings"
          value={loading ? "…" : published.length}
          hint="Visible to job seekers"
          icon={<FileText className="h-4 w-4" />}
        />
        <KpiCard
          label="Applicants"
          value={loading ? "…" : overview?.total_applicants ?? totalApplicants}
          hint={overview ? `${overview.new_7d} new this week` : "Across all postings"}
          icon={<Users className="h-4 w-4" />}
          tone="zinc"
        />
        <KpiCard
          label="Awaiting review"
          value={loading ? "…" : overview?.waiting_review ?? 0}
          hint="Applications you have not opened yet"
          icon={<Clock className="h-4 w-4" />}
          tone="amber"
        />
        <KpiCard
          label="Hired"
          value={loading ? "…" : overview?.counts.hired ?? 0}
          hint={`${drafts.length} drafts · ${closed.length} closed / archived`}
          icon={<Award className="h-4 w-4" />}
          tone="teal"
        />
      </div>

      {/* No company profile yet */}
      {!loading && company === null && (
        <div className="mt-6 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3">
          <div className="flex items-center gap-2 text-sm text-amber-200">
            <AlertTriangle className="h-4 w-4 text-amber-400" />
            Your account has no company profile yet — add one so your postings show a company name.
          </div>
          <Link
            href="/hiring/company"
            className="text-sm font-semibold text-amber-300 underline underline-offset-4 hover:text-amber-200"
          >
            Set up company →
          </Link>
        </div>
      )}

      {/* Moderation: removed or reported postings */}
      {!loading && removed.length > 0 && (
        <div className="mt-6 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-rose-200">
            <ShieldAlert className="h-4 w-4 text-rose-400" />
            {removed.length} posting{removed.length > 1 ? "s were" : " was"} removed by our moderation team
          </div>
          <ul className="mt-2 space-y-1 text-xs text-rose-200/80">
            {removed.map((j) => (
              <li key={j.id}>
                <span className="font-semibold">{j.title}</span>
                {j.moderation_note ? ` — ${j.moderation_note}` : ""}
              </li>
            ))}
          </ul>
          <p className="mt-2 text-[11px] text-rose-200/60">
            Removed postings are hidden from job seekers and cannot be republished. Contact support if
            you believe this is a mistake.
          </p>
        </div>
      )}
      {!loading && reported.length > 0 && (
        <div className="mt-4 rounded-xl border border-amber-500/30 bg-amber-500/10 p-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-amber-200">
            <Flag className="h-4 w-4 text-amber-400" />
            {reported.length} posting{reported.length > 1 ? "s have" : " has"} been reported by job seekers
          </div>
          <p className="mt-1 text-xs text-amber-200/70">
            Our team is reviewing: {reported.map((j) => j.title).join(", ")}. Make sure pay, location and
            requirements are accurate, and never ask candidates for fees.
          </p>
        </div>
      )}

      {/* Funnel + who is waiting on you */}
      {!loading && overview && (
        <div className="mt-6 grid gap-4 lg:grid-cols-5">
          <div className="lg:col-span-3">
            <HiringFunnel overview={overview} />
          </div>
          <div className="rounded-xl border border-zinc-800/80 bg-zinc-900/40 p-5 lg:col-span-2">
            <div className="flex items-center justify-between">
              <h2 className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
                <Clock className="h-4 w-4 text-amber-400" /> Waiting for your review
              </h2>
              <span className="text-xs text-zinc-500">{overview.waiting_review} total</span>
            </div>
            {overview.needs_review.length === 0 ? (
              <p className="mt-4 text-xs text-zinc-500">You are all caught up.</p>
            ) : (
              <ul className="mt-3 divide-y divide-zinc-800/60">
                {overview.needs_review.map((a) => (
                  <li key={a.application_id}>
                    <Link
                      href={`/hiring/postings/${a.job_id}/applicants`}
                      className="flex items-center justify-between gap-3 py-2 text-xs hover:text-teal-300"
                    >
                      <span className="min-w-0">
                        <span className="block truncate font-semibold text-zinc-100">{a.candidate_name}</span>
                        <span className="block truncate text-zinc-500">
                          {a.headline ? `${a.headline} · ` : ""}
                          {a.job_title}
                        </span>
                      </span>
                      <span
                        className={`shrink-0 rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                          a.days_waiting >= 7 ? "bg-rose-500/10 text-rose-300" : "bg-zinc-800 text-zinc-400"
                        }`}
                      >
                        {a.days_waiting === 0 ? "today" : `${a.days_waiting}d`}
                      </span>
                    </Link>
                  </li>
                ))}
              </ul>
            )}
            {overview.recent_hires.length > 0 && (
              <div className="mt-4 border-t border-zinc-800/60 pt-3">
                <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-teal-400">
                  <Award className="h-3.5 w-3.5" /> Recent hires
                </p>
                <ul className="mt-1.5 space-y-1 text-xs text-zinc-300">
                  {overview.recent_hires.map((h) => (
                    <li key={h.application_id} className="truncate">
                      {h.candidate_name} <span className="text-zinc-500">· {h.job_title}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Postings needing attention */}
      {!loading && noApplicants.length > 0 && (
        <div className="mt-6 rounded-xl border border-zinc-800/80 bg-zinc-900/40 p-4">
          <div className="flex items-center gap-2 text-sm font-semibold text-zinc-200">
            <AlertTriangle className="h-4 w-4 text-amber-400" />
            {noApplicants.length} published posting
            {noApplicants.length > 1 ? "s have" : " has"} no applicants yet
          </div>
          <p className="mt-1 text-xs text-zinc-500">
            Open a posting&apos;s <span className="text-zinc-300">Talent</span> tab to see matching candidates, or{" "}
            <span className="text-zinc-300">What-if</span> to see how relaxing a requirement widens your reach:{" "}
            {noApplicants.slice(0, 3).map((j, i) => (
              <span key={j.id}>
                {i > 0 && ", "}
                <Link href={`/hiring/postings/${j.id}/talent`} className="text-teal-400 hover:text-teal-300">
                  {j.title}
                </Link>
              </span>
            ))}
            {noApplicants.length > 3 ? "…" : ""}
          </p>
        </div>
      )}

      {/* Recent postings */}
      <div className="mt-10">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-zinc-100">Your postings</h2>
          {jobs.length > 0 && (
            <Link
              href="/hiring/postings"
              className="inline-flex items-center gap-1 text-sm font-medium text-teal-400 hover:text-teal-300"
            >
              View all
              <ArrowRight className="h-4 w-4" />
            </Link>
          )}
        </div>

        {loading ? (
          <div className="mt-4 h-40 animate-pulse rounded-xl bg-zinc-900/50" />
        ) : jobs.length === 0 ? (
          <div className="mt-4 rounded-xl border border-dashed border-zinc-800 bg-zinc-900/30 p-10 text-center">
            <FileText className="mx-auto h-8 w-8 text-zinc-600" />
            <p className="mt-3 text-sm font-medium text-zinc-300">No postings yet</p>
            <p className="mt-1 text-xs text-zinc-500">
              Publish your first role to start receiving applicants.
            </p>
            {user?.role !== "admin" && (
              <Link
                href="/jobs/new"
                className="mt-4 inline-flex items-center gap-2 rounded-lg bg-teal-500 px-4 py-2 text-xs font-semibold text-zinc-950 hover:bg-teal-400"
              >
                <PlusCircle className="h-3.5 w-3.5" />
                Post your first job
              </Link>
            )}
          </div>
        ) : (
          <div className="mt-4 divide-y divide-zinc-800/60 overflow-hidden rounded-xl border border-zinc-800/80 bg-zinc-900/40">
            {jobs.slice(0, 6).map((job) => (
              <Link
                key={job.id}
                href={`/hiring/postings/${job.id}`}
                className="flex items-center justify-between gap-4 px-4 py-3.5 transition-colors hover:bg-zinc-800/40"
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="truncate text-sm font-semibold text-zinc-100">
                      {job.title}
                    </span>
                    <StatusPill status={job.status} />
                  </div>
                  <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-zinc-500">
                    <span className="inline-flex items-center gap-1">
                      <MapPin className="h-3 w-3" />
                      {job.location}
                      {job.is_remote && " · Remote"}
                    </span>
                    {job.salary_min || job.salary_max ? (
                      <span>
                        {formatSalary(job.salary_min, job.salary_max, job.salary_currency, job.location)}
                      </span>
                    ) : null}
                    <span>
                      Posted {new Date(job.created_at).toLocaleDateString()}
                    </span>
                  </div>
                </div>
                <div className="flex shrink-0 items-center gap-4">
                  <div className="text-right">
                    <div className="text-lg font-bold text-zinc-100">
                      {job.applicant_count ?? 0}
                    </div>
                    <div className="text-[10px] uppercase tracking-wider text-zinc-500">
                      applicants
                    </div>
                  </div>
                  <ArrowRight className="h-4 w-4 text-zinc-600" />
                </div>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
