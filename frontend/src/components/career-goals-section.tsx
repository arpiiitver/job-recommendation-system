"use client";

import { useEffect, useState } from "react";
import { BookOpen, Compass, Sparkles, X } from "lucide-react";
import { getSkills } from "@/lib/api";

/** Common target roles (job families + frequent Indian job titles) for suggestions. */
const ROLE_SUGGESTIONS = [
  "Software Engineer", "Backend Developer", "Frontend Developer", "Full Stack Developer",
  "Data Scientist", "Data Analyst", "Machine Learning Engineer", "AI Engineer",
  "DevOps Engineer", "Cloud Engineer", "Site Reliability Engineer", "Cybersecurity Analyst",
  "QA / Test Engineer", "Mobile App Developer", "Business Analyst", "Product Manager",
  "UI/UX Designer", "Data Engineer",
];

const DOMAIN_SUGGESTIONS = [
  "Machine Learning", "Generative AI", "Data Analytics", "Cloud", "Web Development",
  "Mobile Apps", "Cybersecurity", "FinTech", "HealthTech", "E-commerce", "EdTech", "Gaming",
];

const LIMITS = { roles: 5, interests: 10, learning: 25 } as const;

type TagInputProps = {
  id: string;
  values: string[];
  onChange: (next: string[]) => void;
  placeholder: string;
  suggestions: string[];
  max: number;
  chipClass: string;
  onQuery?: (q: string) => void;
};

function TagInput({ id, values, onChange, placeholder, suggestions, max, chipClass, onQuery }: TagInputProps) {
  const [draft, setDraft] = useState("");
  const add = (raw: string) => {
    const v = raw.trim();
    if (!v || values.length >= max) return;
    if (values.some((x) => x.toLowerCase() === v.toLowerCase())) return;
    onChange([...values, v]);
    setDraft("");
  };
  return (
    <div>
      <div className="flex flex-wrap gap-2">
        {values.map((v) => (
          <span key={v} className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-1 text-xs font-medium ${chipClass}`}>
            {v}
            <button
              type="button"
              aria-label={`Remove ${v}`}
              onClick={() => onChange(values.filter((x) => x !== v))}
              className="opacity-70 hover:opacity-100"
            >
              <X className="h-3 w-3" />
            </button>
          </span>
        ))}
      </div>
      <div className="mt-2 flex gap-2">
        <input
          list={`${id}-list`}
          value={draft}
          disabled={values.length >= max}
          onChange={(e) => {
            setDraft(e.target.value);
            onQuery?.(e.target.value);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === ",") {
              e.preventDefault();
              add(draft);
            }
          }}
          placeholder={values.length >= max ? `Maximum ${max} reached` : placeholder}
          className="flex-1 rounded-lg border border-zinc-800 bg-zinc-950 px-3 py-2 text-sm text-zinc-100 placeholder:text-zinc-600 focus:border-teal-500/60 focus:outline-none disabled:opacity-50"
        />
        <button
          type="button"
          onClick={() => add(draft)}
          disabled={!draft.trim() || values.length >= max}
          className="rounded-lg border border-zinc-700 px-3 py-2 text-xs font-semibold text-zinc-200 hover:border-teal-500/50 hover:text-teal-300 disabled:opacity-40"
        >
          Add
        </button>
        <datalist id={`${id}-list`}>
          {suggestions.map((s) => (
            <option key={s} value={s} />
          ))}
        </datalist>
      </div>
    </div>
  );
}

type Props = {
  targetRoles: string[];
  setTargetRoles: (v: string[]) => void;
  interests: string[];
  setInterests: (v: string[]) => void;
  learning: string[];
  setLearning: (v: string[]) => void;
  /** Skills already on the profile (to keep "learning" distinct from "have"). */
  haveSkills: string[];
};

/**
 * Career goals: what the seeker WANTS next, beyond what the resume shows.
 * Especially for role/domain switchers - recommendations, the live job feed and
 * skill gaps follow these target roles and count "learning" skills as partial.
 */
export default function CareerGoalsSection({
  targetRoles, setTargetRoles, interests, setInterests, learning, setLearning, haveSkills,
}: Props) {
  const [skillSuggestions, setSkillSuggestions] = useState<string[]>([]);
  const [skillQuery, setSkillQuery] = useState("");

  useEffect(() => {
    const q = skillQuery.trim();
    if (q.length < 2) return;
    const t = setTimeout(() => {
      getSkills(q)
        .then((items) => setSkillSuggestions(items.map((s) => s.name)))
        .catch(() => setSkillSuggestions([]));
    }, 250);
    return () => clearTimeout(t);
  }, [skillQuery]);

  const have = new Set(haveSkills.map((s) => s.toLowerCase()));
  const overlap = learning.filter((s) => have.has(s.toLowerCase()));

  return (
    <section id="career-goals" className="scroll-mt-24 rounded-xl border border-teal-500/20 bg-zinc-900/40 p-6 backdrop-blur">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="flex items-center gap-2 text-lg font-semibold text-zinc-100">
            <Compass className="h-5 w-5 text-teal-400" /> Career goals
          </h2>
          <p className="mt-1 max-w-2xl text-xs text-zinc-400">
            Tell us where you want to go, not just where you have been. Switching roles or domains?
            Add the roles you are aiming for and the skills you are learning — recommendations, live
            job results and skill gaps will follow your goals instead of only your resume.
          </p>
        </div>
        <Sparkles className="hidden h-5 w-5 shrink-0 text-teal-400/60 sm:block" />
      </div>

      <div className="mt-5 grid gap-6 lg:grid-cols-3">
        <div>
          <label className="text-sm font-medium text-zinc-200">Roles I&apos;m targeting</label>
          <p className="mb-2 text-[11px] text-zinc-500">Up to {LIMITS.roles}. The first one is used most.</p>
          <TagInput
            id="target-roles"
            values={targetRoles}
            onChange={setTargetRoles}
            placeholder="e.g. Data Scientist"
            suggestions={ROLE_SUGGESTIONS}
            max={LIMITS.roles}
            chipClass="border-teal-500/30 bg-teal-500/10 text-teal-300"
          />
        </div>

        <div>
          <label className="text-sm font-medium text-zinc-200">Domains / interests</label>
          <p className="mb-2 text-[11px] text-zinc-500">Areas you&apos;d like to work in.</p>
          <TagInput
            id="interests"
            values={interests}
            onChange={setInterests}
            placeholder="e.g. Machine Learning"
            suggestions={DOMAIN_SUGGESTIONS}
            max={LIMITS.interests}
            chipClass="border-violet-500/30 bg-violet-500/10 text-violet-300"
          />
        </div>

        <div>
          <label className="flex items-center gap-1.5 text-sm font-medium text-zinc-200">
            <BookOpen className="h-4 w-4 text-sky-400" /> Skills I&apos;m learning
          </label>
          <p className="mb-2 text-[11px] text-zinc-500">
            Shown as &quot;learning&quot; (not missing) on job skill gaps.
          </p>
          <TagInput
            id="learning-skills"
            values={learning}
            onChange={setLearning}
            placeholder="e.g. Python"
            suggestions={skillSuggestions}
            max={LIMITS.learning}
            chipClass="border-sky-500/30 bg-sky-500/10 text-sky-300"
            onQuery={setSkillQuery}
          />
          {overlap.length > 0 && (
            <p className="mt-2 text-[11px] text-amber-400">
              Already in your skills: {overlap.join(", ")} — those count as skills you have.
            </p>
          )}
        </div>
      </div>

      <p className="mt-5 text-[11px] text-zinc-500">
        Have skills that are not on your resume? Add them in <strong>Skills &amp; Proficiencies</strong> above.
        Remember to <strong>Save</strong> your profile.
      </p>
    </section>
  );
}
