import { useStore } from './store';
import { Btn, Eyebrow, Tag } from './ui';
import { goConnect } from './route';

function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="h-8 w-8" fill="none" aria-label="MLPilot">
      <rect width="32" height="32" fill="#e0a24f" />
      <g transform="rotate(38 16 16)">
        <ellipse cx="16" cy="16" rx="7.5" ry="11" fill="#0c0907" />
        <path d="M16 5.6C20.6 11 11.4 21 16 26.4" stroke="#e0a24f" strokeWidth="1.7" strokeLinecap="round" />
      </g>
    </svg>
  );
}

const DIFFS: { n: string; claim: string; sub: React.ReactNode }[] = [
  {
    n: '01',
    claim: 'One hypothesis at a time — with the reasoning',
    sub: 'Every run branches from the current best model and cites what the previous runs taught it. No blind brute-force grids, no 400 silent variations.',
  },
  {
    n: '02',
    claim: 'Leakage is categorized, not just flagged',
    sub: (
      <>
        Each flag names its mechanism with the evidence:
        <span className="ml-2 inline-flex flex-wrap gap-1.5 align-middle">
          {['target', 'temporal', 'missingness', 'contamination', 'preprocessing', 'aggregate'].map((c) => (
            <Tag key={c}>{c}</Tag>
          ))}
        </span>
      </>
    ),
  },
  {
    n: '03',
    claim: 'Your CSV, or your database',
    sub: 'Drop a CSV to run the experiment loop on one table. Or connect a Postgres, SQLite or DuckDB database read-only, ask a prediction question, and confirm the task before anything trains.',
  },
  {
    n: '04',
    claim: 'Leave with a runnable pipeline',
    sub: 'A database run ends in an evidence report and an export bundle: the feature SQL, the model file and a scoring script. CSV runs export their training script. Nothing is a black box you have to stay subscribed to.',
  },
];

/** What the database flow does, page by page (Connect → task → run). */
const STEPS = [
  { n: '01', title: 'Connect', sub: 'Single SELECT statements in a read-only transaction, with a timeout and a row limit. The schema graph shows tables and how they join.' },
  { n: '02', title: 'Ask and confirm', sub: 'Type a prediction question. MLPilot drafts the task: entity, target, horizon and cutoff dates. Nothing trains until you confirm it.' },
  { n: '03', title: 'Run', sub: 'Every training row only sees data from before its cutoff. Features are proposed as readable SQL, one at a time, and kept only when time-based validation shows a gain.' },
  { n: '04', title: 'Take it with you', sub: 'An evidence report, an export bundle with the feature SQL, model and scoring script, and a scored list from your live data.' },
];

export function Landing() {
  const { startApp } = useStore();

  return (
    <div className="relative min-h-screen overflow-y-auto">
      <div className="guides" aria-hidden />

      <header className="mx-auto flex h-14 max-w-[1200px] items-center justify-between px-6">
        <div className="flex items-center gap-3">
          <Logo />
          <span className="font-display text-[20px] font-extrabold tracking-wide text-bone">MLPilot</span>
          <Tag className="max-sm:hidden">agentic experimentation</Tag>
        </div>
        <button onClick={() => startApp('sample')} className="whitespace-nowrap font-mono text-[11px] uppercase tracking-[0.12em] text-mute transition-colors hover:text-copper-2">
          Open with sample data →
        </button>
      </header>

      <main className="mx-auto max-w-[1200px] px-6">
        {/* hero + live preview */}
        <section className="grid grid-cols-1 items-center gap-10 py-14 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
          <div>
            <Eyebrow className="text-copper">Feature engineering, on the agent side</Eyebrow>
            <h1 className="mt-4 font-display text-[clamp(44px,5.6vw,80px)] font-extrabold uppercase leading-[0.92] text-bone">
              The loop is the job. <span className="text-copper-2">MLPilot runs it.</span>
            </h1>
            <p className="mt-5 max-w-xl text-[15px] leading-relaxed text-mute">
              Feature engineering and experimentation eat most of a data scientist's time. MLPilot takes that loop — profile the data, audit leakage,
              propose one feature hypothesis, validate it against the current best, record the decision — and shows its reasoning at every step,
              while keeping you in control of every call.
            </p>
            <div className="mt-7 flex flex-wrap items-center gap-3">
              <Btn variant="primary" size="lg" onClick={() => startApp('sample')}>
                ▶ Start with sample data
              </Btn>
              <Btn size="lg" onClick={() => startApp('upload')}>
                Upload your own CSV
              </Btn>
              <Btn size="lg" onClick={goConnect}>
                Connect a database
              </Btn>
            </div>
            <p className="mt-4 font-mono text-[10.5px] uppercase tracking-[0.1em] text-mute">
              read-only · postgres / sqlite / duckdb · the LLM sees schema and statistics, not rows
            </p>
          </div>

          <div className="ticks relative border border-line bg-panel/70">
            <div className="flex items-center justify-between border-b border-line px-4 py-2">
              <Eyebrow>A run on your database</Eyebrow>
              <Tag tone="copper">read-only</Tag>
            </div>
            <ol>
              {STEPS.map((st) => (
                <li key={st.n} className="grid grid-cols-[38px_minmax(0,1fr)] gap-x-3 border-b border-line px-4 py-4 last:border-b-0">
                  <span className="pt-[5px] font-mono text-[11px] text-copper">{st.n}</span>
                  <div className="min-w-0">
                    <h3 className="font-display text-[20px] font-bold uppercase leading-tight tracking-wide text-bone">{st.title}</h3>
                    <p className="mt-1 text-[12.5px] leading-relaxed text-mute">{st.sub}</p>
                  </div>
                </li>
              ))}
            </ol>
          </div>
        </section>

        {/* differentiators */}
        <section className="border-t border-line py-12">
          <Eyebrow>What is actually different</Eyebrow>
          <div className="mt-5 grid gap-x-10 gap-y-8 md:grid-cols-2">
            {DIFFS.map((d) => (
              <div key={d.n} className="border-l-2 border-line pl-5">
                <span className="font-mono text-[11px] text-copper">{d.n}</span>
                <h3 className="mt-1 font-display text-[24px] font-bold uppercase leading-tight tracking-wide text-bone">{d.claim}</h3>
                <p className="mt-2 max-w-md text-[13px] leading-relaxed text-mute">{d.sub}</p>
              </div>
            ))}
          </div>
        </section>

        {/* final CTA */}
        <section className="border-t border-line py-14 text-center">
          <h2 className="font-display text-[40px] font-extrabold uppercase leading-none tracking-wide text-bone">
            Try it on the sample data.
          </h2>
          <p className="mx-auto mt-3 max-w-md text-[13.5px] leading-relaxed text-mute">
            A telecom churn table, 7,043 customers and 21 columns: profiled, scanned for leakage and trained on the real backend, with feature ideas queued to run.
          </p>
          <div className="mt-6 flex justify-center">
            <Btn variant="primary" size="lg" onClick={() => startApp('sample')}>
              ▶ Start with sample data
            </Btn>
          </div>
          <p className="mt-8 font-mono text-[10px] uppercase tracking-[0.14em] text-mute">
            Self-hosted · read-only on your database · open source
          </p>
        </section>
      </main>
    </div>
  );
}
