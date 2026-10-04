import { useStore } from './store';
import { Btn, Eyebrow, Tag } from './ui';
import { ExperimentGraph } from './components/ExperimentGraph';

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
    claim: 'Your data, or a live source',
    sub: 'Drop a CSV or point MLPilot at a read-only warehouse query. The profile, the leakage scan and every experiment run against what you actually have.',
  },
  {
    n: '04',
    claim: 'Leave with a runnable pipeline',
    sub: 'Any run — champion or rejected — exports a complete, reproducible training script. Nothing in the graph is a black box you have to stay subscribed to.',
  },
];

export function Landing() {
  const { startApp, experiments, loading } = useStore();

  return (
    <div className="relative min-h-screen overflow-y-auto">
      <div className="guides" aria-hidden />

      <header className="mx-auto flex h-14 max-w-[1200px] items-center justify-between px-6">
        <div className="flex items-center gap-3">
          <Logo />
          <span className="font-display text-[20px] font-extrabold tracking-wide text-bone">MLPilot</span>
          <Tag>agentic experimentation</Tag>
        </div>
        <button onClick={() => startApp('sample')} className="font-mono text-[11px] uppercase tracking-[0.12em] text-mute transition-colors hover:text-copper-2">
          Open with sample data →
        </button>
      </header>

      <main className="mx-auto max-w-[1200px] px-6">
        {/* hero + live preview */}
        <section className="grid items-center gap-10 py-14 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
          <div>
            <Eyebrow className="text-copper">Feature engineering, on the agent side</Eyebrow>
            <h1 className="mt-4 font-display text-[clamp(44px,5.6vw,80px)] font-extrabold uppercase leading-[0.92] text-bone">
              The loop is the job. <span className="text-copper-2">MLPilot runs it.</span>
            </h1>
            <p className="mt-5 max-w-xl text-[15px] leading-relaxed text-mute">
              Feature engineering and experimentation eat most of a data scientist's time. MLPilot takes that loop — profile the data, audit leakage,
              propose one feature hypothesis, cross-validate it against the current best, record the decision — and shows its reasoning at every step,
              while keeping you in control of every call.
            </p>
            <div className="mt-7 flex flex-wrap items-center gap-3">
              <Btn variant="primary" size="lg" onClick={() => startApp('sample')}>
                ▶ Start with sample data
              </Btn>
              <Btn size="lg" onClick={() => startApp('upload')}>
                Upload your own CSV
              </Btn>
            </div>
            <p className="mt-4 font-mono text-[10.5px] uppercase tracking-[0.1em] text-mute">
              or connect a read-only source · bigquery / snowflake / postgres
            </p>
          </div>

          <div className="ticks relative border border-line bg-panel/70">
            <div className="flex items-center justify-between border-b border-line px-4 py-2">
              <Eyebrow>Experiment lineage · live</Eyebrow>
              <Tag tone="copper">Real preview</Tag>
            </div>
            <div className="bg-grid relative h-[380px]">
              {loading ? (
                <div className="flex h-full items-center justify-center font-mono text-[11px] uppercase tracking-[0.12em] text-mute">Loading lineage…</div>
              ) : (
                <ExperimentGraph />
              )}
              <button
                onClick={() => startApp('sample')}
                className="absolute inset-0 z-10 flex cursor-pointer items-end justify-center pb-4"
                title="Open the workspace"
              >
                <span className="border border-rule bg-ink/90 px-3 py-1.5 font-mono text-[10.5px] uppercase tracking-[0.12em] text-bone-dim transition-colors hover:border-copper hover:text-copper-2">
                  Click to open the workspace ↗
                </span>
              </button>
            </div>
            <div className="flex items-center justify-between border-t border-line px-4 py-2 font-mono text-[10.5px] text-mute">
              <span>{experiments.length} runs · churn_v3</span>
              <span className="text-copper-2">every edge is a tested feature</span>
            </div>
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

        {/* evidence: accepted vs rejected */}
        <section className="border-t border-line py-12">
          <Eyebrow>Every decision shows its work</Eyebrow>
          <h2 className="mt-2 font-display text-[34px] font-extrabold uppercase leading-none tracking-wide text-bone">
            Accepted. Rejected. With receipts.
          </h2>
          <div className="mt-6 grid gap-4 lg:grid-cols-2">
            <div className="relative border border-sage/40 bg-panel p-5">
              <span className="absolute inset-y-0 left-0 w-[3px] bg-sage" />
              <div className="flex items-center justify-between">
                <Tag tone="sage">Accepted · kept</Tag>
                <code className="font-mono text-[11px] text-mute">exp_021 · F1 0.810 → 0.840</code>
              </div>
              <code className="mt-3 block font-mono text-[14px] font-semibold text-bone">support_ticket_velocity</code>
              <div className="relative mt-2 border border-line bg-ink">
                <span className="absolute inset-y-0 left-0 w-[2px] bg-copper" />
                <code className="block py-2 pl-3 pr-3 font-mono text-[11.5px] text-copper-2">count(tickets, 30d) / count(tickets, 90d)</code>
              </div>
              <p className="mt-3 text-[13px] leading-relaxed text-bone-dim">
                Reasoning on record: a rising short-term ticket rate signals frustration that precedes cancellation.
              </p>
              <p className="mt-2 font-mono text-[11px] text-mute">
                evidence · 5-fold CV beat parent by +0.030 · monotonic with churn in sample
              </p>
            </div>
            <div className="relative border border-clay/40 bg-panel p-5">
              <span className="absolute inset-y-0 left-0 w-[3px] bg-clay" />
              <div className="flex items-center justify-between">
                <Tag tone="clay">Rejected · as feature</Tag>
                <code className="font-mono text-[11px] text-mute">leakage report · lk_01</code>
              </div>
              <code className="mt-3 block font-mono text-[14px] font-semibold text-bone">churn_date</code>
              <div className="relative mt-2 border border-line bg-ink">
                <span className="absolute inset-y-0 left-0 w-[2px] bg-clay" />
                <code className="block py-2 pl-3 pr-3 font-mono text-[11.5px] text-clay">87.4% null · 100% of non-null rows are positive</code>
              </div>
              <p className="mt-3 text-[13px] leading-relaxed text-bone-dim">
                Reasoning on record: only populated for customers who already churned — the column encodes the label.
              </p>
              <p className="mt-2 font-mono text-[11px] text-mute">evidence · category: target · including it would inflate F1 to 0.91 offline</p>
            </div>
          </div>
        </section>

        {/* final CTA */}
        <section className="border-t border-line py-14 text-center">
          <h2 className="font-display text-[40px] font-extrabold uppercase leading-none tracking-wide text-bone">
            The sample dataset is already loaded.
          </h2>
          <p className="mx-auto mt-3 max-w-md text-[13.5px] leading-relaxed text-mute">
            Fourteen thousand rows, one leaky column the agent will catch, and a queue of feature hypotheses ready to run.
          </p>
          <div className="mt-6 flex justify-center">
            <Btn variant="primary" size="lg" onClick={() => startApp('sample')}>
              ▶ Start with sample data
            </Btn>
          </div>
          <p className="mt-8 font-mono text-[10px] uppercase tracking-[0.14em] text-mute">
            Local-first demo · sample data included · every run exports a training script
          </p>
        </section>
      </main>
    </div>
  );
}
