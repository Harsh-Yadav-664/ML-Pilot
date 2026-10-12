import { useMemo, useRef, useState } from 'react';
import { FeedItem, useStore } from '../store';
import { LeakageWarning } from '../types';
import { algoLabel, f3, pct, shortId, timeAgo } from '../lib';
import { cn } from '../utils/cn';
import {
  Btn,
  CardHeader,
  Delta,
  Eyebrow,
  Led,
  Leader,
  LineChart,
  MiniHist,
  Num,
  Panel,
  Reveal,
  SectionHead,
  STATUS,
  Tag,
  VolumeBars,
  useNow,
} from '../ui';
import { ExperimentGraph } from '../components/ExperimentGraph';

const dtypeColor: Record<string, string> = {
  int: 'text-copper-2 border-copper/40',
  float: 'text-copper-2 border-copper/40',
  category: 'text-bone-dim border-rule',
  datetime: 'text-sage border-sage/40',
  bool: 'text-bone border-rule',
  string: 'text-mute border-line',
};

/* ---------------------------------------------------------------- upload */
export function UploadModal() {
  const { uploadOpen, setUploadOpen, ingestCsv } = useStore();
  const [over, setOver] = useState(false);
  const ref = useRef<HTMLInputElement>(null);
  if (!uploadOpen) return null;

  const take = (f?: File) => {
    if (f && /\.csv$/i.test(f.name)) void ingestCsv(f);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4" onMouseDown={() => setUploadOpen(false)}>
      <div className="pop ticks w-full max-w-lg border border-rule bg-panel p-6 shadow-[8px_8px_0_0_rgba(0,0,0,0.6)]" onMouseDown={(e) => e.stopPropagation()}>
        <Eyebrow className="text-copper">Ingest</Eyebrow>
        <h2 className="mt-1 font-display text-[34px] font-extrabold uppercase leading-none tracking-wide text-bone">Drop a dataset</h2>
        <p className="mt-2 text-[13px] text-mute">CSV only. Columns are profiled and the target is guessed from the header.</p>
        <button
          onDragOver={(e) => {
            e.preventDefault();
            setOver(true);
          }}
          onDragLeave={() => setOver(false)}
          onDrop={(e) => {
            e.preventDefault();
            setOver(false);
            take(e.dataTransfer.files[0]);
          }}
          onClick={() => ref.current?.click()}
          className={cn(
            'mt-5 flex h-40 w-full flex-col items-center justify-center gap-2 border border-dashed transition-colors',
            over ? 'border-copper bg-copper/10' : 'border-rule bg-ink hover:border-copper/70'
          )}
        >
          <span className="font-display text-[26px] font-bold uppercase tracking-wide text-copper-2">{over ? 'Release to load' : '.csv'}</span>
          <span className="font-mono text-[11px] uppercase tracking-[0.1em] text-mute">drag here or click to browse</span>
        </button>
        <input ref={ref} type="file" accept=".csv" className="hidden" onChange={(e) => take(e.target.files?.[0])} />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ hero */
function Hero() {
  const {
    metrics,
    champion,
    baseline,
    loading,
    live,
    demo,
    warnings,
    columns,
    experiments,
    agentStatus,
    hypotheses,
    autoClean,
    cleaning,
    runNext,
    setUploadOpen,
    setView,
    datasetName,
    fileName,
    target,
  } = useStore();

  const bySeverity = (s: LeakageWarning['severity']) => warnings.filter((w) => w.severity === s).length;
  const [high, medium, low] = [bySeverity('high'), bySeverity('medium'), bySeverity('low')];
  const running = experiments.find((e) => e.status === 'running');
  const finished = experiments.filter((e) => e.status === 'completed').length;
  const championIsBaseline = !!champion && champion.id === baseline?.id;
  const lift =
    !championIsBaseline && champion?.metrics.f1 !== undefined && baseline?.metrics.f1 !== undefined ? champion.metrics.f1 - baseline.metrics.f1 : undefined;

  const headline =
    agentStatus === 'autopilot'
      ? `Autopilot is testing ${hypotheses} hypotheses`
      : high > 0
      ? `${high} high-risk leakage finding${high > 1 ? 's' : ''}`
      : running
      ? `${algoLabel(running.model_name)} is training`
      : champion
      ? `${algoLabel(champion.model_name)} holds the lead`
      : 'No completed runs yet';

  const sub =
    agentStatus === 'autopilot'
      ? 'Each hypothesis branches off the current champion and reports back to the queue.'
      : high > 0
      ? 'A column that leaks the label scores well offline and fails in use. Read the findings below before trusting a score.'
      : running
      ? `${running.title ?? shortId(running.id)}, ${running.runtime_seconds.toFixed(1)}s so far.`
      : champion && lift !== undefined
      ? `Champion ${shortId(champion.id)} is ${lift >= 0 ? '+' : ''}${lift.toFixed(3)} F1 against baseline ${shortId(baseline!.id)}.`
      : championIsBaseline
      ? finished > 1
        ? 'The baseline still leads: no feature idea has beaten it yet.'
        : 'The baseline is the only finished run. Run a feature idea to try to beat it.'
      : 'Run a hypothesis from the agent queue to seed the lineage.';

  const scanned = live || demo;
  const scanTone = high ? 'text-clay' : medium ? 'text-copper-2' : 'text-sage';
  const features = columns.filter((c) => c.role === 'feature').length;

  return (
    <section className="ticks relative border border-line bg-panel/70">
      <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="min-w-0 p-6 sm:p-8">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[10.5px] uppercase tracking-[0.12em] text-mute">
            <span className="text-copper">Command deck</span>
            <span className="text-rule">//</span>
            <span className="break-all text-bone-dim">{fileName ?? datasetName}</span>
            <span>→</span>
            <span className="text-copper-2">{target}</span>
            <span className="ml-auto flex items-center gap-1.5">
              <Led color={live ? '#b2cd8b' : demo ? '#f7cd82' : '#e0705a'} />
              {live ? 'backend live' : demo ? 'demo · sample data' : 'not connected'}
            </span>
          </div>

          <h1 className="mt-5 font-display text-[clamp(40px,6.2vw,92px)] font-extrabold uppercase leading-[0.9] text-bone">
            {loading ? 'Profiling dataset' : headline}
          </h1>
          <p className="mt-4 max-w-xl text-[14px] leading-relaxed text-mute">{sub}</p>

          <div className="mt-6 flex flex-wrap gap-2.5">
            <Btn variant="primary" size="lg" onClick={autoClean} disabled={cleaning}>
              {cleaning ? 'Asking for a recipe…' : 'Apply AI auto-clean'}
            </Btn>
            <Btn size="lg" onClick={runNext}>
              ▶ Run next idea
            </Btn>
            <Btn size="lg" variant="ghost" onClick={() => setView('experiments')}>
              Lineage →
            </Btn>
            <Btn size="lg" variant="ghost" onClick={() => setUploadOpen(true)}>
              Replace CSV
            </Btn>
          </div>
        </div>

        <div className="flex flex-col border-t border-line p-6 lg:border-l lg:border-t-0">
          <Eyebrow>Leakage scan</Eyebrow>
          <div className="mt-2 flex items-end gap-2">
            <span className={cn('font-display text-[96px] font-extrabold leading-[0.82] tabular-nums', scanned ? scanTone : 'text-rule')}>
              {scanned && !loading ? <Num value={warnings.length} /> : '—'}
            </span>
            <span className="pb-1 font-mono text-[12px] text-mute">finding{warnings.length === 1 ? '' : 's'}</span>
          </div>
          <div className="mt-3 flex gap-3 font-mono text-[10.5px] uppercase tracking-[0.1em]">
            <span className={high ? 'text-clay' : 'text-mute'}>{high} high</span>
            <span className={medium ? 'text-copper-2' : 'text-mute'}>{medium} medium</span>
            <span className="text-mute">{low} low</span>
          </div>

          <div className="mt-4 divide-y divide-transparent border-t border-line pt-2">
            <Leader k="Rows" v={metrics ? <Num value={metrics.total_rows} /> : '—'} />
            <Leader k="Columns" v={metrics ? <Num value={metrics.total_columns} /> : '—'} />
            <Leader
              k="Missing"
              v={metrics ? <span className={metrics.missing_data_percent > 0.05 ? 'text-copper-2' : ''}><Num value={metrics.missing_data_percent * 100} decimals={2} suffix="%" /></span> : '—'}
            />
            <Leader
              k="Duplicate rows"
              v={metrics ? <span className={metrics.duplicate_rows > 0 ? 'text-copper-2' : ''}><Num value={metrics.duplicate_rows} /></span> : '—'}
            />
            <Leader k="Feature columns" v={metrics ? features : '—'} />
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 divide-x divide-line border-t border-line sm:grid-cols-4 max-sm:[&>*:nth-child(3)]:border-l-0 max-sm:[&>*:nth-child(n+3)]:border-t max-sm:[&>*:nth-child(n+3)]:border-line">
        <div className="min-w-0 px-5 py-4">
          <Eyebrow>Champion f1</Eyebrow>
          <div className="mt-1 font-display text-[44px] font-extrabold leading-none tabular-nums text-bone">
            {loading ? <span className="skeleton inline-block h-9 w-24" /> : f3(champion?.metrics.f1)}
          </div>
        </div>
        <div className="min-w-0 px-5 py-4">
          <Eyebrow>Vs baseline</Eyebrow>
          <div className="mt-2 font-display text-[30px] font-bold leading-none">
            {lift !== undefined ? <Delta value={lift} className="!text-[22px] !font-bold" /> : <span className="text-rule">—</span>}
          </div>
          <div className="mt-1.5 truncate font-mono text-[10.5px] text-mute" title={baseline?.id}>
            {championIsBaseline ? 'champion is the baseline' : baseline ? `baseline ${shortId(baseline.id)}` : 'no baseline'}
          </div>
        </div>
        <div className="min-w-0 px-5 py-4">
          <Eyebrow>Algorithm</Eyebrow>
          <div className="mt-1 font-display text-[28px] font-bold uppercase leading-none tracking-wide text-bone">
            {champion ? algoLabel(champion.model_name) : '—'}
          </div>
          <div className="mt-1.5 font-mono text-[10.5px] text-mute">{pct(champion?.metrics.accuracy)} accuracy</div>
        </div>
        <div className="min-w-0 px-5 py-4">
          <Eyebrow>Run</Eyebrow>
          <div className="mt-1 truncate font-mono text-[22px] font-medium leading-none text-copper-2" title={champion?.id}>
            {champion ? shortId(champion.id) : '—'}
          </div>
          <div className="mt-1.5 font-mono text-[10.5px] text-mute">{champion ? `${champion.runtime_seconds.toFixed(1)}s runtime` : 'no run'}</div>
        </div>
      </div>
    </section>
  );
}

/* --------------------------------------------------------------- leakage */
function Leakage() {
  const { warnings, loading, connectionError } = useStore();
  const high = warnings.filter((w) => w.severity === 'high').length;
  const sev = { high: '#e0705a', medium: '#f7cd82', low: '#a08d76' } as const;

  return (
    <div className="min-w-0">
      <SectionHead
        n="01"
        title="Target leakage"
        right={<Tag tone={high ? 'clay' : warnings.length ? 'copper' : 'sage'}>{loading ? 'scanning' : warnings.length ? `${warnings.length} found` : 'clean'}</Tag>}
      />
      <Panel className={cn('mt-3', high > 0 && 'border-clay/40')}>
        {connectionError && !loading && (
          <p className="px-4 py-5 text-[13px] text-mute">Not available: the backend is not connected, so no leakage scan has run.</p>
        )}
        {warnings.length === 0 && !loading && !connectionError && (
          <p className="px-4 py-5 text-[13px] text-mute">No leakage detected. Every feature is available at prediction time.</p>
        )}
        {loading && <div className="skeleton m-4 h-16" />}
        {warnings.map((w) => (
          <div key={w.id} className="relative border-b border-line px-5 py-3.5 last:border-b-0">
            <span className="absolute inset-y-0 left-0 w-[3px]" style={{ background: sev[w.severity] }} />
            <div className="flex flex-wrap items-center gap-2.5">
              <code className="font-mono text-[13px] font-medium text-bone">{w.column}</code>
              <Tag tone={w.severity === 'high' ? 'clay' : w.severity === 'medium' ? 'copper' : 'neutral'}>{w.severity}</Tag>
              <Tag>{w.category}</Tag>
            </div>
            <p className="mt-1.5 text-[12.5px] leading-relaxed text-bone-dim">{w.message}</p>
          </div>
        ))}
      </Panel>
    </div>
  );
}

/* ------------------------------------------------------------ auto-clean */
function AutoClean() {
  const { cleaning, autoClean } = useStore();
  return (
    <div className="min-w-0">
      <SectionHead n="02" title="Auto-clean" right={<Tag tone={cleaning ? 'copper' : 'neutral'}>{cleaning ? 'asking' : 'on demand'}</Tag>} />
      <Panel className="mt-3 p-4">
        <p className="text-[12.5px] leading-relaxed text-bone-dim">
          The LLM reads the column profile and proposes a cleaning recipe: imputation, encoding and transforms. It is trained as a new baseline run, so
          it is compared like any other run in the lineage.
        </p>
        <Btn variant="primary" size="md" className="mt-3 w-full" onClick={autoClean} disabled={cleaning}>
          {cleaning ? 'Asking for a recipe…' : 'Apply AI auto-clean'}
        </Btn>
      </Panel>
    </div>
  );
}

/* ----------------------------------------------------------- agent queue */
function AgentQueue() {
  const { feed, runSuggestion, agentStatus, setAgentOpen, setView, select, experiments, demo } = useStore();
  const proposals = useMemo(() => feed.filter((f): f is Extract<FeedItem, { kind: 'suggestion' }> => f.kind === 'suggestion'), [feed]);
  const pending = proposals.filter((p) => !p.expId).slice(0, 3);
  const busy = agentStatus !== 'idle';

  return (
    <div className="flex min-w-0 flex-col">
      <SectionHead
        n="04"
        title="Agent queue"
        right={
          <button onClick={() => setAgentOpen(true)} className="font-mono text-[10.5px] uppercase tracking-[0.1em] text-mute transition-colors hover:text-copper-2">
            Open panel ↗
          </button>
        }
      />
      <Panel className="mt-3 flex flex-1 flex-col">
        <div className="flex items-center gap-2 border-b border-line px-4 py-2.5 font-mono text-[10.5px] uppercase tracking-[0.1em]">
          <Led color={busy ? '#f7cd82' : '#b2cd8b'} pulse={busy} />
          <span className={busy ? 'text-copper-2' : 'text-bone-dim'}>{agentStatus === 'autopilot' ? 'autopilot' : agentStatus === 'thinking' ? 'thinking' : 'idle'}</span>
          <span className="text-mute">· {pending.length} pending</span>
        </div>

        <div className="flex-1 space-y-2.5 p-3">
          {pending.length === 0 && <p className="px-1 py-6 text-center text-[12px] text-mute">Queue is empty: every feature idea has a run.</p>}
          {pending.map((p) => (
            <div key={p.id} className="border border-line bg-ink p-3 transition-colors hover:border-copper/50">
              <div className="flex items-start justify-between gap-2">
                <code className="truncate font-mono text-[12px] font-semibold text-bone">{p.s.name}</code>
                {(p.s.impact ?? (demo ? 0.02 : undefined)) !== undefined && (
                  <span className="shrink-0 font-display text-[19px] font-bold leading-none text-sage">+{(p.s.impact ?? 0.02).toFixed(3)}</span>
                )}
              </div>
              <code className="mt-2 block truncate border-l-2 border-copper bg-panel py-1 pl-2.5 font-mono text-[11px] text-copper-2">{p.s.formula}</code>
              <p className="mt-2 line-clamp-2 text-[12px] leading-relaxed text-mute">{p.s.reason}</p>
              <Btn variant="outline" size="sm" className="mt-2.5 w-full" onClick={() => runSuggestion(p.id)}>
                ▶ Run experiment
              </Btn>
            </div>
          ))}
          {proposals
            .filter((p) => p.expId)
            .slice(-2)
            .map((p) => {
              const exp = experiments.find((e) => e.id === p.expId);
              if (!exp) return null;
              const col = exp.status === 'completed' ? '#b2cd8b' : exp.status === 'running' ? '#f7cd82' : '#a08d76';
              return (
                <button
                  key={p.id}
                  onClick={() => {
                    setView('experiments');
                    select(exp.id);
                  }}
                  className="flex w-full items-center gap-2.5 border border-line px-3 py-2 text-left transition-colors hover:border-rule"
                >
                  <Led color={col} pulse={exp.status === 'running'} />
                  <code className="truncate font-mono text-[11.5px] text-bone-dim">{p.s.name}</code>
                  <span className="ml-auto shrink-0 font-mono text-[12px] text-bone">{f3(exp.metrics.f1)}</span>
                </button>
              );
            })}
        </div>
      </Panel>
    </div>
  );
}

/* ------------------------------------------------------------- production */
function Prod() {
  const { endpoints, drift, volume, champion, deploy, setView } = useStore();
  const ep = endpoints[0];
  const psiNow = drift[drift.length - 1]?.psi ?? 0;
  const drifted = psiNow > 0.2;
  const pts = useMemo(() => drift.map((d, i) => ({ id: String(i), y: d.psi, label: `d-${drift.length - i}` })), [drift]);

  return (
    <div>
      <SectionHead
        n="07"
        title="In production"
        right={
          <button onClick={() => setView('deployments')} className="font-mono text-[10.5px] uppercase tracking-[0.1em] text-mute transition-colors hover:text-copper-2">
            Details ↗
          </button>
        }
      />
      <Panel className="mt-3 grid grid-cols-1 divide-y divide-line lg:grid-cols-[minmax(0,0.9fr)_minmax(0,1.2fr)_minmax(0,1fr)] lg:divide-x lg:divide-y-0">
        <div className="p-4">
          <Eyebrow>Endpoint</Eyebrow>
          {ep ? (
            <>
              <div className="mt-2 flex items-center gap-2">
                <Led color={ep.status === 'healthy' ? '#b2cd8b' : '#f7cd82'} pulse={ep.status !== 'healthy'} />
                <span className="font-display text-[22px] font-bold uppercase tracking-wide text-bone">{algoLabel(ep.model_name)}</span>
              </div>
              <code className="mt-1 block truncate font-mono text-[11px] text-copper-2">{ep.url}</code>
              <div className="mt-3">
                <Leader k="Requests / min" v={ep.rpm.toLocaleString()} />
                <Leader k="p95 latency" v={`${ep.p95_ms} ms`} />
                <Leader k="Status" v={ep.status} />
              </div>
            </>
          ) : (
            <>
              <p className="mt-2 text-[13px] text-bone-dim">Nothing serving yet.</p>
              <Btn variant="outline" size="sm" className="mt-3" onClick={() => (champion ? deploy(champion.id) : setView('experiments'))}>
                {champion ? `Deploy ${champion.id}` : 'Open lineage'}
              </Btn>
            </>
          )}
        </div>
        <div className="p-4">
          <div className="flex items-center justify-between">
            <Eyebrow>Data drift · PSI</Eyebrow>
            <Tag tone={drifted ? 'clay' : 'sage'}>
              {psiNow.toFixed(3)} {drifted ? 'alert' : 'stable'}
            </Tag>
          </div>
          <div className="mt-2">
            <LineChart points={pts} baseline={0.2} color={drifted ? '#e0705a' : '#e0a24f'} height={112} yLabel={false} />
          </div>
        </div>
        <div className="p-4">
          <Eyebrow>Requests · 24h</Eyebrow>
          <div className="mt-3">
            <VolumeBars data={volume} />
          </div>
        </div>
      </Panel>
    </div>
  );
}

/* --------------------------------------------------------------- profile */
function Profile() {
  const { columns } = useStore();
  const [q, setQ] = useState('');
  const rows = useMemo(() => columns.filter((c) => !q || c.name.toLowerCase().includes(q.toLowerCase())), [columns, q]);
  /* Last track is flexible so the histogram always fits the pane. */
  const grid = 'grid grid-cols-[minmax(110px,1.1fr)_58px_52px_minmax(84px,0.9fr)_46px_minmax(60px,0.7fr)] items-center gap-x-2.5 px-4';

  return (
    <div className="min-w-0">
      <SectionHead
        n="05"
        title="Column profile"
        right={
          <input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="filter"
            aria-label="Filter columns"
            className="h-7 w-32 border border-line bg-transparent px-2 font-mono text-[11px] text-bone placeholder:text-mute focus:border-copper focus:outline-none"
          />
        }
      />
      <Panel className="mt-3 overflow-x-auto">
        <div className="min-w-[460px]">
          <div className={cn(grid, 'border-b border-line py-2')}>
            <Eyebrow>Column</Eyebrow>
            <Eyebrow>Type</Eyebrow>
            <Eyebrow>Role</Eyebrow>
            <Eyebrow>Missing</Eyebrow>
            <Eyebrow className="text-right">Uniq</Eyebrow>
            <Eyebrow>Dist</Eyebrow>
          </div>
          {rows.map((c) => (
            <div key={c.name} className="group relative border-b border-line/70 py-[7px] transition-colors last:border-b-0 hover:bg-panel-2/60">
              <span className="absolute inset-y-0 left-0 w-[2px] origin-top scale-y-0 bg-copper transition-transform duration-200 group-hover:scale-y-100" />
              <div className={grid}>
                <code className="truncate font-mono text-[12px] text-bone" title={c.name}>
                  {c.name}
                </code>
                <span className={cn('w-fit border px-1 font-mono text-[9.5px]', dtypeColor[c.dtype])}>{c.dtype}</span>
                <span className="truncate font-mono text-[10.5px] text-mute">{c.role}</span>
                <div className="flex items-center gap-1.5">
                  <div className="h-[3px] min-w-0 flex-1 bg-panel-2">
                    <div className={cn('h-full transition-[width] duration-700', c.missing_pct > 20 ? 'bg-copper-2' : 'bg-mute')} style={{ width: `${Math.min(100, c.missing_pct)}%` }} />
                  </div>
                  <span className="w-9 shrink-0 text-right font-mono text-[10.5px] tabular-nums text-mute">{c.missing_pct.toFixed(1)}%</span>
                </div>
                <span className="text-right font-mono text-[10.5px] tabular-nums text-mute">
                  {c.unique > 9999 ? `${(c.unique / 1000).toFixed(0)}k` : c.unique.toLocaleString()}
                </span>
                <MiniHist data={c.dist} />
              </div>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

/* ------------------------------------------------------------------ view */
export function Dashboard() {
  const { experiments, setView, select, champion, baseline, loading, demo } = useStore();
  useNow();
  const completed = experiments.filter((e) => e.status === 'completed' && e.metrics.f1 !== undefined);
  const recent = [...experiments].sort((a, b) => +new Date(b.created_at) - +new Date(a.created_at)).slice(0, 6);
  const inspect = (id: string) => {
    setView('experiments');
    select(id);
  };

  return (
    <div className="mx-auto max-w-[1280px] space-y-10 p-4 sm:p-6">
      <Reveal>
        <Hero />
      </Reveal>

      <Reveal delay={40}>
        <div className="grid grid-cols-1 items-start gap-8 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
          <Leakage />
          <AutoClean />
        </div>
      </Reveal>

      <Reveal delay={60}>
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
          <div className="flex min-w-0 flex-col">
            <SectionHead
              n="03"
              title="Experiment lineage"
              right={
                <button onClick={() => setView('experiments')} className="font-mono text-[10.5px] uppercase tracking-[0.1em] text-mute transition-colors hover:text-copper-2">
                  Open workspace ↗
                </button>
              }
            />
            <Panel ticks className="mt-3 flex flex-1 flex-col">
              <div className="bg-grid min-h-[400px] flex-1">
                {loading ? <div className="flex h-full items-center justify-center font-mono text-[11px] uppercase tracking-[0.1em] text-mute">loading lineage…</div> : <ExperimentGraph focusOnSelect />}
              </div>
              <div className="flex items-center justify-between gap-3 border-t border-line px-4 py-2 font-mono text-[10.5px] text-mute">
                <span>Click a node to inspect it</span>
                {champion && (
                  <span className="truncate text-copper-2">
                    {shortId(champion.id)} · f1 {f3(champion.metrics.f1)} · {pct(champion.metrics.accuracy)} acc
                  </span>
                )}
              </div>
            </Panel>
          </div>
          <AgentQueue />
        </div>
      </Reveal>

      <Reveal>
        <div className="grid grid-cols-1 gap-8 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
          <Profile />
          <div className="flex min-w-0 flex-col">
            <SectionHead n="06" title="Score history" />
            <Panel className="mt-3 flex flex-1 flex-col">
              <CardHeader title="F1 per finished run" sub={`${completed.length} completed run${completed.length === 1 ? '' : 's'}`} />
              <div className="p-4">
                <LineChart
                  points={[...completed]
                    .sort((a, b) => +new Date(a.created_at) - +new Date(b.created_at))
                    .map((e) => ({ id: e.id, y: e.metrics.f1!, label: shortId(e.id) }))}
                  baseline={baseline?.metrics.f1}
                  onSelect={inspect}
                />
              </div>
              <div className="border-t border-line">
                <Eyebrow className="block px-4 pb-1 pt-3">Recent runs</Eyebrow>
                <ul>
                  {recent.map((e) => (
                    <li key={e.id} className="border-t border-line/60 first:border-t-0">
                      <button onClick={() => inspect(e.id)} className="flex w-full items-center gap-2.5 px-4 py-2 text-left text-[12px] transition-colors hover:bg-panel-2/60">
                        <Led color={STATUS[e.status].hex} pulse={e.status === 'running'} />
                        <span className="min-w-0 flex-1 truncate text-bone-dim">{e.title ?? algoLabel(e.model_name)}</span>
                        <span className="shrink-0 font-mono text-[11px] tabular-nums text-bone">{e.status === 'completed' ? f3(e.metrics.f1) : STATUS[e.status].label}</span>
                        <span className="w-14 shrink-0 text-right font-mono text-[10px] tabular-nums text-mute">{timeAgo(e.created_at)}</span>
                      </button>
                    </li>
                  ))}
                  {!recent.length && <li className="px-4 py-3 text-[12px] text-mute">No runs yet.</li>}
                </ul>
              </div>
            </Panel>
          </div>
        </div>
      </Reveal>

      {demo && (
        <Reveal>
          <Prod />
        </Reveal>
      )}
    </div>
  );
}
