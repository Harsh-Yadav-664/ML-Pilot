import { useMemo, useState } from 'react';
import { useStore } from '../store';
import { Experiment, ExperimentStatus } from '../types';
import { algoLabel, f3, pct, trainingScript } from '../lib';
import { cn } from '../utils/cn';
import { Btn, Delta, Eyebrow, Led, Panel, STATUS, StatusChip, Tag } from '../ui';
import { ExperimentGraph } from '../components/ExperimentGraph';
import { Inspector } from '../components/Inspector';

type SortKey = 'f1' | 'accuracy' | 'runtime' | 'id';

export function ExperimentsView() {
  const { experiments, champion, baseline, selectedId, select, compareBaseline, setCompareBaseline, deploy, excluded, target, setDecision } =
    useStore();
  const [mode, setMode] = useState<'graph' | 'table'>('graph');
  const [status, setStatus] = useState<ExperimentStatus | 'all'>('all');
  const [sort, setSort] = useState<{ key: SortKey; dir: 1 | -1 }>({ key: 'f1', dir: -1 });

  const rows = useMemo(() => {
    const val = (e: Experiment) =>
      sort.key === 'f1' ? e.metrics.f1 ?? -1 : sort.key === 'accuracy' ? e.metrics.accuracy ?? -1 : sort.key === 'runtime' ? e.runtime_seconds : e.id;
    return experiments
      .filter((e) => status === 'all' || e.status === status)
      .sort((a, b) => (val(a) > val(b) ? 1 : val(a) < val(b) ? -1 : 0) * sort.dir);
  }, [experiments, status, sort]);

  const counts = (s: ExperimentStatus) => experiments.filter((e) => e.status === s).length;
  const grid = 'grid grid-cols-[minmax(130px,1.5fr)_minmax(90px,1fr)_62px_56px_82px_50px_82px_150px] items-center gap-x-3 px-4';

  const th = (label: string, key?: SortKey) => (
    <button
      disabled={!key}
      onClick={() => key && setSort((s) => ({ key, dir: s.key === key ? (s.dir === 1 ? -1 : 1) : -1 }))}
      className={cn('text-left font-mono text-[10px] font-medium uppercase tracking-[0.14em]', key ? 'hover:text-bone' : '', sort.key === key ? 'text-copper-2' : 'text-mute')}
    >
      {label}
      {sort.key === key && (sort.dir === -1 ? ' ↓' : ' ↑')}
    </button>
  );

  const scriptHref = champion ? `data:text/x-python,${encodeURIComponent(trainingScript(champion, target, [...excluded]))}` : undefined;

  return (
    <div className="fade-up flex h-full min-h-0 flex-col p-5">
      <div className="mb-4 flex flex-wrap items-end justify-between gap-4">
        <div>
          <Eyebrow className="text-copper">Experiments</Eyebrow>
          <h1 className="mt-1 font-display text-[40px] font-extrabold uppercase leading-none tracking-wide text-bone">Lineage & results</h1>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {champion && (
            <>
              <a
                href={scriptHref}
                download={`${champion.id}_train.py`}
                className="inline-flex h-9 items-center border border-rule px-4 font-mono text-[11px] font-semibold uppercase tracking-[0.1em] text-bone transition-[transform,border-color,color,box-shadow] hover:-translate-x-px hover:-translate-y-px hover:border-copper hover:text-copper-2 hover:shadow-[3px_3px_0_0_rgba(224,162,79,0.28)]"
              >
                Training script ↓
              </a>
              <Btn variant="primary" onClick={() => deploy(champion.id)}>
                Deploy champion ↗
              </Btn>
            </>
          )}
        </div>
      </div>

      <div className="mb-3 flex flex-wrap items-center justify-between gap-3 border-b border-line">
        <div className="flex">
          {(
            [
              ['graph', 'Lineage'],
              ['table', 'Leaderboard'],
            ] as const
          ).map(([k, label]) => (
            <button
              key={k}
              onClick={() => setMode(k)}
              className={cn(
                'relative px-4 py-2.5 font-mono text-[11px] uppercase tracking-[0.1em] transition-colors',
                mode === k ? 'text-bone' : 'text-mute hover:text-bone'
              )}
            >
              {label}
              {mode === k && <span className="absolute inset-x-0 -bottom-px h-[2px] bg-copper" />}
            </button>
          ))}
        </div>
        <button onClick={() => setCompareBaseline(!compareBaseline)} className="flex items-center gap-2.5 pb-1 font-mono text-[10.5px] uppercase tracking-[0.1em] text-mute transition-colors hover:text-bone">
          Compare to baseline
          <span className={cn('relative h-[18px] w-9 border transition-colors', compareBaseline ? 'border-copper bg-copper' : 'border-rule')}>
            <span className={cn('absolute top-[2px] h-[12px] w-[12px] transition-all', compareBaseline ? 'left-[20px] bg-ink' : 'left-[2px] bg-mute')} />
          </span>
        </button>
      </div>

      {mode === 'graph' ? (
        <div className="flex min-h-0 flex-1 flex-col">
          <Panel ticks className="bg-grid min-h-[300px] flex-1 overflow-hidden">
            <ExperimentGraph />
          </Panel>
          {selectedId && <Inspector />}
        </div>
      ) : (
        <div className="min-h-0 flex-1 overflow-auto">
          <div className="mb-3 flex flex-wrap gap-1.5">
            {(['all', 'completed', 'running', 'queued', 'failed'] as const).map((s) => (
              <button
                key={s}
                onClick={() => setStatus(s)}
                className={cn(
                  'flex h-7 items-center gap-1.5 border px-2.5 font-mono text-[10.5px] uppercase tracking-[0.1em] transition-colors',
                  status === s ? 'border-copper bg-copper text-ink' : 'border-line text-mute hover:border-rule hover:text-bone'
                )}
              >
                {s !== 'all' && status !== s && <Led color={STATUS[s].hex} />}
                {s}
                <span className={status === s ? 'text-ink/70' : 'text-rule'}>{s === 'all' ? experiments.length : counts(s)}</span>
              </button>
            ))}
          </div>

          <Panel className="overflow-x-auto">
            <div className="min-w-[780px]">
              <div className={cn(grid, 'border-b border-line py-2.5')}>
                {th('Run', 'id')}
                {th('Algorithm')}
                {th('F1', 'f1')}
                {th('Acc', 'accuracy')}
                {th('Δ base')}
                {th('Time', 'runtime')}
                {th('Status')}
                {th('Decision')}
              </div>
              {rows.map((e) => {
                const dBase = compareBaseline && e.metrics.f1 !== undefined && baseline?.metrics.f1 !== undefined ? e.metrics.f1 - baseline.metrics.f1 : undefined;
                const sel = selectedId === e.id;
                return (
                  <div
                    key={e.id}
                    onClick={() => select(sel ? null : e.id)}
                    className={cn('group relative cursor-pointer border-b border-line/70 py-2.5 transition-colors last:border-b-0 hover:bg-panel-2/60', sel && 'bg-copper/[0.06]')}
                  >
                    <span className={cn('absolute inset-y-0 left-0 w-[3px] origin-top bg-copper transition-transform duration-200', sel ? 'scale-y-100' : 'scale-y-0 group-hover:scale-y-100')} />
                    <div className={grid}>
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <code className="font-mono text-[12px] text-bone">{e.id}</code>
                          {champion?.id === e.id && <Tag tone="copper">Champion</Tag>}
                        </div>
                        <div className="truncate font-mono text-[10.5px] text-mute">{e.title ?? '—'}</div>
                      </div>
                      <span className="truncate text-[12.5px] text-bone-dim">{algoLabel(e.model_name)}</span>
                      <span className="font-mono text-[13px] font-semibold tabular-nums text-bone">{f3(e.metrics.f1)}</span>
                      <span className="font-mono text-[11.5px] tabular-nums text-mute">{pct(e.metrics.accuracy)}</span>
                      <span>{dBase !== undefined && e.id !== baseline?.id ? <Delta value={dBase} /> : <span className="text-rule">—</span>}</span>
                      <span className="font-mono text-[11.5px] tabular-nums text-mute">{e.runtime_seconds.toFixed(1)}s</span>
                      <StatusChip status={e.status} />
                      <div className="flex" onClick={(ev) => ev.stopPropagation()}>
                        {(['keep', 'reject', 'baseline'] as const).map((d, i) => (
                          <button
                            key={d}
                            onClick={() => setDecision(e.id, d)}
                            className={cn(
                              'h-6 flex-1 border font-mono text-[9.5px] font-semibold uppercase tracking-[0.08em] transition-colors',
                              i > 0 && '-ml-px',
                              e.decision === d ? (d === 'reject' ? 'z-10 border-clay bg-clay text-ink' : 'z-10 border-copper bg-copper text-ink') : 'border-line text-mute hover:text-bone'
                            )}
                          >
                            {d === 'baseline' ? 'base' : d}
                          </button>
                        ))}
                      </div>
                    </div>
                  </div>
                );
              })}
              {!rows.length && <div className="px-4 py-10 text-center font-mono text-[12px] text-mute">no runs match this filter</div>}
            </div>
          </Panel>
          {selectedId && <Inspector />}
        </div>
      )}
    </div>
  );
}
