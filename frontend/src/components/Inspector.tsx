import { useMemo, useState } from 'react';
import { MODELS, algoLabel, f3, importances, makeCurve, pct, timeAgo, trainingScript } from '../lib';
import { useStore } from '../store';
import { connection } from '../api';
import { Decision } from '../types';
import { cn } from '../utils/cn';
import { Btn, Curve, Delta, Eyebrow, Leader, StatusChip, Tag } from '../ui';

type Tab = 'overview' | 'explain' | 'script';

export function Inspector() {
  const {
    experiments,
    selectedId,
    select,
    champion,
    baseline,
    columns,
    excluded,
    target,
    promote,
    retry,
    cancel,
    remove,
    fork,
    setDecision,
    deploy,
    compareBaseline,
  } = useStore();
  const [tab, setTab] = useState<Tab>('overview');
  const [model, setModel] = useState('xgboost');
  const [copied, setCopied] = useState(false);

  const exp = experiments.find((e) => e.id === selectedId);
  const parent = experiments.find((e) => e.id === exp?.parent_id);
  const lineage = useMemo(() => {
    const chain = [];
    let cur = exp;
    while (cur) {
      chain.unshift(cur);
      cur = cur.parent_id ? experiments.find((e) => e.id === cur!.parent_id) : undefined;
    }
    return chain;
  }, [exp, experiments]);

  if (!exp) return null;

  const vs = compareBaseline ? baseline : parent;
  const d = (k: 'f1' | 'accuracy' | 'precision' | 'recall') =>
    exp.metrics[k] !== undefined && vs?.metrics[k] !== undefined ? exp.metrics[k]! - vs.metrics[k]! : undefined;
  const isChampion = champion?.id === exp.id;
  const hasKids = experiments.some((e) => e.parent_id === exp.id);
  // Learning curves and importances are illustrative sample data: Demo mode only.
  const curve = connection.demo && exp.metrics.f1 !== undefined ? makeCurve(exp.id, exp.metrics.f1) : null;
  const imps = connection.demo ? importances(exp, columns) : [];
  const script = trainingScript(exp, target, [...excluded]);
  const mx = Math.max(...imps.map((i) => i.weight), 0.01);

  const copy = async (text: string) => {
    await navigator.clipboard?.writeText(text).catch(() => undefined);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  const decisions: { k: Decision; label: string }[] = [
    { k: 'keep', label: 'Keep' },
    { k: 'reject', label: 'Reject' },
    { k: 'baseline', label: 'Baseline' },
  ];

  const metricCells = [
    ['F1', f3(exp.metrics.f1), d('f1')],
    ['Accuracy', pct(exp.metrics.accuracy), d('accuracy')],
    ['Precision', f3(exp.metrics.precision), d('precision')],
    ['Recall', f3(exp.metrics.recall), d('recall')],
  ] as const;

  return (
    <section className="sheet-in ticks mt-3 flex max-h-[46vh] shrink-0 flex-col border border-line bg-panel">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line pl-4">
        <div className="flex min-w-0 items-center gap-3 py-2.5">
          <code className="font-mono text-[11px] text-mute">{exp.id}</code>
          <span className="font-display text-[24px] font-bold uppercase leading-none tracking-wide text-bone">{algoLabel(exp.model_name)}</span>
          <StatusChip status={exp.status} />
          {isChampion && <Tag tone="copper">Champion</Tag>}
        </div>
        <div className="flex items-stretch">
          {(['overview', 'explain', 'script'] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={cn(
                'relative px-3.5 font-mono text-[10.5px] uppercase tracking-[0.1em] transition-colors',
                tab === t ? 'text-bone' : 'text-mute hover:text-bone'
              )}
            >
              {t === 'explain' ? 'Importance' : t}
              {tab === t && <span className="absolute inset-x-0 -bottom-px h-[2px] bg-copper" />}
            </button>
          ))}
          <button onClick={() => select(null)} className="border-l border-line px-3.5 font-mono text-[12px] text-mute transition-colors hover:bg-panel-2 hover:text-bone" title="Close (Esc)">
            ✕
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {tab === 'overview' && (
          <div className="grid gap-6 lg:grid-cols-[1.25fr_1fr]">
            <div className="min-w-0">
              <p className="text-[13px] text-bone-dim">{exp.title ?? exp.model_name}</p>
              <div className="mt-2 flex flex-wrap items-center gap-1 font-mono text-[11px]">
                {lineage.map((l, i) => (
                  <span key={l.id} className="flex items-center gap-1">
                    {i > 0 && <span className="text-rule">→</span>}
                    <button
                      onClick={() => select(l.id)}
                      className={cn('px-1.5 py-0.5 transition-colors', l.id === exp.id ? 'bg-copper text-ink' : 'text-mute hover:text-bone')}
                    >
                      {l.id.replace('exp_', '#')}
                    </button>
                  </span>
                ))}
              </div>

              {exp.status === 'failed' && exp.error && (
                <div className="mt-3 border-l-[3px] border-clay bg-clay/5 p-3 font-mono text-[11.5px] leading-relaxed text-clay">{exp.error}</div>
              )}

              <div className="mt-4 grid grid-cols-2 divide-x divide-line border border-line sm:grid-cols-4">
                {metricCells.map(([l, v, delta]) => (
                  <div key={l} className="px-3.5 py-3">
                    <Eyebrow>{l}</Eyebrow>
                    <div className="mt-1 font-display text-[32px] font-extrabold leading-none tabular-nums text-bone">{v}</div>
                    <div className="mt-1.5 h-4">{delta !== undefined && <Delta value={delta} />}</div>
                  </div>
                ))}
              </div>
              <p className="mt-2 font-mono text-[10.5px] text-mute">
                Δ vs {compareBaseline ? 'baseline' : 'parent'} {vs?.id ?? '—'} · {timeAgo(exp.created_at)} · {exp.runtime_seconds.toFixed(1)}s
              </p>

              {exp.status === 'running' && (
                <div className="mt-3">
                  <div className="mb-1 flex justify-between font-mono text-[11px] text-mute">
                    <span>training</span>
                    <span>{Math.round(exp.progress ?? 0)}%</span>
                  </div>
                  <div className="h-[4px] bg-panel-2">
                    <div className="stripes h-full bg-copper-2 transition-[width] duration-700" style={{ width: `${exp.progress ?? 0}%` }} />
                  </div>
                </div>
              )}
              {curve && (
                <div className="mt-4 border border-line bg-ink p-2">
                  <Eyebrow className="mb-1 block px-1">Validation F1 / epoch</Eyebrow>
                  <Curve data={curve} />
                </div>
              )}
            </div>

            <div className="min-w-0 space-y-4">
              <div>
                <Eyebrow className="mb-1.5 block">Decision</Eyebrow>
                <div className="flex">
                  {decisions.map((x, i) => (
                    <button
                      key={x.k}
                      onClick={() => setDecision(exp.id, x.k)}
                      className={cn(
                        'h-9 flex-1 border font-mono text-[10.5px] font-semibold uppercase tracking-[0.1em] transition-colors',
                        i > 0 && '-ml-px',
                        exp.decision === x.k
                          ? x.k === 'reject'
                            ? 'z-10 border-clay bg-clay text-ink'
                            : 'z-10 border-copper bg-copper text-ink'
                          : 'border-rule text-mute hover:text-bone'
                      )}
                    >
                      {x.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <Leader k="Estimator" v={exp.model_name} />
                <Leader k="Feature" v={exp.feature ?? '—'} />
                {Object.entries(exp.params ?? {})
                  .slice(0, 4)
                  .map(([k, v]) => (
                    <Leader key={k} k={k} v={String(v)} />
                  ))}
              </div>

              <div className="flex flex-wrap gap-2">
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="h-8 cursor-pointer border border-rule bg-ink px-2 font-mono text-[11px] text-bone focus:border-copper focus:outline-none"
                >
                  {MODELS.map((m) => (
                    <option key={m.key} value={m.key}>
                      {m.label}
                    </option>
                  ))}
                </select>
                <Btn size="sm" onClick={() => fork(exp.id, model)}>
                  Fork run
                </Btn>
                {exp.status === 'completed' && (
                  <>
                    {!isChampion && (
                      <Btn variant="primary" size="sm" onClick={() => promote(exp.id)}>
                        Promote
                      </Btn>
                    )}
                    <Btn size="sm" onClick={() => deploy(exp.id)}>
                      Deploy ↗
                    </Btn>
                  </>
                )}
                {exp.status === 'failed' && (
                  <Btn variant="primary" size="sm" onClick={() => retry(exp.id)}>
                    Retry
                  </Btn>
                )}
                {(exp.status === 'running' || exp.status === 'queued') && (
                  <Btn variant="danger" size="sm" onClick={() => cancel(exp.id)}>
                    Cancel
                  </Btn>
                )}
                <Btn variant="ghost" size="sm" onClick={() => remove(exp.id)} disabled={hasKids || !exp.parent_id} title="Delete run">
                  Delete
                </Btn>
              </div>
            </div>
          </div>
        )}

        {tab === 'explain' && (
          <div>
            <p className="mb-4 text-[12.5px] text-mute">
              {connection.demo
                ? 'Top feature importances for this run. Engineered features, when present, tend to surface first.'
                : 'Not available yet: feature importances are not recorded for real runs.'}
            </p>
            <div className="space-y-3">
              {imps.map((f, i) => (
                <div key={f.name} className="grid grid-cols-[150px_minmax(0,1fr)_54px] items-center gap-3">
                  <code className="truncate font-mono text-[12px] text-bone" title={f.name}>
                    {f.name}
                  </code>
                  <div className="h-3 bg-panel-2">
                    <div className="bar-in h-full bg-copper" style={{ width: `${(f.weight / mx) * 100}%`, animationDelay: `${i * 60}ms` }} />
                  </div>
                  <span className="text-right font-mono text-[12px] tabular-nums text-bone-dim">{(f.weight * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {tab === 'script' && (
          <div>
            <div className="mb-2 flex items-center justify-between">
              <span className="font-mono text-[11px] text-mute">train_{exp.id}.py</span>
              <div className="flex gap-2">
                <Btn size="sm" onClick={() => copy(script)}>
                  {copied ? 'Copied' : 'Copy'}
                </Btn>
                <a
                  href={`data:text/x-python,${encodeURIComponent(script)}`}
                  download={`${exp.id}_train.py`}
                  className="chamfer inline-flex h-7 items-center bg-copper px-3 font-mono text-[10.5px] font-semibold uppercase tracking-[0.1em] text-ink transition-colors hover:bg-copper-2"
                >
                  Download ↓
                </a>
              </div>
            </div>
            <pre className="overflow-x-auto border border-line border-l-copper bg-ink p-3 font-mono text-[11.5px] leading-relaxed text-bone-dim [border-left-width:3px]">{script}</pre>
          </div>
        )}
      </div>
    </section>
  );
}
