import { useEffect, useMemo, useState } from 'react';
import { algoLabel, downloadText, f3, importances, makeCurve, pct, shortId, timeAgo } from '../lib';
import { useStore } from '../store';
import { connection, describeError, exportExperiment } from '../api';
import { Experiment } from '../types';
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
    deploy,
    compareBaseline,
  } = useStore();
  const [tab, setTab] = useState<Tab>('overview');
  const [copied, setCopied] = useState(false);
  // The training script comes from the backend (GET .../experiments/{id}/export), fetched when the tab opens.
  const [script, setScript] = useState<{ id: string; filename?: string; text?: string; error?: string } | null>(null);
  const scriptId = tab === 'script' && !connection.demo ? selectedId : null;
  useEffect(() => {
    if (!scriptId) return;
    let dead = false;
    setScript({ id: scriptId });
    exportExperiment(scriptId)
      .then((r) => !dead && setScript({ id: scriptId, filename: r.filename, text: r.script }))
      .catch((e) => !dead && setScript({ id: scriptId, error: describeError(e) }));
    return () => {
      dead = true;
    };
  }, [scriptId]);

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
  const d = (k: keyof Experiment['metrics']) =>
    vs && vs.id !== exp.id && exp.metrics[k] !== undefined && vs.metrics[k] !== undefined ? exp.metrics[k]! - vs.metrics[k]! : undefined;
  const isChampion = champion?.id === exp.id;
  // Learning curves and importances are illustrative sample data: Demo mode only.
  const curve = connection.demo && exp.metrics.f1 !== undefined ? makeCurve(exp.id, exp.metrics.f1) : null;
  const imps = connection.demo ? importances(exp, columns) : [];
  const tabs: Tab[] = connection.demo ? ['overview', 'explain'] : ['overview', 'script'];
  const shown = script?.id === exp.id ? script : null;
  const mx = Math.max(...imps.map((i) => i.weight), 0.01);

  const copy = async (text: string) => {
    await navigator.clipboard?.writeText(text).catch(() => undefined);
    setCopied(true);
    setTimeout(() => setCopied(false), 1200);
  };

  // What a business user acts on first (binary tasks): ranking quality and lift over the
  // base rate. Label metrics at the 0.5 cut-off stay visible but secondary.
  const metricCells = [
    ['PR-AUC', f3(exp.metrics.pr_auc), d('pr_auc')],
    ['Lift @ top 10%', exp.metrics.lift_at_10pct === undefined ? '—' : `${exp.metrics.lift_at_10pct.toFixed(2)}×`, d('lift_at_10pct')],
    ['Base rate', pct(exp.metrics.base_rate), undefined],
    ['F1', f3(exp.metrics.f1), d('f1')],
  ] as const;
  const secondaryCells = [
    ['Accuracy', pct(exp.metrics.accuracy)],
    ['Precision', f3(exp.metrics.precision)],
    ['Recall', f3(exp.metrics.recall)],
  ] as const;

  return (
    <section className="sheet-in ticks mt-3 flex max-h-[46vh] shrink-0 flex-col border border-line bg-panel">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line pl-4">
        <div className="flex min-w-0 items-center gap-3 py-2.5">
          <code className="font-mono text-[11px] text-mute" title={exp.id}>
            {shortId(exp.id)}
          </code>
          <span className="font-display text-[24px] font-bold uppercase leading-none tracking-wide text-bone">{algoLabel(exp.model_name)}</span>
          <StatusChip status={exp.status} />
          {isChampion && <Tag tone="copper">Champion</Tag>}
        </div>
        <div className="flex items-stretch">
          {tabs.map((t) => (
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
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
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
                      {shortId(l.id).replace('exp_', '#')}
                    </button>
                  </span>
                ))}
              </div>

              {exp.hypothesis_mode === 'fallback' && (
                <div className="mt-3 border-l-[3px] border-copper bg-copper/5 p-3 font-mono text-[11.5px] leading-relaxed text-copper">
                  LLM fallback: this idea came from the offline stub provider, not a real model.
                </div>
              )}

              {(exp.status === 'failed' || exp.status === 'rejected_invalid') && exp.error && (
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
                Test split.{' '}
                {secondaryCells.map(([l, v]) => (
                  <span key={l} className="mr-3">
                    {l} {v}
                  </span>
                ))}
              </p>
              <p className="mt-1 font-mono text-[10.5px] text-mute">
                Δ vs {compareBaseline ? 'baseline' : 'parent'} {vs ? shortId(vs.id) : '—'} · {timeAgo(exp.created_at)} · {exp.runtime_seconds.toFixed(1)}s
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
                <div className="flex items-center gap-2">
                  {exp.decision && exp.decision !== 'none' ? (
                    <Tag tone={exp.decision === 'reject' ? 'clay' : exp.decision === 'keep' ? 'sage' : 'neutral'}>{exp.decision}</Tag>
                  ) : (
                    <Tag>pending</Tag>
                  )}
                  {!connection.demo && <span className="font-mono text-[10.5px] text-mute">recorded by the backend</span>}
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

              {connection.demo && exp.status === 'completed' && (
                <Btn size="sm" onClick={() => deploy(exp.id)}>
                  Deploy ↗
                </Btn>
              )}
            </div>
          </div>
        )}

        {tab === 'explain' && (
          <div>
            <p className="mb-4 text-[12.5px] text-mute">Demo mode: sample importances for this run, not measured.</p>
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
            {!shown?.text && (
              <p className="text-[12.5px] text-mute">{shown?.error ? `Not available: ${shown.error}` : 'Loading the training script…'}</p>
            )}
            {shown?.text && (
              <>
                <div className="mb-2 flex items-center justify-between gap-3">
                  <span className="truncate font-mono text-[11px] text-mute">{shown.filename}</span>
                  <div className="flex shrink-0 gap-2">
                    <Btn size="sm" onClick={() => copy(shown.text!)}>
                      {copied ? 'Copied' : 'Copy'}
                    </Btn>
                    <Btn variant="primary" size="sm" onClick={() => downloadText(shown.filename ?? `train_${exp.id}.py`, shown.text!)}>
                      Download ↓
                    </Btn>
                  </div>
                </div>
                <pre className="overflow-x-auto border border-line border-l-copper bg-ink p-3 font-mono text-[11.5px] leading-relaxed text-bone-dim [border-left-width:3px]">{shown.text}</pre>
              </>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
