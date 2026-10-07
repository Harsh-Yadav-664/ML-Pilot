import { useMemo, useState } from 'react';
import type { RunFeature } from '../api/runs';
import { Tag } from '../ui';

type Status = 'baseline' | 'accepted' | 'rejected_gain' | 'rejected_guard' | 'rejected_duplicate' | 'vetoed' | 'proposed';

const LABEL: Record<Status, string> = {
  baseline: 'baseline',
  accepted: 'accepted',
  rejected_gain: 'rejected: no gain',
  rejected_guard: 'rejected: guard',
  rejected_duplicate: 'rejected: duplicate',
  vetoed: 'vetoed by you',
  proposed: 'proposed',
};
const TONE: Record<Status, 'sage' | 'copper' | 'clay' | 'neutral'> = {
  baseline: 'neutral',
  accepted: 'sage',
  rejected_gain: 'clay',
  rejected_guard: 'clay',
  rejected_duplicate: 'clay',
  vetoed: 'copper',
  proposed: 'copper',
};

/** A baseline feature is `accepted` in the table but was not tested one at a time: it is shown as the baseline. */
export const statusOf = (f: RunFeature): Status => (f.kind === 'dfs' ? 'baseline' : (f.status as Status));
const SOURCE: Record<string, string> = {
  dfs: 'DFS',
  llm_sql: 'LLM',
  formula: 'formula',
  user: 'user',
};

type Gain = {
  mean_gain?: number;
  ci95?: [number, number];
  margin?: number;
  importance?: number;
};
const signed = (x: number) => `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(4)}`;

function gainText(f: RunFeature): string {
  const g = (f.gain ?? null) as Gain | null;
  if (!g) return '–';
  if (g.mean_gain != null && g.ci95) return `${signed(g.mean_gain)} (${signed(g.ci95[0])} to ${signed(g.ci95[1])})`;
  if (g.importance != null) return `share ${(g.importance * 100).toFixed(1)}%`;
  return '–';
}

/**
 * One row per feature: name, English description, source, status, the paired gain with its 95% interval (LLM
 * proposals) or the model's gain share (baseline features), the SQL and what the guards said. Filter by status.
 * Every value is the API's.
 */
export function FeatureLeaderboard({ features }: { features: RunFeature[] }) {
  const [filter, setFilter] = useState<Status | 'all'>('all');
  const counts = useMemo(() => {
    const c = new Map<Status, number>();
    for (const f of features) c.set(statusOf(f), (c.get(statusOf(f)) ?? 0) + 1);
    return c;
  }, [features]);
  const rows = features.filter((f) => filter === 'all' || statusOf(f) === filter);
  return (
    <div data-testid="leaderboard" className="space-y-2">
      <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Filter by status">
        <button
          onClick={() => setFilter('all')}
          className={`border px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.1em] ${filter === 'all' ? 'border-copper text-copper-2' : 'border-line text-mute'}`}
        >
          all {features.length}
        </button>
        {[...counts.entries()].map(([s, n]) => (
          <button
            key={s}
            onClick={() => setFilter(s)}
            className={`border px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.1em] ${filter === s ? 'border-copper text-copper-2' : 'border-line text-mute'}`}
          >
            {LABEL[s]} {n}
          </button>
        ))}
      </div>
      <div className="max-h-[28rem] overflow-auto border border-line">
        <table className="w-full font-mono text-[11px]">
          <thead className="sticky top-0 bg-ink text-left text-mute">
            <tr>
              <th className="px-2 py-1">feature</th>
              <th className="px-2 py-1">source</th>
              <th className="px-2 py-1">status</th>
              <th className="px-2 py-1">gain (PR-AUC, 95% interval)</th>
            </tr>
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr>
                <td colSpan={4} className="px-2 py-3 text-mute">
                  Nothing yet.
                </td>
              </tr>
            )}
            {rows.map((f) => {
              const s = statusOf(f);
              const stage = (f.guard_results ?? {}) as {
                stage?: string | null;
                reasons?: string[];
              };
              return (
                <tr key={f.id} data-testid="feature-row" data-status={s} className="border-t border-line align-top text-bone-dim">
                  <td className="px-2 py-1.5">
                    <div className="text-bone">{f.name}</div>
                    {f.description && <div className="text-[11px] text-mute">{f.description}</div>}
                    {(f.sql || (stage.reasons?.length ?? 0) > 0) && (
                      <details className="mt-1">
                        <summary className="cursor-pointer text-[10px] uppercase tracking-[0.1em] text-mute">SQL and checks</summary>
                        {f.sql && <pre className="mt-1 max-w-xl overflow-x-auto border border-line p-2 text-[10.5px]">{f.sql}</pre>}
                        {stage.stage && (
                          <p className="mt-1 text-clay">
                            stopped at the {stage.stage} check: {stage.reasons?.[0]}
                          </p>
                        )}
                      </details>
                    )}
                  </td>
                  <td className="px-2 py-1.5">{SOURCE[f.kind] ?? f.kind}</td>
                  <td className="px-2 py-1.5">
                    <Tag tone={TONE[s]}>{LABEL[s]}</Tag>
                  </td>
                  <td className="px-2 py-1.5 tabular-nums">{gainText(f)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
