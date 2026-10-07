import type { NarrationItem } from '../api/runs';

type Step = { label: string; value: number; accepted: boolean };

/** The champion's validation PR-AUC after each round, read from the run's own events. */
export function championSteps(items: NarrationItem[]): {
  steps: Step[];
  baseRate: number | null;
} {
  const steps: Step[] = [];
  let baseRate: number | null = null;
  for (const it of items) {
    const p = it.payload as Record<string, unknown>;
    if (it.type === 'baseline' && typeof p.val_pr_auc === 'number') {
      steps.push({ label: 'baseline', value: p.val_pr_auc, accepted: true });
      baseRate = typeof p.base_rate === 'number' ? p.base_rate : null;
    } else if (it.type === 'feature_decision' && p.status !== 'no_llm' && typeof p.champion_val_pr_auc === 'number') {
      steps.push({
        label: `round ${p.round}${p.accepted ? `: ${p.name}` : ''}`,
        value: p.champion_val_pr_auc,
        accepted: p.accepted === true,
      });
    }
  }
  return { steps, baseRate };
}

/** Validation PR-AUC over rounds; a marker where a feature was accepted. The dashed line is the base rate. */
export function ChampionPath({ items }: { items: NarrationItem[] }) {
  const { steps, baseRate } = championSteps(items);
  if (steps.length === 0) return <p className="text-[12px] text-mute">The baseline is not trained yet.</p>;
  const W = 520;
  const H = 120;
  const pad = 8;
  const values = [...steps.map((s) => s.value), ...(baseRate != null ? [baseRate] : [])];
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const span = hi - lo || 1;
  const x = (i: number) => pad + (steps.length === 1 ? (W - 2 * pad) / 2 : (i / (steps.length - 1)) * (W - 2 * pad));
  const y = (v: number) => H - pad - ((v - lo) / span) * (H - 2 * pad);
  // a step line: the champion keeps its value until a feature is accepted
  const path = steps.map((s, i) => `${i === 0 ? 'M' : 'L'}${x(i)},${y(s.value)}`).join(' ');
  return (
    <div data-testid="champion-path" className="space-y-2">
      <svg viewBox={`0 0 ${W} ${H}`} className="h-32 w-full" role="img" aria-label="Validation PR-AUC of the champion over rounds">
        {baseRate != null && <line x1={pad} x2={W - pad} y1={y(baseRate)} y2={y(baseRate)} stroke="#a08d76" strokeDasharray="4 4" strokeWidth="1" />}
        <path d={path} fill="none" stroke="#e0a24f" strokeWidth="1.6" />
        {steps.map((s, i) => (
          <circle key={`${s.label}-${i}`} cx={x(i)} cy={y(s.value)} r={s.accepted ? 4 : 2} fill={s.accepted ? '#b2cd8b' : '#a08d76'}>
            <title>{`${s.label}: ${s.value.toFixed(4)}`}</title>
          </circle>
        ))}
      </svg>
      <ol className="space-y-0.5 font-mono text-[11px] text-bone-dim">
        {steps.map((s, i) => (
          <li key={`${s.label}-${i}`} data-testid="champion-step" className="flex justify-between gap-3">
            <span className={s.accepted ? 'text-bone' : 'text-mute'}>{s.label}</span>
            <span className="tabular-nums">{s.value.toFixed(4)}</span>
          </li>
        ))}
        {baseRate != null && (
          <li className="flex justify-between gap-3 text-mute">
            <span>base rate (validation)</span>
            <span className="tabular-nums">{baseRate.toFixed(4)}</span>
          </li>
        )}
      </ol>
    </div>
  );
}
