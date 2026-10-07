import type { LabelPreview, SplitPreview } from '../api/runs';
import { Tag } from '../ui';

const day = (iso: string) => iso.slice(0, 10);
const pct = (x: number | null | undefined) => (x == null ? '–' : `${(x * 100).toFixed(1)}%`);

const PART: Record<SplitPreview['timeline'][number]['part'], { label: string; bar: string; tone: 'sage' | 'copper' | 'clay' | 'neutral' }> = {
  train: { label: 'train', bar: 'bg-sage/70', tone: 'sage' },
  val: { label: 'validation', bar: 'bg-copper/80', tone: 'copper' },
  test: { label: 'test', bar: 'bg-clay/80', tone: 'clay' },
  purged_train: {
    label: 'left out (crosses validation)',
    bar: 'bg-mute/40',
    tone: 'neutral',
  },
  purged_val: {
    label: 'left out (crosses test)',
    bar: 'bg-mute/40',
    tone: 'neutral',
  },
};

/**
 * The cutoffs of the task in time order, each marked train, validation, test or left out, with its rows and
 * (binary tasks) its base rate. The counts are the backend's own (`preview-split`, `preview-labels`).
 */
export function SplitTimeline({ split, labels }: { split: SplitPreview; labels: LabelPreview | null }) {
  const base = new Map((labels?.cutoffs ?? []).map((c) => [day(c.cutoff), c.base_rate] as const));
  const widest = Math.max(1, ...split.timeline.map((t) => t.rows));
  return (
    <div data-testid="split-timeline" className="space-y-2">
      <div className="flex flex-wrap gap-2 font-mono text-[10px] text-mute">
        <Tag tone="sage">train {split.n_train.toLocaleString()} rows</Tag>
        <Tag tone="copper">
          validation {split.n_val.toLocaleString()} rows (from {day(split.val_from)})
        </Tag>
        <Tag tone="clay">
          test {split.n_test.toLocaleString()} rows (from {day(split.test_from)})
        </Tag>
        {split.n_purged_train + split.n_purged_val > 0 && <Tag>left out {(split.n_purged_train + split.n_purged_val).toLocaleString()}</Tag>}
      </div>
      <div className="max-h-56 overflow-auto border border-line">
        <table className="w-full font-mono text-[11px]">
          <thead className="sticky top-0 bg-ink text-left text-mute">
            <tr>
              <th className="px-2 py-1">cutoff</th>
              <th className="px-2 py-1">part</th>
              <th className="w-40 px-2 py-1">rows</th>
              <th className="px-2 py-1 text-right">base rate</th>
            </tr>
          </thead>
          <tbody>
            {split.timeline.map((t) => (
              <tr key={t.cutoff} className="border-t border-line text-bone-dim">
                <td className="px-2 py-1">{day(t.cutoff)}</td>
                <td className="px-2 py-1">{PART[t.part].label}</td>
                <td className="px-2 py-1">
                  <div className="h-2 bg-line" title={`${t.rows} rows`}>
                    <div className={`h-2 ${PART[t.part].bar}`} style={{ width: `${(t.rows / widest) * 100}%` }} />
                  </div>
                </td>
                <td className="px-2 py-1 text-right">{pct(base.get(day(t.cutoff)))}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="text-[12px] text-mute">{split.note}</p>
    </div>
  );
}
