import type { LabelPreview as Preview } from '../api/tasks';
import { Tag } from '../ui';

const pct = (x: number | null | undefined) => (x == null ? '–' : `${(x * 100).toFixed(1)}%`);
const day = (iso: string) => iso.slice(0, 10);

/**
 * The labels a spec would produce: rows and feasibility at the top, one bar row per cutoff (eligible
 * entities, and the share that is positive for a binary task), cutoffs left out, and the label SQL.
 */
export function LabelPreview({ preview }: { preview: Preview }) {
  const regression = preview.cutoffs[0]?.mean_label != null;
  const widest = Math.max(1, ...preview.cutoffs.map((c) => c.eligible));
  return (
    <div data-testid="task-preview" className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <div className="font-mono text-[10px] uppercase tracking-[0.1em] text-mute">
          Labels: {preview.total_rows.toLocaleString()} rows, {preview.cutoffs.length} cutoffs
        </div>
        <Tag tone={preview.feasibility.status === 'block' ? 'clay' : preview.feasibility.status === 'warn' ? 'copper' : 'sage'}>
          feasibility: {preview.feasibility.status}
        </Tag>
      </div>
      {preview.feasibility.reasons.length > 0 && (
        <ul className="list-disc pl-5 text-[12px] text-bone-dim">
          {preview.feasibility.reasons.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ul>
      )}
      <div className="max-h-64 overflow-auto border border-line">
        <table className="w-full font-mono text-[11px]">
          <thead className="sticky top-0 bg-ink text-left text-mute">
            <tr>
              <th className="px-2 py-1">cutoff</th>
              <th className="px-2 py-1">{regression ? 'size' : 'share positive'}</th>
              <th className="px-2 py-1 text-right">eligible</th>
              <th className="px-2 py-1 text-right">{regression ? 'mean label' : 'positives'}</th>
              <th className="px-2 py-1 text-right">{regression ? '' : 'balance'}</th>
            </tr>
          </thead>
          <tbody>
            {preview.cutoffs.map((c) => (
              <tr key={c.cutoff} className="border-t border-line text-bone-dim">
                <td className="px-2 py-1">{day(c.cutoff)}</td>
                <td className="w-40 px-2 py-1">
                  <div className="h-2 bg-line" title={`${c.eligible} eligible`}>
                    <div className="relative h-2 bg-mute/60" style={{ width: `${(c.eligible / widest) * 100}%` }}>
                      {!regression && c.eligible > 0 && (
                        <div className="absolute inset-y-0 left-0 bg-copper" style={{ width: `${((c.positives ?? 0) / c.eligible) * 100}%` }} />
                      )}
                    </div>
                  </div>
                </td>
                <td className="px-2 py-1 text-right">{c.eligible.toLocaleString()}</td>
                <td className="px-2 py-1 text-right">{regression ? c.mean_label?.toFixed(2) : (c.positives ?? 0).toLocaleString()}</td>
                <td className="px-2 py-1 text-right">{regression ? '' : pct(c.base_rate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {preview.dropped_cutoffs.length > 0 && (
        <p className="text-[12px] text-copper-2">{preview.dropped_cutoffs.length} cutoffs left out: their label window is not complete in the data.</p>
      )}
      <details>
        <summary className="cursor-pointer font-mono text-[11px] uppercase tracking-[0.1em] text-mute">The label SQL</summary>
        <pre className="mt-2 overflow-x-auto border border-line p-3 font-mono text-[11px] text-bone-dim">{preview.sql}</pre>
      </details>
    </div>
  );
}
