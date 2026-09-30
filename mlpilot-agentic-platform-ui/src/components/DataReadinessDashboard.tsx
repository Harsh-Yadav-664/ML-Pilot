import { useState, useEffect } from 'react';
import {
  AlertTriangle,
  ShieldCheck,
  ArrowUpRight,
  Rows3,
  Columns3,
  CircleDashed,
  CopyMinus,
} from 'lucide-react';
import { getDataMetrics, getLeakageWarnings } from '../api';
import { DataMetrics, LeakageWarning } from '../types';

const nf = new Intl.NumberFormat('en-US');

interface DataReadinessDashboardProps {
  datasetPath: string;
  targetColumn: string;
}

export const DataReadinessDashboard = ({ datasetPath, targetColumn }: DataReadinessDashboardProps) => {
  const [metrics, setMetrics] = useState<DataMetrics | null>(null);
  const [warnings, setWarnings] = useState<LeakageWarning[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      setLoading(true);
      const [m, w] = await Promise.all([
        getDataMetrics(datasetPath, targetColumn), 
        getLeakageWarnings(datasetPath, targetColumn)
      ]);
      setMetrics(m);
      setWarnings(w);
      setLoading(false);
    })();
  }, [datasetPath, targetColumn]);

  const highCount = warnings.filter((w) => w.severity === 'high').length;

  const cells = [
    {
      label: 'Rows',
      value: metrics ? nf.format(metrics.total_rows) : '—',
      icon: Rows3,
      sub: 'customer records',
    },
    {
      label: 'Columns',
      value: metrics ? nf.format(metrics.total_columns) : '—',
      icon: Columns3,
      sub: '38 numeric · 4 categorical',
    },
    {
      label: 'Missing',
      value: metrics ? `${(metrics.missing_data_percent * 100).toFixed(2)}%` : '—',
      icon: CircleDashed,
      sub: 'across all cells',
      tone: metrics && metrics.missing_data_percent > 0.05 ? 'warn' : 'ok',
    },
    {
      label: 'Duplicates',
      value: metrics ? nf.format(metrics.duplicate_rows) : '—',
      icon: CopyMinus,
      sub: 'exact row matches',
      tone: metrics && metrics.duplicate_rows > 0 ? 'warn' : 'ok',
    },
  ] as const;

  return (
    <section className="space-y-4">
      {/* Metrics strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 divide-x divide-y lg:divide-y-0 divide-zinc-800/80 rounded-xl border border-zinc-800/80 bg-zinc-900/30 overflow-hidden">
        {cells.map((c) => {
          const Icon = c.icon;
          const warn = 'tone' in c && c.tone === 'warn';
          return (
            <div key={c.label} className="p-4 sm:p-5 group">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-medium uppercase tracking-[0.14em] text-zinc-500">
                  {c.label}
                </span>
                <Icon
                  className={`h-3.5 w-3.5 ${
                    warn ? 'text-amber-400/80' : 'text-zinc-600 group-hover:text-zinc-400'
                  } transition-colors`}
                />
              </div>
              <div
                className={`mt-2 font-mono text-2xl font-semibold tracking-tight ${
                  loading ? 'text-zinc-700 animate-pulse' : warn ? 'text-amber-200' : 'text-zinc-50'
                }`}
              >
                {loading ? '····' : c.value}
              </div>
              <div className="mt-1 text-[11px] text-zinc-600">{c.sub}</div>
            </div>
          );
        })}
      </div>

      {/* Leakage warnings */}
      <div className="rounded-xl border border-zinc-800/80 bg-zinc-900/30">
        <div className="flex items-center justify-between border-b border-zinc-800/80 px-4 py-3">
          <div className="flex items-center gap-2">
            {highCount > 0 ? (
              <AlertTriangle className="h-4 w-4 text-red-400" />
            ) : (
              <ShieldCheck className="h-4 w-4 text-emerald-400" />
            )}
            <h3 className="text-sm font-medium text-zinc-200">Target leakage scan</h3>
          </div>
          <span
            className={`rounded-full px-2 py-0.5 text-[11px] font-medium ${
              highCount > 0
                ? 'bg-red-500/10 text-red-300 ring-1 ring-inset ring-red-500/20'
                : 'bg-emerald-500/10 text-emerald-300 ring-1 ring-inset ring-emerald-500/20'
            }`}
          >
            {loading ? 'scanning…' : `${warnings.length} flagged`}
          </span>
        </div>

        <div className="divide-y divide-zinc-800/60">
          {warnings.map((w) => (
            <div key={w.id} className="flex items-start gap-3 px-4 py-3.5 hover:bg-zinc-800/20">
              <span
                className={`mt-1.5 h-1.5 w-1.5 flex-shrink-0 rounded-full ${
                  w.severity === 'high'
                    ? 'bg-red-400'
                    : w.severity === 'medium'
                    ? 'bg-amber-400'
                    : 'bg-sky-400'
                }`}
              />
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <code className="font-mono text-[13px] text-zinc-100">{w.column}</code>
                  <span
                    className={`rounded px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide ${
                      w.severity === 'high'
                        ? 'bg-red-500/10 text-red-300'
                        : w.severity === 'medium'
                        ? 'bg-amber-500/10 text-amber-300'
                        : 'bg-sky-500/10 text-sky-300'
                    }`}
                  >
                    {w.severity}
                  </span>
                </div>
                <p className="mt-1 text-[13px] leading-relaxed text-zinc-400">
                  {w.message}
                </p>
                <div className="mt-2 rounded bg-black/20 p-2 text-[12px] text-zinc-500">
                  <span className="font-semibold text-zinc-400">Why it matters:</span> Target leakage happens when the model learns to "cheat" using a column that wouldn't actually be available in the real world at prediction time, or using IDs that let it memorize individual rows instead of learning patterns. Dropping it prevents fake high accuracy.
                </div>
              </div>
              <button className="flex flex-shrink-0 items-center gap-1 rounded-md px-2 py-1 text-[11px] text-zinc-500 hover:bg-zinc-800 hover:text-zinc-200">
                Drop
                <ArrowUpRight className="h-3 w-3" />
              </button>
            </div>
          ))}
          {!loading && warnings.length === 0 && (
            <div className="px-4 py-6 text-center text-[13px] text-zinc-500">
              No leakage detected. Feature set is clean.
            </div>
          )}
        </div>
      </div>
    </section>
  );
};
