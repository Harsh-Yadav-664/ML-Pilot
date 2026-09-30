import { useEffect, useState } from 'react';
import { getExperimentTree } from '../api';
import { Experiment } from '../types';
import { CheckCircle2, XCircle, Clock, ArrowUp, ArrowDown } from 'lucide-react';

export const ExperimentResultsPanel = ({ refreshTrigger }: { refreshTrigger: number }) => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [compareMode, setCompareMode] = useState(false);

  useEffect(() => {
    const fetchExps = async () => {
      setLoading(true);
      const data = await getExperimentTree();
      setExperiments(data);
      setLoading(false);
    };
    fetchExps();
  }, [refreshTrigger]);

  if (loading && experiments.length === 0) {
    return (
      <div className="mt-8 flex h-48 items-center justify-center rounded-xl border border-zinc-800/80 bg-zinc-900/20">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-zinc-700 border-t-emerald-400" />
      </div>
    );
  }

  if (experiments.length === 0) {
    return (
      <div className="mt-8 flex h-48 items-center justify-center rounded-xl border border-zinc-800/80 bg-zinc-900/20 text-sm text-zinc-500">
        No experiments run yet.
      </div>
    );
  }

  const baseline = experiments.find(e => !e.parent_id) || experiments[0];
  const sorted = [...experiments].sort((a, b) => {
    const f1A = a.metrics?.f1 ?? -1;
    const f1B = b.metrics?.f1 ?? -1;
    return f1B - f1A;
  });

  const getDecision = (exp: Experiment) => {
    if (exp.status === 'queued' || exp.status === 'running') return 'PENDING';
    if (exp.status === 'failed') return 'REJECT';
    if (exp.id === baseline.id) return 'BASELINE';
    const baseF1 = baseline.metrics?.f1 ?? 0;
    const expF1 = exp.metrics?.f1 ?? 0;
    return expF1 > baseF1 ? 'KEEP' : 'REJECT';
  };

  const getDecisionColor = (decision: string) => {
    switch (decision) {
      case 'KEEP': return 'text-emerald-400 bg-emerald-400/10 ring-emerald-400/20';
      case 'REJECT': return 'text-red-400 bg-red-400/10 ring-red-400/20';
      case 'PENDING': return 'text-yellow-400 bg-yellow-400/10 ring-yellow-400/20';
      case 'BASELINE': return 'text-indigo-400 bg-indigo-400/10 ring-indigo-400/20';
      default: return 'text-zinc-400 bg-zinc-400/10 ring-zinc-400/20';
    }
  };

  const renderDelta = (val: number | undefined, baseVal: number | undefined, higherIsBetter = true) => {
    if (!compareMode || val === undefined || baseVal === undefined) return null;
    const delta = val - baseVal;
    if (Math.abs(delta) < 0.001) return null;
    const isPositive = delta > 0;
    const isGood = isPositive === higherIsBetter;
    
    return (
      <span className={`ml-2 text-[10px] flex items-center gap-0.5 ${isGood ? 'text-emerald-400' : 'text-red-400'}`}>
        {isPositive ? <ArrowUp className="w-3 h-3" /> : <ArrowDown className="w-3 h-3" />}
        {Math.abs(delta).toFixed(3)}
      </span>
    );
  };

  return (
    <div className="mt-8 rounded-xl border border-zinc-800/80 bg-zinc-900/20 overflow-hidden">
      <div className="flex items-center justify-between border-b border-zinc-800/80 px-4 py-3 bg-zinc-900/50">
        <h3 className="text-sm font-medium text-zinc-200">Experiment Leaderboard</h3>
        <button 
          onClick={() => setCompareMode(!compareMode)}
          className={`px-3 py-1 text-xs rounded-md border transition-colors ${compareMode ? 'bg-indigo-500/20 border-indigo-500/30 text-indigo-300' : 'bg-zinc-800/50 border-zinc-700/50 text-zinc-400 hover:bg-zinc-800 hover:text-zinc-300'}`}
        >
          {compareMode ? 'Comparing to Baseline' : 'Compare to Baseline'}
        </button>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="bg-zinc-900/30 text-[11px] uppercase tracking-wider text-zinc-500 border-b border-zinc-800/80">
            <tr>
              <th className="px-4 py-3 font-medium">Feature / Model Name</th>
              <th className="px-4 py-3 font-medium">F1 Score</th>
              <th className="px-4 py-3 font-medium">Accuracy</th>
              <th className="px-4 py-3 font-medium">Runtime</th>
              <th className="px-4 py-3 font-medium">Status</th>
              <th className="px-4 py-3 font-medium text-right">Decision</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-800/80">
            {sorted.map(exp => {
              const decision = getDecision(exp);
              const f1 = exp.metrics?.f1;
              const acc = exp.metrics?.accuracy;
              const baseF1 = baseline.metrics?.f1;
              const baseAcc = baseline.metrics?.accuracy;
              
              return (
                <tr key={exp.id} className={`hover:bg-zinc-800/20 transition-colors ${exp.id === baseline.id ? 'bg-indigo-500/5 hover:bg-indigo-500/10' : ''}`}>
                  <td className="px-4 py-3">
                    <div className="flex flex-col">
                      <span className="font-medium text-zinc-200">{exp.model_name}</span>
                      <span className="text-[10px] font-mono text-zinc-500">{exp.id}</span>
                    </div>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <div className="flex items-center">
                      <span className="font-mono text-zinc-300">{f1 !== undefined ? f1.toFixed(4) : '—'}</span>
                      {exp.id !== baseline.id && renderDelta(f1, baseF1, true)}
                    </div>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <div className="flex items-center">
                      <span className="font-mono text-zinc-300">{acc !== undefined ? (acc * 100).toFixed(2) + '%' : '—'}</span>
                      {exp.id !== baseline.id && renderDelta(acc, baseAcc, true)}
                    </div>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-zinc-400 font-mono text-xs">
                    {exp.status === 'queued' || exp.runtime_seconds == null ? '—' : `${exp.runtime_seconds.toFixed(1)}s`}
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap">
                    <span className="text-zinc-400 text-xs capitalize flex items-center gap-1.5">
                      {exp.status === 'running' && <span className="w-2 h-2 rounded-full bg-sky-400 animate-pulse" />}
                      {exp.status === 'completed' && <span className="w-2 h-2 rounded-full bg-emerald-400" />}
                      {exp.status === 'failed' && <span className="w-2 h-2 rounded-full bg-red-400" />}
                      {exp.status === 'queued' && <span className="w-2 h-2 rounded-full bg-zinc-500" />}
                      {exp.status}
                    </span>
                  </td>
                  <td className="px-4 py-3 whitespace-nowrap text-right">
                    <span className={`inline-flex items-center gap-1 rounded-md px-2 py-1 text-[10px] font-bold tracking-wider ring-1 ring-inset ${getDecisionColor(decision)}`}>
                      {decision === 'KEEP' && <CheckCircle2 className="w-3 h-3" />}
                      {decision === 'REJECT' && <XCircle className="w-3 h-3" />}
                      {decision === 'PENDING' && <Clock className="w-3 h-3" />}
                      {decision}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
