import { useState, useEffect } from 'react';
import { Sparkles, TriangleAlert, Play, Loader2, CornerDownLeft } from 'lucide-react';
import { getFeatureSuggestions, runExperiment } from '../api';
import { FeatureSuggestion } from '../types';

interface AIAgentPanelProps {
  onExperimentStart?: () => void;
  datasetPath: string;
  targetColumn: string;
}

export const AIAgentPanel = ({ onExperimentStart, datasetPath, targetColumn }: AIAgentPanelProps) => {
  const [suggestions, setSuggestions] = useState<FeatureSuggestion[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState<string | null>(null);
  const [launched, setLaunched] = useState<Set<string>>(new Set());

  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await getFeatureSuggestions(datasetPath, targetColumn);
        if (!res || res.length === 0) {
          setError("The AI returned 0 candidates. This usually means your API key hit a rate limit (like Gemini's free tier) or the AI Gateway could not reach the provider.");
        }
        setSuggestions(res || []);
      } catch (err: any) {
        setError(err.message || "Failed to connect to the AI Gateway. Check your backend logs or API keys.");
        setSuggestions([]);
      }
      setLoading(false);
    })();
  }, [datasetPath, targetColumn]);

  const handleRun = async (s: FeatureSuggestion) => {
    setRunning(s.name);
    try {
      await runExperiment(s, datasetPath);
      setLaunched((prev) => new Set(prev).add(s.name));
      onExperimentStart?.();
    } catch (e) {
      alert("Experiment failed to run. Check backend logs.");
    }
    setRunning(null);
  };

  return (
    <div className="flex h-full flex-col">
      {/* Panel header */}
      <div className="flex items-center gap-3 border-b border-zinc-800/80 px-4 py-3.5">
        <div className="relative">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-400/20 to-emerald-500/5 ring-1 ring-inset ring-emerald-500/30">
            <Sparkles className="h-4 w-4 text-emerald-300" />
          </div>
          <span className="pulse-dot absolute -bottom-0.5 -right-0.5 h-2.5 w-2.5 rounded-full border-2 border-zinc-950 bg-emerald-400" />
        </div>
        <div className="min-w-0">
          <div className="text-sm font-medium text-zinc-100">Data Scientist Agent</div>
          <div className="text-[11px] text-zinc-500">Proposing features · churn_v3</div>
        </div>
      </div>

      {/* Feed */}
      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
        {/* Agent intro message */}
        <div className="flex gap-2.5">
          <div className="mt-0.5 flex h-6 w-6 flex-shrink-0 items-center justify-center rounded-md bg-zinc-800 text-[10px] font-semibold text-emerald-300">
            AI
          </div>
          <div className="rounded-lg rounded-tl-sm border border-zinc-800/80 bg-zinc-900/40 px-3 py-2 text-[13px] leading-relaxed text-zinc-300">
            {error ? (
              <span className="text-red-400">Oops, I ran into an issue: {error}</span>
            ) : (
              <>
                I analyzed the feature space and found{' '}
                <span className="font-medium text-zinc-100">
                  {loading ? '...' : suggestions.length}
                </span>{' '}
                candidates likely to lift F1. Review and dispatch any to the experiment queue.
              </>
            )}
          </div>
        </div>

        {loading &&
          [0, 1].map((i) => (
            <div
              key={i}
              className="h-28 animate-pulse rounded-xl border border-zinc-800/60 bg-zinc-900/30"
            />
          ))}

        {suggestions.map((s) => (
          <div
            key={s.name}
            className="group rounded-xl border border-zinc-800/80 bg-zinc-900/40 transition-colors hover:border-zinc-700"
          >
            <div className="flex items-center justify-between px-3.5 pt-3">
              <div className="flex items-center gap-2">
                <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
                <code className="font-mono text-[13px] font-medium text-zinc-100">{s.name}</code>
              </div>
              <span className="text-[10px] uppercase tracking-wider text-zinc-600">feature</span>
            </div>

            {/* formula */}
            <div className="mx-3.5 mt-2.5 overflow-x-auto rounded-lg border border-zinc-800/80 bg-black/40 px-3 py-2">
              <code className="whitespace-nowrap font-mono text-[12px] text-emerald-300/90">
                <span className="text-zinc-600">λ </span>
                {s.formula}
              </code>
            </div>

            <p className="px-3.5 pt-2.5 text-[12.5px] leading-relaxed text-zinc-400">{s.reason}</p>

            {s.risk && (
              <div className="mx-3.5 mt-2.5 flex items-start gap-2 rounded-lg bg-amber-500/[0.06] px-2.5 py-2">
                <TriangleAlert className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 text-amber-400/80" />
                <span className="text-[12px] leading-relaxed text-amber-200/80">{s.risk}</span>
              </div>
            )}

            <div className="flex items-center justify-between px-3.5 py-3">
              <span className="text-[11px] text-zinc-600">est. +0.02–0.05 F1</span>
              <button
                onClick={() => handleRun(s)}
                disabled={running === s.name || launched.has(s.name)}
                className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[12px] font-medium transition-all ${
                  launched.has(s.name)
                    ? 'bg-zinc-800 text-zinc-500'
                    : 'bg-emerald-500/90 text-emerald-950 hover:bg-emerald-400 disabled:opacity-60'
                }`}
              >
                {running === s.name ? (
                  <>
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                    Dispatching
                  </>
                ) : launched.has(s.name) ? (
                  'Queued ✓'
                ) : (
                  <>
                    <Play className="h-3 w-3 fill-current" />
                    Run experiment
                  </>
                )}
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Composer */}
      <div className="border-t border-zinc-800/80 p-3">
        <div className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-2 focus-within:border-zinc-700">
          <input
            type="text"
            placeholder="Ask the agent for a feature idea…"
            className="flex-1 bg-transparent text-[13px] text-zinc-200 placeholder:text-zinc-600 focus:outline-none"
          />
          <kbd className="flex items-center gap-1 rounded border border-zinc-700 px-1.5 py-0.5 text-[10px] text-zinc-500">
            <CornerDownLeft className="h-3 w-3" />
          </kbd>
        </div>
      </div>
    </div>
  );
};
