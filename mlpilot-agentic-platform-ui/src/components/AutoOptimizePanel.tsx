import { useState, useEffect } from 'react';
import { Zap, CheckCircle2, AlertTriangle, Loader2 } from 'lucide-react';
import { runAutoOptimize, getOptimizeStatus } from '../api';

interface AutoOptimizePanelProps {
  datasetPath: string;
  targetColumn: string;
  onComplete?: () => void;
}

export const AutoOptimizePanel = ({ datasetPath, targetColumn, onComplete }: AutoOptimizePanelProps) => {
  const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'failed'>('idle');
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (status === 'running' && jobId) {
      interval = setInterval(async () => {
        try {
          const res = await getOptimizeStatus(jobId);
          if (res.status === 'completed') {
            setStatus('completed');
            setResult(res.result);
            clearInterval(interval);
            onComplete?.();
          } else if (res.status === 'failed') {
            setStatus('failed');
            setError(res.error || 'Unknown error');
            clearInterval(interval);
          }
        } catch (e: any) {
          console.error("Poll failed", e);
        }
      }, 3000);
    }
    return () => clearInterval(interval);
  }, [status, jobId, onComplete]);

  const handleRun = async () => {
    try {
      setStatus('running');
      setError(null);
      setResult(null);
      const res = await runAutoOptimize(datasetPath, targetColumn, 5);
      setJobId(res.job_id);
    } catch (e: any) {
      setStatus('failed');
      setError(e.message);
    }
  };

  return (
    <div className="border border-zinc-800/80 bg-zinc-900/40 rounded-xl p-4 m-4">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h3 className="text-zinc-100 font-medium flex items-center gap-2">
            <Zap className="w-4 h-4 text-emerald-400" />
            Auto-Optimize
          </h3>
          <p className="text-zinc-400 text-sm mt-1">
            Let the agent autonomously generate, execute, and evaluate experiments.
          </p>
        </div>
        {status === 'idle' && (
          <button 
            onClick={handleRun}
            className="flex items-center gap-2 bg-emerald-500/90 hover:bg-emerald-400 text-emerald-950 font-medium px-4 py-2 rounded-lg text-sm transition-colors"
          >
            <Zap className="w-4 h-4 fill-current" />
            Run Full Autopilot
          </button>
        )}
      </div>

      {status === 'running' && (
        <div className="flex items-center gap-3 text-emerald-400 text-sm p-4 bg-emerald-500/10 rounded-lg border border-emerald-500/20">
          <Loader2 className="w-4 h-4 animate-spin" />
          Agent is running 5 experiments autonomously...
        </div>
      )}

      {status === 'failed' && (
        <div className="flex items-center gap-3 text-red-400 text-sm p-4 bg-red-500/10 rounded-lg border border-red-500/20">
          <AlertTriangle className="w-4 h-4" />
          Failed: {error}
        </div>
      )}

      {status === 'completed' && result && (
        <div className="space-y-4">
          <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
            <div className="flex items-center gap-2 text-emerald-400 font-medium mb-2">
              <CheckCircle2 className="w-4 h-4" />
              Optimization Complete
            </div>
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div className="bg-black/40 p-3 rounded-lg border border-zinc-800/80">
                <div className="text-zinc-500 text-xs uppercase tracking-wider mb-1">Best F1 Score</div>
                <div className="text-2xl font-semibold text-zinc-100">{result.best_f1?.toFixed(4) || 0}</div>
              </div>
              <div className="bg-black/40 p-3 rounded-lg border border-zinc-800/80">
                <div className="text-zinc-500 text-xs uppercase tracking-wider mb-1">Winner Features</div>
                <div className="text-sm text-zinc-300">
                  {result.winner_features?.length > 0 
                    ? result.winner_features.join(', ') 
                    : 'None'}
                </div>
              </div>
            </div>
          </div>
          <div className="bg-black/40 p-3 rounded-lg border border-zinc-800/80">
            <div className="text-zinc-500 text-xs uppercase tracking-wider mb-2">AI Summary</div>
            <p className="text-sm text-zinc-300 leading-relaxed">
              {result.summary}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
