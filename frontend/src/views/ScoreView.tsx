import { useState } from 'react';
import { describeError } from '../api';
import { RunScore, downloadScores, scoreRun } from '../api/runs';
import { Btn, CardHeader, Panel } from '../ui';

const TOP_K = 300;

/**
 * Score the entities eligible on one day with the run's champion (backend #62). Nothing here is computed in the
 * browser: the list, the reasons, the summary and the warnings are what the API returned.
 */
export function ScoreView({ runId }: { runId: string }) {
  const [cutoff, setCutoff] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RunScore | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setResult(await scoreRun(runId, cutoff ? `${cutoff}T00:00:00` : null, TOP_K));
    } catch (e) {
      setResult(null);
      setError(describeError(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <Panel>
      <CardHeader title="Score" sub="rank the entities eligible on a day, read from the live database" />
      <div className="space-y-3 p-4">
        <div className="flex flex-wrap items-end gap-3">
          <label className="text-[12px] text-mute">
            Day (empty: last complete day)
            <input
              data-testid="score-cutoff"
              type="date"
              value={cutoff}
              onChange={(e) => setCutoff(e.target.value)}
              className="ml-2 rounded border border-line bg-transparent px-2 py-1 text-bone"
            />
          </label>
          <Btn size="sm" data-testid="score-run" disabled={busy} onClick={() => void run()}>
            {busy ? 'Scoring…' : 'Score'}
          </Btn>
        </div>
        {error && (
          <p data-testid="score-error" className="text-[12.5px] text-clay">
            {error}
          </p>
        )}
        {result && (
          <div data-testid="score-result" className="space-y-3">
            <p className="text-[12.5px] text-bone-dim">
              {Number(result.summary.n_scored).toLocaleString()} entities scored on {result.cutoff.slice(0, 10)}. Sum of the model's
              probabilities in the top {String(result.summary.top_k)}:{' '}
              {Number(result.summary.expected_positives_in_top_k_uncalibrated).toFixed(1)} (not calibrated, so not a forecast).
            </p>
            {result.warnings.map((w) => (
              <p key={`${w.name}-${w.detail}`} data-testid="score-warning" className="text-[12px] text-copper">
                {w.detail}
              </p>
            ))}
            <div className="overflow-auto">
              <table className="w-full font-mono text-[11px]">
                <thead className="text-left text-mute">
                  <tr>
                    <th>rank</th>
                    <th>entity</th>
                    <th>score</th>
                    <th>decile</th>
                    <th>strongest reasons</th>
                  </tr>
                </thead>
                <tbody>
                  {result.preview.map((r) => (
                    <tr key={String(r.entity_id)} className="border-t border-line/60">
                      <td>{String(r.rank)}</td>
                      <td>{String(r.entity_id)}</td>
                      <td className="tabular-nums">{Number(r.score).toFixed(3)}</td>
                      <td>{String(r.decile)}</td>
                      <td className="text-bone-dim">{[r.reason_1, r.reason_2, r.reason_3].filter(Boolean).join('; ')}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <Btn
              variant="ghost"
              size="sm"
              data-testid="score-csv"
              onClick={() => void downloadScores(runId, result.csv).catch((e) => setError(describeError(e)))}
            >
              Download the full list (CSV)
            </Btn>
          </div>
        )}
      </div>
    </Panel>
  );
}
