import { useCallback, useEffect, useRef, useState } from 'react';
import { describeError } from '../api';
import {
  Checkpoint,
  LabelPreview,
  NarrationItem,
  RunFeature,
  RunState,
  SplitPreview,
  cancelRun,
  downloadBundle,
  openReport,
  getCheckpoints,
  getNarration,
  getRun,
  getRunFeatures,
  previewLabels,
  previewSplit,
} from '../api/runs';
import { DbVersion, listDbVersions } from '../api/snapshots';
import { ChampionPath } from '../components/ChampionPath';
import { FeatureLeaderboard } from '../components/FeatureLeaderboard';
import { SplitTimeline } from '../components/SplitTimeline';
import { SteerPanel } from '../components/SteerPanel';
import { leaveRoute, runIdFromHash } from '../route';
import { ScoreView } from './ScoreView';
import { Btn, CardHeader, Eyebrow, Panel, Tag } from '../ui';

const POLL_MS = 1500;
const LIVE = ['queued', 'running'];
const day = (iso: string) => iso.slice(0, 10);

const TONE: Record<string, 'sage' | 'copper' | 'clay' | 'neutral'> = {
  created: 'neutral',
  queued: 'copper',
  running: 'copper',
  completed: 'sage',
  stopped: 'sage',
  failed: 'clay',
  cancelled: 'clay',
};

function MetricList({ metrics, testId }: { metrics: Record<string, number>; testId: string }) {
  const entries = Object.entries(metrics).filter(([k]) => !['best_iteration'].includes(k));
  return (
    <dl data-testid={testId} className="grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-[11px]">
      {entries.map(([k, v]) => (
        <div key={k} className="flex justify-between gap-3 border-b border-line/60 py-0.5">
          <dt className="text-mute">{k}</dt>
          <dd data-metric={k} className="tabular-nums text-bone">
            {k.startsWith('n_') ? v.toLocaleString() : v.toFixed(4)}
          </dd>
        </div>
      ))}
    </dl>
  );
}

/**
 * A relational run on one page: the task in words, the data it reads, the split, every feature that was tried
 * and why it was kept or not, the champion's path, a running log in words, and the test score once the run ends.
 * Every number on the page is read from the API (rule 5); nothing is computed or invented here.
 */
export function RunView() {
  const runId = runIdFromHash();
  const [run, setRun] = useState<RunState | null>(null);
  const [features, setFeatures] = useState<RunFeature[]>([]);
  const [narration, setNarration] = useState<NarrationItem[]>([]);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [split, setSplit] = useState<SplitPreview | null>(null);
  const [labels, setLabels] = useState<LabelPreview | null>(null);
  const [version, setVersion] = useState<DbVersion | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [splitError, setSplitError] = useState<string | null>(null);
  const loadedTraining = useRef(false);

  const refresh = useCallback(async () => {
    try {
      const state = await getRun(runId);
      const [f, n, c] = await Promise.all([getRunFeatures(runId), getNarration(runId), getCheckpoints(runId)]);
      setRun(state);
      setFeatures(f);
      setNarration(n);
      setCheckpoints(c);
      setError(null);
      return state;
    } catch (e) {
      setError(describeError(e));
      return null;
    }
  }, [runId]);

  // Poll while the run is live, once more when it has ended.
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      const state = await refresh();
      if (stopped) return;
      if (state === null || LIVE.includes(state.status) || state.status === 'created') timer = setTimeout(() => void tick(), POLL_MS);
    };
    void tick();
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [refresh]);

  // The training table and the data version do not change while the run goes on: load them once.
  useEffect(() => {
    if (!run?.task_id || loadedTraining.current) return;
    loadedTraining.current = true;
    const taskId = run.task_id;
    const versionId = run.data_version_id ?? null;
    void Promise.all([previewSplit(taskId, versionId), previewLabels(taskId, versionId)])
      .then(([s, l]) => {
        setSplit(s);
        setLabels(l);
      })
      .catch((e) => setSplitError(describeError(e)));
    void listDbVersions()
      .then((all) => setVersion(all.find((v) => v.id === versionId) ?? null))
      .catch(() => setVersion(null));
  }, [run?.task_id, run?.data_version_id]);

  const live = run !== null && LIVE.includes(run.status);
  const ended = run !== null && !live && run.status !== 'created';
  const lastDecision = [...narration].reverse().find((n) => n.type === 'feature_decision');
  const usedNow =
    (run && Object.keys(run.budget_used).length > 0
      ? run.budget_used
      : (lastDecision?.payload as { budget_used?: Record<string, number> } | undefined)?.budget_used) ?? {};
  const budget = run?.budget as Record<string, number | string | null> | undefined;

  return (
    <div className="relative min-h-screen overflow-y-auto text-bone">
      <div className="guides" aria-hidden />
      <header className="mx-auto flex h-14 max-w-[1400px] items-center justify-between px-6">
        <Btn variant="ghost" size="sm" onClick={leaveRoute}>
          ← Back
        </Btn>
        {run && (
          <div className="flex items-center gap-2" data-testid="run-status" data-status={run.status}>
            <Tag tone={TONE[run.status] ?? 'neutral'}>{run.status}</Tag>
            {run.stop_reason && <Tag>{run.stop_reason}</Tag>}
          </div>
        )}
      </header>
      <main className="mx-auto max-w-[1400px] space-y-6 px-6 pb-16">
        {error && (
          <div role="alert" data-testid="run-error" className="border border-clay/50 bg-clay/5 p-3 text-[13px] text-bone">
            {error}
          </div>
        )}
        {!run && !error && <p className="text-mute">Loading the run…</p>}
        {run && (
          <>
            <section>
              <Eyebrow className="text-copper">Relational run</Eyebrow>
              <h1 data-testid="run-task" className="mt-1 font-display text-[40px] font-extrabold uppercase leading-none tracking-wide text-bone">
                {run.task_name ?? 'Run'}
              </h1>
              {run.question && <p className="mt-3 max-w-3xl text-[14px] leading-relaxed text-mute">“{run.question}”</p>}
              <div className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 font-mono text-[11px] text-bone-dim">
                <span>
                  data: {version ? `${version.mode} ${version.short_hash}` : run.data_version_id?.slice(0, 12)}
                  {run.as_of ? `, as of ${day(run.as_of)}` : ''}
                </span>
                <span data-testid="run-budget">
                  rounds {run.rounds} ({run.accepted} accepted)
                  {usedNow.cost_usd != null ? ` · model cost $${Number(usedNow.cost_usd).toFixed(4)}` : ''}
                  {usedNow.seconds != null ? ` · ${Number(usedNow.seconds).toFixed(0)}s` : ''}
                  {budget?.max_cost_usd != null ? ` (limit $${Number(budget.max_cost_usd).toFixed(2)})` : ''}
                </span>
                {live && run.job_id && (
                  <Btn
                    size="sm"
                    variant="danger"
                    onClick={() => {
                      void cancelRun(run.job_id as string).catch((e) => setError(describeError(e)));
                    }}
                  >
                    Cancel
                  </Btn>
                )}
              </div>
              {run.error && (
                <p data-testid="run-failure" className="mt-3 border border-clay/50 bg-clay/5 p-3 text-[13px] text-bone">
                  {run.error}
                </p>
              )}
            </section>

            <div className="grid gap-6 lg:grid-cols-2">
              <Panel>
                <CardHeader title="Training table" sub="time-based split, never random" />
                <div className="space-y-3 p-4">
                  {splitError && <p className="text-[12px] text-clay">{splitError}</p>}
                  {!split && !splitError && <p className="text-[12px] text-mute">Building the cutoffs…</p>}
                  {split && <SplitTimeline split={split} labels={labels} />}
                  {labels && (
                    <div className="flex flex-wrap items-center gap-2" data-testid="run-feasibility">
                      <Tag tone={labels.feasibility.status === 'block' ? 'clay' : labels.feasibility.status === 'warn' ? 'copper' : 'sage'}>
                        feasibility: {labels.feasibility.status}
                      </Tag>
                      <span className="font-mono text-[11px] text-mute">{labels.total_rows.toLocaleString()} label rows</span>
                    </div>
                  )}
                </div>
              </Panel>
              <Panel>
                <CardHeader title="Champion path" sub="validation PR-AUC after each round" />
                <div className="p-4">
                  <ChampionPath items={narration} />
                </div>
              </Panel>
            </div>

            <Panel>
              <CardHeader title="Features" sub="what was tried, what was kept and why" />
              <div className="p-4">
                <FeatureLeaderboard features={features} />
              </div>
            </Panel>

            <div className="grid gap-6 lg:grid-cols-2">
              <Panel>
                <CardHeader title="Log" sub="the run's events, in words" />
                <ol data-testid="run-log" className="max-h-80 space-y-1.5 overflow-auto p-4 text-[12.5px] leading-snug text-bone-dim">
                  {narration.map((n) => (
                    <li key={`${n.type}-${n.seq}-${n.at}`} data-event={n.type}>
                      {n.text}
                    </li>
                  ))}
                  {narration.length === 0 && <li className="text-mute">Nothing yet.</li>}
                </ol>
              </Panel>
              <SteerPanel runId={runId} checkpoints={checkpoints} live={live} onChange={() => void refresh()} />
            </div>

            <Panel>
              <CardHeader title="Result" sub={ended ? 'the run has ended' : 'the test score appears when the run ends'} />
              <div className="space-y-4 p-4">
                {run.champion_val_metrics && (
                  <div>
                    <div className="mb-1 font-mono text-[10px] uppercase tracking-[0.1em] text-mute">Champion, validation rows (used to choose)</div>
                    <MetricList metrics={run.champion_val_metrics} testId="run-validation-metrics" />
                  </div>
                )}
                {ended && run.test_metrics && (
                  <div>
                    <div className="mb-1 font-mono text-[10px] uppercase tracking-[0.1em] text-mute">Test rows: scored once, after the last round</div>
                    <MetricList metrics={run.test_metrics} testId="run-test-metrics" />
                  </div>
                )}
                {ended && !run.test_metrics && (
                  <p data-testid="run-no-test" className="text-[12px] text-mute">
                    {run.test_error ?? 'The test rows were not scored.'}
                  </p>
                )}
                {!ended && <p className="text-[12px] text-mute">The test rows are not touched until the run ends.</p>}
                {run.notes.map((n) => (
                  <p key={n} className="text-[12px] text-mute">
                    {n}
                  </p>
                ))}
                <div className="flex flex-wrap gap-2">
                  {(['html', 'markdown'] as const).map((f) => (
                    <Btn
                      key={f}
                      variant="ghost"
                      size="sm"
                      data-testid={`report-${f}`}
                      onClick={() => void openReport(run.id, f).catch((e) => setError(describeError(e)))}
                    >
                      Evidence report ({f === 'html' ? 'HTML' : 'Markdown'})
                    </Btn>
                  ))}
                  <Btn
                    variant="ghost"
                    size="sm"
                    data-testid="export-bundle"
                    disabled={!run.champion_val_metrics}
                    onClick={() => void downloadBundle(run.id).catch((e) => setError(describeError(e)))}
                  >
                    Export bundle (zip)
                  </Btn>
                </div>
              </div>
            </Panel>
            {ended && run.champion_val_metrics && <ScoreView runId={run.id} />}
          </>
        )}
      </main>
    </div>
  );
}
