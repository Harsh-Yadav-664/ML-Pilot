import { useCallback, useEffect, useRef, useState } from 'react';
import { describeError } from '../api';
import { DbVersion, listDbVersions, takeSnapshot, waitForJob } from '../api/snapshots';
import { Btn, CardHeader, Panel, Tag } from '../ui';

/**
 * Version what a run will read from this database: a snapshot copies the tables up to a cutoff into the
 * project (same data, same id); a live record copies nothing and says it cannot be reproduced exactly.
 */
export function DataVersions({ connectionId, tables }: { connectionId: string; tables: string[] }) {
  const [mode, setMode] = useState<'snapshot' | 'live'>('snapshot');
  const [versions, setVersions] = useState<DbVersion[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [fraction, setFraction] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const alive = useRef(true);

  const refresh = useCallback(async () => {
    try {
      setVersions((await listDbVersions()).filter((v) => v.connection.id === connectionId));
    } catch (e) {
      setError(describeError(e));
    }
  }, [connectionId]);

  useEffect(() => {
    alive.current = true;
    void refresh();
    return () => {
      alive.current = false;
    };
  }, [refresh]);

  async function run() {
    setError(null);
    setBusy(mode === 'live' ? 'Reading the source' : 'Starting');
    setFraction(0);
    try {
      const { version, jobId } = await takeSnapshot(connectionId, { mode, tables });
      if (jobId) {
        const outcome = await waitForJob(jobId, (step, f) => { setBusy(step); setFraction(f); }, () => !alive.current);
        if (!outcome.ok && outcome.error !== 'stopped') setError(outcome.error ?? 'The snapshot failed');
      } else if (!version) {
        setError('The server returned neither a version nor a job');
      }
      await refresh();
    } catch (e) {
      setError(describeError(e));
    } finally {
      if (alive.current) setBusy(null);
    }
  }

  return (
    <Panel ticks>
      <CardHeader title="Data versions" sub="What exactly a run reads from this database" />
      <div className="space-y-3 p-4" data-testid="data-versions">
        <div className="flex flex-wrap items-center gap-3">
          <select aria-label="Version mode" value={mode} onChange={(e) => setMode(e.target.value as 'snapshot' | 'live')}
            className="h-9 border border-line bg-ink px-2 font-mono text-[12px] text-bone" disabled={busy !== null}>
            <option value="snapshot">Snapshot (copy the tables)</option>
            <option value="live">Live (record the state only)</option>
          </select>
          <Btn onClick={() => void run()} disabled={busy !== null || tables.length === 0}>
            {mode === 'snapshot' ? 'Take snapshot' : 'Record live state'}
          </Btn>
          {busy && (
            <span className="font-mono text-[11px] text-mute" role="status">
              {busy}{mode === 'snapshot' ? ` · ${Math.round(fraction * 100)}%` : ''}
            </span>
          )}
        </div>
        <p className="text-[12px] text-bone-dim">
          {mode === 'snapshot'
            ? 'Copies every table of this schema, up to the time you click, into the project. The same data always gets the same version id.'
            : 'Copies nothing. Records the row counts and latest event times now; the same run cannot be reproduced exactly if the data changes.'}
        </p>
        {error && <div role="alert" className="border border-clay/50 bg-clay/5 p-3 text-[13px] text-bone">{error}</div>}
        {versions.length > 0 && (
          <ul className="divide-y divide-rule border border-rule">
            {versions.map((v) => (
              <li key={v.id} className="flex flex-wrap items-center gap-x-4 gap-y-1 px-3 py-2 font-mono text-[11.5px] text-bone-dim" data-testid="db-version">
                <span className="text-bone">{v.short_hash}</span>
                <Tag>{v.mode}</Tag>
                <span>as of {v.as_of.slice(0, 19).replace('T', ' ')} UTC</span>
                <span>{v.n_rows.toLocaleString('en-US')} rows · {Object.keys(v.tables).length} tables</span>
                <span className={v.reproducible ? 'text-sage' : 'text-clay'}>
                  {v.reproducible ? 'reproducible' : 'not reproducible'}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </Panel>
  );
}
