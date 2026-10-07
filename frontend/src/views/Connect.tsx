import { useCallback, useEffect, useState } from 'react';
import { describeError } from '../api';
import {
  ConnectionCreate,
  ConnectionRead,
  ConnectionTestResult,
  SchemaGraphData,
  TableStatsData,
  createConnection,
  deleteConnection,
  getSchema,
  getTableStats,
  listConnections,
  patchSchema,
  testConnection,
  updateConnection,
} from '../api/connections';
import { PrivacyRead, getPrivacy, putPrivacy } from '../api/privacy';
import { DataVersions } from '../components/DataVersions';
import { SchemaGraph, compact, describeEdge } from '../components/SchemaGraph';
import { Btn, CardHeader, Eyebrow, Leader, Panel, Tag } from '../ui';
import { cn } from '../utils/cn';
import { leaveRoute } from '../route';

type Dialect = ConnectionCreate['dialect'];
type SslMode = NonNullable<ConnectionCreate['ssl_mode']>;

const DIALECTS: { key: Dialect; label: string }[] = [
  { key: 'postgres', label: 'Postgres' },
  { key: 'sqlite', label: 'SQLite' },
  { key: 'duckdb', label: 'DuckDB' },
];
const SSL_MODES: SslMode[] = ['prefer', 'disable', 'require', 'verify-ca', 'verify-full'];

const READ_ONLY_ROLE_SQL = `CREATE ROLE mlpilot_ro LOGIN PASSWORD '<choose one>';
GRANT CONNECT ON DATABASE <your_database> TO mlpilot_ro;
GRANT USAGE ON SCHEMA public TO mlpilot_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO mlpilot_ro;`;

const inputCls =
  'h-9 w-full border border-line bg-ink px-3 font-mono text-[12.5px] text-bone placeholder:text-mute/60 focus:border-copper focus:outline-none';

function Field({ label, children, hint }: { label: string; children: React.ReactNode; hint?: string }) {
  return (
    <div>
      <label className="block">
        <span className="mb-1 block font-mono text-[10px] uppercase tracking-[0.12em] text-mute">{label}</span>
        {children}
      </label>
      {hint && <p className="mt-1 text-[11px] text-mute">{hint}</p>}
    </div>
  );
}

const pct1 = (x: number) => `${(x * 100).toFixed(x > 0 && x < 0.001 ? 2 : 1)}%`;
const num = (x: number | null | undefined) =>
  x == null ? '—' : Math.abs(x) >= 1e6 || (x !== 0 && Math.abs(x) < 0.01) ? x.toExponential(2) : Number.isInteger(x) ? x.toLocaleString('en-US') : x.toFixed(2);

/* ------------------------------------------------------------ test result */
function TestResult({ result }: { result: ConnectionTestResult }) {
  if (!result.ok) {
    return (
      <div role="alert" data-testid="connect-error" className="border border-clay/50 bg-clay/5 p-4">
        <Tag tone="clay">{result.error_code ?? 'failed'}</Tag>
        <p className="mt-2 text-[13px] text-bone">{result.message ?? 'The connection failed.'}</p>
      </div>
    );
  }
  return (
    <div className="space-y-3">
      <div data-testid="connect-ok" className="border border-sage/40 bg-sage/5 p-4">
        <Tag tone="sage">connected</Tag>
        <div className="mt-2 grid gap-x-8 sm:grid-cols-2">
          <Leader k="Server" v={result.server_version ?? 'unknown'} />
          <Leader k="Latency" v={result.latency_ms != null ? `${Math.round(result.latency_ms)} ms` : '—'} />
        </div>
      </div>
      {result.can_write ? (
        <div role="alert" data-testid="write-warning" className="border border-clay/60 bg-clay/10 p-4">
          <Tag tone="clay">this role can write</Tag>
          <p className="mt-2 text-[13px] leading-relaxed text-bone">
            MLPilot only reads: every query is checked to be a single SELECT and runs in a read-only transaction. A login that can also change data is still a
            risk if anything else uses it, so use a read-only role for this connection.
          </p>
          {result.privilege_notes.length > 0 && (
            <ul className="mt-2 list-disc pl-5 text-[12px] text-bone-dim">
              {result.privilege_notes.map((n) => (
                <li key={n}>{n}</li>
              ))}
            </ul>
          )}
          <details className="mt-3">
            <summary className="cursor-pointer font-mono text-[10.5px] uppercase tracking-[0.1em] text-copper-2">How to create a read-only role (Postgres)</summary>
            <pre className="mt-2 overflow-x-auto border border-line bg-ink p-3 font-mono text-[11.5px] text-copper-2">{READ_ONLY_ROLE_SQL}</pre>
          </details>
        </div>
      ) : (
        result.can_write === false && (
          <div data-testid="read-only-ok" className="border border-sage/40 bg-sage/5 p-3 text-[13px] text-bone">
            <Tag tone="sage">read-only role</Tag>
            <span className="ml-2">This login cannot change data.</span>
          </div>
        )
      )}
    </div>
  );
}

/* ------------------------------------------------------------ table panel */
function TablePanel({
  connectionId,
  graph,
  tableKey,
  onGraph,
  privacy,
  onNeverSend,
}: {
  connectionId: string;
  graph: SchemaGraphData;
  tableKey: string;
  onGraph: (g: SchemaGraphData) => void;
  privacy: PrivacyRead | null;
  onNeverSend: (table: string, column: string, on: boolean) => void;
}) {
  const table = graph.tables.find((t) => t.key === tableKey);
  const [stats, setStats] = useState<TableStatsData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    let live = true;
    setStats(null);
    setError(null);
    getTableStats(connectionId, tableKey)
      .then((s) => live && setStats(s))
      .catch((e) => live && setError(describeError(e)));
    return () => {
      live = false;
    };
  }, [connectionId, tableKey]);

  if (!table) return null;
  const byName = new Map(stats?.stats.columns.map((c) => [c.name, c]));
  const timeOptions = [...new Set([...(table.time_candidates ?? []), ...table.columns.filter((c) => c.hint === 'time').map((c) => c.name)])];

  const save = async (overrides: Parameters<typeof patchSchema>[1]) => {
    setSaving(true);
    setError(null);
    try {
      onGraph(await patchSchema(connectionId, overrides));
    } catch (e) {
      setError(describeError(e));
    } finally {
      setSaving(false);
    }
  };

  const excluded = new Set((privacy?.never_send ?? []).map((x) => x.toLowerCase()));
  const everywhere = new Set([...excluded].filter((x) => !x.includes('.')));
  const edges = graph.edges.filter((e) => e.from_table === tableKey || e.to_table === tableKey);

  return (
    <Panel className="flex max-h-[640px] flex-col" ticks>
      <CardHeader title={table.key} sub={`${table.row_count_estimated ? '~' : ''}${compact(table.row_count)} rows`} />
      <div className="space-y-4 overflow-y-auto p-4" data-testid="table-details">
        <div className="space-y-2 border border-line bg-ink p-3">
          <Eyebrow>Event time</Eyebrow>
          <select
            aria-label="Time column"
            disabled={saving}
            value={table.time_column ?? ''}
            onChange={(e) => void save({ time_columns: { [table.key]: e.target.value || null } })}
            className={inputCls}
          >
            <option value="">(no time column)</option>
            {timeOptions.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <p className="text-[11px] text-mute">
            {table.time_column
              ? table.time_column_source === 'user'
                ? 'Chosen by you.'
                : 'Guessed from the column names; change it if it is wrong.'
              : 'Rows of a table without a time column cannot be limited to “before the cutoff”.'}
          </p>
          {table.time_leakage_hint && <p className="border border-clay/50 p-2 text-[11px] text-clay">{table.time_leakage_hint}</p>}
          <label className="flex items-center gap-2 text-[12px] text-bone-dim">
            <input
              type="checkbox"
              disabled={saving}
              checked={table.is_static === true}
              onChange={(e) => void save({ static_tables: { [table.key]: e.target.checked } })}
            />
            Static table (no event times; its rows do not change over time)
          </label>
        </div>

        {edges.length > 0 && (
          <div>
            <Eyebrow>Relationships</Eyebrow>
            <ul className="mt-1 space-y-1 font-mono text-[10.5px] text-bone-dim">
              {edges.map((e, i) => (
                <li key={i}>{describeEdge(e)}</li>
              ))}
            </ul>
          </div>
        )}

        <div>
          <div className="flex items-center justify-between">
            <Eyebrow>Columns</Eyebrow>
            {stats && (
              <span className="font-mono text-[10px] text-mute" data-testid="stats-note">
                {stats.stats.sampled
                  ? `statistics from a ${stats.stats.sample_method} sample of ${compact(stats.stats.profiled_rows)} rows`
                  : `statistics over all ${compact(stats.stats.profiled_rows)} rows`}
                {stats.cached ? ' · cached' : ''}
              </span>
            )}
          </div>
          {error && (
            <p role="alert" className="mt-2 border border-clay/50 p-2 text-[12px] text-clay">
              {error}
            </p>
          )}
          {!stats && !error && <p className="mt-2 font-mono text-[11px] text-mute">Computing column statistics in the database…</p>}
          <table className="mt-2 w-full text-left font-mono text-[11px]">
            <thead className="text-mute">
              <tr>
                <th className="py-1 pr-2 font-medium">column</th>
                <th className="pr-2 font-medium">null</th>
                <th className="pr-2 font-medium">distinct</th>
                <th className="pr-2 font-medium">range</th>
                <th className="font-medium" title="Never include this column in a prompt to the LLM">
                  never send
                </th>
              </tr>
            </thead>
            <tbody>
              {table.columns.map((c) => {
                const s = byName.get(c.name);
                const range = s ? (s.time_min ? `${s.time_min.slice(0, 10)} → ${(s.time_max ?? '').slice(0, 10)}` : s.min != null ? `${num(s.min)} → ${num(s.max)}` : '') : '';
                return (
                  <tr key={c.name} data-testid={`col-${c.name}`} className="border-t border-line align-top">
                    <td className="py-1 pr-2 text-bone">
                      {c.name}
                      {c.is_primary_key && <span className="ml-1 text-copper-2">PK</span>}
                      <div className="text-[10px] text-mute">
                        {c.type} · {c.hint}
                      </div>
                    </td>
                    <td className="pr-2 text-bone-dim">{s ? pct1(s.null_fraction) : ''}</td>
                    <td className="pr-2 text-bone-dim">{s ? num(s.distinct) : ''}</td>
                    <td className="pr-2 text-bone-dim">{range}</td>
                    <td>
                      <input
                        type="checkbox"
                        aria-label={`Never send ${c.name} to the LLM`}
                        disabled={!privacy || everywhere.has(c.name.toLowerCase())}
                        title={everywhere.has(c.name.toLowerCase()) ? 'Excluded for every table (Settings)' : undefined}
                        checked={!!privacy && (excluded.has(`${table.key}.${c.name}`.toLowerCase()) || everywhere.has(c.name.toLowerCase()))}
                        onChange={(e) => onNeverSend(table.key, c.name, e.target.checked)}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="mt-2 text-[10.5px] text-mute">Only counts and ranges are shown. No row values leave the database.</p>
        </div>
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------- page */
export function ConnectView() {
  const [dialect, setDialect] = useState<Dialect>('postgres');
  const [name, setName] = useState('');
  const [host, setHost] = useState('');
  const [port, setPort] = useState('5432');
  const [database, setDatabase] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [passwordEnv, setPasswordEnv] = useState('');
  const [useEnv, setUseEnv] = useState(false);
  const [ssl, setSsl] = useState<SslMode>('prefer');

  const [saved, setSaved] = useState<ConnectionRead | null>(null);
  const [existing, setExisting] = useState<ConnectionRead[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ConnectionTestResult | null>(null);
  const [graph, setGraph] = useState<SchemaGraphData | null>(null);
  const [graphError, setGraphError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [privacy, setPrivacy] = useState<PrivacyRead | null>(null);

  useEffect(() => {
    getPrivacy()
      .then(setPrivacy)
      .catch(() => setPrivacy(null));
  }, []);

  const setNeverSend = async (table: string, column: string, on: boolean) => {
    if (!privacy) return;
    const key = `${table}.${column}`;
    const rest = (privacy.never_send ?? []).filter((x) => x.toLowerCase() !== key.toLowerCase());
    try {
      setPrivacy(await putPrivacy({ level: privacy.level, never_send: on ? [...rest, key] : rest }));
    } catch (e) {
      setError(describeError(e));
    }
  };

  const refreshExisting = useCallback(() => {
    listConnections()
      .then(setExisting)
      .catch(() => setExisting([]));
  }, []);
  useEffect(refreshExisting, [refreshExisting]);

  const loadGraph = async (id: string) => {
    setGraph(null);
    setGraphError(null);
    setSelected(null);
    try {
      setGraph(await getSchema(id));
    } catch (e) {
      setGraphError(describeError(e));
    }
  };

  const submit = async (ev: React.FormEvent) => {
    ev.preventDefault();
    setBusy(true);
    setError(null);
    setResult(null);
    setGraph(null);
    setGraphError(null);
    try {
      const fields = {
        name: name.trim() || database.split('/').pop() || 'database',
        database: database.trim(),
        ...(dialect === 'postgres'
          ? {
              host: host.trim(),
              port: Number(port) || 5432,
              username: username.trim(),
              ssl_mode: ssl,
              ...(useEnv ? { password_env: passwordEnv.trim() } : password ? { password } : {}),
            }
          : {}),
      };
      let conn = saved;
      if (conn && conn.dialect !== dialect) {
        await deleteConnection(conn.id);
        conn = null;
      }
      conn = conn ? await updateConnection(conn.id, fields) : await createConnection({ dialect, ...fields });
      setSaved(conn);
      setPassword('');
      refreshExisting();
      const test = await testConnection(conn.id);
      setResult(test);
      if (test.ok) await loadGraph(conn.id);
    } catch (e) {
      setError(describeError(e));
    } finally {
      setBusy(false);
    }
  };

  const open = async (c: ConnectionRead) => {
    setSaved(c);
    setError(null);
    setResult(null);
    setBusy(true);
    try {
      const test = await testConnection(c.id);
      setResult(test);
      if (test.ok) await loadGraph(c.id);
    } catch (e) {
      setError(describeError(e));
    } finally {
      setBusy(false);
    }
  };

  const remove = async (c: ConnectionRead) => {
    try {
      await deleteConnection(c.id);
      if (saved?.id === c.id) {
        setSaved(null);
        setResult(null);
        setGraph(null);
      }
      refreshExisting();
    } catch (e) {
      setError(describeError(e));
    }
  };

  const canSubmit =
    !busy && database.trim() !== '' && (dialect !== 'postgres' || (host.trim() !== '' && username.trim() !== '' && (!useEnv || passwordEnv.trim() !== '')));

  return (
    <div className="relative min-h-screen overflow-y-auto text-bone">
      <div className="guides" aria-hidden />
      <header className="mx-auto flex h-14 max-w-[1200px] items-center justify-between px-6">
        <Btn variant="ghost" size="sm" onClick={leaveRoute}>
          ← Back
        </Btn>
        <Tag tone="copper">read-only</Tag>
      </header>

      <main className="mx-auto max-w-[1200px] space-y-8 px-6 pb-16">
        <section>
          <Eyebrow className="text-copper">Data source</Eyebrow>
          <h1 className="mt-1 font-display text-[44px] font-extrabold uppercase leading-none tracking-wide text-bone">Connect a database</h1>
          <p className="mt-3 max-w-2xl text-[14px] leading-relaxed text-mute">
            MLPilot reads your tables where they are. It sends single SELECT statements only, inside a read-only transaction with a timeout and a row limit. This
            page shows the tables, how they relate and column statistics. Turning them into a prediction task comes in a later step: nothing here trains a model
            yet.
          </p>
        </section>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,420px)_minmax(0,1fr)]">
          <form onSubmit={submit} className="space-y-4" aria-label="Connection details">
            <Panel ticks>
              <CardHeader title="Connection" />
              <div className="space-y-3 p-4">
                <div className="flex gap-1" role="radiogroup" aria-label="Database type">
                  {DIALECTS.map((d) => (
                    <button
                      key={d.key}
                      type="button"
                      role="radio"
                      aria-checked={dialect === d.key}
                      onClick={() => setDialect(d.key)}
                      className={cn(
                        'h-9 flex-1 border font-mono text-[11px] uppercase tracking-[0.1em] transition-colors',
                        dialect === d.key ? 'border-copper bg-copper/10 text-copper-2' : 'border-line text-mute hover:text-bone'
                      )}
                    >
                      {d.label}
                    </button>
                  ))}
                </div>
                <Field label="Name">
                  <input className={inputCls} value={name} onChange={(e) => setName(e.target.value)} placeholder="shop-production" />
                </Field>
                {dialect === 'postgres' ? (
                  <>
                    <div className="grid grid-cols-[1fr_96px] gap-3">
                      <Field label="Host">
                        <input className={inputCls} value={host} onChange={(e) => setHost(e.target.value)} placeholder="db.example.com" autoComplete="off" />
                      </Field>
                      <Field label="Port">
                        <input className={inputCls} value={port} onChange={(e) => setPort(e.target.value)} inputMode="numeric" />
                      </Field>
                    </div>
                    <Field label="Database">
                      <input className={inputCls} value={database} onChange={(e) => setDatabase(e.target.value)} placeholder="shop" autoComplete="off" />
                    </Field>
                    <Field label="User">
                      <input className={inputCls} value={username} onChange={(e) => setUsername(e.target.value)} placeholder="read-only login" autoComplete="off" />
                    </Field>
                    {useEnv ? (
                      <Field label="Password variable" hint="The name of an environment variable on the machine running MLPilot. Nothing secret is stored.">
                        <input className={inputCls} value={passwordEnv} onChange={(e) => setPasswordEnv(e.target.value)} placeholder="SHOP_DB_PASSWORD" autoComplete="off" />
                      </Field>
                    ) : (
                      <Field label="Password" hint="Stored encrypted when MLPILOT_SECRET_KEY is set; never shown again.">
                        <input
                          className={inputCls}
                          type="password"
                          value={password}
                          onChange={(e) => setPassword(e.target.value)}
                          autoComplete="new-password"
                        />
                      </Field>
                    )}
                    <label className="flex items-center gap-2 text-[12px] text-bone-dim">
                      <input type="checkbox" checked={useEnv} onChange={(e) => setUseEnv(e.target.checked)} />
                      Read the password from an environment variable instead
                    </label>
                    <Field label="SSL">
                      <select className={inputCls} value={ssl} onChange={(e) => setSsl(e.target.value as SslMode)}>
                        {SSL_MODES.map((m) => (
                          <option key={m} value={m}>
                            {m}
                          </option>
                        ))}
                      </select>
                    </Field>
                  </>
                ) : (
                  <Field label="File path" hint="A path on the machine that runs the MLPilot backend. The file is opened read-only.">
                    <input
                      className={inputCls}
                      value={database}
                      onChange={(e) => setDatabase(e.target.value)}
                      placeholder={dialect === 'sqlite' ? '/data/shop.sqlite' : '/data/shop.duckdb'}
                      autoComplete="off"
                    />
                  </Field>
                )}
                <Btn variant="primary" size="lg" type="submit" disabled={!canSubmit} className="w-full">
                  {busy ? 'Connecting…' : saved ? 'Save & test again' : 'Save & test'}
                </Btn>
              </div>
            </Panel>

            {existing.length > 0 && (
              <Panel>
                <CardHeader title="Saved connections" />
                <ul className="divide-y divide-line">
                  {existing.map((c) => (
                    <li key={c.id} className="flex items-center justify-between gap-2 px-4 py-2" data-testid={`saved-${c.name}`}>
                      <div className="min-w-0">
                        <div className="truncate font-mono text-[12px] text-bone">{c.name}</div>
                        <div className="truncate font-mono text-[10px] text-mute">
                          {c.dialect} · {c.host ? `${c.host}/` : ''}
                          {c.database}
                        </div>
                      </div>
                      <div className="flex shrink-0 gap-1">
                        <Btn size="sm" onClick={() => void open(c)} disabled={busy}>
                          Open
                        </Btn>
                        <Btn size="sm" variant="danger" onClick={() => void remove(c)} aria-label={`Delete ${c.name}`}>
                          ×
                        </Btn>
                      </div>
                    </li>
                  ))}
                </ul>
              </Panel>
            )}
          </form>

          <div className="min-w-0 space-y-4">
            {error && (
              <div role="alert" data-testid="connect-error" className="border border-clay/50 bg-clay/5 p-4 text-[13px] text-bone">
                {error}
              </div>
            )}
            {result && <TestResult result={result} />}
            {!result && !error && (
              <div className="border border-dashed border-rule p-8 text-center font-mono text-[11px] uppercase tracking-[0.1em] text-mute">
                Fill in the connection to see the tables and how they relate.
              </div>
            )}

            {result?.ok && !graph && !graphError && <p className="font-mono text-[11px] text-mute">Reading the schema…</p>}
            {graphError && (
              <div role="alert" data-testid="schema-error" className="border border-clay/50 bg-clay/5 p-4 text-[13px] text-bone">
                {graphError}
              </div>
            )}
            {graph && saved && (
              <div className="space-y-3">
                <Panel ticks>
                  <CardHeader
                    title="Schema graph"
                    sub={`${graph.tables.length} tables · ${graph.edges.length} relationships`}
                    right={
                      <div className="hidden items-center gap-3 font-mono text-[10px] text-mute sm:flex">
                        <span className="text-copper-2">━ declared</span>
                        <span>┅ inferred (confidence)</span>
                        <span className="text-sage">━ added by you</span>
                      </div>
                    }
                  />
                  <div className="bg-grid h-[460px]" data-testid="schema-graph">
                    <SchemaGraph graph={graph} selected={selected} onSelect={setSelected} />
                  </div>
                </Panel>
                {graph.warnings && graph.warnings.length > 0 && (
                  <ul className="border border-rule p-3 text-[12px] text-bone-dim">
                    {graph.warnings.map((w) => (
                      <li key={w}>{w}</li>
                    ))}
                  </ul>
                )}
                <DataVersions connectionId={saved.id} tables={graph.tables.map((t) => t.key)} />
                {selected ? (
                  <TablePanel connectionId={saved.id} graph={graph} tableKey={selected} onGraph={setGraph} privacy={privacy} onNeverSend={(t, c, on) => void setNeverSend(t, c, on)} />
                ) : (
                  <p className="font-mono text-[11px] uppercase tracking-[0.1em] text-mute">Click a table to see its columns, statistics and event-time column.</p>
                )}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}
