import { useEffect, useState } from 'react';
import { describeError } from '../api';
import { PrivacyLevel, PrivacyRead, getPrivacy, putPrivacy } from '../api/privacy';
import { Btn, Eyebrow, Panel, Tag } from '../ui';
import { cn } from '../utils/cn';

const LEVELS: { key: PrivacyLevel; label: string }[] = [
  { key: 'schema_only', label: 'Schema only' },
  { key: 'schema_and_stats', label: 'Schema and statistics' },
  { key: 'allow_category_labels', label: 'Also category labels' },
  { key: 'allow_sample_values', label: 'Also sample values' },
];

/** What the LLM may see of the data, and the columns it never sees. */
export function PrivacyPanel() {
  const [privacy, setPrivacy] = useState<PrivacyRead | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [entry, setEntry] = useState('');

  useEffect(() => {
    getPrivacy()
      .then(setPrivacy)
      .catch((e) => setError(describeError(e)));
  }, []);

  const save = async (level: PrivacyLevel, neverSend: string[]) => {
    setSaving(true);
    setError(null);
    try {
      setPrivacy(await putPrivacy({ level, never_send: neverSend }));
    } catch (e) {
      setError(describeError(e));
    } finally {
      setSaving(false);
    }
  };

  const neverSend = privacy?.never_send ?? [];
  if (!privacy) {
    return <Panel className="mt-3 p-5 font-mono text-[11px] text-mute">{error ?? 'Loading privacy settings…'}</Panel>;
  }

  return (
    <Panel ticks className="mt-3">
      <div role="radiogroup" aria-label="What the LLM may see">
        {LEVELS.map((l) => {
          const on = privacy.level === l.key;
          return (
            <button
              key={l.key}
              role="radio"
              aria-checked={on}
              disabled={saving}
              onClick={() => void save(l.key, neverSend)}
              className={cn(
                'relative flex w-full items-start gap-4 border-b border-line px-5 py-3 text-left transition-colors',
                on ? 'bg-panel-2' : 'hover:bg-panel-2/50'
              )}
            >
              {on && <span className="absolute inset-y-0 left-0 w-[3px] bg-copper" />}
              <span className={cn('mt-1 flex h-4 w-4 shrink-0 items-center justify-center border', on ? 'border-copper' : 'border-rule')}>
                {on && <span className="h-2 w-2 bg-copper" />}
              </span>
              <span className="min-w-0 flex-1">
                <span className="block font-display text-[19px] font-bold uppercase leading-none tracking-wide text-bone">
                  {l.label}
                  {l.key === 'schema_and_stats' && <Tag className="ml-2">default</Tag>}
                </span>
                <span className="mt-1 block text-[12px] text-mute">{privacy.level_summaries[l.key]}</span>
              </span>
            </button>
          );
        })}
      </div>
      {privacy.warning && (
        <p role="alert" className="border-b border-clay/50 bg-clay/10 px-5 py-3 text-[12px] text-clay">
          {privacy.warning}
        </p>
      )}
      <div className="space-y-3 p-5">
        <Eyebrow>Never send these columns</Eyebrow>
        <p className="text-[12px] text-mute">
          Left out of every prompt at every level, and their names are replaced in questions and history. Enter <code className="font-mono">column</code> for any table or{' '}
          <code className="font-mono">table.column</code>. You can also tick columns on the schema graph.
        </p>
        <div className="flex flex-wrap gap-2">
          {neverSend.length === 0 && <span className="font-mono text-[11px] text-mute">none</span>}
          {neverSend.map((c) => (
            <span key={c} className="inline-flex items-center gap-1 border border-rule px-2 py-1 font-mono text-[11px] text-bone">
              {c}
              <button
                aria-label={`Remove ${c}`}
                disabled={saving}
                onClick={() => void save(privacy.level, neverSend.filter((x) => x !== c))}
                className="text-mute hover:text-clay"
              >
                ×
              </button>
            </span>
          ))}
        </div>
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (!entry.trim()) return;
            void save(privacy.level, [...neverSend, entry.trim()]).then(() => setEntry(''));
          }}
        >
          <input
            aria-label="Column to never send"
            value={entry}
            onChange={(e) => setEntry(e.target.value)}
            placeholder="email or customers.email"
            className="h-9 min-w-0 flex-1 border border-line bg-ink px-3 font-mono text-[12.5px] text-bone placeholder:text-mute/60 focus:border-copper focus:outline-none"
          />
          <Btn type="submit" disabled={saving || !entry.trim()}>
            Add
          </Btn>
        </form>
        {error && (
          <p role="alert" className="border border-clay/50 p-2 text-[12px] text-clay">
            {error}
          </p>
        )}
        <p className="border-t border-line pt-3 text-[11.5px] text-mute">
          Nothing has to leave your network: set <code className="font-mono">OLLAMA_BASE_URL</code> to use a local model, and every prompt stays on your machine. The prompt log below shows what was
          sent.
        </p>
      </div>
    </Panel>
  );
}
