import { useState } from 'react';
import { describeError } from '../api';
import { Checkpoint, SettingsPreview, addSuggestion, answerCheckpoint, applySettings, previewSettings } from '../api/runs';
import { Btn, Panel, CardHeader } from '../ui';

const inputCls = 'w-full border border-line bg-ink px-3 py-2 font-mono text-[12px] text-bone placeholder:text-mute focus:border-copper focus:outline-none';

/**
 * What a person can do while the run goes on: approve or veto the feature the run is waiting on, suggest an idea,
 * change a setting in a sentence (shown as a change first, applied on confirmation).
 */
export function SteerPanel({ runId, checkpoints, live, onChange }: { runId: string; checkpoints: Checkpoint[]; live: boolean; onChange: () => void }) {
  const waiting = checkpoints.filter((c) => c.state === 'pending');
  const [idea, setIdea] = useState('');
  const [message, setMessage] = useState('');
  const [preview, setPreview] = useState<SettingsPreview | null>(null);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function attempt(fn: () => Promise<void>) {
    setError(null);
    try {
      await fn();
      onChange();
    } catch (e) {
      setError(describeError(e));
    }
  }

  return (
    <Panel>
      <CardHeader title="Steer the run" sub={live ? 'it reads these before its next round' : 'the run has ended'} />
      <div className="space-y-4 p-4" data-testid="steer-panel">
        {error && (
          <div role="alert" className="border border-clay/50 bg-clay/5 p-2 text-[12px] text-bone">
            {error}
          </div>
        )}
        {waiting.map((c) => (
          <div key={c.id} data-testid="checkpoint" className="space-y-2 border border-copper/40 bg-copper/5 p-3">
            <div className="text-[13px] text-bone">
              Round {c.round}: approve <span className="font-mono">{String((c.payload as { name?: string }).name)}</span>?{' '}
              {String((c.payload as { description?: string }).description ?? '')}
            </div>
            <div className="flex gap-2">
              <Btn size="sm" variant="primary" onClick={() => void attempt(async () => void (await answerCheckpoint(runId, c.id, 'approve')))}>
                Approve
              </Btn>
              <Btn size="sm" variant="danger" onClick={() => void attempt(async () => void (await answerCheckpoint(runId, c.id, 'veto')))}>
                Veto
              </Btn>
              <span className="self-center font-mono text-[10px] text-mute">
                no answer in {c.timeout_seconds}s: {c.recommended}
              </span>
            </div>
          </div>
        ))}
        <div className="space-y-2">
          <label className="font-mono text-[10px] uppercase tracking-[0.1em] text-mute" htmlFor="idea">
            Suggest a feature
          </label>
          <div className="flex gap-2">
            <input
              id="idea"
              className={inputCls}
              value={idea}
              onChange={(e) => setIdea(e.target.value)}
              placeholder="Something about support tickets"
              disabled={!live}
            />
            <Btn
              size="sm"
              disabled={!live || idea.trim() === ''}
              onClick={() =>
                void attempt(async () => {
                  await addSuggestion(runId, idea.trim());
                  setIdea('');
                  setNote('Suggestion sent: the model sees it in its next round, and what it proposes is checked like any proposal.');
                })
              }
            >
              Send
            </Btn>
          </div>
        </div>
        <div className="space-y-2">
          <label className="font-mono text-[10px] uppercase tracking-[0.1em] text-mute" htmlFor="settings">
            Change a setting
          </label>
          <div className="flex gap-2">
            <input
              id="settings"
              className={inputCls}
              value={message}
              onChange={(e) => {
                setMessage(e.target.value);
                setPreview(null);
              }}
              placeholder="budget $1, at most 5 rounds, ask me before each feature"
              disabled={!live}
            />
            <Btn
              size="sm"
              variant="outline"
              disabled={!live || message.trim() === ''}
              onClick={() => void attempt(async () => setPreview(await previewSettings(runId, message.trim())))}
            >
              Preview
            </Btn>
          </div>
          {preview && (
            <div data-testid="settings-preview" className="space-y-2 border border-line p-3 text-[12px] text-bone-dim">
              <div>{preview.summary}</div>
              {preview.unrecognised.length > 0 && <div className="text-copper-2">Not understood: {preview.unrecognised.join('; ')}</div>}
              {preview.changes.length > 0 && (
                <Btn
                  size="sm"
                  variant="primary"
                  onClick={() =>
                    void attempt(async () => {
                      const done = await applySettings(runId, message.trim());
                      setNote(done.applied ? `Applied: ${done.summary}` : 'Nothing changed.');
                      setPreview(null);
                      setMessage('');
                    })
                  }
                >
                  Apply
                </Btn>
              )}
            </div>
          )}
        </div>
        {note && <p className="text-[12px] text-sage">{note}</p>}
      </div>
    </Panel>
  );
}
