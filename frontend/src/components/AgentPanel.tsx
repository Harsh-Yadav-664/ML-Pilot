import { useEffect, useRef, useState } from 'react';
import { FeedItem, useStore } from '../store';
import { MODELS, estLift, f3 } from '../lib';
import { connection } from '../api';
import { cn } from '../utils/cn';
import { Btn, Eyebrow, Led, STATUS, Tag } from '../ui';

function Stream({ text, on }: { text: string; on: boolean }) {
  const [n, setN] = useState(on ? 0 : text.length);
  useEffect(() => {
    if (!on) return;
    const t = setInterval(() => setN((v) => (v >= text.length ? (clearInterval(t), v) : v + 2)), 16);
    return () => clearInterval(t);
  }, [text, on]);
  return <span className={n < text.length ? 'caret' : ''}>{text.slice(0, n)}</span>;
}

/** A proposed feature, laid out like a spec sheet. */
function SpecSheet({ item }: { item: Extract<FeedItem, { kind: 'suggestion' }> }) {
  const { runSuggestion, dismissSuggestion, experiments, select, setView, toast, treeOnly } = useStore();
  const [model, setModel] = useState('xgboost');
  const [copied, setCopied] = useState(false);
  const s = item.s;
  const exp = item.expId ? experiments.find((e) => e.id === item.expId) : undefined;
  // Estimated lift is sample data unless the backend supplies one.
  const lift = s.impact ?? (connection.demo ? estLift(s.name) : undefined);
  const available = connection.demo ? MODELS : MODELS.filter((m) => m.key !== 'catboost');
  const models = treeOnly ? available.filter((m) => m.tree) : available;

  const copy = () => {
    navigator.clipboard?.writeText(s.formula).catch(() => undefined);
    setCopied(true);
    toast('Formula copied', 'info');
    setTimeout(() => setCopied(false), 1400);
  };

  return (
    <div className="fade-up border border-line bg-panel transition-colors hover:border-rule">
      <div className="flex items-center justify-between border-b border-line px-3 py-1.5">
        <Eyebrow>Feature proposal</Eyebrow>
        {!exp && (
          <button onClick={() => dismissSuggestion(item.id)} className="font-mono text-[10px] text-mute transition-colors hover:text-clay" title="Dismiss">
            ✕
          </button>
        )}
      </div>

      <div className="px-3 pt-3">
        <code className="font-mono text-[13px] font-semibold text-bone">{s.name}</code>
        <div className="relative mt-2 border border-line bg-ink">
          <span className="absolute inset-y-0 left-0 w-[2px] bg-copper" />
          <code className="block overflow-x-auto whitespace-nowrap py-2 pl-3 pr-12 font-mono text-[11.5px] text-copper-2">{s.formula}</code>
          <button onClick={copy} className="absolute right-2 top-2 font-mono text-[9.5px] uppercase tracking-[0.1em] text-mute transition-colors hover:text-bone">
            {copied ? 'copied' : 'copy'}
          </button>
        </div>
      </div>

      <dl className="space-y-2 px-3 pt-3 text-[12.5px] leading-relaxed">
        <div className="grid grid-cols-[44px_1fr] gap-2">
          <dt className="pt-[3px]">
            <Eyebrow>Why</Eyebrow>
          </dt>
          <dd className="text-bone-dim">{s.reason}</dd>
        </div>
        <div className="grid grid-cols-[44px_1fr] gap-2">
          <dt className="pt-[3px]">
            <Eyebrow className="text-clay">Risk</Eyebrow>
          </dt>
          <dd className="text-mute">{s.risk}</dd>
        </div>
      </dl>

      <div className="mt-3 flex items-center justify-between gap-2 border-t border-dashed border-rule px-3 py-2.5">
        {exp ? (
          <>
            <div className="flex min-w-0 items-center gap-2 font-mono text-[11px]">
              <Led color={STATUS[exp.status].hex} pulse={exp.status === 'running'} />
              <code className="text-mute">{exp.id}</code>
              <span style={{ color: STATUS[exp.status].hex }} className="uppercase tracking-[0.08em]">
                {STATUS[exp.status].label}
              </span>
              {exp.metrics.f1 !== undefined && <span className="text-bone">F1 {f3(exp.metrics.f1)}</span>}
            </div>
            <button
              onClick={() => {
                setView('experiments');
                select(exp.id);
              }}
              className="shrink-0 font-mono text-[10.5px] uppercase tracking-[0.1em] text-mute transition-colors hover:text-copper-2"
            >
              View ↗
            </button>
          </>
        ) : (
          <>
            <div className="leading-none">
              <Eyebrow className="block text-[9px]">Est. F1</Eyebrow>
              {lift !== undefined ? (
                <span className="font-display text-[24px] font-bold text-sage">+{lift.toFixed(3)}</span>
              ) : (
                <span className="font-mono text-[11px] text-mute">not estimated</span>
              )}
            </div>
            <div className="flex items-center gap-1.5">
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="h-7 cursor-pointer border border-rule bg-ink px-1.5 font-mono text-[10.5px] text-bone-dim focus:border-copper focus:outline-none"
              >
                {models.map((m) => (
                  <option key={m.key} value={m.key}>
                    {m.label}
                  </option>
                ))}
              </select>
              <Btn variant="primary" size="sm" onClick={() => runSuggestion(item.id, model)}>
                ▶ Run experiment
              </Btn>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/** Human-in-the-loop checkpoint, raised directly in the thread. */
function CheckpointCard({ item }: { item: Extract<FeedItem, { kind: 'checkpoint' }> }) {
  const { answerCheckpoint } = useStore();
  const resolved = item.choice !== undefined;
  return (
    <div className={cn('fade-up border bg-copper/[0.05]', resolved ? 'border-line opacity-70' : 'border-copper/60')}>
      <div className="flex items-center gap-2 border-b border-copper/30 px-3 py-1.5">
        <Eyebrow className="text-copper-2">Decision needed — waiting on you</Eyebrow>
      </div>
      <div className="px-3 py-3">
        <p className="text-[13px] font-medium leading-snug text-bone">{item.question}</p>
        <p className="mt-1.5 text-[11.5px] leading-relaxed text-mute">{item.detail}</p>
        <div className="mt-3 flex flex-col gap-1.5">
          {item.options.map((o, i) => (
            <Btn key={o} size="sm" variant={resolved && item.choice === i ? 'primary' : 'outline'} className="justify-start" disabled={resolved} onClick={() => answerCheckpoint(item.id, i)}>
              {o}
              {resolved && item.choice === i ? ' ✓' : ''}
            </Btn>
          ))}
        </div>
      </div>
    </div>
  );
}

function Entry({ who, tone, children }: { who: string; tone: 'agent' | 'user'; children: React.ReactNode }) {
  return (
    <div className="fade-up grid grid-cols-[40px_1fr] gap-x-2">
      <span className={cn('pt-[3px] font-mono text-[9.5px] font-semibold uppercase tracking-[0.12em]', tone === 'agent' ? 'text-copper' : 'text-bone')}>{who}</span>
      <div className={cn('border-l pl-3 text-[13px] leading-relaxed whitespace-pre-line', tone === 'agent' ? 'border-line text-bone-dim' : 'border-copper text-bone')}>
        {children}
      </div>
    </div>
  );
}

export function AgentPanel() {
  const {
    feed,
    askAgent,
    loading,
    hypotheses,
    setHypotheses,
    runAutopilot,
    agentStatus,
    providerModel,
    optimizeFor,
    treeOnly,
    budgetMin,
  } = useStore();
  const [text, setText] = useState('');
  const end = useRef<HTMLDivElement>(null);
  const thinking = agentStatus === 'thinking' || feed.some((f) => f.kind === 'thinking');
  const busy = agentStatus !== 'idle';

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [feed.length]);

  const send = (t: string) => {
    if (!t.trim() || thinking) return;
    setText('');
    void askAgent(t);
  };

  const statusLabel = agentStatus === 'autopilot' ? 'Autopilot' : agentStatus === 'thinking' ? 'Thinking' : 'Idle';

  return (
    <div className="flex h-full w-[380px] flex-col bg-ink">
      {/* header */}
      <div className="border-b border-line px-4 py-3">
        <div className="flex items-center gap-2">
          <h2 className="font-display text-[22px] font-extrabold uppercase leading-none tracking-wide text-bone">Pilot</h2>
          <span className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-[0.12em]" style={{ color: busy ? '#f7cd82' : '#b2cd8b' }}>
            <Led color={busy ? '#f7cd82' : '#b2cd8b'} pulse={busy} />
            {statusLabel}
          </span>
        </div>
        <p className="mt-1 font-mono text-[10px] text-mute">AI data scientist · {providerModel}</p>
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          <Tag>opt {optimizeFor}</Tag>
          <Tag>{treeOnly ? 'tree-only' : 'all models'}</Tag>
          {budgetMin ? <Tag>stop {budgetMin}m</Tag> : null}
        </div>
      </div>

      {/* autopilot */}
      <div className="border-b border-line px-4 py-3.5">
        <div className="mb-2 flex items-center justify-between">
          <Eyebrow>Hypotheses to test</Eyebrow>
          <span className="font-mono text-[11px] text-bone">
            {hypotheses}
            <span className="text-mute"> / 8</span>
          </span>
        </div>
        <div className="flex gap-[3px]">
          {Array.from({ length: 8 }).map((_, i) => (
            <button
              key={i}
              onClick={() => setHypotheses(i + 1)}
              aria-label={`${i + 1} hypotheses`}
              className={cn(
                'h-6 flex-1 border font-mono text-[10px] transition-colors',
                i < hypotheses ? 'border-copper bg-copper text-ink' : 'border-line text-mute hover:border-rule hover:text-bone'
              )}
            >
              {i + 1}
            </button>
          ))}
        </div>
        <Btn variant="primary" size="lg" className="relative mt-3 w-full overflow-hidden" onClick={runAutopilot} disabled={agentStatus === 'autopilot'}>
          {agentStatus === 'autopilot' && <span className="stripes absolute inset-0 opacity-70" />}
          <span className="relative">{agentStatus === 'autopilot' ? 'Autopilot running…' : 'Run full autopilot  ▶'}</span>
        </Btn>
      </div>

      {/* feed */}
      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
        {loading && [0, 1, 2].map((i) => <div key={i} className="skeleton h-24" style={{ animationDelay: `${i * 140}ms` }} />)}
        {feed.map((f) => {
          if (f.kind === 'suggestion') return <SpecSheet key={f.id} item={f} />;
          if (f.kind === 'checkpoint') return <CheckpointCard key={f.id} item={f} />;
          if (f.kind === 'thinking')
            return (
              <Entry key={f.id} who="Pilot" tone="agent">
                <span className="sq-dots flex h-5 items-center gap-1">
                  {[0, 1, 2].map((i) => (
                    <span key={i} className="h-1.5 w-1.5 bg-copper" />
                  ))}
                </span>
              </Entry>
            );
          if (f.from === 'user')
            return (
              <Entry key={f.id} who="You" tone="user">
                {f.text}
              </Entry>
            );
          return (
            <Entry key={f.id} who="Pilot" tone="agent">
              <Stream text={f.text} on={!!f.stream} />
            </Entry>
          );
        })}
        <div ref={end} />
      </div>

      {/* composer */}
      <form
        onSubmit={(e) => {
          e.preventDefault();
          send(text);
        }}
        className="border-t border-line"
      >
        <textarea
          rows={3}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              send(text);
            }
          }}
          placeholder="Steer the agent (e.g., focus on temporal features)..."
          className="block w-full resize-none bg-transparent px-4 pt-3 text-[13px] leading-relaxed text-bone placeholder:text-mute/70 focus:outline-none"
        />
        <div className="flex items-center justify-between px-4 pb-3">
          <span className="font-mono text-[10px] uppercase tracking-[0.1em] text-mute">⇧ + ↵ newline</span>
          <Btn type="submit" variant="primary" size="sm" disabled={!text.trim() || thinking}>
            Enter ↵
          </Btn>
        </div>
      </form>
    </div>
  );
}
