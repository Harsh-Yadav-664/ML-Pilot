import { useEffect, useMemo, useRef, useState } from 'react';
import { WORKERS, useStore } from '../store';
import { View } from '../types';
import { API_BASE_URL } from '../api';
import { algoLabel, f3 } from '../lib';
import { cn } from '../utils/cn';
import { Eyebrow, Kbd, Led } from '../ui';

export function Toasts() {
  const { toasts, dismissToast } = useStore();
  const bar = { ok: '#b2cd8b', err: '#e0705a', warn: '#f7cd82', info: '#a08d76' } as const;
  return (
    <div className="pointer-events-none fixed bottom-10 left-16 z-50 flex flex-col items-start gap-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          className="toast-in pointer-events-auto relative flex items-center gap-3 border border-rule bg-panel py-2 pl-4 pr-2 text-[12.5px] text-bone shadow-[4px_4px_0_0_rgba(0,0,0,0.55)]"
        >
          <span className="absolute inset-y-0 left-0 w-[3px]" style={{ background: bar[t.tone] }} />
          <span>{t.text}</span>
          {t.action && (
            <button
              onClick={() => {
                t.action!.run();
                dismissToast(t.id);
              }}
              className="border border-copper/60 px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-[0.1em] text-copper-2 transition-colors hover:bg-copper hover:text-ink"
            >
              {t.action.label}
            </button>
          )}
          <button onClick={() => dismissToast(t.id)} className="px-1 font-mono text-[11px] text-mute transition-colors hover:text-bone">
            ✕
          </button>
        </div>
      ))}
    </div>
  );
}

export function StatusBar() {
  const { live, experiments, champion, readiness, setPaletteOpen } = useStore();
  const running = experiments.filter((e) => e.status === 'running').length;
  const queued = experiments.filter((e) => e.status === 'queued').length;
  return (
    <footer className="flex h-7 shrink-0 items-center justify-between border-t border-line bg-ink px-3 font-mono text-[10px] uppercase tracking-[0.1em] text-mute">
      <div className="flex items-center gap-5">
        <span className="flex items-center gap-1.5" title={API_BASE_URL}>
          <Led color={live ? '#b2cd8b' : '#f7cd82'} />
          {live ? 'api connected' : 'api · sample data'}
        </span>
        <span className="hidden items-center gap-2 sm:flex">
          workers
          <span className="flex gap-[2px]">
            {Array.from({ length: WORKERS }).map((_, i) => (
              <span key={i} className={cn('h-2.5 w-2 transition-colors', i < running ? 'bg-copper-2' : 'bg-line')} />
            ))}
          </span>
          {running}/{WORKERS}
          {queued > 0 && <span className="text-copper-2/80">· {queued} queued</span>}
        </span>
      </div>
      <div className="flex items-center gap-5">
        <span className="hidden md:inline">readiness {readiness}</span>
        {champion && <span className="hidden text-copper-2 md:inline">best f1 {f3(champion.metrics.f1)}</span>}
        <button onClick={() => setPaletteOpen(true)} className="flex items-center gap-1.5 transition-colors hover:text-bone">
          commands <Kbd className="h-4 min-w-4">⌘K</Kbd>
        </button>
      </div>
    </footer>
  );
}

interface Cmd {
  id: string;
  group: string;
  label: string;
  hint?: string;
  run: () => void;
}

export function CommandPalette() {
  const s = useStore();
  const [q, setQ] = useState('');
  const [idx, setIdx] = useState(0);
  const list = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (s.paletteOpen) {
      setQ('');
      setIdx(0);
    }
  }, [s.paletteOpen]);

  const cmds = useMemo<Cmd[]>(() => {
    const failed = s.experiments.filter((e) => e.status === 'failed');
    const nav: [View, string, string][] = [
      ['dashboard', 'Go to Dashboard', '1'],
      ['experiments', 'Go to Experiments', '2'],
      ['deployments', 'Go to Deployments', '3'],
      ['settings', 'Go to Settings', '4'],
    ];
    const base: Cmd[] = [
      ...nav.map(([v, label, hint]) => ({ id: v, group: 'Navigate', label, hint, run: () => s.setView(v) })),
      { id: 'a1', group: 'Actions', label: 'Run next feature idea', run: s.runNext },
      { id: 'a2', group: 'Actions', label: 'Run full Autopilot', run: s.runAutopilot },
      { id: 'a3', group: 'Actions', label: s.agentOpen ? 'Hide agent panel' : 'Show agent panel', hint: '\\', run: () => s.setAgentOpen(!s.agentOpen) },
      { id: 'a4', group: 'Actions', label: 'Upload dataset', run: () => s.setUploadOpen(true) },
      { id: 'a5', group: 'Actions', label: 'Apply AI Auto-Clean', run: s.autoClean },
    ];
    failed.forEach((e) =>
      base.push({ id: `r${e.id}`, group: 'Actions', label: `Retry ${e.id} · ${algoLabel(e.model_name)}`, run: () => s.retry(e.id) })
    );
    if (s.champion)
      base.push({
        id: 'ch',
        group: 'Actions',
        label: `Inspect champion ${s.champion.id}`,
        hint: `F1 ${f3(s.champion.metrics.f1)}`,
        run: () => {
          s.setView('experiments');
          s.select(s.champion!.id);
        },
      });
    s.experiments.forEach((e) =>
      base.push({
        id: `e${e.id}`,
        group: 'Runs',
        label: `${e.id} · ${e.title ?? algoLabel(e.model_name)}`,
        hint: e.status === 'completed' ? `F1 ${f3(e.metrics.f1)}` : e.status,
        run: () => {
          s.setView('experiments');
          s.select(e.id);
        },
      })
    );
    return base;
  }, [s]);

  const filtered = useMemo(() => {
    const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    return cmds.filter((c) => terms.every((t) => (c.label + ' ' + c.group).toLowerCase().includes(t)));
  }, [cmds, q]);

  useEffect(() => setIdx(0), [q]);
  useEffect(() => {
    list.current?.querySelector(`[data-i="${idx}"]`)?.scrollIntoView({ block: 'nearest' });
  }, [idx]);

  if (!s.paletteOpen) return null;
  const close = () => s.setPaletteOpen(false);
  const exec = (c?: Cmd) => {
    if (!c) return;
    close();
    c.run();
  };

  let lastGroup = '';
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-black/70 px-4 pt-[14vh]" onMouseDown={close}>
      <div className="pop ticks w-full max-w-[580px] border border-rule bg-panel shadow-[8px_8px_0_0_rgba(0,0,0,0.6)]" onMouseDown={(e) => e.stopPropagation()}>
        <div className="flex items-center gap-3 border-b border-line px-4">
          <span className="font-mono text-[12px] text-copper">›</span>
          <input
            autoFocus
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'ArrowDown') {
                e.preventDefault();
                setIdx((i) => Math.min(filtered.length - 1, i + 1));
              } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                setIdx((i) => Math.max(0, i - 1));
              } else if (e.key === 'Enter') exec(filtered[idx]);
              else if (e.key === 'Escape') close();
            }}
            placeholder="Search commands, runs, actions"
            className="h-12 flex-1 bg-transparent font-mono text-[13px] text-bone placeholder:text-mute focus:outline-none"
          />
          <Kbd>esc</Kbd>
        </div>
        <div ref={list} className="max-h-[340px] overflow-y-auto py-1">
          {filtered.map((c, i) => {
            const head = c.group !== lastGroup;
            lastGroup = c.group;
            return (
              <div key={c.id}>
                {head && <Eyebrow className="block px-4 pb-1 pt-3">{c.group}</Eyebrow>}
                <button
                  data-i={i}
                  onMouseMove={() => setIdx(i)}
                  onClick={() => exec(c)}
                  className={cn(
                    'relative flex w-full items-center justify-between px-4 py-2 text-left text-[13px] transition-colors',
                    i === idx ? 'bg-panel-2 text-bone' : 'text-bone-dim'
                  )}
                >
                  {i === idx && <span className="absolute inset-y-0 left-0 w-[3px] bg-copper" />}
                  <span className="truncate">{c.label}</span>
                  <span className="ml-3 shrink-0 font-mono text-[10.5px] text-mute">{c.hint}</span>
                </button>
              </div>
            );
          })}
          {!filtered.length && <div className="px-4 py-8 text-center font-mono text-[12px] text-mute">no results for “{q}”</div>}
        </div>
        <div className="flex items-center gap-4 border-t border-line px-4 py-2 font-mono text-[10px] uppercase tracking-[0.1em] text-mute">
          <span className="flex items-center gap-1"><Kbd>↑</Kbd><Kbd>↓</Kbd> move</span>
          <span className="flex items-center gap-1"><Kbd>↵</Kbd> run</span>
        </div>
      </div>
    </div>
  );
}
