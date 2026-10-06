import { useEffect } from 'react';
import { Database, GitBranch, Radio, Settings as SettingsIcon } from 'lucide-react';
import { StoreProvider, useStore } from './store';
import { View } from './types';
import { cn } from './utils/cn';
import { Btn, Kbd, Led } from './ui';
import { Dashboard, UploadModal } from './views/Dashboard';
import { ExperimentsView } from './views/ExperimentsView';
import { Deployments } from './views/Deployments';
import { Settings } from './views/Settings';
import { AgentPanel } from './components/AgentPanel';
import { connection } from './api';
import { CommandPalette, ConnectionBanner, StatusBar, Toasts } from './components/Chrome';
import { Landing } from './Landing';
import { ConnectView } from './views/Connect';
import { useRoute } from './route';

/** A coffee bean in a caramel block. */
function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="h-8 w-8" fill="none" aria-label="MLPilot">
      <rect width="32" height="32" fill="#e0a24f" />
      <g transform="rotate(38 16 16)">
        <ellipse cx="16" cy="16" rx="7.5" ry="11" fill="#0c0907" />
        <path d="M16 5.6C20.6 11 11.4 21 16 26.4" stroke="#e0a24f" strokeWidth="1.7" strokeLinecap="round" />
      </g>
    </svg>
  );
}

const ALL_NAV: { key: View; label: string; icon: typeof Database }[] = [
  { key: 'dashboard', label: 'Dashboard', icon: Database },
  { key: 'experiments', label: 'Experiments', icon: GitBranch },
  { key: 'deployments', label: 'Deployments', icon: Radio },
  { key: 'settings', label: 'Settings', icon: SettingsIcon },
];
// Deployments has no backend behind it yet: shown only in Demo mode.
const NAV = ALL_NAV.filter((n) => connection.demo || n.key !== 'deployments');

function Shell() {
  const { started } = useStore();
  const route = useRoute();
  if (route === 'connect') return <ConnectView />;
  if (!started) return <Landing />;
  return <Workspace />;
}

function Workspace() {
  const s = useStore();
  const {
    view,
    setView,
    agentOpen,
    setAgentOpen,
    setPaletteOpen,
    selectedId,
    select,
    runNext,
    datasetName,
    target,
    setTarget,
    columns,
    agentStatus,
    setUploadOpen,
  } = s;
  const idx = Math.max(0, NAV.findIndex((n) => n.key === view));

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        setPaletteOpen(!s.paletteOpen);
        return;
      }
      const t = e.target as HTMLElement;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(t.tagName) || s.paletteOpen) return;
      if (e.key === 'Escape' && selectedId) select(null);
      if (e.key === '1') setView('dashboard');
      if (e.key === '2') setView('experiments');
      if (e.key === '3' && connection.demo) setView('deployments');
      if (e.key === '4') setView('settings');
      if (e.key === '\\') setAgentOpen(!agentOpen);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [s.paletteOpen, selectedId, agentOpen, select, setView, setAgentOpen, setPaletteOpen]);

  const busy = agentStatus !== 'idle';
  const statusLabel = agentStatus === 'autopilot' ? 'Autopilot' : agentStatus === 'thinking' ? 'Thinking' : 'Idle';

  return (
    <div className="flex h-screen flex-col overflow-hidden text-bone">
      <div className="guides" aria-hidden />
      <ConnectionBanner />
      <div className="flex min-h-0 flex-1">
        {/* ---------- rail ---------- */}
        <aside className="flex w-14 shrink-0 flex-col items-center border-r border-line bg-ink py-3">
          <Logo />
          <nav className="relative mt-7 flex flex-col">
            <span
              className="absolute -left-[13px] top-0 h-10 w-[3px] bg-copper transition-transform duration-300 ease-[cubic-bezier(.22,1,.36,1)]"
              style={{ transform: `translateY(${idx * 40}px)` }}
            />
            {NAV.map((n, i) => {
              const Icon = n.icon;
              const on = n.key === view;
              return (
                <button
                  key={n.key}
                  onClick={() => setView(n.key)}
                  className={cn(
                    'group relative flex h-10 w-10 items-center justify-center transition-colors',
                    on ? 'bg-panel-2 text-copper-2' : 'text-mute hover:bg-panel hover:text-bone'
                  )}
                >
                  <Icon className="h-[17px] w-[17px]" strokeWidth={1.6} />
                  <span className="pointer-events-none absolute left-full z-30 ml-3 flex -translate-x-1 items-center gap-2 whitespace-nowrap border border-rule bg-panel px-2 py-1 font-mono text-[10px] uppercase tracking-[0.1em] text-bone opacity-0 transition-all group-hover:translate-x-0 group-hover:opacity-100">
                    {n.label}
                    <Kbd>{i + 1}</Kbd>
                  </span>
                </button>
              );
            })}
          </nav>
          <span className="mt-auto rotate-180 font-display text-[15px] font-bold uppercase tracking-[0.34em] text-rule [writing-mode:vertical-rl]">
            MLPilot
          </span>
        </aside>

        {/* ---------- main column ---------- */}
        <div className="flex min-w-0 flex-1 flex-col">
          <header className="flex h-11 shrink-0 items-center justify-between gap-3 border-b border-line bg-ink px-4">
            <div className="flex min-w-0 items-center gap-2.5 font-mono text-[11px] uppercase tracking-[0.1em]">
              <span className="font-display text-[17px] font-extrabold normal-case tracking-wide text-bone">MLPilot</span>
              <span className="text-rule">/</span>
              <button onClick={() => setUploadOpen(true)} className="truncate text-mute transition-colors hover:text-copper-2" title="Replace dataset">
                {datasetName}
              </button>
              <span className="text-rule">/</span>
              <select
                value={target}
                onChange={(e) => setTarget(e.target.value)}
                className="max-w-[150px] cursor-pointer truncate bg-transparent text-copper-2 focus:outline-none"
                title="Target column"
              >
                {columns.map((c) => (
                  <option key={c.name} value={c.name} className="bg-panel text-bone">
                    {c.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center gap-2.5">
              <div className="hidden items-center gap-2 border border-line px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.12em] sm:flex">
                <span className="text-mute">Agent</span>
                <Led color={busy ? '#f7cd82' : '#b2cd8b'} pulse={busy} />
                <span className={busy ? 'text-copper-2' : 'text-bone-dim'}>{statusLabel}</span>
              </div>
              <button
                onClick={() => setPaletteOpen(true)}
                className="hidden h-7 items-center gap-2 border border-line px-2 font-mono text-[10px] uppercase tracking-[0.1em] text-mute transition-colors hover:border-rule hover:text-bone sm:flex"
              >
                Search <Kbd>⌘K</Kbd>
              </button>
              <Btn variant="primary" size="sm" onClick={runNext}>
                ▶ Run next
              </Btn>
              <button
                onClick={() => setAgentOpen(!agentOpen)}
                title="Toggle agent panel (\)"
                className={cn(
                  'h-7 border px-2 font-mono text-[10px] uppercase tracking-[0.1em] transition-colors',
                  agentOpen ? 'border-copper/60 text-copper-2' : 'border-line text-mute hover:text-bone'
                )}
              >
                Agent {agentOpen ? '▸' : '◂'}
              </button>
            </div>
          </header>

          <div className="relative flex min-h-0 flex-1">
            <div className="relative min-w-0 flex-1">
              <main className="absolute inset-0 overflow-y-auto">
                {view === 'dashboard' && <Dashboard />}
                {view === 'experiments' && <ExperimentsView />}
                {view === 'deployments' && connection.demo && <Deployments />}
                {view === 'settings' && <Settings />}
              </main>
            </div>
            <aside
              className={cn(
                'shrink-0 overflow-hidden border-l border-line bg-ink transition-[width] duration-300 ease-[cubic-bezier(.22,1,.36,1)]',
                agentOpen ? 'w-[380px]' : 'w-0 border-l-0',
                agentOpen && 'max-xl:absolute max-xl:inset-y-0 max-xl:right-0 max-xl:z-30 max-xl:shadow-2xl max-xl:shadow-black'
              )}
            >
              <AgentPanel />
            </aside>
          </div>
        </div>
      </div>
      <StatusBar />
      <CommandPalette />
      <Toasts />
      <UploadModal />
    </div>
  );
}

export default function App() {
  return (
    <StoreProvider>
      <Shell />
    </StoreProvider>
  );
}
