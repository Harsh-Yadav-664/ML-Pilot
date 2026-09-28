import { useCallback, useState } from 'react';
import {
  GitBranch,
  LayoutGrid,
  Database,
  Sparkles,
  Settings,
  Search,
  ChevronDown,
  GitCommitHorizontal,
  Command,
} from 'lucide-react';
import { DataReadinessDashboard } from './components/DataReadinessDashboard';
import { AIAgentPanel } from './components/AIAgentPanel';
import { ExperimentGraph } from './components/ExperimentGraph';
import { DatasetUploader } from './components/DatasetUploader';

export default function App() {
  const [datasetPath, setDatasetPath] = useState<string | null>(null);
  const [targetColumn, setTargetColumn] = useState<string | null>(null);
  const [showUploader, setShowUploader] = useState(true); // show by default on load

  const [graphKey, setGraphKey] = useState(0);
  const handleExperimentStart = useCallback(() => setGraphKey((k) => k + 1), []);

  const rail = [
    { icon: LayoutGrid, active: true },
    { icon: GitBranch, active: false },
    { icon: Database, active: false },
    { icon: Sparkles, active: false },
  ];

  return (
    <div className="flex h-screen overflow-hidden bg-zinc-950 text-zinc-100">
      {showUploader && (
        <DatasetUploader 
          onSuccess={(path, target) => {
            setDatasetPath(path);
            setTargetColumn(target);
            setShowUploader(false);
          }}
          onCancel={() => {
            if (datasetPath && targetColumn) setShowUploader(false);
          }}
        />
      )}

      {/* Icon rail */}
      <aside className="flex w-14 flex-col items-center border-r border-zinc-800/80 py-4">
        <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-400 to-teal-500 text-zinc-950 shadow-lg shadow-emerald-500/10">
          <GitCommitHorizontal className="h-5 w-5" strokeWidth={2.5} />
        </div>
        <nav className="mt-6 flex flex-1 flex-col items-center gap-1">
          {rail.map((r, i) => {
            const Icon = r.icon;
            return (
              <button
                key={i}
                className={`flex h-9 w-9 items-center justify-center rounded-lg transition-colors ${
                  r.active
                    ? 'bg-zinc-800 text-zinc-100'
                    : 'text-zinc-500 hover:bg-zinc-800/60 hover:text-zinc-300'
                }`}
              >
                <Icon className="h-[18px] w-[18px]" />
              </button>
            );
          })}
        </nav>
        <button className="flex h-9 w-9 items-center justify-center rounded-lg text-zinc-500 hover:bg-zinc-800/60 hover:text-zinc-300">
          <Settings className="h-[18px] w-[18px]" />
        </button>
        <div className="mt-3 h-7 w-7 rounded-full bg-gradient-to-br from-indigo-400 to-violet-500 ring-2 ring-zinc-800" />
      </aside>

      {/* Main workspace */}
      <div className="flex min-w-0 flex-1 flex-col">
        {/* Top bar */}
        <header className="flex h-14 flex-shrink-0 items-center justify-between border-b border-zinc-800/80 px-5">
          <div className="flex items-center gap-2 text-sm">
            <span className="font-semibold tracking-tight text-zinc-100">MLPilot</span>
            <span className="text-zinc-700">/</span>
            <button 
              onClick={() => setShowUploader(true)}
              className="flex items-center gap-1.5 rounded-md px-2 py-1 text-zinc-300 hover:bg-zinc-800"
            >
              {targetColumn ? `Target: ${targetColumn}` : 'Select dataset...'}
              <ChevronDown className="h-3.5 w-3.5 text-zinc-500" />
            </button>
            <span className="ml-1 rounded-full bg-emerald-500/10 px-2 py-0.5 text-[11px] font-medium text-emerald-300 ring-1 ring-inset ring-emerald-500/20">
              v1.0
            </span>
          </div>

          <div className="flex items-center gap-3">
            <button className="flex items-center gap-2 rounded-lg border border-zinc-800 bg-zinc-900/60 px-3 py-1.5 text-[13px] text-zinc-500 hover:border-zinc-700">
              <Search className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Search runs</span>
              <kbd className="ml-4 hidden items-center gap-0.5 rounded border border-zinc-700 px-1 py-0.5 text-[10px] sm:flex">
                <Command className="h-2.5 w-2.5" />K
              </kbd>
            </button>
            <div className="flex items-center gap-1.5 text-[11px] text-zinc-500">
              <span className="h-1.5 w-1.5 rounded-full bg-emerald-400" />
              active
            </div>
          </div>
        </header>

        {/* Body: content + agent split */}
        <div className="flex min-h-0 flex-1">
          {/* Center column */}
          <main className="min-w-0 flex-1 overflow-y-auto bg-grid p-5">
            <div className="mb-5">
              <h1 className="text-lg font-semibold tracking-tight text-zinc-100">
                Data readiness
              </h1>
              <p className="mt-0.5 text-[13px] text-zinc-500">
                Snapshot & leakage audit for the current training window.
              </p>
            </div>

            {datasetPath && targetColumn ? (
              <DataReadinessDashboard datasetPath={datasetPath} targetColumn={targetColumn} />
            ) : (
              <div className="rounded-xl border border-dashed border-zinc-800 p-12 text-center text-zinc-500">
                Please upload a dataset to see metrics.
              </div>
            )}

            <div className="mb-4 mt-8 flex items-baseline justify-between">
              <div>
                <h2 className="text-lg font-semibold tracking-tight text-zinc-100">
                  Experiments
                </h2>
                <p className="mt-0.5 text-[13px] text-zinc-500">
                  Branching lineage from the baseline model.
                </p>
              </div>
            </div>

            <div className="h-[560px] overflow-hidden rounded-xl border border-zinc-800/80 bg-zinc-900/20">
              <ExperimentGraph key={graphKey} />
            </div>
          </main>

          {/* Agent panel */}
          <aside className="hidden w-[380px] flex-shrink-0 border-l border-zinc-800/80 bg-zinc-900/20 xl:flex xl:flex-col">
            {datasetPath && targetColumn ? (
              <AIAgentPanel 
                onExperimentStart={handleExperimentStart} 
                datasetPath={datasetPath}
                targetColumn={targetColumn}
              />
            ) : (
              <div className="flex h-full items-center justify-center p-8 text-center text-sm text-zinc-500">
                Upload a dataset to wake up the AI Data Scientist.
              </div>
            )}
          </aside>
        </div>
      </div>
    </div>
  );
}
