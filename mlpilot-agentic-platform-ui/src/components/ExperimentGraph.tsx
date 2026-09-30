import { useState, useEffect, useCallback } from 'react';
import ReactFlow, {
  Controls,
  Background,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
  Node,
  Edge,
  MarkerType,
  NodeTypes,
  Handle,
  Position,
} from 'reactflow';
import { getExperimentTree } from '../api';
import { Experiment } from '../types';

/* ---------- status meta ---------- */
const STATUS = {
  completed: { label: 'Completed', dot: 'bg-emerald-400', text: 'text-emerald-300', ring: 'ring-emerald-500/25', glow: '' },
  running: { label: 'Running', dot: 'bg-sky-400', text: 'text-sky-300', ring: 'ring-sky-500/40', glow: 'shadow-[0_0_0_1px_rgba(56,189,248,0.15)]' },
  failed: { label: 'Failed', dot: 'bg-red-400', text: 'text-red-300', ring: 'ring-red-500/25', glow: '' },
  queued: { label: 'Queued', dot: 'bg-zinc-500', text: 'text-zinc-400', ring: 'ring-zinc-600/40', glow: '' },
} as const;

/* ---------- custom node ---------- */
const ExperimentNode = ({ data }: { data: { exp: Experiment; isBaseline: boolean } }) => {
  const { exp, isBaseline } = data;
  const s = STATUS[exp.status];
  const ensF1 = exp.metrics?.ensemble_f1;
  const ensAcc = exp.metrics?.ensemble_accuracy;
  const f1 = exp.metrics?.f1;
  const acc = exp.metrics?.accuracy;
  
  const displayF1 = ensF1 !== undefined ? ensF1 : f1;
  const displayAcc = ensAcc !== undefined ? ensAcc : acc;

  const titleStr = exp.metrics
    ? `Single F1: ${f1?.toFixed(4)}\nSingle Acc: ${acc?.toFixed(4)}\nEnsemble F1: ${ensF1 !== undefined ? ensF1.toFixed(4) : 'N/A'}\nEnsemble Acc: ${ensAcc !== undefined ? ensAcc.toFixed(4) : 'N/A'}`
    : '';

  return (
    <div
      title={titleStr}
      className={`w-[228px] rounded-xl border border-zinc-800 bg-zinc-900/90 backdrop-blur ring-1 ring-inset ${s.ring} ${s.glow} transition-transform hover:-translate-y-0.5`}
    >
      <Handle type="target" position={Position.Top} className="!h-1.5 !w-1.5 !border-0 !bg-zinc-600" />

      <div className="flex items-center justify-between px-3 pt-2.5">
        <div className="flex items-center gap-1.5">
          {isBaseline && (
            <span className="rounded bg-indigo-500/15 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-indigo-300">
              Baseline
            </span>
          )}
          {ensF1 !== undefined && (
            <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 text-[9px] font-semibold uppercase tracking-wider text-emerald-300">
              ENS
            </span>
          )}
          <code className="font-mono text-[10px] text-zinc-500">{exp.id}</code>
        </div>
        <div className="flex items-center gap-1.5">
          <span className={`h-1.5 w-1.5 rounded-full ${s.dot} ${exp.status === 'running' ? 'animate-pulse' : ''}`} />
          <span className={`text-[10px] font-medium ${s.text}`}>{s.label}</span>
        </div>
      </div>

      <div className="px-3 pb-1 pt-1.5">
        <div className="text-[13px] font-semibold text-zinc-100">{exp.model_name}</div>
      </div>

      {/* metrics */}
      <div className="mx-3 mb-3 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-zinc-800 bg-zinc-800">
        <div className="bg-zinc-900 px-2.5 py-2">
          <div className="text-[9px] uppercase tracking-wider text-zinc-600">F1</div>
          <div className="font-mono text-[15px] font-semibold text-zinc-50">
            {displayF1 !== undefined ? displayF1.toFixed(3) : <span className="text-zinc-600">—</span>}
          </div>
        </div>
        <div className="bg-zinc-900 px-2.5 py-2">
          <div className="text-[9px] uppercase tracking-wider text-zinc-600">Accuracy</div>
          <div className="font-mono text-[15px] font-semibold text-zinc-50">
            {displayAcc !== undefined ? `${(displayAcc * 100).toFixed(1)}%` : <span className="text-zinc-600">—</span>}
          </div>
        </div>
      </div>

      <div className="flex items-center justify-between border-t border-zinc-800/80 px-3 py-2 text-[10px] text-zinc-500">
        <span className="font-mono">
          {exp.status === 'queued' ? 'pending' : `${exp.runtime_seconds.toFixed(1)}s`}
        </span>
        <span className="font-mono">{exp.model_name.slice(0, 3).toUpperCase()}</span>
      </div>

      <Handle type="source" position={Position.Bottom} className="!h-1.5 !w-1.5 !border-0 !bg-zinc-600" />
    </div>
  );
};

const nodeTypes: NodeTypes = { experiment: ExperimentNode };

/* ---------- graph ---------- */
export const ExperimentGraph = () => {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [loading, setLoading] = useState(true);

  const layout = useCallback((exps: Experiment[]) => {
    const children: Record<string, string[]> = {};
    const byId: Record<string, Experiment> = {};
    exps.forEach((e) => {
      byId[e.id] = e;
      if (e.parent_id) (children[e.parent_id] ??= []).push(e.id);
    });

    const roots = exps.filter((e) => !e.parent_id || !byId[e.parent_id]);
    const outNodes: Node[] = [];
    const outEdges: Edge[] = [];
    const NODE_W = 260;
    const NODE_H = 175;

    // compute subtree widths for tidy spacing
    const widthOf = (id: string): number => {
      const kids = children[id] ?? [];
      if (!kids.length) return NODE_W;
      return kids.reduce((sum, k) => sum + widthOf(k), 0);
    };

    const place = (id: string, left: number, depth: number, parent?: string) => {
      const w = widthOf(id);
      const x = left + w / 2 - NODE_W / 2;
      const y = depth * NODE_H;
      outNodes.push({
        id,
        type: 'experiment',
        position: { x, y },
        data: { exp: byId[id], isBaseline: !byId[id].parent_id },
        draggable: true,
      });
      if (parent) {
        outEdges.push({
          id: `${parent}-${id}`,
          source: parent,
          target: id,
          type: 'smoothstep',
          animated: byId[id].status === 'running',
          markerEnd: { type: MarkerType.ArrowClosed, color: '#3f3f46', width: 14, height: 14 },
          style: { stroke: '#3f3f46', strokeWidth: 1.5 },
        });
      }
      let cursor = left;
      (children[id] ?? []).forEach((k) => {
        place(k, cursor, depth + 1, id);
        cursor += widthOf(k);
      });
    };

    let cursor = 0;
    roots.forEach((r) => {
      place(r.id, cursor, 0);
      cursor += widthOf(r.id);
    });

    return { outNodes, outEdges };
  }, []);

  useEffect(() => {
    (async () => {
      setLoading(true);
      const data = await getExperimentTree();
      const { outNodes, outEdges } = layout(data);
      setNodes(outNodes);
      setEdges(outEdges);
      setLoading(false);
    })();
  }, [layout, setNodes, setEdges]);

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-zinc-800/80 px-4 py-3">
        <div className="flex items-center gap-2.5">
          <h3 className="text-sm font-medium text-zinc-200">Experiment lineage</h3>
          <span className="rounded-full bg-zinc-800 px-2 py-0.5 font-mono text-[10px] text-zinc-400">
            {nodes.length} runs
          </span>
        </div>
        <div className="flex items-center gap-3 text-[10px] text-zinc-500">
          {(['completed', 'running', 'queued', 'failed'] as const).map((k) => (
            <span key={k} className="flex items-center gap-1.5">
              <span className={`h-1.5 w-1.5 rounded-full ${STATUS[k].dot}`} />
              {STATUS[k].label}
            </span>
          ))}
        </div>
      </div>

      <div className="relative flex-1">
        {loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-zinc-950/40">
            <div className="h-6 w-6 animate-spin rounded-full border-2 border-zinc-700 border-t-emerald-400" />
          </div>
        )}
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.35 }}
          minZoom={0.4}
          maxZoom={1.5}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#1c1c1f" />
          <Controls showInteractive={false} position="bottom-right" />
        </ReactFlow>
      </div>
    </div>
  );
};
