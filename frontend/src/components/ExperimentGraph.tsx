import { useEffect, useMemo } from 'react';
import ReactFlow, {
  Background,
  BackgroundVariant,
  Controls,
  Edge,
  Handle,
  MarkerType,
  Node,
  NodeProps,
  Position,
  ReactFlowProvider,
  useReactFlow,
} from 'reactflow';
import { useStore } from '../store';
import { Experiment } from '../types';
import { algoLabel, pct } from '../lib';
import { cn } from '../utils/cn';
import { Delta, Eyebrow, STATUS, StatusChip } from '../ui';

interface NodeData {
  exp: Experiment;
  isBaseline: boolean;
  isChampion: boolean;
  delta?: number;
  selected: boolean;
}

/** Each run is a tag: status spine, id strip, big condensed score. */
function ExpNode({ data }: NodeProps<NodeData>) {
  const { exp, isBaseline, isChampion, delta, selected } = data;
  const st = STATUS[exp.status];
  const f1 = exp.metrics.f1;
  return (
    <div
      className={cn(
        'node-in relative w-[248px] cursor-pointer border bg-panel transition-[transform,box-shadow,border-color] duration-200 hover:-translate-x-px hover:-translate-y-px',
        selected ? 'border-copper shadow-[5px_5px_0_0_rgba(224,162,79,0.22)]' : 'border-line hover:border-rule'
      )}
    >
      <span className="absolute inset-y-0 left-0 w-[3px]" style={{ background: st.hex }} />
      {(isChampion || isBaseline) && (
        <span
          className={cn(
            'absolute -top-[9px] right-3 px-1.5 font-mono text-[9px] font-semibold uppercase tracking-[0.14em]',
            isChampion ? 'bg-copper text-ink' : 'bg-rule text-bone-dim'
          )}
        >
          {isChampion ? 'Champion' : 'Baseline'}
        </span>
      )}
      <Handle type="target" position={Position.Left} />
      <div className="flex items-center justify-between px-4 pt-3">
        <code className="font-mono text-[10px] text-mute">{exp.id}</code>
        <StatusChip status={exp.status} />
      </div>
      <div className="px-4 pt-2">
        <div className="font-display text-[23px] font-bold uppercase leading-none tracking-wide text-bone">
          {algoLabel(exp.model_name)}
        </div>
        <div className="mt-1 truncate font-mono text-[10.5px] text-mute">{exp.title ?? exp.model_name}</div>
      </div>
      <div className="flex items-end justify-between px-4 pb-3 pt-2.5">
        <div className="flex items-baseline gap-2">
          <span className="font-display text-[40px] font-extrabold leading-none tabular-nums text-bone">
            {f1 !== undefined ? f1.toFixed(3) : <span className="text-rule">———</span>}
          </span>
          {delta !== undefined && <Delta value={delta} />}
        </div>
        <div className="text-right">
          <Eyebrow className="block text-[9px]">Acc</Eyebrow>
          <div className="font-mono text-[12px] tabular-nums text-bone-dim">{pct(exp.metrics.accuracy)}</div>
        </div>
      </div>
      {exp.status === 'running' ? (
        <div className="border-t border-dashed border-rule px-4 py-2">
          <div className="mb-1 flex justify-between font-mono text-[10px] text-mute">
            <span>{exp.runtime_seconds.toFixed(1)}s</span>
            <span>{Math.round(exp.progress ?? 0)}%</span>
          </div>
          <div className="h-[3px] bg-panel-2">
            <div className="stripes h-full bg-copper-2 transition-[width] duration-700 ease-out" style={{ width: `${exp.progress ?? 0}%` }} />
          </div>
        </div>
      ) : (
        <div className="flex justify-between border-t border-dashed border-rule px-4 py-1.5 font-mono text-[10px] text-mute">
          <span>{exp.status === 'queued' ? 'waiting for worker' : `${exp.runtime_seconds.toFixed(1)}s runtime`}</span>
          {exp.decision && exp.decision !== 'none' && (
            <span className="uppercase tracking-[0.12em] text-copper-2/90">[{exp.decision}]</span>
          )}
        </div>
      )}
      <Handle type="source" position={Position.Right} />
    </div>
  );
}

const nodeTypes = { exp: ExpNode };

const NODE_W = 248;
const COL = NODE_W + 124;
const ROW = 186;

function layout(
  exps: Experiment[],
  championId: string | undefined,
  selectedId: string | null
): { nodes: Node<NodeData>[]; edges: Edge[] } {
  const byId = new Map(exps.map((e) => [e.id, e]));
  const kids: Record<string, string[]> = {};
  exps.forEach((e) => {
    if (e.parent_id && byId.has(e.parent_id)) (kids[e.parent_id] ??= []).push(e.id);
  });
  const roots = exps.filter((e) => !e.parent_id || !byId.has(e.parent_id));
  const heightOf = (id: string): number => {
    const k = kids[id] ?? [];
    return k.length ? k.reduce((s, x) => s + heightOf(x), 0) : ROW;
  };
  const nodes: Node<NodeData>[] = [];
  const edges: Edge[] = [];
  const place = (id: string, top: number, depth: number, parent?: string) => {
    const e = byId.get(id)!;
    const h = heightOf(id);
    const p = parent ? byId.get(parent) : undefined;
    const delta = e.metrics.f1 !== undefined && p?.metrics.f1 !== undefined ? e.metrics.f1 - p.metrics.f1 : undefined;
    nodes.push({
      id,
      type: 'exp',
      position: { x: depth * COL, y: top + h / 2 - ROW / 2 },
      draggable: false,
      data: {
        exp: e,
        isBaseline: e.decision === 'baseline' || !e.parent_id,
        isChampion: id === championId,
        delta,
        selected: id === selectedId,
      },
    });
    if (parent) {
      const color = e.status === 'completed' ? '#5a4a38' : STATUS[e.status].hex;
      edges.push({
        id: `${parent}>${id}`,
        source: parent,
        target: id,
        type: 'step',
        animated: e.status === 'running',
        label: e.feature ? `+ ${e.feature}` : e.title,
        labelStyle: { fill: '#d8c9b4', fontSize: 10, fontFamily: 'Spline Sans Mono, monospace' },
        labelBgStyle: { fill: '#0c0907', fillOpacity: 1 },
        labelBgPadding: [6, 3],
        labelBgBorderRadius: 0,
        markerEnd: { type: MarkerType.ArrowClosed, color, width: 12, height: 12 },
        style: { stroke: color, strokeWidth: 1.5, strokeDasharray: e.status === 'failed' ? '4 4' : undefined },
      });
    }
    let cur = top;
    (kids[id] ?? []).forEach((k) => {
      place(k, cur, depth + 1, id);
      cur += heightOf(k);
    });
  };
  let top = 0;
  roots.forEach((r) => {
    place(r.id, top, 0);
    top += heightOf(r.id);
  });
  return { nodes, edges };
}

function Canvas({ focusOnSelect }: { focusOnSelect?: boolean }) {
  const { experiments, champion, selectedId, select, setView } = useStore();
  const { fitView } = useReactFlow();
  const { nodes, edges } = useMemo(
    () => layout(experiments, champion?.id, selectedId),
    [experiments, champion?.id, selectedId]
  );
  const shape = experiments.map((e) => e.id).join(',');

  useEffect(() => {
    const t = setTimeout(() => fitView({ padding: 0.2, duration: 650, maxZoom: 1.05 }), 60);
    return () => clearTimeout(t);
  }, [shape, fitView]);

  return (
    <div className="relative h-full w-full">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onNodeClick={(_, n) => {
          if (focusOnSelect) {
            select(n.id);
            setView('experiments');
            return;
          }
          select(n.id === selectedId ? null : n.id);
        }}
        onPaneClick={() => select(null)}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        minZoom={0.3}
        maxZoom={1.4}
        fitView
        fitViewOptions={{ padding: 0.2, maxZoom: 1.05 }}
        proOptions={{ hideAttribution: true }}
      >
        <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#2b2118" />
        <Controls showInteractive={false} position="bottom-left" />
      </ReactFlow>
      <button
        onClick={() => fitView({ padding: 0.2, duration: 500, maxZoom: 1.05 })}
        className="absolute right-3 top-3 border border-line bg-panel px-2 py-1 font-mono text-[10px] uppercase tracking-[0.12em] text-mute transition-colors hover:border-copper hover:text-copper-2"
      >
        Fit view
      </button>
    </div>
  );
}

export function ExperimentGraph({ focusOnSelect }: { focusOnSelect?: boolean } = {}) {
  return (
    <ReactFlowProvider>
      <Canvas focusOnSelect={focusOnSelect} />
    </ReactFlowProvider>
  );
}
