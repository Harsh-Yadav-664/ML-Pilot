import { useMemo } from 'react';
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
} from 'reactflow';
import type { SchemaEdge, SchemaGraphData, SchemaTable } from '../api/connections';
import { cn } from '../utils/cn';

const NODE_W = 232;
const COL_GAP = 96;
const ROW_GAP = 40;
const NODE_H = 96;

interface TableNodeData {
  table: SchemaTable;
  selected: boolean;
}

export const compact = (n: number) =>
  n >= 1e9 ? `${(n / 1e9).toFixed(1)}B` : n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : n >= 1e4 ? `${Math.round(n / 1e3)}k` : n.toLocaleString('en-US');

function TableNode({ data }: NodeProps<TableNodeData>) {
  const { table, selected } = data;
  return (
    <div
      data-testid={`table-${table.key}`}
      className={cn(
        'relative cursor-pointer border bg-panel transition-[border-color,box-shadow] duration-150',
        selected ? 'border-copper shadow-[4px_4px_0_0_rgba(224,162,79,0.22)]' : 'border-line hover:border-rule'
      )}
      style={{ width: NODE_W, minHeight: NODE_H }}
    >
      <Handle type="target" position={Position.Right} />
      <Handle type="source" position={Position.Left} />
      <div className="flex items-baseline justify-between gap-2 px-3 pt-2.5">
        <code className="truncate font-mono text-[12.5px] font-semibold text-bone">{table.key}</code>
        <span className="shrink-0 font-mono text-[10px] text-mute">
          {table.row_count_estimated ? '~' : ''}
          {compact(table.row_count)} rows
        </span>
      </div>
      <div className="px-3 pb-1 pt-1 font-mono text-[10px] text-mute">
        {table.columns.length} columns
        {table.primary_key.length > 0 && <> · key {table.primary_key.join(', ')}</>}
      </div>
      <div className="flex flex-wrap gap-1 px-3 pb-2.5">
        {table.time_column ? (
          <span className="border border-sage/50 px-1 font-mono text-[9.5px] text-sage">time: {table.time_column}</span>
        ) : table.is_static ? (
          <span className="border border-rule px-1 font-mono text-[9.5px] text-mute">static</span>
        ) : (
          <span className="border border-rule px-1 font-mono text-[9.5px] text-mute">no time column</span>
        )}
        {table.time_leakage_hint && (
          <span className="border border-clay/60 px-1 font-mono text-[9.5px] text-clay" title={table.time_leakage_hint}>
            leakage risk
          </span>
        )}
      </div>
    </div>
  );
}

const nodeTypes = { table: TableNode };

/** Parents (referenced tables) go to the left, the tables that reference them to the right. */
export function layoutLevels(tables: SchemaTable[], edges: SchemaEdge[]): Map<string, number> {
  const level = new Map(tables.map((t) => [t.key, 0]));
  // Longest path from a root; the bound keeps a cycle of tables from looping.
  for (let pass = 0; pass < tables.length; pass++) {
    let changed = false;
    for (const e of edges) {
      if (e.from_table === e.to_table) continue;
      const want = (level.get(e.to_table) ?? 0) + 1;
      if (want > (level.get(e.from_table) ?? 0) && want < tables.length) {
        level.set(e.from_table, want);
        changed = true;
      }
    }
    if (!changed) break;
  }
  return level;
}

const edgeStyle = (e: SchemaEdge) =>
  e.source === 'declared'
    ? { stroke: '#e0a24f', strokeWidth: 1.6 }
    : e.source === 'user'
      ? { stroke: '#8fae8b', strokeWidth: 1.6 }
      : { stroke: '#a08d76', strokeWidth: 1.4, strokeDasharray: '6 4' };

export function describeEdge(e: SchemaEdge): string {
  const cols = `${e.from_table}.${e.from_columns.join(',')} → ${e.to_table}.${e.to_columns.join(',')}`;
  const how =
    e.source === 'declared'
      ? 'declared foreign key'
      : e.source === 'user'
        ? 'added by you'
        : `inferred, ${Math.round(e.confidence * 100)}% confidence${e.overlap != null ? `, ${Math.round(e.overlap * 100)}% of sampled keys found in the parent` : ''}`;
  return `${cols} (${how}, ${e.cardinality})`;
}

function GraphInner({ graph, selected, onSelect }: { graph: SchemaGraphData; selected: string | null; onSelect: (key: string) => void }) {
  const { nodes, edges } = useMemo(() => {
    const levels = layoutLevels(graph.tables, graph.edges);
    const rows = new Map<number, number>();
    const nodes: Node<TableNodeData>[] = [...graph.tables]
      .sort((a, b) => (levels.get(a.key) ?? 0) - (levels.get(b.key) ?? 0) || b.row_count - a.row_count)
      .map((table) => {
        const lvl = levels.get(table.key) ?? 0;
        const row = rows.get(lvl) ?? 0;
        rows.set(lvl, row + 1);
        return {
          id: table.key,
          type: 'table',
          position: { x: lvl * (NODE_W + COL_GAP), y: row * (NODE_H + ROW_GAP) },
          data: { table, selected: table.key === selected },
          draggable: true,
        };
      });
    const edges: Edge[] = graph.edges.map((e, i) => ({
      id: `e${i}`,
      source: e.from_table,
      target: e.to_table,
      label: e.source === 'inferred' ? `${Math.round(e.confidence * 100)}%` : undefined,
      labelStyle: { fill: '#a08d76', fontSize: 10, fontFamily: 'monospace' },
      labelBgStyle: { fill: '#0c0907' },
      style: edgeStyle(e),
      markerEnd: { type: MarkerType.ArrowClosed, color: e.source === 'declared' ? '#e0a24f' : e.source === 'user' ? '#8fae8b' : '#a08d76' },
      data: { description: describeEdge(e) },
    }));
    return { nodes, edges };
  }, [graph, selected]);

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      nodeTypes={nodeTypes}
      onNodeClick={(_, node) => onSelect(node.id)}
      fitView
      fitViewOptions={{ padding: 0.15 }}
      minZoom={0.2}
      nodesConnectable={false}
      proOptions={{ hideAttribution: true }}
    >
      <Background variant={BackgroundVariant.Dots} gap={22} size={1} color="#2a2118" />
      <Controls showInteractive={false} />
    </ReactFlow>
  );
}

/** Tables as nodes, foreign keys as edges: solid copper = declared, dashed = inferred (with confidence), green = added by the user. */
export function SchemaGraph(props: { graph: SchemaGraphData; selected: string | null; onSelect: (key: string) => void }) {
  return (
    <ReactFlowProvider>
      <GraphInner {...props} />
    </ReactFlowProvider>
  );
}
