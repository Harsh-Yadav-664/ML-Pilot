import type { components } from './schema';
import api, { ensureProject } from './index';

/** Saved database connections, their schema graph and column statistics (backend issues #43, #45, #96). */
type S = components['schemas'];
export type ConnectionCreate = S['ConnectionCreate'];
export type ConnectionRead = S['ConnectionRead'];
export type ConnectionUpdate = S['ConnectionUpdate'];
export type ConnectionTestResult = S['ConnectionTestResult'];
export type SchemaGraphData = S['SchemaGraph'];
export type SchemaOverrides = S['SchemaOverrides'];
export type SchemaTable = S['Table'];
export type SchemaEdge = S['Edge'];
export type TableStatsData = S['TableStatsRead'];

async function base(): Promise<string> {
  const project = await ensureProject();
  return `/projects/${project.id}/connections`;
}

export async function listConnections(): Promise<ConnectionRead[]> {
  return (await api.get<ConnectionRead[]>(`${await base()}/`)).data;
}

export async function createConnection(body: ConnectionCreate): Promise<ConnectionRead> {
  return (await api.post<ConnectionRead>(`${await base()}/`, body)).data;
}

export async function updateConnection(id: string, body: ConnectionUpdate): Promise<ConnectionRead> {
  return (await api.patch<ConnectionRead>(`${await base()}/${id}`, body)).data;
}

export async function deleteConnection(id: string): Promise<void> {
  await api.delete(`${await base()}/${id}`);
}

/** Opens the database as the stored role and reports version, latency and whether the role can write. */
export async function testConnection(id: string): Promise<ConnectionTestResult> {
  // Reaching a remote database can take a while: allow longer than the default 10 s.
  return (await api.post<ConnectionTestResult>(`${await base()}/${id}/test`, undefined, { timeout: 60_000 })).data;
}

export async function getSchema(id: string): Promise<SchemaGraphData> {
  return (await api.get<SchemaGraphData>(`${await base()}/${id}/schema`, { timeout: 120_000 })).data;
}

/** Corrections the user makes to the inferred graph; the response is the graph with them applied. */
export async function patchSchema(id: string, overrides: SchemaOverrides): Promise<SchemaGraphData> {
  return (await api.patch<SchemaGraphData>(`${await base()}/${id}/schema`, overrides, { timeout: 120_000 })).data;
}

export async function getTableStats(id: string, table: string, refresh = false): Promise<TableStatsData> {
  return (
    await api.get<TableStatsData>(`${await base()}/${id}/tables/${encodeURIComponent(table)}/stats`, {
      params: { refresh },
      timeout: 120_000,
    })
  ).data;
}
