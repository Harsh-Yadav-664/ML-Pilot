import type { components } from './schema';
import api, { ensureProject } from './index';

/** Prediction task specs: draft from a question, save, preview the labels, confirm (backend issues #49, #50, #53). */
type S = components['schemas'];
export type TaskDraft = S['TaskDraft'];
export type TaskSpecRead = S['TaskSpecRead'];
export type LabelPreview = S['LabelPreview'];
export type SpecIssue = S['SpecIssueRead'];

async function base(): Promise<string> {
  const project = await ensureProject();
  return `/projects/${project.id}/tasks`;
}

/** Nothing is saved: the answer is a spec to read, a question to ask the user, or the problems that remain. */
export async function draftTask(question: string, connectionId: string, dataVersionId: string | null): Promise<TaskDraft> {
  // Profiling a database's columns for the prompt can take a while on a first call.
  return (
    await api.post<TaskDraft>(
      `${await base()}/draft`,
      { question, connection_id: connectionId, data_version_id: dataVersionId },
      { timeout: 180_000 },
    )
  ).data;
}

type SaveInput = { yaml: string; connectionId: string; dataVersionId: string | null; draftSource: TaskDraft['source'] | null };

/** Saves the spec as a draft (a new task, or a new save of the same task when `taskId` is given). */
export async function saveTask(input: SaveInput, taskId: string | null): Promise<TaskSpecRead> {
  const body = {
    yaml: input.yaml,
    connection_id: input.connectionId,
    data_version_id: input.dataVersionId,
    draft_source: input.draftSource,
  };
  const url = taskId ? `${await base()}/${taskId}` : `${await base()}/`;
  return (await (taskId ? api.put<TaskSpecRead>(url, body) : api.post<TaskSpecRead>(url, body))).data;
}

/** Builds the labels of a saved task: the SQL, eligible entities and positives per cutoff, feasibility. */
export async function previewLabels(taskId: string, dataVersionId: string | null): Promise<LabelPreview> {
  return (
    await api.post<LabelPreview>(`${await base()}/${taskId}/preview-labels`, { data_version_id: dataVersionId }, { timeout: 300_000 })
  ).data;
}

export async function confirmTask(taskId: string, dataVersionId: string | null): Promise<TaskSpecRead> {
  return (await api.post<TaskSpecRead>(`${await base()}/${taskId}/confirm`, { data_version_id: dataVersionId })).data;
}
