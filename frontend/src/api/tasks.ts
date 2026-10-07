import axios from 'axios';
import type { components } from './schema';
import api, { ensureProject } from './index';

/** Prediction task specs: draft from a question, save, preview the labels, confirm (backend issues #49, #50, #53). */
type S = components['schemas'];
export type TaskDraft = S['TaskDraft'];
export type TaskSpecRead = S['TaskSpecRead'];
export type LabelPreview = S['LabelPreview'];
export type SpecIssue = S['SpecIssueRead'];
export type SpecCheck = S['SpecCheck'];
export type TaskSpecBody = { [key: string]: unknown };

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

/**
 * What the editor shows on every change, without saving: the spec's problems with their field paths, its
 * canonical YAML and, when it has no errors, the labels per cutoff and the feasibility checks. Send the form's
 * `spec` or the YAML tab's `yaml`.
 */
export async function checkSpec(
  input: { spec?: TaskSpecBody; yaml?: string; connectionId: string; dataVersionId: string | null },
  signal?: AbortSignal,
): Promise<SpecCheck> {
  return (
    await api.post<SpecCheck>(
      `${await base()}/check`,
      { spec: input.spec, yaml: input.yaml, connection_id: input.connectionId, data_version_id: input.dataVersionId },
      { timeout: 300_000, signal },
    )
  ).data;
}

export async function listTasks(): Promise<TaskSpecRead[]> {
  return (await api.get<TaskSpecRead[]>(`${await base()}/`, { params: { latest_only: true } })).data;
}

/**
 * Save by task name: a new task, or, when the name exists (asking the same question twice, editing a
 * spec again), the next save of that task: a draft is edited in place, a confirmed task gets a new version.
 */
export async function saveTaskNamed(input: SaveInput, name: string, taskId: string | null): Promise<TaskSpecRead> {
  if (taskId) return saveTask(input, taskId);
  try {
    return await saveTask(input, null);
  } catch (e) {
    if (axios.isAxiosError(e) && e.response?.status === 409) {
      const existing = (await listTasks()).find((t) => t.name === name);
      if (existing) return saveTask(input, existing.id);
    }
    throw e;
  }
}
