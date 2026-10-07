import type { components } from './schema';
import api, { cancelJob, ensureProject } from './index';
import { takeSnapshot, waitForJob } from './snapshots';

/** A relational run: start it, read where it stands, steer it (backend issues #58, #59). Every number shown comes from here. */
type S = components['schemas'];
export type RunState = S['RunStateRead'];
export type RunFeature = S['RunFeatureRead'];
export type NarrationItem = S['NarrationItem'];
export type Checkpoint = S['CheckpointRead'];
export type Suggestion = S['SuggestionRead'];
export type SettingsPreview = S['SettingsPreview'];
export type SplitPreview = S['SplitPreview'];
export type LabelPreview = S['LabelPreview'];
export type TaskSpecRead = S['TaskSpecRead'];

export type ApprovalMode = 'auto' | 'confirm_task' | 'approve_each_feature';

async function runs(runId: string): Promise<string> {
  const project = await ensureProject();
  return `/projects/${project.id}/runs/${runId}`;
}

/**
 * Snapshot the connected database (the loop reads a snapshot, never the live one), record a run of the confirmed
 * task on it and queue the loop. Returns the run to open. `dataVersionId` is an existing snapshot to reuse.
 */
export async function startRun(
  task: { id: string },
  connectionId: string,
  dataVersionId: string | null,
  approvalMode: ApprovalMode,
  onStep: (step: string) => void,
): Promise<string> {
  const project = await ensureProject();
  let version = dataVersionId;
  if (!version) {
    onStep('Taking a snapshot of the database');
    const { version: v, jobId } = await takeSnapshot(connectionId, {
      mode: 'snapshot',
    });
    version = v?.id ?? null;
    if (jobId) {
      const done = await waitForJob(
        jobId,
        (step) => onStep(step),
        () => false,
      );
      if (!done.ok) throw new Error(done.error ?? 'The snapshot failed');
      version = done.versionId;
    }
  }
  if (!version) throw new Error('No snapshot to run on');
  onStep('Recording the run');
  const run = (await api.post<S['TaskRunRead']>(`/projects/${project.id}/tasks/${task.id}/runs`, { data_version_id: version })).data;
  onStep('Starting the run');
  await api.post<S['RunLoopStarted']>(`/projects/${project.id}/runs/${run.id}/start`, { approval_mode: approvalMode });
  return run.id;
}

export async function getRun(runId: string): Promise<RunState> {
  return (await api.get<RunState>(await runs(runId))).data;
}

export async function getRunFeatures(runId: string): Promise<RunFeature[]> {
  return (await api.get<S['RunFeaturesRead']>(`${await runs(runId)}/features`)).data.features;
}

export async function getNarration(runId: string): Promise<NarrationItem[]> {
  return (await api.get<S['RunNarration']>(`${await runs(runId)}/narration`)).data.items;
}

export async function getCheckpoints(runId: string): Promise<Checkpoint[]> {
  return (await api.get<Checkpoint[]>(`${await runs(runId)}/checkpoints`)).data;
}

export async function answerCheckpoint(runId: string, id: string, decision: 'approve' | 'veto'): Promise<Checkpoint> {
  return (
    await api.post<Checkpoint>(`${await runs(runId)}/checkpoints/${id}`, {
      decision,
    })
  ).data;
}

export async function addSuggestion(runId: string, text: string): Promise<Suggestion> {
  return (await api.post<Suggestion>(`${await runs(runId)}/suggestions`, { text })).data;
}

/** Says what a sentence would change; nothing changes until `applySettings`. */
export async function previewSettings(runId: string, message: string): Promise<SettingsPreview> {
  return (
    await api.post<SettingsPreview>(`${await runs(runId)}/settings`, {
      message,
    })
  ).data;
}

export async function applySettings(runId: string, message: string): Promise<SettingsPreview> {
  return (
    await api.post<SettingsPreview>(`${await runs(runId)}/settings/apply`, {
      message,
    })
  ).data;
}

export async function cancelRun(jobId: string): Promise<void> {
  await cancelJob(jobId);
}

export async function getTask(taskId: string): Promise<TaskSpecRead> {
  const project = await ensureProject();
  return (await api.get<TaskSpecRead>(`/projects/${project.id}/tasks/${taskId}`)).data;
}

/** The task's cutoffs divided into train, validation and test, on the run's data version. */
export async function previewSplit(taskId: string, dataVersionId: string | null): Promise<SplitPreview> {
  const project = await ensureProject();
  return (
    await api.post<SplitPreview>(
      `/projects/${project.id}/tasks/${taskId}/preview-split`,
      { data_version_id: dataVersionId, materialize: false },
      { timeout: 300_000 },
    )
  ).data;
}

/** Labels per cutoff (eligible entities, positives, base rate) on the run's data version. */
export async function previewLabels(taskId: string, dataVersionId: string | null): Promise<LabelPreview> {
  const project = await ensureProject();
  return (
    await api.post<LabelPreview>(
      `/projects/${project.id}/tasks/${taskId}/preview-labels`,
      { data_version_id: dataVersionId, materialize: false },
      { timeout: 300_000 },
    )
  ).data;
}
