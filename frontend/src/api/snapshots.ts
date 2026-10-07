import type { components } from './schema';
import api, { ensureProject, getJob } from './index';

/** Snapshots and live records of a connected database: what exactly a run reads (backend issue #97). */
type S = components['schemas'];
export type DbVersion = S['DbVersionRead'];
export type SnapshotRequest = S['SnapshotRequest'];

export async function listDbVersions(): Promise<DbVersion[]> {
  const project = await ensureProject();
  return (await api.get<DbVersion[]>(`/projects/${project.id}/db-versions`)).data;
}

/** A live record comes back at once; a snapshot is a job to follow with `waitForJob`. */
export async function takeSnapshot(
  connectionId: string,
  body: SnapshotRequest,
): Promise<{ version: DbVersion | null; jobId: string | null }> {
  const project = await ensureProject();
  const { data } = await api.post<S['SnapshotResponse']>(
    `/projects/${project.id}/connections/${connectionId}/snapshots`,
    body,
    { timeout: 60_000 },
  );
  return { version: data.version ?? null, jobId: data.job?.id ?? null };
}

export async function waitForJob(
  jobId: string,
  onProgress: (step: string, fraction: number) => void,
  stopped: () => boolean,
): Promise<{ ok: boolean; error: string | null; versionId: string | null }> {
  for (;;) {
    const job = await getJob(jobId);
    onProgress(job.current_step ?? 'Waiting', job.progress);
    if (['succeeded', 'failed', 'cancelled'].includes(job.status)) {
      const result = (job.result ?? null) as { data_version_id?: string } | null;
      return { ok: job.status === 'succeeded', error: job.error ?? null, versionId: result?.data_version_id ?? null };
    }
    if (stopped()) return { ok: false, error: 'stopped', versionId: null };
    await new Promise((r) => setTimeout(r, 600));
  }
}
