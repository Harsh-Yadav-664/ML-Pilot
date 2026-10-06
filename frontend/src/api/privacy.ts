import type { components } from './schema';
import api, { ensureProject } from './index';

/** Privacy settings of the active project and the log of every prompt sent to an LLM (backend issue #48). */
type S = components['schemas'];
export type PrivacyRead = S['PrivacyRead'];
export type PrivacySettings = S['PrivacySettings'];
export type PrivacyLevel = PrivacySettings['level'];
export type LLMCall = S['LLMCallRead'];
export type LLMCallDetail = S['LLMCallDetail'];

async function base(): Promise<string> {
  return `/projects/${(await ensureProject()).id}`;
}

export async function getPrivacy(): Promise<PrivacyRead> {
  return (await api.get<PrivacyRead>(`${await base()}/privacy`)).data;
}

export async function putPrivacy(settings: PrivacySettings): Promise<PrivacyRead> {
  return (await api.put<PrivacyRead>(`${await base()}/privacy`, settings)).data;
}

export async function listLlmCalls(limit = 100): Promise<LLMCall[]> {
  return (await api.get<LLMCall[]>(`${await base()}/llm-calls`, { params: { limit } })).data;
}

export async function getLlmCall(id: string): Promise<LLMCallDetail> {
  return (await api.get<LLMCallDetail>(`${await base()}/llm-calls/${id}`)).data;
}
