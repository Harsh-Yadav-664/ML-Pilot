import client from './client';
import type { Project, ProjectCreate, PaginatedResponse } from '../types';

export const projectsApi = {
  list: async (page = 1, pageSize = 20): Promise<PaginatedResponse<Project>> => {
    const { data } = await client.get('/projects/', { params: { page, page_size: pageSize } });
    return data;
  },

  get: async (id: string): Promise<Project> => {
    const { data } = await client.get(`/projects/${id}`);
    return data;
  },

  create: async (payload: ProjectCreate): Promise<Project> => {
    const { data } = await client.post('/projects/', payload);
    return data;
  },

  update: async (id: string, payload: Partial<ProjectCreate>): Promise<Project> => {
    const { data } = await client.patch(`/projects/${id}`, payload);
    return data;
  },

  delete: async (id: string): Promise<void> => {
    await client.delete(`/projects/${id}`);
  },
};
