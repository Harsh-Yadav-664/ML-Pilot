import client from './client';
import type { Experiment, PaginatedResponse } from '../types';

export const experimentsApi = {
  list: async (projectId: string, page = 1, pageSize = 20): Promise<PaginatedResponse<Experiment>> => {
    const { data } = await client.get(`/experiments/project/${projectId}`, {
      params: { page, page_size: pageSize },
    });
    return data;
  },

  get: async (id: string): Promise<Experiment> => {
    const { data } = await client.get(`/experiments/${id}`);
    return data;
  },

  create: async (payload: Partial<Experiment>): Promise<Experiment> => {
    const { data } = await client.post('/experiments/', payload);
    return data;
  },

  update: async (id: string, payload: Partial<Experiment>): Promise<Experiment> => {
    const { data } = await client.patch(`/experiments/${id}`, payload);
    return data;
  },
};
