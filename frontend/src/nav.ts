import { Database, GitBranch, Radio, Settings } from 'lucide-react';
import { connection } from './api';
import type { View } from './types';

const ALL: { key: View; label: string; icon: typeof Database }[] = [
  { key: 'dashboard', label: 'Dashboard', icon: Database },
  { key: 'experiments', label: 'Experiments', icon: GitBranch },
  { key: 'deployments', label: 'Deployments', icon: Radio },
  { key: 'settings', label: 'Settings', icon: Settings },
];

/** The workspace views in rail order; key N opens NAV[N - 1]. Deployments is sample data: Demo mode only. */
export const NAV = ALL.filter((n) => connection.demo || n.key !== 'deployments');
