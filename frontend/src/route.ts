import { useEffect, useState } from 'react';

/** Pages that live outside the workspace store. Only the hash changes, so a reload keeps the page. */
export type Route = 'connect' | 'tasks' | 'run' | null;

const read = (): Route => {
  if (window.location.hash === '#/connect') return 'connect';
  if (window.location.hash === '#/tasks') return 'tasks';
  if (window.location.hash.startsWith('#/run/')) return 'run';
  return null;
};

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return route;
}

/** The run in `#/run/<id>`. */
export const runIdFromHash = (): string => decodeURIComponent(window.location.hash.slice('#/run/'.length));
export const goRun = (id: string) => {
  window.location.hash = `#/run/${encodeURIComponent(id)}`;
};
export const goConnect = () => {
  window.location.hash = '#/connect';
};
export const goTasks = () => {
  window.location.hash = '#/tasks';
};
export const leaveRoute = () => {
  // Drop the hash without leaving a bare "#" in the URL.
  history.pushState(null, '', window.location.pathname + window.location.search);
  window.dispatchEvent(new HashChangeEvent('hashchange'));
};
