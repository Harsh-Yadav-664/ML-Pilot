import { useEffect, useState } from 'react';

/** Pages that live outside the workspace store. Only the hash changes, so a reload keeps the page. */
export type Route = 'connect' | null;

const read = (): Route => (window.location.hash === '#/connect' ? 'connect' : null);

export function useRoute(): Route {
  const [route, setRoute] = useState<Route>(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return route;
}

export const goConnect = () => {
  window.location.hash = '#/connect';
};
export const leaveRoute = () => {
  // Drop the hash without leaving a bare "#" in the URL.
  history.pushState(null, '', window.location.pathname + window.location.search);
  window.dispatchEvent(new HashChangeEvent('hashchange'));
};
