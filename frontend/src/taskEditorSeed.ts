import type { TaskDraft } from './api/tasks';

/** A spec handed to the task editor from another page (the question card's "Open in the editor"). */
export type EditorSeed = {
  connectionId: string;
  dataVersionId: string | null;
  yaml: string;
  taskId: string | null;
  draftSource: TaskDraft['source'] | null;
};

let seed: EditorSeed | null = null;

export const setEditorSeed = (s: EditorSeed | null) => {
  seed = s;
};

/** The editor reads the seed once on opening. */
export const takeEditorSeed = (): EditorSeed | null => {
  const s = seed;
  seed = null;
  return s;
};
