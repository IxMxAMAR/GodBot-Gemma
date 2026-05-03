// extensions/vscode/src/state.ts
import { GodBotClient } from './client';

export interface ExtensionState {
  client: GodBotClient | null;
  sessionId: string | null;
  baseUrl: string;
  active: boolean;  // true while a stream is in flight
  abortController: AbortController | null;
}

export function createState(baseUrl: string): ExtensionState {
  return {
    client: null,
    sessionId: null,
    baseUrl,
    active: false,
    abortController: null,
  };
}
