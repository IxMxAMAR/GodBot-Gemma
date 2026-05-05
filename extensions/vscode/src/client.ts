// extensions/vscode/src/client.ts
import { request } from 'undici';

export interface TokenEvent { type: 'token'; text: string; }
export interface ToolCallEvent { type: 'tool_call'; id: string; name: string; args: Record<string, unknown>; }
export interface ToolResultEvent { type: 'tool_result'; id: string; preview: string; blob: string | null; duration_ms: number; }
export interface GateEvent { type: 'gate'; id: string; name: string; args: Record<string, unknown>; }
export interface ErrorEv { type: 'agent_error'; message: string; recoverable: boolean; }
export interface DoneEvent { type: 'done'; step_count: number; }
export type GodbotEvent = TokenEvent | ToolCallEvent | ToolResultEvent | GateEvent | ErrorEv | DoneEvent;

export class ClientError extends Error {
  constructor(public statusCode: number, message: string) {
    super(`[${statusCode}] ${message}`);
  }
}

export class GodBotClient {
  constructor(private baseUrl: string = 'http://127.0.0.1:7878') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  async health(): Promise<boolean> {
    try {
      const { statusCode } = await request(`${this.baseUrl}/api/health`, { method: 'GET' });
      return statusCode === 200;
    } catch {
      return false;
    }
  }

  async newSession(
    model = 'auto',
    workspace?: string,
    autoApproveInSandbox = false,
  ): Promise<string> {
    const body: Record<string, unknown> = { model };
    if (workspace) body.workspace = workspace;
    if (autoApproveInSandbox) body.auto_approve_in_sandbox = true;
    const r = await this._req('POST', '/api/sessions/new', body);
    return (r as { session_id: string }).session_id;
  }

  async listSessions(): Promise<Array<{ id: string; model: string }>> {
    return await this._req('GET', '/api/sessions') as Array<{ id: string; model: string }>;
  }

  async getSession(sid: string): Promise<unknown> {
    return await this._req('GET', `/api/sessions/${sid}`);
  }

  async send(sid: string, message: string): Promise<void> {
    await this._req('POST', '/api/chat', { session_id: sid, message });
  }

  async resolveGate(sid: string, callId: string, decision: 'allow' | 'deny' | 'always'): Promise<void> {
    await this._req('POST', `/api/gate/${callId}`, { session_id: sid, decision });
  }

  async stop(sid: string): Promise<void> {
    await this._req('POST', '/api/stop', { session_id: sid });
  }

  async listTools(): Promise<Array<{ name: string; description: string; dangerous: boolean }>> {
    return await this._req('GET', '/api/tools') as Array<{ name: string; description: string; dangerous: boolean }>;
  }

  async toggleTool(sid: string, name: string, enabled: boolean): Promise<void> {
    await this._req('POST', '/api/tools/toggle', { session_id: sid, name, enabled });
  }

  async listRagCollections(): Promise<string[]> {
    return await this._req('GET', '/api/rag/collections') as string[];
  }

  async useRagCollection(sid: string, collection: string | null): Promise<void> {
    await this._req('POST', '/api/rag/use', { session_id: sid, collection });
  }

  async abortAll(): Promise<{ ok: boolean; sessions_signaled: number; tasks_signaled: number }> {
    return await this._req('POST', '/api/agent/abort_all') as {
      ok: boolean;
      sessions_signaled: number;
      tasks_signaled: number;
    };
  }

  async listWorkspaces(): Promise<Array<{ path: string; sessions: number; last_activity: string }>> {
    const r = await this._req('GET', '/api/workspaces') as {
      workspaces: Array<{ path: string; sessions: number; last_activity: string }>;
    };
    return r.workspaces;
  }

  /**
   * Stream SSE events from /api/chat/stream until DoneEvent or close.
   * Yields parsed event objects. Drops `ping` heartbeats.
   */
  async *stream(sid: string, signal?: AbortSignal): AsyncIterable<GodbotEvent> {
    const url = `${this.baseUrl}/api/chat/stream?session_id=${encodeURIComponent(sid)}`;
    const { statusCode, body } = await request(url, { method: 'GET', signal });
    if (statusCode >= 400) {
      const text = await body.text();
      throw new ClientError(statusCode, text);
    }
    let eventName: string | null = null;
    let dataBuf: string[] = [];
    let pending = '';
    for await (const chunk of body) {
      pending += chunk.toString('utf-8');
      let nl: number;
      while ((nl = pending.indexOf('\n')) >= 0) {
        const line = pending.slice(0, nl).replace(/\r$/, '');
        pending = pending.slice(nl + 1);
        if (line === '') {
          if (eventName && dataBuf.length) {
            const dataStr = dataBuf.join('\n');
            try {
              const payload = JSON.parse(dataStr);
              if (eventName !== 'ping') {
                yield payload as GodbotEvent;
                if ((payload as GodbotEvent).type === 'done') return;
              }
            } catch { /* drop malformed line */ }
          }
          eventName = null;
          dataBuf = [];
        } else if (line.startsWith(':')) {
          // SSE comment
        } else if (line.startsWith('event:')) {
          eventName = line.slice(6).trim();
        } else if (line.startsWith('data:')) {
          dataBuf.push(line.slice(5).replace(/^ /, ''));
        }
      }
    }
  }

  private async _req(method: string, path: string, body?: unknown): Promise<unknown> {
    const { statusCode, body: respBody } = await request(`${this.baseUrl}${path}`, {
      method: method as 'GET' | 'POST',
      headers: body ? { 'content-type': 'application/json' } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
    const text = await respBody.text();
    if (statusCode >= 400) {
      throw new ClientError(statusCode, text);
    }
    return text ? JSON.parse(text) : null;
  }
}
