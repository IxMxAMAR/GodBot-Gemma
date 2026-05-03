// extensions/vscode/src/chatProvider.ts
import * as vscode from 'vscode';
import { GodBotClient, GodbotEvent, ClientError } from './client';
import { ExtensionState } from './state';

export class ChatViewProvider implements vscode.WebviewViewProvider {
  public static readonly viewId = 'godbot.chatView';
  private view?: vscode.WebviewView;

  constructor(
    private readonly extensionUri: vscode.Uri,
    private readonly state: ExtensionState,
  ) {}

  resolveWebviewView(webviewView: vscode.WebviewView): void {
    this.view = webviewView;
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(this.extensionUri, 'media')],
    };
    webviewView.webview.html = this.htmlFor(webviewView.webview);
    webviewView.webview.onDidReceiveMessage((msg) => this.handleMessage(msg));
    void this.bootstrap();
  }

  /** Send a user message from the editor command path. */
  async send(text: string): Promise<void> {
    if (!this.view) {
      await vscode.commands.executeCommand('godbot.chatView.focus');
    }
    this.postToWebview({ type: 'user_message', text });
    await this.startStream(text);
  }

  /** Reveal the panel. */
  async reveal(): Promise<void> {
    await vscode.commands.executeCommand('godbot.chatView.focus');
  }

  /** Force a new session (used by the New Session command). */
  async newSession(): Promise<void> {
    if (!this.state.client) return;
    const sid = await this.state.client.newSession();
    this.state.sessionId = sid;
    this.postToWebview({ type: 'session', sid, mode: 'daemon' });
    this.postToWebview({ type: 'cleared' });
  }

  private async bootstrap(): Promise<void> {
    if (!this.state.client) return;
    if (!this.state.sessionId) {
      try {
        this.state.sessionId = await this.state.client.newSession();
      } catch (e) {
        this.postToWebview({ type: 'error', message: `daemon error: ${(e as Error).message}` });
        return;
      }
    }
    this.postToWebview({ type: 'session', sid: this.state.sessionId, mode: 'daemon' });
    try {
      const tools = await this.state.client.listTools();
      this.postToWebview({ type: 'tools', tools });
    } catch { /* best effort */ }
  }

  private postToWebview(msg: unknown): void {
    void this.view?.webview.postMessage(msg);
  }

  private async handleMessage(msg: any): Promise<void> {
    if (!this.state.client || !this.state.sessionId) return;
    switch (msg.type) {
      case 'send':
        this.postToWebview({ type: 'user_message', text: msg.text });
        await this.startStream(String(msg.text));
        break;
      case 'gate_decision':
        await this.state.client.resolveGate(this.state.sessionId, msg.callId, msg.decision);
        break;
      case 'stop':
        await this.state.client.stop(this.state.sessionId);
        if (this.state.abortController) this.state.abortController.abort();
        break;
      case 'new_session':
        await this.newSession();
        break;
      case 'toggle_tool':
        await this.state.client.toggleTool(this.state.sessionId, msg.name, msg.enabled);
        break;
    }
  }

  private async startStream(text: string): Promise<void> {
    if (!this.state.client || !this.state.sessionId) return;
    if (this.state.active) {
      this.postToWebview({ type: 'error', message: 'already streaming; wait for current turn to finish' });
      return;
    }
    this.state.active = true;
    this.state.abortController = new AbortController();
    try {
      await this.state.client.send(this.state.sessionId, text);
      for await (const ev of this.state.client.stream(this.state.sessionId, this.state.abortController.signal)) {
        this.postToWebview({ type: 'event', payload: ev });
      }
    } catch (e) {
      const msg = e instanceof ClientError ? e.message : String(e);
      this.postToWebview({ type: 'error', message: msg });
    } finally {
      this.state.active = false;
      this.state.abortController = null;
    }
  }

  private htmlFor(webview: vscode.Webview): string {
    const nonce = randomNonce();
    const cssUri = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', 'chat.css'));
    const jsUri = webview.asWebviewUri(vscode.Uri.joinPath(this.extensionUri, 'media', 'chat.js'));
    return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy"
      content="default-src 'none'; img-src ${webview.cspSource}; style-src ${webview.cspSource}; script-src 'nonce-${nonce}';">
<link rel="stylesheet" href="${cssUri}">
<title>GodBot</title>
</head>
<body>
<div id="topbar">
  <span id="title">GodBot</span>
  <span id="session-id">no session</span>
  <button id="new">⊕</button>
  <button id="stop">⏹</button>
  <button id="cog">⚙</button>
</div>
<details id="settings">
  <summary>Tools / Sessions</summary>
  <ul id="tool-list"></ul>
</details>
<div id="conversation"></div>
<div id="composer">
  <textarea id="input" placeholder="type a message..."></textarea>
  <button id="send">Send</button>
</div>
<script type="module" nonce="${nonce}" src="${jsUri}"></script>
</body>
</html>`;
  }
}

function randomNonce(): string {
  const bytes = new Uint8Array(16);
  for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
  return Buffer.from(bytes).toString('base64');
}
