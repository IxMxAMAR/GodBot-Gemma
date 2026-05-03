// extensions/vscode/src/chatPanel.ts
import * as vscode from 'vscode';
import { ChatViewProvider } from './chatProvider';
import { ExtensionState } from './state';

let panelSingleton: vscode.WebviewPanel | undefined;

export function showChatPanel(extensionUri: vscode.Uri, state: ExtensionState): void {
  if (panelSingleton) {
    panelSingleton.reveal(vscode.ViewColumn.Active);
    return;
  }
  const panel = vscode.window.createWebviewPanel(
    'godbotChat', 'GodBot Chat', vscode.ViewColumn.Active,
    {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(extensionUri, 'media')],
      retainContextWhenHidden: true,
    },
  );
  panelSingleton = panel;
  panel.onDidDispose(() => { panelSingleton = undefined; });

  // Reuse the provider's HTML + message routing: instantiate a transient
  // provider bound to this panel's webview. The panel exposes `webview` with
  // the same `postMessage`/`onDidReceiveMessage`/`html` shape as a
  // WebviewView, so we duck-type the panel as the view sink.
  const provider = new ChatViewProvider(extensionUri, state);
  // Manually wire the panel as the view sink:
  (provider as any).view = panel;  // duck-typed; both have webview.postMessage
  panel.webview.options = {
    enableScripts: true,
    localResourceRoots: [vscode.Uri.joinPath(extensionUri, 'media')],
  };
  panel.webview.html = (provider as any).htmlFor(panel.webview);
  panel.webview.onDidReceiveMessage((m: unknown) => (provider as any).handleMessage(m));
  void (provider as any).bootstrap();
}
