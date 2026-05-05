// extensions/vscode/src/extension.ts
import * as vscode from 'vscode';
import * as childProcess from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import { GodBotClient } from './client';
import { ChatViewProvider } from './chatProvider';
import { showChatPanel } from './chatPanel';
import { createState } from './state';
import { readConfig } from './config';

let providerSingleton: ChatViewProvider | undefined;

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  const cfg = readConfig();
  const state = createState(cfg.baseUrl, cfg.workspace, cfg.autoApproveInSandbox);
  state.client = new GodBotClient(cfg.baseUrl);

  // Ensure daemon is up.
  const alive = await state.client.health();
  if (!alive) {
    if (cfg.autoLaunch) {
      const ok = await tryLaunchDaemon(cfg);
      if (!ok) {
        vscode.window.showErrorMessage(
          `GodBot daemon failed to start at ${cfg.baseUrl}. Check ~/.godbot/daemon.log or set godbot.pythonPath.`,
          'Open Settings',
        ).then((choice) => {
          if (choice === 'Open Settings') {
            void vscode.commands.executeCommand('workbench.action.openSettings', 'godbot.pythonPath');
          }
        });
      }
    } else {
      vscode.window.setStatusBarMessage('GodBot: daemon down (autoLaunch off)', 5000);
    }
  }

  providerSingleton = new ChatViewProvider(context.extensionUri, state);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider(ChatViewProvider.viewId, providerSingleton),
    vscode.commands.registerCommand('godbot.openChat', () => providerSingleton!.reveal()),
    vscode.commands.registerCommand('godbot.openInEditor', () =>
      showChatPanel(context.extensionUri, state),
    ),
    vscode.commands.registerCommand('godbot.newSession', () => providerSingleton!.newSession()),
    vscode.commands.registerCommand('godbot.askSelection', async () => {
      const editor = vscode.window.activeTextEditor;
      if (!editor) {
        vscode.window.showInformationMessage('GodBot: no active editor');
        return;
      }
      const sel = editor.document.getText(editor.selection);
      if (!sel.trim()) {
        vscode.window.showInformationMessage('GodBot: selection is empty');
        return;
      }
      const path = vscode.workspace.asRelativePath(editor.document.uri);
      const startLine = editor.selection.start.line + 1;
      const endLine = editor.selection.end.line + 1;
      const prefix = `From ${path}:${startLine}-${endLine}\n\`\`\`\n${sel}\n\`\`\`\n\nQuestion: `;
      const text = await vscode.window.showInputBox({
        prompt: 'Ask GodBot about the selection',
        placeHolder: 'e.g. "explain this function"',
      });
      if (!text) return;
      await providerSingleton!.send(prefix + text);
    }),
    vscode.commands.registerCommand('godbot.abortAll', async () => {
      if (!state.client) {
        vscode.window.showErrorMessage('GodBot: client not initialized');
        return;
      }
      try {
        const res = await state.client.abortAll();
        vscode.window.showInformationMessage(
          `GodBot: aborted ${res.sessions_signaled} session(s), ${res.tasks_signaled} task(s)`,
        );
      } catch (e) {
        vscode.window.showErrorMessage(`GodBot abortAll failed: ${(e as Error).message}`);
      }
    }),
    vscode.commands.registerCommand('godbot.openWorkspace', async () => {
      if (!state.client) {
        vscode.window.showErrorMessage('GodBot: client not initialized');
        return;
      }
      let workspaces: Array<{ path: string; sessions: number; last_activity: string }>;
      try {
        workspaces = await state.client.listWorkspaces();
      } catch (e) {
        vscode.window.showErrorMessage(`GodBot listWorkspaces failed: ${(e as Error).message}`);
        return;
      }
      if (workspaces.length === 0) {
        vscode.window.showInformationMessage('GodBot: no workspaces recorded yet');
        return;
      }
      const items: vscode.QuickPickItem[] = workspaces.map((w) => ({
        label: w.path,
        description: `${w.sessions} session${w.sessions === 1 ? '' : 's'}`,
        detail: w.last_activity ? `last activity ${w.last_activity}` : undefined,
      }));
      const pick = await vscode.window.showQuickPick(items, {
        placeHolder: 'Select a GodBot workspace to open',
        matchOnDescription: true,
        matchOnDetail: true,
      });
      if (!pick) return;
      const uri = vscode.Uri.file(pick.label);
      await vscode.commands.executeCommand('vscode.openFolder', uri);
    }),
  );
}

export function deactivate(): void {
  // No-op; daemon stays running for other surfaces.
}

async function tryLaunchDaemon(cfg: ReturnType<typeof readConfig>): Promise<boolean> {
  const cwd = cfg.sessionsRoot || vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || process.cwd();
  const args = ['-m', 'godbot.interfaces.web'];
  if (cfg.sessionsRoot) {
    args.push('--sessions-root', cfg.sessionsRoot);
  }
  try {
    const proc = childProcess.spawn(cfg.pythonPath, args, {
      cwd,
      detached: true,
      stdio: 'ignore',
    });
    proc.unref();
  } catch {
    return false;
  }
  // Poll health up to 15s.
  const client = new GodBotClient(cfg.baseUrl);
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    if (await client.health()) return true;
    await delay(500);
  }
  return false;
}
