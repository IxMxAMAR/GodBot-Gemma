// extensions/vscode/src/config.ts
import * as vscode from 'vscode';

export interface GodbotConfig {
  baseUrl: string;
  autoLaunch: boolean;
  sessionsRoot: string;
  pythonPath: string;
}

export function readConfig(): GodbotConfig {
  const c = vscode.workspace.getConfiguration('godbot');
  return {
    baseUrl: c.get<string>('baseUrl', 'http://127.0.0.1:7878'),
    autoLaunch: c.get<boolean>('autoLaunch', true),
    sessionsRoot: c.get<string>('sessionsRoot', ''),
    pythonPath: c.get<string>('pythonPath', '') || (process.platform === 'win32' ? 'python' : 'python3'),
  };
}
