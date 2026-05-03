// extensions/vscode/src/__tests__/client.test.ts
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Readable } from 'node:stream';

// Test the SSE parser logic in isolation by simulating a body stream.
// The actual GodBotClient.stream() takes a body from undici; here we hand-craft an async iterator.

async function* parseSseStream(body: AsyncIterable<Buffer>): AsyncIterable<{ event: string; data: any }> {
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
            yield { event: eventName, data: payload };
          } catch { /* skip */ }
        }
        eventName = null;
        dataBuf = [];
      } else if (line.startsWith(':')) {
      } else if (line.startsWith('event:')) {
        eventName = line.slice(6).trim();
      } else if (line.startsWith('data:')) {
        dataBuf.push(line.slice(5).replace(/^ /, ''));
      }
    }
  }
}

test('parses single token event', async () => {
  const body = Readable.from([Buffer.from('event: token\ndata: {"type":"token","text":"hi"}\n\n')]);
  const events: any[] = [];
  for await (const ev of parseSseStream(body as any)) events.push(ev);
  assert.deepEqual(events, [{ event: 'token', data: { type: 'token', text: 'hi' } }]);
});

test('parses multiple events including done', async () => {
  const body = Readable.from([Buffer.from(
    'event: token\ndata: {"type":"token","text":"a"}\n\nevent: done\ndata: {"type":"done","step_count":1}\n\n'
  )]);
  const events: any[] = [];
  for await (const ev of parseSseStream(body as any)) events.push(ev);
  assert.equal(events.length, 2);
  assert.equal(events[1].event, 'done');
});

test('skips comment lines', async () => {
  const body = Readable.from([Buffer.from(
    ': keepalive\nevent: token\ndata: {"type":"token","text":"x"}\n\n'
  )]);
  const events: any[] = [];
  for await (const ev of parseSseStream(body as any)) events.push(ev);
  assert.equal(events.length, 1);
});
