#!/usr/bin/env node
// Validate the shipped WebUX page in Chromium against controlled SSE responses.
// Usage: node eng/validation/validate-webui-steering.mjs \
//   --playwright-module /path/to/playwright-core/index.mjs --browser /path/to/chrome
// This checks browser rendering and request handling, not real model inference.
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = fileURLToPath(new URL('../../', import.meta.url));
const args = process.argv.slice(2);
const option = name => args[args.indexOf(name) + 1];
const modulePath = args.includes('--playwright-module') ? option('--playwright-module') : 'playwright-core';
const browserPath = args.includes('--browser') ? option('--browser') : undefined;
const output = path.resolve(root, args.includes('--output') ? option('--output') : 'artifacts/validation/webui-steering');
const { chromium } = await import(path.isAbsolute(modulePath) ? pathToFileURL(modulePath).href : modulePath);
const streams = [];
const sessions = [];
const server = createServer(async (request, response) => {
  const reply = data => {
    response.writeHead(200, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify(data));
  };
  try {
    if (request.url === '/') {
      response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      response.end(await readFile(path.join(root, 'TensorSharp.Server.Host/wwwroot/index.html')));
    } else if (request.url === '/api/models') {
      reply({ loaded: 'Steering browser fixture', architecture: 'qwen35', defaultMaxTokens: 64 });
    } else if (request.url === '/api/queue/status') {
      reply({ processing: 0, pending_requests: 0 });
    } else if (request.url === '/api/sessions') {
      const sessionId = 'fixture-session-' + (sessions.length + 1);
      sessions.push(sessionId);
      reply({ sessionId });
    } else if (request.url.startsWith('/api/sessions/')) {
      reply({});
    } else if (request.url === '/api/chat') {
      let body = '';
      for await (const chunk of request) body += chunk;
      response.writeHead(200, { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' });
      response.flushHeaders();
      streams.push({ payload: JSON.parse(body), response, closed: false });
      const stream = streams.at(-1);
      response.on('close', () => { stream.closed = true; });
    } else if (/^\/images\/[a-zA-Z0-9_.-]+\.png$/.test(request.url)) {
      response.writeHead(200, { 'Content-Type': 'image/png' });
      response.end(await readFile(path.join(root, 'TensorSharp.Server.Host/wwwroot', request.url)));
    } else {
      response.writeHead(404);
      response.end();
    }
  } catch (error) {
    response.writeHead(500);
    response.end(String(error));
  }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const browser = await chromium.launch({ executablePath: browserPath, headless: true });
const page = await browser.newPage({ viewport: { width: 360, height: 800 } });
const errors = [];
page.on('pageerror', error => errors.push(String(error)));
const send = (index, data) => streams[index].response.write('data: ' + JSON.stringify(data) + '\n\n');
const waitForRequest = async count => {
  await page.waitForFunction(expected => chatHistory.some(message => message.role === 'user') && isGenerating, count);
  const deadline = Date.now() + 10000;
  while (streams.length < count && Date.now() < deadline) await new Promise(resolve => setTimeout(resolve, 25));
  assert.equal(streams.length, count);
};

try {
  await page.goto('http://127.0.0.1:' + server.address().port);
  await page.waitForFunction(() => currentSessionId && currentLoadedModel);
  await page.locator('#message-input').fill('Explain the design');
  await page.locator('#btn-send').click();
  await waitForRequest(1);
  send(0, { thinking: 'Considering the design.' });
  send(0, { token: 'The initial answer' });
  await page.waitForFunction(() => document.querySelector('.assistant .bubble-text').textContent === 'The initial answer');
  await page.locator('#message-input').fill('Focus on concurrency');
  const bounds = await page.evaluate(() => {
    const input = document.getElementById('message-input');
    const area = input.getBoundingClientRect();
    const controls = document.querySelector('.input-actions').getBoundingClientRect();
    return { width: area.width, padding: parseFloat(getComputedStyle(input).paddingRight), controls: area.right - controls.left };
  });
  assert.ok(bounds.padding >= bounds.controls + 8, `composer text overlaps its controls: ${JSON.stringify(bounds)}`);
  assert.ok(bounds.width - bounds.padding > 100, 'narrow composer must leave readable text space');
  await mkdir(output, { recursive: true });
  await page.screenshot({ path: path.join(output, 'narrow-steering.png'), fullPage: true });
  await page.locator('#message-input').press('Enter');
  await waitForRequest(2);
  assert.deepEqual(streams[1].payload.messages, [
    { role: 'user', content: 'Explain the design' },
    { role: 'assistant', content: 'The initial answer', thinking: 'Considering the design.' },
    { role: 'user', content: 'Focus on concurrency' },
  ]);
  assert.equal(streams[1].payload.replaceActive, true);
  assert.equal(streams[1].payload.newChat, undefined);
  send(1, { thinking: 'Reviewing concurrency.' });
  await page.waitForFunction(() => document.querySelectorAll('.thinking-content').length === 2);
  await page.locator('#message-input').fill('Keep it concise');
  await page.locator('#btn-send').click();
  await waitForRequest(3);
  assert.equal(streams[2].payload.messages.at(-1).content, 'Keep it concise');
  send(2, { token: 'Use a serialized queue.' });
  await page.waitForFunction(() => document.querySelectorAll('.assistant .bubble-text')[2].textContent === 'Use a serialized queue.');
  await page.locator('#btn-stop').click();
  await page.waitForFunction(() => !isGenerating && document.getElementById('btn-stop').hidden);
  assert.equal(await page.evaluate(() => chatHistory.at(-1).content), 'Use a serialized queue.');
  assert.equal(streams.length, 3);
  await page.locator('#btn-clear').click();
  await page.waitForFunction(() => chatHistory.length === 0 && currentSessionId === 'fixture-session-2');
  assert.deepEqual(errors, []);
  const summary = {
    passed: true, viewport: '360x800', requests: streams.length, bounds,
    scenarios: ['partial answer steering', 'reasoning-only steering', 'Enter and Send', 'explicit Stop', 'New Chat', 'narrow composer'],
    limitations: 'Controlled SSE fixture; no real model, backend session coordination, or device performance coverage.',
  };
  await writeFile(path.join(output, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
  console.log(JSON.stringify({ ...summary, output }, null, 2));
} finally {
  await browser.close();
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}
