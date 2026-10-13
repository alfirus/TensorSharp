#!/usr/bin/env node
// Exercise the shipped WebUX handlers against a controlled HTTP/SSE boundary.
// No model, native inference library, or browser installation is required.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import vm from 'node:vm';

const html = readFileSync(new URL('../../TensorSharp.Server.Host/wwwroot/index.html', import.meta.url), 'utf8');

function section(start, end) {
  const first = html.indexOf(start);
  const last = html.indexOf(end, first);
  assert.ok(first >= 0 && last > first, `Missing shipped handler: ${start}`);
  return html.slice(first, last);
}

class Element {
  constructor(className = '') {
    this.className = className;
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.textContent = '';
    this.attributes = {};
    this.listeners = {};
  }
  appendChild(child) { child.parentNode = this; this.children.push(child); return child; }
  insertBefore(child, before) { child.parentNode = this; this.children.splice(this.children.indexOf(before), 0, child); }
  remove() { this.removed = true; if (this.parentNode) this.parentNode.children.splice(this.parentNode.children.indexOf(this), 1); }
  setAttribute(name, value) { this.attributes[name] = value; }
  addEventListener(name, action) { this.listeners[name] = action; }
  querySelector(selector) {
    const name = selector.slice(1);
    for (const child of this.children) {
      if (child.className.split(' ').includes(name)) return child;
      const nested = child.querySelector(selector);
      if (nested) return nested;
    }
    return null;
  }
  set innerHTML(value) {
    this._html = value;
    this.children = [];
    if (this.className === 'thinking-block') this.appendChild(new Element('thinking-content'));
  }
  get innerHTML() { return this._html || ''; }
}

class Stream {
  constructor(signal, deferAbort) {
    this.signal = signal;
    this.deferAbort = deferAbort;
    this.chunks = [];
    signal.addEventListener('abort', () => {
      if (!deferAbort && this.waiter) {
        this.waiter.reject(new DOMException('Canceled', 'AbortError'));
        this.waiter = null;
      }
    });
  }
  read() {
    if (this.signal.aborted && !this.deferAbort) return Promise.reject(new DOMException('Canceled', 'AbortError'));
    if (this.chunks.length) return Promise.resolve(this.chunks.shift());
    return new Promise((resolve, reject) => { this.waiter = { resolve, reject }; });
  }
  deliver(chunk) {
    if (this.waiter) {
      this.waiter.resolve(chunk);
      this.waiter = null;
    } else this.chunks.push(chunk);
  }
  frame(data) { this.deliver({ done: false, value: new TextEncoder().encode('data: ' + JSON.stringify(data) + '\n\n') }); }
  close() { this.deliver({ done: true }); }
}

function page({ deferAbort = false, sessionReplies = [] } = {}) {
  const elements = {
    'message-input': new Element(), 'btn-send': new Element('send-btn'),
    'btn-stop': new Element('stop-btn'), 'reasoning-toggle': { checked: true },
  };
  elements['message-input'].value = '';
  const assistantBubbles = [], userBubbles = [], calls = [], streams = [], alerts = [];
  const frames = new Map();
  let frameId = 0;
  const container = new Element();
  const context = vm.createContext({
    AbortController, DOMException, TextDecoder, Promise, Map,
    isGenerating: false, currentLoadedModel: 'fixture', currentArchitecture: 'qwen35',
    currentSessionId: 'chat-session', DEFAULT_SESSION_ID: '__default__',
    chatHistory: [], pendingAttachments: [], activeImageEditTarget: null, pendingUploadCount: 0,
    needsCacheReset: false, shouldAutoScroll: true, serverDefaultMaxTokens: 64,
    selectedSkills: [], VIDEO_ARCHITECTURES: new Set(['wan']), editOriginalHtml: new Map(),
    messageInput: elements['message-input'], attachmentsDiv: new Element(), chatContainer: container,
    document: { getElementById: id => elements[id], createElement: () => new Element() },
    console, alert: value => alerts.push(value),
    requestAnimationFrame: action => { frames.set(++frameId, action); return frameId; },
    cancelAnimationFrame: id => frames.delete(id),
    scrollChatToBottom() {}, renderMarkdown: (element, content) => { element.textContent = content; },
    clearCurrentActivity() {}, renderEmptyState: () => container.appendChild(new Element('empty-state')),
    describeAttachments: () => 'Use these files', isQwenImageModel: () => false,
    addUserBubble: (content, attachments, index) => userBubbles.push({ content, attachments, index }),
    addAssistantBubble() {
      const root = new Element('message assistant');
      const bubble = root.appendChild(new Element('bubble'));
      bubble.appendChild(new Element('bubble-text'));
      bubble.appendChild(new Element('stats'));
      bubble.appendChild(new Element('typing-indicator'));
      container.appendChild(root);
      assistantBubbles.push(root);
      return root;
    },
    fetch: async (url, options = {}) => {
      calls.push({ url, ...options, payload: options.body && JSON.parse(options.body) });
      if (url === '/api/chat') {
        const stream = new Stream(options.signal, deferAbort);
        streams.push(stream);
        return { ok: true, status: 200, body: { getReader: () => stream } };
      }
      if (url === '/api/sessions') {
        const reply = sessionReplies.shift() || { sessionId: 'new-session' };
        return { ok: true, json: async () => await reply };
      }
      return { ok: true, json: async () => ({}) };
    },
  });
  vm.runInContext(section('async function sendMessage()', 'async function runImageEdit('), context);
  vm.runInContext(section('async function requestAssistantResponse()', 'function describeAttachments()'), context);
  vm.runInContext(section('async function createSession(', 'let uploadQueue ='), context);
  vm.runInContext(section("messageInput.addEventListener('keydown'", "document.getElementById('btn-clear')"), context);
  return {
    context, elements, assistantBubbles, userBubbles, calls, streams, alerts, frames, container,
    read: expression => vm.runInContext(expression, context),
    get history() { return JSON.parse(JSON.stringify(context.chatHistory)); },
    get requests() { return calls.filter(call => call.url === '/api/chat').map(call => call.payload); },
    submit(content) { elements['message-input'].value = content; return context.sendMessage(); },
    paint() { const callbacks = [...frames.values()]; frames.clear(); callbacks.forEach(callback => callback()); },
  };
}

async function settle() { await new Promise(setImmediate); }

function complete(stream, token = 'Finished') {
  stream.frame({ token });
  stream.frame({ done: true, sessionId: 'chat-session', tokenCount: 2, elapsed: 1, tokPerSec: 2 });
  stream.close();
}

test('Send stays available while reasoning, preserving partial output before steering', async () => {
  const ui = page();
  const first = ui.submit('Explain the design');
  await settle();
  ui.streams[0].frame({ thinking: 'Considering several designs.' });
  ui.streams[0].frame({ token: 'The first design' });
  await settle();
  assert.match(ui.elements['btn-send'].title, /steer/);
  assert.equal(ui.elements['btn-send'].className, 'send-btn');
  assert.equal(ui.elements['btn-stop'].hidden, false);
  assert.equal(ui.elements['message-input'].style.paddingRight, '148px');
  const second = ui.submit('Focus on concurrency');
  await settle();
  assert.equal(ui.streams[0].signal.aborted, true);
  assert.equal(ui.requests.length, 2);
  assert.deepEqual(ui.requests[1].messages, [
    { role: 'user', content: 'Explain the design' },
    { role: 'assistant', content: 'The first design', thinking: 'Considering several designs.' },
    { role: 'user', content: 'Focus on concurrency' },
  ]);
  assert.equal(ui.requests[1].replaceActive, true);
  assert.equal(ui.requests[1].sessionId, 'chat-session');
  assert.equal(ui.requests[1].newChat, undefined, 'steering preserves normal KV reuse');
  assert.equal(ui.assistantBubbles[0].querySelector('.bubble-text').textContent, 'The first design');
  assert.equal(ui.assistantBubbles[0].querySelector('.stats').textContent, 'Steered by new prompt');
  assert.equal(ui.frames.size, 0, 'pending paint of the interrupted reply was canceled');
  complete(ui.streams[1], 'Use a serialized queue.');
  await Promise.all([first, second]);
  assert.equal(ui.history.at(-1).content, 'Use a serialized queue.');
  assert.equal(ui.elements['btn-stop'].hidden, true);
  assert.equal(ui.elements['message-input'].style.paddingRight, '');
});

test('rapid guidance is ordered, coalesced, and ignores frames arriving after cancellation', async () => {
  const ui = page({ deferAbort: true });
  const first = ui.submit('Initial prompt');
  await settle();
  ui.streams[0].frame({ thinking: 'Planning' });
  await settle();
  const second = ui.submit('Use a short answer');
  const third = ui.submit('Include one example');
  assert.equal(ui.requests.length, 1, 'successor waits for partial-history finalization');
  ui.streams[0].frame({ token: 'STALE CONTENT', done: true, sessionId: 'stale-session' });
  await settle();
  assert.equal(ui.requests.length, 2);
  assert.deepEqual(ui.requests[1].messages.map(message => message.content), ['Initial prompt', '', 'Use a short answer', 'Include one example']);
  assert.equal(ui.requests[1].messages[1].thinking, 'Planning');
  assert.equal(ui.context.currentSessionId, 'chat-session');
  assert.equal(ui.requests[1].messages.some(message => message.content.includes('STALE')), false);
  complete(ui.streams[1]);
  await Promise.all([first, second, third]);
});

test('Stop after accepted guidance preserves the guidance and suppresses its restart', async () => {
  const ui = page({ deferAbort: true });
  const first = ui.submit('Initial prompt');
  await settle();
  ui.streams[0].frame({ token: 'Partial answer' });
  await settle();
  const next = ui.submit('Make it shorter');
  ui.context.abortGeneration();
  ui.streams[0].close();
  await Promise.all([first, next]);
  assert.equal(ui.requests.length, 1);
  assert.deepEqual(ui.history.map(message => message.content), ['Initial prompt', 'Partial answer', 'Make it shorter']);
  assert.equal(ui.assistantBubbles[0].querySelector('.stats').textContent, 'Stopped by user');
  assert.equal(ui.elements['btn-stop'].hidden, true);
  const resumed = ui.submit('Please continue');
  await settle();
  assert.equal(ui.requests.length, 2);
  complete(ui.streams[1]);
  await resumed;
});

test('New Chat discards interrupted history and queued steering before creating its session', async () => {
  const ui = page({ deferAbort: true });
  const first = ui.submit('Old prompt');
  await settle();
  ui.streams[0].frame({ token: 'Old answer' });
  await settle();
  const steering = ui.submit('Old guidance');
  const clear = ui.context.clearChat();
  ui.streams[0].frame({ token: 'Stale answer' });
  await Promise.all([first, steering, clear]);
  assert.deepEqual(ui.history, []);
  assert.equal(ui.requests.length, 1);
  assert.equal(ui.context.currentSessionId, 'new-session');
  assert.ok(ui.container.querySelector('.empty-state'), 'queued worker must preserve the empty state');
  assert.equal(ui.calls.find(call => call.method === 'DELETE').url, '/api/sessions/chat-session');
  const next = ui.submit('Fresh prompt');
  await settle();
  assert.deepEqual(ui.requests[1].messages, [{ role: 'user', content: 'Fresh prompt' }]);
  assert.equal(ui.requests[1].sessionId, 'new-session');
  complete(ui.streams[1]);
  await next;
});

test('a new prompt during New Chat session setup waits and uses the new session', async () => {
  let releaseSession;
  const ui = page({ sessionReplies: [new Promise(resolve => { releaseSession = resolve; })] });
  const clear = ui.context.clearChat();
  await settle();
  const next = ui.submit('Use the new chat');
  await settle();
  assert.equal(ui.requests.length, 0);
  releaseSession({ sessionId: 'created-session' });
  await clear;
  await settle();
  assert.equal(ui.requests[0].sessionId, 'created-session');
  complete(ui.streams[0]);
  await next;
});

test('late session creation is disposed after New Chat and cannot replace the current session', async () => {
  let releaseSession;
  const ui = page({ sessionReplies: [new Promise(resolve => { releaseSession = resolve; }), { sessionId: 'current-session' }] });
  ui.context.currentSessionId = null;
  const creating = ui.context.createSession();
  const clear = ui.context.clearChat();
  await settle();
  await clear;
  releaseSession({ sessionId: 'obsolete-session' });
  await creating;
  assert.equal(ui.context.currentSessionId, 'current-session');
  assert.ok(ui.calls.some(call => call.url === '/api/sessions/obsolete-session' && call.method === 'DELETE'));
});

test('startup and first prompt share one pending session creation, even with steering', async () => {
  let releaseSession;
  const ui = page({ sessionReplies: [new Promise(resolve => { releaseSession = resolve; })] });
  ui.context.currentSessionId = null;
  const creating = ui.context.createSession();
  const first = ui.submit('Initial prompt');
  await settle();
  const steering = ui.submit('Guidance during startup');
  assert.equal(ui.calls.filter(call => call.url === '/api/sessions').length, 1);
  releaseSession({ sessionId: 'shared-session' });
  await creating;
  await settle();
  assert.equal(ui.requests.length, 1, 'canceled initial request cannot start inference after session setup');
  assert.equal(ui.requests[0].sessionId, 'shared-session');
  assert.deepEqual(ui.requests[0].messages.map(message => message.content), ['Initial prompt', 'Guidance during startup']);
  complete(ui.streams[0]);
  await Promise.all([first, steering]);
});

test('Enter submits guidance while generating, but IME composition, repeats and Shift+Enter do not', async () => {
  const ui = page();
  const first = ui.submit('Initial prompt');
  await settle();
  const key = ui.elements['message-input'].listeners.keydown;
  ui.elements['message-input'].value = 'Guidance';
  for (const options of [{ isComposing: true }, { keyCode: 229 }, { repeat: true }, { shiftKey: true }]) {
    key({ key: 'Enter', preventDefault() {}, ...options });
  }
  assert.equal(ui.streams[0].signal.aborted, false);
  key({ key: 'Enter', preventDefault() {} });
  await settle();
  assert.equal(ui.requests.length, 2);
  assert.equal(ui.requests[1].messages.at(-1).content, 'Guidance');
  complete(ui.streams[1]);
  await first;
});

test('edits and reverts cannot mutate a reply while guidance is being applied', async () => {
  const ui = page();
  const first = ui.submit('Initial prompt');
  await settle();
  ui.context.revertFrom(0);
  ui.context.startEdit(0);
  await ui.context.submitEdit(0);
  assert.deepEqual(ui.history, [{ role: 'user', content: 'Initial prompt' }]);
  complete(ui.streams[0]);
  await first;
});

test('the WebUX controls expose Send and Stop as separate accessible actions', () => {
  assert.match(html, /id="btn-stop"[^>]*onclick="abortGeneration\(\)"[^>]*aria-label="Stop generating"/);
  assert.match(html, /id="btn-send"[^>]*onclick="sendMessage\(\)"[^>]*aria-label="Send message"/);
});
