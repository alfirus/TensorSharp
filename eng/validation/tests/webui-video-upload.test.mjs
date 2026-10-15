// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
// Licensed under the BSD-3-Clause license found in the repository's LICENSE file.

// Run with: node --test eng/validation/tests/webui-video-upload.test.mjs
// These execute the shipped WebUX functions, including the upload-to-history
// handoff, without loading a model or replacing request construction with a copy.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const html = readFileSync(new URL('../../../TensorSharp.Server.Host/wwwroot/index.html', import.meta.url), 'utf8');

function sourceBetween(startMarker, endMarker) {
  const start = html.indexOf(startMarker);
  const end = html.indexOf(endMarker, start + startMarker.length);
  assert.ok(start >= 0 && end > start, `Cannot find shipped WebUX function: ${startMarker}`);
  return html.slice(start, end);
}

const sendSource = sourceBetween('async function sendMessage()', '// Finish the interrupted assistant entry');
const videoSource = sourceBetween('async function runVideoGenerate(', 'async function requestAssistantResponse()');
const referenceCaps = { supportsReferenceConditioning: true, maxReferenceImages: 3 };
const keyframeCaps = { supportsImageConditioning: true, supportsEndImageConditioning: true };
const wanCaps = { supportsImageConditioning: true, supportsEndImageConditioning: false };
const picture = file => ({ file, fileName: 'banner_1.png', mediaType: 'image' });

async function postUploadedAttachments(caps, attachments) {
  const requests = [];
  const bubble = { textContent: '' };
  const stats = { textContent: '' };
  const generation = { controller: new AbortController() };
  const context = vm.createContext({
    currentVideoCaps: caps,
    currentLoadedModel: { architecture: 'video' },
    pendingAttachments: attachments,
    pendingUploadCount: 0,
    activeImageEditTarget: null,
    messageInput: { value: '保持人物全身入镜，让角色移动双手。', style: {} },
    attachmentsDiv: { innerHTML: '' },
    queuedChatMessages: [],
    queuedResponseRequested: false,
    describeAttachments: () => 'Animate the uploaded image.',
    isQwenImageModel: () => false,
    cancelCurrentGeneration: () => {},
    setSendButtonState: () => {},
    ensureCurrentGeneration: () => {},
    alert: message => { throw new Error(message); },
    performance,
    TextDecoder,
    fetch: async (url, request) => {
      requests.push({ url, method: request.method, payload: JSON.parse(request.body) });
      assert.equal(request.headers['Content-Type'], 'application/json');
      assert.equal(request.signal, generation.controller.signal);
      return { ok: true, body: { getReader: () => ({ read: async () => ({ done: true }) }) } };
    },
  });
  vm.runInContext(sendSource + '\n' + videoSource, context);
  let userMessage;
  context.drainChatSubmissions = async () => {
    assert.equal(context.queuedChatMessages.length, 1);
    userMessage = context.queuedChatMessages[0].message;
    await context.runVideoGenerate(userMessage, bubble, stats, generation);
  };
  await context.sendMessage();
  assert.equal(requests.length, 1, `Expected video request; UI showed: ${bubble.textContent}`);
  assert.equal(requests[0].url, '/api/video-generate/stream');
  assert.equal(requests[0].method, 'POST');
  assert.notEqual(stats.textContent, 'Failed', bubble.textContent);
  assert.equal(context.pendingAttachments.length, 0, 'Submitted uploads should leave the composer');
  return { payload: requests[0].payload, stats: stats.textContent, userMessage };
}

test('Ref2VA uploaded image carries reference mode, overriding a startup i2v default', async () => {
  const { payload } = await postUploadedAttachments(referenceCaps, [picture('upload-banner.png')]);
  assert.deepEqual(payload, {
    prompt: '保持人物全身入镜，让角色移动双手。',
    referenceImages: ['upload-banner.png'],
    videoMode: 'ref',
  });
});

test('Ref2VA keeps two uploaded images as references', async () => {
  const { payload } = await postUploadedAttachments(referenceCaps, [picture('first.png'), picture('second.png')]);
  assert.deepEqual(payload.referenceImages, ['first.png', 'second.png']);
  assert.equal(payload.videoMode, 'ref');
  assert.ok(!('imagePath' in payload) && !('endImage' in payload));
});

test('Ref2VA clip-only upload keeps the clip, excluding sampled frames from still references', async () => {
  const { payload, userMessage } = await postUploadedAttachments(referenceCaps, [
    { file: 'upload-clip.mp4', fileName: 'clip.mp4', mediaType: 'video', frames: ['clip-frame-1.png', 'clip-frame-2.png'] },
  ]);
  assert.equal(userMessage.imagePaths.length, 2, 'The upload handoff must retain sampled frames for vision chat');
  assert.deepEqual(payload.referenceVideos, ['upload-clip.mp4']);
  assert.equal(payload.videoMode, 'ref');
  assert.ok(!('referenceImages' in payload));
});

test('Ref2VA sends each uploaded media kind through its matching reference field', async () => {
  const { payload } = await postUploadedAttachments(referenceCaps, [
    picture('photo.png'),
    { file: 'clip.mp4', fileName: 'clip.mp4', mediaType: 'video', frames: ['clip-frame.png'] },
    { file: 'sound.wav', fileName: 'sound.wav', mediaType: 'audio' },
  ]);
  assert.deepEqual(payload.referenceImages, ['photo.png']);
  assert.deepEqual(payload.referenceVideos, ['clip.mp4']);
  assert.deepEqual(payload.referenceAudios, ['sound.wav']);
  assert.equal(payload.videoMode, 'ref');
});

test('FL2VA one uploaded image selects image-to-video using the uploaded file', async () => {
  const { payload } = await postUploadedAttachments(keyframeCaps, [picture('upload-first.png')]);
  assert.equal(payload.videoMode, 'i2v');
  assert.equal(payload.imagePath, 'upload-first.png');
  assert.ok(!('endImage' in payload) && !('referenceImages' in payload));
});

test('FL2VA two uploaded images select first-and-last-frame mode', async () => {
  const { payload } = await postUploadedAttachments(keyframeCaps, [picture('first.png'), picture('last.png')]);
  assert.equal(payload.videoMode, 'fl2v');
  assert.equal(payload.imagePath, 'first.png');
  assert.equal(payload.endImage, 'last.png');
});

test('Wan image-to-video sends one conditioning image and reports additional unused images', async () => {
  const { payload, stats } = await postUploadedAttachments(wanCaps, [picture('first.png'), picture('extra.png')]);
  assert.equal(payload.videoMode, 'i2v');
  assert.equal(payload.imagePath, 'first.png');
  assert.ok(!('endImage' in payload));
  assert.match(stats, /1 image not used/);
});

test('Older server without capabilities still receives an uploaded first frame', async () => {
  const { payload } = await postUploadedAttachments(null, [picture('upload.png')]);
  assert.equal(payload.videoMode, 'i2v');
  assert.equal(payload.imagePath, 'upload.png');
});

test('Text-only video requests retain the server startup mode default', async () => {
  const { payload } = await postUploadedAttachments(referenceCaps, []);
  assert.deepEqual(payload, { prompt: '保持人物全身入镜，让角色移动双手。' });
});
