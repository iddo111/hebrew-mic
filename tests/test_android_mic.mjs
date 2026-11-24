// Run with: node --test tests/test_android_mic.mjs
// Exercise the shipped page without a browser, microphone, network or keyboard.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const html = readFileSync(new URL('../android_mic.html', import.meta.url), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function page({ permission, response } = {}) {
  const nodes = new Map();
  const timers = new Map();
  const recognizers = [];
  const requests = [];
  let timerId = 0;
  const track = { stopped: 0, stop() { this.stopped++; } };
  const stream = { getTracks: () => [track] };
  class Recognition {
    constructor() { recognizers.push(this); this.starts = 0; }
    start() { this.starts++; this.onstart?.(); }
    stop() { this.onend?.(); }
    abort() { this.onend?.(); }
    end() { this.onend?.(); }
    result(text, index = 0) {
      const result = [{ transcript: text }];
      result.isFinal = true;
      const results = Array.from({ length: index + 1 }, () => result);
      this.onresult({ resultIndex: index, results });
    }
  }
  const node = (id) => {
    if (!nodes.has(id)) nodes.set(id, {
      value: '', textContent: '', className: '',
      classList: { toggle() {} }, addEventListener() {},
      getContext: () => ({ clearRect() {} }),
    });
    return nodes.get(id);
  };
  const context = vm.createContext({
    window: { SpeechRecognition: Recognition },
    location: { protocol: 'https:', origin: 'https://mic.example', search: '' },
    localStorage: { getItem: () => null, setItem() {} },
    document: { getElementById: node, addEventListener() {} },
    navigator: {
      mediaDevices: { getUserMedia: () => permission ? permission(stream) : Promise.resolve(stream) },
      clipboard: { writeText: async () => {} },
    },
    fetch: async (url, options) => {
      requests.push({ url, text: JSON.parse(options.body).text });
      return response ? response() : { status: 200, json: async () => ({ ok: true }) };
    },
    URLSearchParams,
    setTimeout(fn, delay) { const id = ++timerId; timers.set(id, { fn, delay }); return id; },
    clearTimeout(id) { timers.delete(id); },
    cancelAnimationFrame() {},
    alert() {},
  });
  vm.runInContext(script, context);
  // The visualizer is unrelated to recognition event handling.
  vm.runInContext('startVisualizer = () => {}; stopVisualizer = () => {};', context);
  return {
    context, recognizers, requests, track, node,
    toggle: () => context.toggleListen(),
    tick() {
      const pending = [...timers].filter(([, item]) => item.delay <= 1000);
      for (const [id, item] of pending) { timers.delete(id); item.fn(); }
    },
  };
}

test('a repeated final event is dispatched once, but a new utterance may repeat it', async () => {
  const p = page();
  await p.toggle();
  const r = p.recognizers[0];
  r.result('קלוד תעשה את החדר שלי');
  r.result('קלוד תעשה את החדר שלי');
  assert.equal(p.requests.length, 1);
  r.end();
  p.tick();
  r.result('קלוד תעשה את החדר שלי');
  assert.equal(p.requests.length, 2);
});

test('separate final indices in one recognition session are preserved', async () => {
  const p = page();
  await p.toggle();
  p.recognizers[0].result('קלוד משפט ראשון');
  p.recognizers[0].result('קלוד משפט שני', 1);
  assert.deepEqual(p.requests.map(x => x.text), ['משפט ראשון', 'משפט שני']);
});

for (const error of ['no-speech', 'network']) {
  test(`${error} recovers once after end while listening remains requested`, async () => {
    const p = page();
    await p.toggle();
    const r = p.recognizers[0];
    r.onerror({ error });
    r.end();
    p.tick();
    assert.equal(r.starts, 2);
    assert.equal(p.track.stopped, 0);
  });
}

test('permission failure stops capture and does not restart', async () => {
  const p = page();
  await p.toggle();
  const r = p.recognizers[0];
  r.onerror({ error: 'not-allowed' });
  r.end();
  p.tick();
  assert.equal(r.starts, 1);
  assert.equal(p.track.stopped, 1);
});

test('stop ignores late results and cancels a queued restart', async () => {
  const p = page();
  await p.toggle();
  const r = p.recognizers[0];
  r.end();
  await p.toggle();
  r.result('קלוד הודעה ישנה');
  p.tick();
  assert.equal(p.requests.length, 0);
  assert.equal(r.starts, 1);
});

test('a previous recognizer cannot dispatch or restart a new listening session', async () => {
  const p = page();
  await p.toggle();
  const old = p.recognizers[0];
  old.end();
  await p.toggle();
  await p.toggle();
  old.result('קלוד הודעה ישנה');
  old.end();
  p.tick();
  assert.equal(p.requests.length, 0);
  assert.equal(p.recognizers[1].starts, 1);
});

test('stopping while permission is pending closes the late stream', async () => {
  let allow;
  const p = page({ permission: stream => new Promise(resolve => { allow = () => resolve(stream); }) });
  const starting = p.toggle();
  await p.toggle();
  allow();
  await starting;
  assert.equal(p.track.stopped, 1);
  assert.equal(p.recognizers.length, 0);
});

test('a late dispatch response cannot put a stopped mic back into listening UI', async () => {
  let finish;
  const p = page({ response: () => new Promise(resolve => { finish = resolve; }) });
  await p.toggle();
  p.recognizers[0].result('קלוד הודעה שכבר נשלחה');
  await p.toggle();
  finish({ status: 200, json: async () => ({ ok: true }) });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(p.node('micBtn').className, 'idle');
});
