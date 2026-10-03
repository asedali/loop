/* Contract tests for app.js.
 *
 * There is no browser in the test environment, so rather than pulling in a DOM
 * implementation the pure decision helpers are exercised directly by stubbing the
 * handful of globals app.js touches at load. This covers the part that is
 * easiest to break silently: the action-URL -> stage mapping, which once
 * referenced a route that no longer existed and therefore showed the wrong
 * progress text for the work actually in flight.
 *
 * Run with: node tests/app_js.test.js   (pytest invokes it)
 */
'use strict';

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const SRC = fs.readFileSync(
  path.join(__dirname, '..', 'app', 'static', 'app.js'), 'utf8'
);

/* ---- stageFor(), lifted out of the IIFE by evaluating the real source ---- */

function loadStageFor() {
  // Rewrite the IIFE's tail so the module exports what the tests need, while
  // still executing the file that ships. Anything that touches `document`
  // during load would throw, so give it a minimal stub.
  const module_ = { exports: {} };
  const sandbox = {
    module: module_,
    exports: module_.exports,
    // A bare vm context has no URL, and isPlainSameOriginLink needs one to
    // resolve a relative href.
    URL,
    document: {
      documentElement: { getAttribute: () => null, removeAttribute() {}, setAttribute() {} },
      querySelector: () => null,
      querySelectorAll: () => [],
      addEventListener() {},
      readyState: 'complete'
    },
    window: {
      addEventListener() {},
      matchMedia: () => ({ matches: false, addEventListener() {} }),
      // isPlainSameOriginLink compares against window.location, not an argument,
      // so origin/search have to be present or every check bails out.
      location: { href: 'http://localhost:8000/venture/1', origin: 'http://localhost:8000', pathname: '/venture/1', search: '' }
    },
    localStorage: { getItem: () => null, setItem() {} },
    console
  };
  sandbox.window.document = sandbox.document;
  sandbox.globalThis = sandbox;

  const wrapped = SRC.replace(
    '  if (document.readyState === \'loading\') {',
    '  if (module.exports) { module.exports = { stageFor, isPlainSameOriginLink, STAGES }; }\n  if (document.readyState === \'loading\') {'
  );
  const ctx = vm.createContext(sandbox);
  vm.runInContext(wrapped, ctx, { filename: 'app.js' });
  if (!module_.exports.stageFor) throw new Error('harness failed to capture exports');
  return module_.exports;
}

const { stageFor, STAGES } = loadStageFor();

const form = (action) => ({ getAttribute: (n) => (n === 'action' ? action : null) });

const tests = [];
const test = (name, fn) => tests.push([name, fn]);

/* ---- stage routing ---- */

test('designing a run for a block shows the design stages', () => {
  assert.strictEqual(stageFor(form('/venture/7/segment/customer/run')), 'design');
});

test('designing a run is matched before /log, not after', () => {
  // Both shapes are easy to confuse; getting this wrong is what produced
  // "Reading the outcomes you logged" while the app was writing tasks.
  assert.strictEqual(stageFor(form('/venture/7/segment/customer/run')), 'design');
  assert.strictEqual(stageFor(form('/venture/7/run/12/log')), 'analyze');
});

test('a segment key containing log-ish words still routes as design', () => {
  assert.strictEqual(stageFor(form('/venture/7/segment/key_partners/run')), 'design');
});

test('scoring a run routes to analyze', () => {
  assert.strictEqual(stageFor(form('/venture/7/run/12/log')), 'analyze');
});

test('retrying a verdict routes to retry', () => {
  assert.strictEqual(stageFor(form('/venture/7/run/12/retry-analysis')), 'retry');
});

test('phase 1 extract routes to extract', () => {
  assert.strictEqual(stageFor(form('/phase1/extract')), 'extract');
});

test('phase 3 generation routes to strategy', () => {
  assert.strictEqual(stageFor(form('/venture/7/phase3/generate')), 'strategy');
});

test('no stale route is referenced anywhere in app.js', () => {
  assert.ok(!SRC.includes('/cycle/new'),
    'app.js still references /cycle/new, a route removed in the per-segment loop');
});

/* ---- stage copy must describe the per-segment loop, not the old flat one ---- */

test('stage copy no longer describes the old whole-canvas sweep', () => {
  const flat = JSON.stringify(STAGES);
  assert.ok(!/least-validated/i.test(flat),
    "the 'todos' copy was written for the old flat loop");
  assert.ok(!/persevere/.test(flat),
    'the verdict set is now pass/iterate/fail/pivot');
});

test('the design stage mentions a hypothesis, which is what the run does', () => {
  assert.ok(/hypothesis/i.test(STAGES.design.join(' ')));
});

test('every stage referenced by routing has copy defined', () => {
  ['extract', 'design', 'analyze', 'retry', 'strategy'].forEach((k) => {
    assert.ok(Array.isArray(STAGES[k]) && STAGES[k].length, `no copy for stage '${k}'`);
  });
  // and no orphan copy left behind
  Object.keys(STAGES).forEach((k) => {
    assert.ok(['extract', 'design', 'analyze', 'retry', 'strategy'].includes(k),
      `stage '${k}' is defined but never routed to`);
  });
});

/* ---- navigation interception rules ---- */

const { isPlainSameOriginLink } = loadStageFor();
const link = (href, attrs = {}) => ({
  getAttribute: (n) => (n === 'href' ? href : attrs[n]),
  hasAttribute: (n) => n in attrs,
  target: attrs.target,
  textContent: ''
});
const click = (over = {}) => Object.assign({ button: 0, defaultPrevented: false }, over);
const ev = (a, over) => ({ target: { closest: (sel) => a }, ...click(over) });

test('a plain same-origin link is intercepted', () => {
  assert.strictEqual(isPlainSameOriginLink(ev(link('/dashboard')), link('/dashboard')), true,
    'a plain same-origin click must be intercepted, or navigation stays silent');
});

test('a modified click is left to the browser (new tab / new window)', () => {
  ['metaKey', 'ctrlKey', 'shiftKey', 'altKey'].forEach((mod) => {
    const e = ev(link('/dashboard'), { [mod]: true });
    assert.strictEqual(isPlainSameOriginLink(e, link('/dashboard')), false,
      `${mod} click must not be intercepted`);
  });
});

test('middle click is left to the browser', () => {
  assert.strictEqual(
    isPlainSameOriginLink(ev(link('/dashboard'), { button: 1 }), link('/dashboard')), false);
});

test('target=_blank and download links are left to the browser', () => {
  assert.strictEqual(isPlainSameOriginLink(
    ev(link('/x', { target: '_blank' })), link('/x', { target: '_blank' })), false);
  assert.strictEqual(isPlainSameOriginLink(
    ev(link('/x', { download: '' })), link('/x', { download: '' })), false);
});

test('rel=external, mailto and in-page anchors are not intercepted', () => {
  assert.strictEqual(isPlainSameOriginLink(
    ev(link('/x', { rel: 'external' })), link('/x', { rel: 'external' })), false);
  assert.strictEqual(isPlainSameOriginLink(ev(link('mailto:a@b.c')), link('mailto:a@b.c')), false);
  assert.strictEqual(isPlainSameOriginLink(ev(link('#section')), link('#section')), false);
});

test('a cross-origin link is not intercepted', () => {
  const a = link('https://example.com/thing');
  assert.strictEqual(isPlainSameOriginLink(ev(a), a), false);
});

test('an already-handled click is not re-handled', () => {
  const a = link('/dashboard');
  assert.strictEqual(isPlainSameOriginLink(ev(a, { defaultPrevented: true }), a), false);
});

/* ---- runner ---- */

let failed = 0;
tests.forEach(([name, fn]) => {
  try { fn(); process.stdout.write(`  ok   ${name}\n`); }
  catch (err) { failed++; process.stdout.write(`  FAIL ${name}\n         ${err.message}\n`); }
});
process.stdout.write(`\n${tests.length - failed}/${tests.length} app.js contract tests passed\n`);
process.exit(failed ? 1 : 0);
