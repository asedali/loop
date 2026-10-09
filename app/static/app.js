/* LaunchLoop front-end behaviour.
 *
 * Two jobs, both progressive enhancements — every page works without JS:
 *
 *  1. Thinking overlay. A form that calls the LLM blocks for 10-60s (and the
 *     provider retries internally, so it can be longer). Submitted natively
 *     that reads as a hung page. We intercept those forms, keep the current
 *     page on screen, and swap in the server's response when it lands.
 *
 *  2. Theme toggle, persisted to localStorage.
 *
 * The swap works without touching any backend route: the POSTs already
 * redirect with 303, so fetch() follows the redirect and hands back the fully
 * rendered HTML of wherever we were supposed to end up. We take <main> out of
 * it and drop that into the current document.
 */
(function () {
  'use strict';

  var root = document.documentElement;
  var THEME_KEY = 'loop-theme';

  /* ------------------------------------------------------------------ */
  /* Theme                                                                */
  /* ------------------------------------------------------------------ */

  function syncToggleLabel() {
    var btn = document.querySelector('.theme-toggle');
    if (!btn) return;
    var label = currentPref() === 'light' ? 'Switch to dark theme' : 'Switch to light theme';
    btn.setAttribute('aria-label', label);
    btn.setAttribute('title', label);
  }

  function applyTheme(theme) {
    if (theme === 'light' || theme === 'dark') {
      root.setAttribute('data-theme', theme);
    } else {
      root.removeAttribute('data-theme'); // 'system' — fall back to the OS
    }
    syncToggleLabel();
  }

  function currentPref() {
    return root.getAttribute('data-theme') ||
      (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  }

  function initTheme() {
    var stored = null;
    try { stored = window.localStorage.getItem(THEME_KEY); } catch (e) { /* private mode */ }
    // The inline script in <head> has already resolved data-theme from the OS
    // preference. Only override it when the user has explicitly chosen —
    // calling applyTheme(null) here would strip that and silently force dark.
    if (stored === 'light' || stored === 'dark') applyTheme(stored);
    else syncToggleLabel();

    document.addEventListener('click', function (e) {
      var btn = e.target.closest('.theme-toggle');
      if (!btn) return;
      var next = currentPref() === 'light' ? 'dark' : 'light';
      applyTheme(next);
      try { window.localStorage.setItem(THEME_KEY, next); } catch (e) { /* ignore */ }
    });

    // Follow the OS if the user has never chosen for themselves.
    if (window.matchMedia) {
      var mq = window.matchMedia('(prefers-color-scheme: light)');
      var onChange = function () {
        var s = null;
        try { s = window.localStorage.getItem(THEME_KEY); } catch (e) {}
        if (s !== 'light' && s !== 'dark') { applyTheme(null); syncToggleLabel(); }
      };
      if (mq.addEventListener) mq.addEventListener('change', onChange);
    }
  }

  /* ------------------------------------------------------------------ */
  /* Thinking overlay                                                     */
  /* ------------------------------------------------------------------ */

  // The text cycles so a long wait doesn't look frozen, and each line tells
  // the user what is actually happening rather than just "loading".
  var STAGES = {
    extract: [
      'Reading your research material…',
      'Pulling out the technical claims…',
      'Clustering related claims into ideas…',
      'Framing the commercial angle…',
      'Checking the output is usable…'
    ],
    // One block at a time now, so the copy is about that block's hypothesis
    // rather than about sweeping the whole canvas.
    design: [
      'Reading this block and your canvas…',
      'Framing a falsifiable hypothesis…',
      'Picking a method that fits it…',
      'Writing the tasks and pass criteria…',
      'Checking the output is usable…'
    ],
    analyze: [
      'Reading the outcomes you logged…',
      'Checking them against the success criteria…',
      'Deciding: pass, iterate, fail, or pivot…',
      'Writing the justification…'
    ],
    retry: [
      'Re-reading your saved results…',
      'Checking them against the success criteria…',
      'Deciding: pass, iterate, fail, or pivot…'
    ],
    strategy: [
      'Reading the resolved canvas…',
      'Matching funding categories…',
      'Picking acquisition channels…',
      'Sequencing the action plan…'
    ],
    // Not a model call in the sense the others are — nothing here decides
    // anything — but it is still a call with real latency, so it gets its own
    // copy. Saying "Judging your idea" would be the wrong claim: it is reading a
    // record and asking questions about it.
    challenge: [
      'Reading what you recorded…',
      'Applying the playbook…',
      'Checking the questions are specific…'
    ]
  };

  function stageFor(form) {
    var act = form.getAttribute('action') || '';
    if (act.indexOf('/phase1/extract') !== -1) return 'extract';
    // Designing a run for one block. Checked before /log because the two route
    // shapes are easy to confuse, and this must not fall through to 'analyze'
    // — that shows the wrong stages for the work actually happening.
    if (/\/segment\/[^/]+\/run$/.test(act)) return 'design';
    if (act.indexOf('/retry-analysis') !== -1) return 'retry';
    if (/\/run\/[^/]+\/log$/.test(act)) return 'analyze';
    if (act.indexOf('/phase3/generate') !== -1) return 'strategy';
    // Checked before the catch-all below, and matched on /mentor rather than a
    // full route: the three challenge routes live at different paths (/mentor/…
    // and /venture/{id}/mentor/…) so there is no single suffix to key on.
    if (act.indexOf('/mentor/') !== -1) return 'challenge';
    return 'analyze';
  }

  function buildOverlay(key) {
    var lines = STAGES[key] || STAGES.analyze;
    var el = document.createElement('div');
    el.className = 'thinking';
    el.setAttribute('role', 'status');
    el.setAttribute('aria-live', 'polite');
    el.innerHTML =
      '<div class="thinking-card">' +
        '<div class="thinking-spinner"></div>' +
        '<h2>Thinking…</h2>' +
        '<p class="thinking-stage"></p>' +
        '<div class="thinking-bar"></div>' +
        '<div class="thinking-meta">' +
          '<span class="elapsed">0s</span>' +
          '<span>AI calls can take up to a minute</span>' +
        '</div>' +
        '<p class="thinking-note">Your work is already saved — it\'s safe to leave this page open.</p>' +
        '<button type="button" class="thinking-stop">Stop waiting</button>' +
      '</div>';
    document.body.appendChild(el);

    var stageEl = el.querySelector('.thinking-stage');
    var elapsed = el.querySelector('.elapsed');
    var seconds = 0;
    var i = 0;

    // The server retries transient failures up to 3 times with backoff, so a
    // hard 2-minute ceiling before we call it: better to surface the real
    // error than to spin forever behind a dead socket.
    var MAX_WAIT_MS = 150000;

    var tick = setInterval(function () {
      seconds++;
      elapsed.textContent = seconds + 's';
      // Each line gets more time as the wait goes on; don't cycle a 5-line
      // list every 2s or it reads as frantic nonsense.
      var idx = seconds < 6 ? Math.floor(seconds / 2) : Math.min(lines.length - 1, Math.floor(seconds / 5));
      if (idx !== i && idx < lines.length) {
        i = idx;
        stageEl.classList.add('fading');
        setTimeout(function () {
          stageEl.textContent = lines[i];
          stageEl.classList.remove('fading');
        }, 200);
      }
    }, 1000);

    var timer = setTimeout(function () { stop(true); }, MAX_WAIT_MS);

    function stop(timedOut) {
      clearInterval(tick);
      clearTimeout(timer);
      if (el.parentNode) el.parentNode.removeChild(el);
      document.body.classList.remove('is-busy');
      if (timedOut && window.__llAbort) {
        window.__llAbort.abort();
        window.__llReload = true; // let the real error page through
      }
    }
    el.querySelector('.thinking-stop').addEventListener('click', function () { stop(false); });
    stageEl.textContent = lines[0];
    return { stop: stop, node: el };
  }

  /* ------------------------------------------------------------------ */
  /* Submit interception                                                  */
  /* ------------------------------------------------------------------ */

  function swapIn(html, url, redirected) {
    var doc = new DOMParser().parseFromString(html, 'text/html');
    var nextMain = doc.querySelector('main');
    var curMain = document.querySelector('main');
    if (!nextMain || !curMain) { window.location.href = url; return; }

    curMain.className = nextMain.className;
    curMain.innerHTML = nextMain.innerHTML;
    document.title = doc.title;

    // res.url is absolute; nav hrefs are paths, so compare pathnames or nothing
    // ever matches and every link silently loses its active state.
    var path;
    try { path = new URL(url, window.location.origin).pathname; }
    catch (e) { path = url; }

    // When the server answers the POST in place instead of redirecting (a quota
    // or provider error re-rendering the same page), keep the URL we were
    // already on. Pushing the POST endpoint would put /phase1/extract in the
    // address bar and leave no nav link matching it.
    if (!redirected) path = window.location.pathname;

    // A venture page has no nav link of its own; it belongs under "Ventures".
    var highlight = path.indexOf('/venture/') === 0 ? '/dashboard' : path;

    var links = document.querySelectorAll('nav .links a');
    for (var i = 0; i < links.length; i++) {
      var href = links[i].getAttribute('href');
      links[i].classList.toggle('on', !!href && (highlight === href || (href !== '/' && highlight.indexOf(href) === 0)));
    }
    if (redirected && url !== window.location.href) {
      window.history.pushState({}, '', url);
    }
    window.scrollTo(0, 0);

    // The DOM under <main> was replaced wholesale, so whatever had focus is gone
    // and the browser has fallen back to <body> — a keyboard or screen-reader
    // user who activates a control would be dumped at the top of the document
    // with no announcement. Move focus to the new content instead.
    if (curMain) {
      if (!curMain.hasAttribute('tabindex')) curMain.setAttribute('tabindex', '-1');
      curMain.focus({ preventScroll: true });
    }
  }

  function initForms() {
    document.addEventListener('submit', function (e) {
      var form = e.target;
      if (!form.matches('form[data-thinking]')) return;
      if (form.dataset.busy === '1') { e.preventDefault(); return; } // double-submit guard

      e.preventDefault();
      form.dataset.busy = '1';
      document.body.classList.add('is-busy');

      var ov = buildOverlay(stageFor(form));
      var ctrl = new AbortController();
      window.__llAbort = ctrl;
      window.__llReload = false;

      fetch(form.action, {
        method: 'POST',
        body: new FormData(form),
        redirect: 'follow',
        signal: ctrl.signal
      }).then(function (res) {
        return res.text().then(function (html) {
          return { html: html, url: res.url, redirected: res.redirected };
        });
      }).then(function (r) {
        ov.stop(false);
        form.dataset.busy = '';
        document.body.classList.remove('is-busy');
        window.__llAbort = null;
        if (window.__llReload) { window.location.reload(); return; }
        swapIn(r.html, r.url, r.redirected);
      }).catch(function (err) {
        if (err && err.name === 'AbortError') {
          // "Stop waiting" — the server may still finish, so reload to get
          // whatever the real state is rather than guessing client-side.
          ov.stop(false);
          document.body.classList.remove('is-busy');
          window.location.reload();
          return;
        }
        // Network died before the server answered. The POST may not have
        // reached the app at all, so a plain reload is the safe move.
        ov.stop(false);
        form.dataset.busy = '';
        document.body.classList.remove('is-busy');
        window.__llAbort = null;
        showOfflineNote();
      });
    });
  }

  function showOfflineNote() {
    var main = document.querySelector('main');
    if (!main || main.querySelector('.thinking-offline')) return;
    var box = document.createElement('div');
    box.className = 'error thinking-offline';
    box.textContent = 'Could not reach the server. Check your connection, then reload the page and try again.';
    main.insertBefore(box, main.firstChild);
  }

  /* ------------------------------------------------------------------ */
  /* Page navigation                                                     */
  /* ------------------------------------------------------------------ */

  /* A slim bar rather than the full thinking overlay: most page transitions
     are fast, and a modal that flashes for 200ms reads as a glitch. It reuses
     the same sweep gradient, so it looks like the thinking overlay's progress
     bar and not like a new component. */
  var navBar, navTimer;

  function navStart() {
    if (navBar) return;
    navBar = document.createElement('div');
    navBar.className = 'nav-progress';
    navBar.setAttribute('role', 'status');
    navBar.setAttribute('aria-label', 'Loading page');
    document.body.appendChild(navBar);
    document.body.classList.add('is-navigating');
  }

  function navStop() {
    if (navTimer) { clearTimeout(navTimer); navTimer = null; }
    if (navBar && navBar.parentNode) navBar.parentNode.removeChild(navBar);
    navBar = null;
    document.body.classList.remove('is-navigating');
  }

  /* Bail out on anything that is not a plain same-origin left-click: a modified
     click means "open somewhere else", and those must reach the browser. */
  function isPlainSameOriginLink(e, a) {
    if (e.defaultPrevented || e.button !== 0) return false;
    if (e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return false;
    if (a.target && a.target !== '_self') return false;
    if (a.hasAttribute('download')) return false;
    var rel = (a.getAttribute('rel') || '').toLowerCase();
    if (rel.indexOf('external') !== -1) return false;
    var href = a.getAttribute('href');
    if (!href || href.charAt(0) === '#') return false;
    if (/^(mailto|tel|javascript):/i.test(href)) return false;
    var url;
    try { url = new URL(href, window.location.href); }
    catch (err) { return false; }
    if (url.origin !== window.location.origin) return false;
    // A pure-hash link to this page is a scroll, not a navigation.
    if (url.pathname === window.location.pathname && url.search === window.location.search) return false;
    return true;
  }

  function initNav() {
    document.addEventListener('click', function (e) {
      var a = e.target.closest('a');
      if (!a || !isPlainSameOriginLink(e, a)) return;

      var href = a.getAttribute('href');
      e.preventDefault();
      navStart();
      navTimer = setTimeout(function () {
        // If it is genuinely slow, escalate: the dim tells the user the page is
        // still responding rather than merely slow to change.
        document.body.classList.add('is-navigating-slow');
      }, 2000);

      fetch(href, { headers: { 'X-Requested-With': 'fetch' }, redirect: 'follow' })
        .then(function (res) { return res.text(); })
        .then(function (html) { navStop(); swapIn(html, new URL(href, window.location.href).href, true); })
        .catch(function () {
          navStop();
          // Never strand the user on a stale page: fall back to a real load.
          window.location.href = href;
        });
    });
  }

  /* ------------------------------------------------------------------ */

  function init() {
    initTheme();
    initForms();
    initNav();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Back/forward after an in-page swap: re-render from the new URL.
  window.addEventListener('popstate', function () { window.location.reload(); });
})();
