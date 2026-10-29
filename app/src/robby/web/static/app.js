/* Robson Readiness: small behaviours only. ES5, no dependencies besides htmx.
   Served as a static file: the Content-Security-Policy forbids inline script. */
(function () {
  'use strict';

  /* Every htmx request carries the CSRF token from <meta name="csrf-token">. */
  function csrfHeader() {
    document.body.addEventListener('htmx:configRequest', function (e) {
      var meta = document.querySelector('meta[name="csrf-token"]');
      if (meta) e.detail.headers['X-CSRF-Token'] = meta.getAttribute('content');
    });
  }

  /* [data-open="id"] opens .dialog-backdrop#id; [data-close] or Escape closes it.
     Without JavaScript the same links still work: they load the page with the dialog open
     (?add=1) or closed. */
  function dialogs() {
    var lastOpener = null;
    function close(backdrop) {
      if (!backdrop) return;
      backdrop.classList.remove('open');
      if (lastOpener && lastOpener.focus) lastOpener.focus();
      lastOpener = null;
    }
    document.addEventListener('click', function (e) {
      var opener = e.target.closest ? e.target.closest('[data-open]') : null;
      if (opener) {
        var target = document.getElementById(opener.getAttribute('data-open'));
        if (!target) return;
        e.preventDefault();
        lastOpener = opener;
        target.classList.add('open');
        var first = target.querySelector('input:not([type=hidden]), select, textarea');
        if (first) first.focus();
        return;
      }
      var closer = e.target.closest ? e.target.closest('[data-close]') : null;
      if (closer) {
        e.preventDefault();
        close(closer.closest('.dialog-backdrop'));
      }
    });
    document.addEventListener('keydown', function (e) {
      if (e.key !== 'Escape' && e.keyCode !== 27) return;
      close(document.querySelector('.dialog-backdrop.open'));
    });
    /* A dialog the server re-opened because of a form error: focus the first bad field. */
    var invalid = document.querySelector('.dialog-backdrop.open [aria-invalid="true"]');
    if (invalid) invalid.focus();
  }


  /* Admission form: "Not measured" switches a screening row between value and reason. */
  function syncScreening(row) {
    var box = row.querySelector('.nm input');
    if (!box) return;
    var values = row.querySelectorAll('.pair input, .value input, .value select');
    for (var i = 0; i < values.length; i++) values[i].disabled = box.checked;
    var reason = row.querySelector('.reason');
    if (reason) reason.hidden = !box.checked;
  }

  /* The save button waits for admitted-at and screening (the server checks everything again). */
  function gate() {
    var form = document.querySelector('form[data-gate]');
    if (!form) return;
    var button = form.querySelector('[data-submit]');
    var note = form.querySelector('[data-gate-note]');
    var rows = form.querySelectorAll('.scr');
    for (var r = 0; r < rows.length; r++) syncScreening(rows[r]);
    var left = [];
    var required = form.querySelectorAll('[data-required]');
    for (var q = 0; q < required.length; q++) {
      if (!required[q].value.replace(/\s+/g, '')) left.push(required[q].getAttribute('data-required'));
    }
    for (var s = 0; s < rows.length; s++) {
      var row = rows[s];
      var box = row.querySelector('.nm input');
      var done;
      if (box && box.checked) {
        var sel = row.querySelector('.reason select');
        done = !!(sel && sel.value);
      } else {
        var fields = row.querySelectorAll('.pair input, .value input, .value select');
        done = fields.length > 0;
        for (var f = 0; f < fields.length; f++) if (!fields[f].value) done = false;
      }
      if (!done) left.push(row.getAttribute('data-scr'));
    }
    if (left.length) {
      button.disabled = true;
      note.textContent = left.length + (left.length === 1 ? ' thing left: ' : ' things left: ') + left.join(', ');
    } else {
      button.disabled = false;
      note.textContent = note.getAttribute('data-ready');
    }
  }

  /* Mark the fields the Robson panel names as missing or in conflict. */
  function markFields() {
    var panel = document.getElementById('robson-panel');
    if (!panel) return;
    var old = document.querySelectorAll('.f.missing, .f.conflict.live-mark');
    for (var i = 0; i < old.length; i++) {
      old[i].classList.remove('missing');
      if (old[i].classList.contains('live-mark')) {
        old[i].classList.remove('conflict', 'live-mark');
        var m = old[i].querySelector('.msg.live-msg');
        if (m) m.parentNode.removeChild(m);
      }
    }
    var missing = (panel.getAttribute('data-missing') || '').split(' ');
    for (var a = 0; a < missing.length; a++) {
      var el = missing[a] && document.getElementById('f-' + missing[a]);
      if (el) el.classList.add('missing');
    }
    var conflict = (panel.getAttribute('data-conflict') || '').split(' ').filter(Boolean);
    var names = {parity: 'parity', previous_cs_count: 'previous cesareans', plurality: 'number of fetuses',
      fetal_presentation: 'fetal presentation', gestational_age_weeks: 'gestational age', onset_of_labour: 'onset of labour'};
    for (var c = 0; c < conflict.length; c++) {
      var box = document.getElementById('f-' + conflict[c]);
      if (!box || box.classList.contains('conflict')) continue;
      box.classList.add('conflict', 'live-mark');
      var others = conflict.filter(function (x) { return x !== conflict[c]; }).map(function (x) { return names[x] || x; });
      var msg = document.createElement('div');
      msg.className = 'msg live-msg';
      msg.textContent = '⚠ Conflicts with ' + others.join(', ');
      box.appendChild(msg);
    }
  }

  /* Connection lost: say so, keep the entries, queue nothing. */
  function connection() {
    function banner(show) {
      var b = document.getElementById('conn-banner');
      if (b) b.hidden = !show;
    }
    function offline(isSave) {
      banner(true);
      var live = document.querySelector('[data-live]');
      if (live) { live.classList.add('off'); live.lastChild.textContent = 'Offline'; }
      if (isSave) {
        var button = document.querySelector('[data-submit]');
        var note = document.querySelector('[data-gate-note]');
        if (button) button.textContent = 'Try saving again';
        if (note) note.textContent = 'Not saved yet';
      }
    }
    function isSave(e) {
      var cfg = e.detail && e.detail.requestConfig;
      return !!(cfg && cfg.path && cfg.path.indexOf('/preview') === -1 && cfg.verb === 'post');
    }
    document.body.addEventListener('htmx:sendError', function (e) { offline(isSave(e)); });
    document.body.addEventListener('htmx:responseError', function (e) { offline(isSave(e)); });
    document.body.addEventListener('htmx:timeout', function (e) { offline(isSave(e)); });
    document.body.addEventListener('htmx:afterRequest', function (e) {
      if (!e.detail.successful) return;
      banner(false);
      var live = document.querySelector('[data-live]');
      if (live && live.classList.contains('off')) { live.classList.remove('off'); live.lastChild.textContent = 'Live'; }
    });
  }

  function admissionForm() {
    document.addEventListener('change', function (e) {
      var row = e.target.closest ? e.target.closest('.scr') : null;
      if (row) syncScreening(row);
      gate();
    });
    document.addEventListener('input', function () { gate(); });
    document.body.addEventListener('htmx:afterSettle', function () { gate(); markFields(); });
    document.addEventListener('click', function (e) {
      var p = e.target.closest ? e.target.closest('[data-print]') : null;
      if (p) { e.preventDefault(); window.print(); }
    });
    gate();
    markFields();
    connection();
  }

  /* <select data-searchable> becomes a box you type in to filter the list (WAI-ARIA combobox).
     The select stays in the form, hidden, and carries the value; without JavaScript it is a
     plain select. data-submit-on-pick submits the form as soon as an option is picked. */
  var comboCount = 0;
  function searchable(select) {
    var n = ++comboCount;
    var options = Array.prototype.map.call(select.options, function (o) {
      return { value: o.value, text: o.text };
    }).filter(function (o) { return o.value !== ''; });
    var wrap = document.createElement('div');
    wrap.className = 'combo';
    var input = document.createElement('input');
    input.type = 'text';
    input.id = select.id ? select.id + '-search' : 'combo-' + n;
    input.setAttribute('role', 'combobox');
    input.setAttribute('aria-autocomplete', 'list');
    input.setAttribute('aria-expanded', 'false');
    input.setAttribute('aria-controls', 'combo-list-' + n);
    input.setAttribute('autocomplete', 'off');
    input.placeholder = 'Type to search';
    var list = document.createElement('ul');
    list.id = 'combo-list-' + n;
    list.className = 'combo-list';
    list.setAttribute('role', 'listbox');
    list.hidden = true;
    wrap.appendChild(input);
    wrap.appendChild(list);
    select.parentNode.insertBefore(wrap, select);
    select.hidden = true;
    var label = select.id ? document.querySelector('label[for="' + select.id + '"]') : null;
    if (label) label.setAttribute('for', input.id);
    var shown = [];
    var active = -1;

    function selectedText() {
      var o = select.options[select.selectedIndex];
      return o && o.value !== '' ? o.text : '';
    }
    function setActive(i) {
      active = i;
      Array.prototype.forEach.call(list.children, function (li, k) {
        li.setAttribute('aria-selected', k === i ? 'true' : 'false');
        if (k === i && li.scrollIntoView) li.scrollIntoView({ block: 'nearest' });
      });
      if (i >= 0 && list.children[i]) input.setAttribute('aria-activedescendant', list.children[i].id);
      else input.removeAttribute('aria-activedescendant');
    }
    function render(filter) {
      var words = filter.toLowerCase().split(/\s+/).filter(Boolean);
      shown = options.filter(function (o) {
        var t = o.text.toLowerCase();
        return words.every(function (w) { return t.indexOf(w) !== -1; });
      });
      while (list.firstChild) list.removeChild(list.firstChild);
      shown.forEach(function (o, k) {
        var li = document.createElement('li');
        li.id = list.id + '-' + k;
        li.setAttribute('role', 'option');
        li.textContent = o.text;
        list.appendChild(li);
      });
      if (!shown.length) {
        var none = document.createElement('li');
        none.className = 'combo-none';
        none.textContent = 'No match';
        list.appendChild(none);
      }
      list.hidden = false;
      input.setAttribute('aria-expanded', 'true');
      setActive(shown.length ? 0 : -1);
    }
    function close() {
      list.hidden = true;
      input.setAttribute('aria-expanded', 'false');
      input.removeAttribute('aria-activedescendant');
    }
    function pick(k) {
      var o = shown[k];
      if (!o) return;
      select.value = o.value;
      input.value = o.text;
      close();
      select.dispatchEvent(new Event('change', { bubbles: true }));
      if (select.hasAttribute('data-submit-on-pick') && select.form) {
        if (select.form.requestSubmit) select.form.requestSubmit(); else select.form.submit();
      }
    }
    input.value = selectedText();
    input.addEventListener('focus', function () { input.select(); render(''); });
    input.addEventListener('input', function () { render(input.value); });
    input.addEventListener('keydown', function (e) {
      if (e.key === 'ArrowDown') { e.preventDefault(); if (list.hidden) render(''); else setActive(Math.min(active + 1, shown.length - 1)); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(Math.max(active - 1, 0)); }
      else if (e.key === 'Enter') { if (!list.hidden && active >= 0) { e.preventDefault(); pick(active); } }
      else if (e.key === 'Escape') { input.value = selectedText(); close(); }
    });
    list.addEventListener('mousedown', function (e) {
      var li = e.target.closest ? e.target.closest('li[role=option]') : null;
      if (!li) return;
      e.preventDefault();
      pick(Array.prototype.indexOf.call(list.children, li));
    });
    input.addEventListener('blur', function () { input.value = selectedText(); close(); });
  }

  document.addEventListener('DOMContentLoaded', function () {
    csrfHeader();
    dialogs();
    admissionForm();
    Array.prototype.forEach.call(document.querySelectorAll('select[data-searchable]'), searchable);
  });
})();
