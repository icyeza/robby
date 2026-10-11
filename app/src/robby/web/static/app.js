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

  document.addEventListener('DOMContentLoaded', function () {
    csrfHeader();
    dialogs();
  });
})();
