/* Robson Readiness mockups: small behaviours only. ES5, no dependencies. */
(function () {
  var FIGURE = '<svg viewBox="0 0 12 22" aria-hidden="true"><circle cx="6" cy="4" r="3.4"/>' +
    '<path d="M6 8.6c-2.6 0-4.2 1.9-4.6 4.6L.9 18.6c-.1.9.5 1.5 1.3 1.5h7.6c.8 0 1.4-.6 1.3-1.5l-.5-5.4C10.2 10.5 8.6 8.6 6 8.6z"/></svg>';

  /* <div class="icon-array" data-filled="71"></div> -> 100 figures, first N dark */
  function iconArrays() {
    var nodes = document.querySelectorAll('.icon-array[data-filled]');
    for (var n = 0; n < nodes.length; n++) {
      var filled = parseInt(nodes[n].getAttribute('data-filled'), 10), html = '';
      for (var i = 0; i < 100; i++) {
        html += FIGURE.replace('<svg ', '<svg fill="' + (i < filled ? 'var(--fig)' : 'var(--fig-off)') + '" ');
      }
      nodes[n].innerHTML = html;
      nodes[n].setAttribute('role', 'img');
      nodes[n].setAttribute('aria-label', filled + ' of 100 figures filled');
    }
  }

  /* .bar[data-tip] -> hover tooltip */
  function tooltips() {
    var tip = document.createElement('div');
    tip.id = 'tip';
    document.body.appendChild(tip);
    document.addEventListener('mousemove', function (e) {
      var el = e.target.closest ? e.target.closest('[data-tip]') : null;
      if (!el) { tip.style.display = 'none'; return; }
      tip.innerHTML = el.getAttribute('data-tip');
      tip.style.display = 'block';
      tip.style.left = (e.clientX + 12) + 'px';
      tip.style.top = (e.clientY + 12) + 'px';
    });
  }

  /* .scr: ticking "Not measured" disables the value and shows the reason select */
  function notMeasured() {
    var rows = document.querySelectorAll('.scr');
    for (var r = 0; r < rows.length; r++) {
      (function (row) {
        var box = row.querySelector('.nm input');
        if (!box) return;
        function sync() {
          var values = row.querySelectorAll('.pair input, .value input, .value select');
          for (var i = 0; i < values.length; i++) values[i].disabled = box.checked;
          var reason = row.querySelector('.reason');
          if (reason) reason.style.display = box.checked ? '' : 'none';
        }
        box.addEventListener('change', sync);
        sync();
      })(rows[r]);
    }
  }

  /* segmented buttons: clicking one selects it within its group */
  function segments() {
    document.addEventListener('click', function (e) {
      var b = e.target.closest ? e.target.closest('.seg button') : null;
      if (!b) return;
      e.preventDefault();
      var all = b.parentNode.querySelectorAll('button');
      for (var i = 0; i < all.length; i++) all[i].classList.remove('on');
      b.classList.add('on');
    });
  }

  /* [data-open="id"] opens .dialog-backdrop#id; [data-close] closes its backdrop */
  function dialogs() {
    document.addEventListener('click', function (e) {
      var opener = e.target.closest ? e.target.closest('[data-open]') : null;
      if (opener) {
        e.preventDefault();
        document.getElementById(opener.getAttribute('data-open')).classList.add('open');
        return;
      }
      var closer = e.target.closest ? e.target.closest('[data-close]') : null;
      if (closer) {
        e.preventDefault();
        closer.closest('.dialog-backdrop').classList.remove('open');
      }
    });
  }

  document.addEventListener('DOMContentLoaded', function () {
    iconArrays(); tooltips(); notMeasured(); segments(); dialogs();
  });
})();
