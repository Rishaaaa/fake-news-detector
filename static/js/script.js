/* ==========================================================================
   Fake News Detection System - front-end behaviour

   Contents:
     1. Theme toggle (light / dark, persisted)
     2. Character counter for the prediction textarea
     3. Clear / sample buttons
     4. Confirmation for destructive actions
     5. Animated meter + progress bars
     6. Dashboard charts (Chart.js)

   No framework: everything here is plain ES6 so the code stays readable for
   a student defending the project.
   ========================================================================== */
(function () {
  'use strict';

  /* ---------- 1. Theme toggle ---------------------------------------- */
  var root = document.documentElement;

  function currentTheme() {
    return root.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
  }

  function setTheme(theme) {
    root.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('theme', theme);
    } catch (e) {
      /* Storage can be unavailable in private mode - the theme still applies
         for this page view, it just will not be remembered. */
    }
    // Charts read their colours from CSS variables, so they must be rebuilt.
    if (window.initDashboardCharts) { window.initDashboardCharts(); }
  }

  var themeToggle = document.getElementById('themeToggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', function () {
      setTheme(currentTheme() === 'dark' ? 'light' : 'dark');
    });
  }

  /* ---------- 2. Character counter ------------------------------------ */
  var textarea = document.getElementById('news_text');
  var counter = document.getElementById('charCounter');
  var analyzeBtn = document.getElementById('analyzeBtn');

  function readLimits() {
    var max = textarea ? parseInt(textarea.getAttribute('maxlength'), 10) : 0;
    return { max: max || 20000 };
  }

  function updateCounter() {
    if (!textarea || !counter) { return; }
    var limits = readLimits();
    var length = textarea.value.length;

    counter.textContent = length.toLocaleString() + ' / ' + limits.max.toLocaleString();
    counter.classList.toggle('is-warning', length > limits.max * 0.9 && length <= limits.max);
    counter.classList.toggle('is-over', length > limits.max);
  }

  if (textarea) {
    textarea.addEventListener('input', updateCounter);
    updateCounter();

    // Ctrl/Cmd + Enter submits, which is the expected shortcut in a textarea.
    textarea.addEventListener('keydown', function (event) {
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') {
        var form = document.getElementById('predictForm');
        if (form && analyzeBtn && !analyzeBtn.disabled) { form.requestSubmit(); }
      }
    });
  }

  /* ---------- 3. Clear and sample buttons ----------------------------- */
  var clearBtn = document.getElementById('clearBtn');
  if (clearBtn && textarea) {
    clearBtn.addEventListener('click', function () {
      textarea.value = '';
      updateCounter();
      textarea.focus();
    });
  }

  // A neutral sample so a reviewer can try the app without hunting for text.
  var SAMPLE_TEXT =
    'Washington (Reuters) - The Senate voted on Tuesday to approve a bipartisan ' +
    'spending agreement, with lawmakers backing the measure 71 to 28 after weeks ' +
    'of negotiation. The bill extends federal funding through the end of the ' +
    'fiscal year and includes additional money for disaster relief, according to ' +
    'two congressional aides familiar with the talks. It now moves to the House, ' +
    'where leaders have said they expect a vote before the end of the week.';

  var sampleBtn = document.getElementById('sampleBtn');
  if (sampleBtn && textarea) {
    sampleBtn.addEventListener('click', function () {
      textarea.value = SAMPLE_TEXT;
      updateCounter();
      textarea.focus();
    });
  }

  /* ---------- 4. Confirm destructive actions -------------------------- */
  // Any form carrying data-confirm asks before submitting.
  document.querySelectorAll('form[data-confirm]').forEach(function (form) {
    form.addEventListener('submit', function (event) {
      if (!window.confirm(form.getAttribute('data-confirm'))) {
        event.preventDefault();
      }
    });
  });

  // Disable the submit button while the request is in flight, so a slow
  // prediction cannot be submitted twice.
  var predictForm = document.getElementById('predictForm');
  if (predictForm && analyzeBtn) {
    predictForm.addEventListener('submit', function () {
      if (textarea && textarea.value.trim().length === 0) { return; }
      analyzeBtn.disabled = true;
      analyzeBtn.textContent = 'Analyzing…';
    });
  }

  /* ---------- 5. Animate meters on load ------------------------------- */
  // Bars are rendered at width 0 and given their real width here, which makes
  // the CSS transition run and produces the fill animation.
  window.requestAnimationFrame(function () {
    document.querySelectorAll('[data-width]').forEach(function (element) {
      element.style.width = element.getAttribute('data-width') + '%';
    });
  });

  /* ---------- 6. Per-device usage statistics --------------------------- */
  // The stat tiles on the home page and the dashboard are filled from the
  // visitor's own browser storage, because history is never sent to the server.
  function renderUsageStats() {
    if (!window.HistoryStore) { return null; }
    var stats = window.HistoryStore.stats();

    function put(id, value) {
      var node = document.getElementById(id);
      if (node) { node.textContent = value; }
    }

    put('statTotal', stats.total.toLocaleString());
    put('statFake', stats.fake_count.toLocaleString());
    put('statReal', stats.real_count.toLocaleString());
    put('statConf', stats.total ? (stats.avg_confidence * 100).toFixed(1) + '%' : '\u2014');
    put('statFakePct', stats.fake_percentage);
    put('statRealPct', stats.real_percentage);

    // "Recent analyses" list on the home page.
    var list = document.getElementById('recentList');
    var card = document.getElementById('recentCard');
    if (list && card) {
      list.textContent = '';
      if (stats.recent.length) {
        card.hidden = false;
        stats.recent.forEach(function (row) {
          var line = document.createElement('div');
          line.style.cssText =
            'display:flex;gap:0.6rem;align-items:center;font-size:0.83rem;';

          var badge = document.createElement('span');
          badge.className = 'badge badge-' + row.prediction.toLowerCase();
          badge.textContent = row.prediction;

          // textContent, not innerHTML - stored text is never treated as markup.
          var text = document.createElement('span');
          text.className = 'text-muted';
          text.style.cssText =
            'overflow:hidden;text-overflow:ellipsis;white-space:nowrap;';
          text.textContent = row.input_text.slice(0, 48);

          line.appendChild(badge);
          line.appendChild(text);
          list.appendChild(line);
        });
      } else {
        card.hidden = true;
      }
    }
    return stats;
  }

  renderUsageStats();

  /* ---------- 7. Dashboard charts ------------------------------------- */
  var chartInstances = [];

  function cssVar(name) {
    return getComputedStyle(root).getPropertyValue(name).trim();
  }

  function destroyCharts() {
    chartInstances.forEach(function (chart) { chart.destroy(); });
    chartInstances = [];
  }

  window.initDashboardCharts = function () {
    var dataElement = document.getElementById('dashboardData');
    if (!dataElement || typeof window.Chart === 'undefined') { return; }


    var data;
    try {
      data = JSON.parse(dataElement.textContent);
    } catch (e) {
      return;  // Malformed payload - leave the table views in place.
    }

    destroyCharts();

    // Colours come from the stylesheet so the charts follow the active theme.
    var ink = cssVar('--ink-secondary');
    var muted = cssVar('--ink-muted');
    var grid = cssVar('--gridline');
    var surface = cssVar('--surface');
    var fake = cssVar('--fake');
    var real = cssVar('--real');

    // Categorical slots 1-4 from the validated palette, per theme.
    var isDark = currentTheme() === 'dark';
    var series = isDark
      ? ['#3987e5', '#d95926', '#199e70', '#c98500']
      : ['#2a78d6', '#eb6834', '#1baf7a', '#eda100'];

    Chart.defaults.font.family =
      'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
    Chart.defaults.color = muted;

    var tooltipStyle = {
      backgroundColor: isDark ? '#222221' : '#0b0b0b',
      titleColor: '#ffffff',
      bodyColor: '#ffffff',
      padding: 10,
      cornerRadius: 8,
      displayColors: true,
      boxPadding: 4
    };

    /* --- Fake vs real doughnut (from this device's history) --- */
    var usage = renderUsageStats() || { fake_count: 0, real_count: 0, total: 0 };
    var splitWrap = document.getElementById('splitChartWrap');
    var splitEmpty = document.getElementById('splitEmpty');
    if (splitWrap && splitEmpty) {
      splitWrap.hidden = usage.total === 0;
      splitEmpty.hidden = usage.total > 0;
    }
    var splitCaption = document.getElementById('splitCaption');
    if (splitCaption) {
      splitCaption.textContent =
        usage.fake_count.toLocaleString() + ' fake \u00B7 ' +
        usage.real_count.toLocaleString() + ' real';
    }

    var splitCanvas = document.getElementById('chartSplit');
    if (splitCanvas && usage.total > 0) {
      chartInstances.push(new Chart(splitCanvas, {
        type: 'doughnut',
        data: {
          labels: ['FAKE', 'REAL'],
          datasets: [{
            data: [usage.fake_count, usage.real_count],
            // Status colours: these are states, not arbitrary series.
            backgroundColor: [fake, real],
            borderColor: surface,
            borderWidth: 2,          // 2px surface gap between segments
            hoverOffset: 6
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: '62%',
          plugins: {
            legend: {
              position: 'bottom',
              labels: { usePointStyle: true, pointStyle: 'circle', padding: 16, color: ink }
            },
            tooltip: Object.assign({}, tooltipStyle, {
              callbacks: {
                label: function (context) {
                  var total = context.dataset.data.reduce(function (a, b) { return a + b; }, 0);
                  var pct = total ? (context.parsed / total * 100).toFixed(1) : '0.0';
                  return ' ' + context.label + ': ' + context.parsed.toLocaleString() +
                         ' (' + pct + '%)';
                }
              }
            })
          }
        }
      }));
    }

    /* --- Model comparison grouped bars --- */
    var modelsCanvas = document.getElementById('chartModels');
    if (modelsCanvas && data.models.length) {
      var metricKeys = ['accuracy', 'precision', 'recall', 'f1'];
      var metricLabels = ['Accuracy', 'Precision', 'Recall', 'F1-score'];

      chartInstances.push(new Chart(modelsCanvas, {
        type: 'bar',
        data: {
          labels: data.models.map(function (row) { return row.name; }),
          datasets: metricKeys.map(function (key, index) {
            return {
              label: metricLabels[index],
              data: data.models.map(function (row) { return row[key]; }),
              backgroundColor: series[index],
              borderColor: surface,
              borderWidth: 1,
              borderRadius: 4,       // 4px rounded data-end
              borderSkipped: 'bottom',
              maxBarThickness: 26
            };
          })
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          scales: {
            y: {
              beginAtZero: true, max: 1,
              grid: { color: grid, drawTicks: false },
              border: { display: false },
              ticks: { stepSize: 0.2, callback: function (v) { return (v * 100) + '%'; } }
            },
            x: { grid: { display: false }, ticks: { color: ink } }
          },
          plugins: {
            legend: {
              position: 'bottom',
              labels: { usePointStyle: true, pointStyle: 'rectRounded', padding: 14, color: ink }
            },
            tooltip: Object.assign({}, tooltipStyle, {
              callbacks: {
                label: function (context) {
                  return ' ' + context.dataset.label + ': ' + context.parsed.y.toFixed(4);
                }
              }
            })
          }
        }
      }));
    }

    /* --- Per-class performance --- */
    var perClassCanvas = document.getElementById('chartPerClass');
    var classNames = Object.keys(data.perClass || {});
    if (perClassCanvas && classNames.length) {
      chartInstances.push(new Chart(perClassCanvas, {
        type: 'bar',
        data: {
          labels: ['Precision', 'Recall', 'F1-score'],
          datasets: classNames.map(function (name) {
            return {
              label: name,
              data: [
                data.perClass[name].precision,
                data.perClass[name].recall,
                data.perClass[name].f1
              ],
              backgroundColor: name === 'FAKE' ? fake : real,
              borderColor: surface,
              borderWidth: 1,
              borderRadius: 4,
              borderSkipped: 'start',
              maxBarThickness: 22
            };
          })
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          interaction: { mode: 'index', intersect: false },
          scales: {
            x: {
              beginAtZero: true, max: 1,
              grid: { color: grid, drawTicks: false },
              border: { display: false },
              ticks: { stepSize: 0.2, callback: function (v) { return (v * 100) + '%'; } }
            },
            y: { grid: { display: false }, ticks: { color: ink } }
          },
          plugins: {
            legend: {
              position: 'bottom',
              labels: { usePointStyle: true, pointStyle: 'rectRounded', padding: 14, color: ink }
            },
            tooltip: Object.assign({}, tooltipStyle, {
              callbacks: {
                label: function (context) {
                  return ' ' + context.dataset.label + ': ' + context.parsed.x.toFixed(4);
                }
              }
            })
          }
        }
      }));
    }
  };

  // Run once on load in case the inline call fired before this file parsed.
  if (document.getElementById('dashboardData')) {
    window.initDashboardCharts();
  }
})();
