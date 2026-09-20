/* ==========================================================================
   Prediction history - browser localStorage store

   History lives on the visitor's own device rather than on the server. That
   choice has three consequences worth understanding:

     * Each visitor sees only their own history. On a shared demo link nobody
       sees anyone else's submissions, which is better for privacy.
     * Nothing is lost when the server restarts or redeploys. Free hosting
       tiers have an ephemeral filesystem, so a server-side SQLite file would
       be wiped on every deploy.
     * History does NOT follow the user to another browser or device, and
       clearing site data erases it.

   Every access is wrapped in try/catch: localStorage throws in private mode
   and when site data is blocked, and the app must keep working without it.
   ========================================================================== */
window.HistoryStore = (function () {
  'use strict';

  var KEY = 'fnd.history.v1';
  var MAX_ENTRIES = 500;   // keeps us well inside the ~5 MB localStorage quota
  var MAX_TEXT = 1000;     // mirrors the old server-side truncation

  function available() {
    try {
      var probe = '__fnd_probe__';
      window.localStorage.setItem(probe, '1');
      window.localStorage.removeItem(probe);
      return true;
    } catch (e) {
      return false;
    }
  }

  function readAll() {
    try {
      var raw = window.localStorage.getItem(KEY);
      if (!raw) { return []; }
      var parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch (e) {
      // Corrupt or unreadable payload - start clean rather than crash.
      return [];
    }
  }

  function writeAll(entries) {
    try {
      window.localStorage.setItem(KEY, JSON.stringify(entries));
      return true;
    } catch (e) {
      // Most likely QuotaExceededError. Drop the oldest half and retry once.
      try {
        window.localStorage.setItem(
          KEY, JSON.stringify(entries.slice(0, Math.floor(MAX_ENTRIES / 2)))
        );
        return true;
      } catch (e2) {
        return false;
      }
    }
  }

  function nextId(entries) {
    return entries.reduce(function (max, row) {
      return row.id > max ? row.id : max;
    }, 0) + 1;
  }

  return {
    isAvailable: available,

    /** Newest first. */
    all: function () {
      return readAll().sort(function (a, b) { return b.id - a.id; });
    },

    add: function (entry) {
      var entries = readAll();
      var text = String(entry.input_text || '');
      entries.push({
        id: nextId(entries),
        input_text: text.length > MAX_TEXT ? text.slice(0, MAX_TEXT) + '...' : text,
        prediction: entry.prediction,
        confidence: Number(entry.confidence) || 0,
        model_name: entry.model_name || '',
        word_count: Number(entry.word_count) || 0,
        low_signal: !!entry.low_signal,
        out_of_domain: !!entry.out_of_domain,
        created_at: new Date().toISOString()
      });
      // Cap growth by discarding the oldest entries.
      if (entries.length > MAX_ENTRIES) {
        entries = entries.slice(entries.length - MAX_ENTRIES);
      }
      return writeAll(entries);
    },

    remove: function (id) {
      var kept = readAll().filter(function (row) { return row.id !== id; });
      return writeAll(kept);
    },

    clear: function () {
      var count = readAll().length;
      try {
        window.localStorage.removeItem(KEY);
      } catch (e) { /* nothing we can do */ }
      return count;
    },

    /**
     * Filter by free-text search and/or FAKE/REAL label.
     * Mirrors the query the server used to run, so the UI behaves identically.
     */
    query: function (search, label) {
      var needle = (search || '').toLowerCase();
      return this.all().filter(function (row) {
        if (label && row.prediction !== label) { return false; }
        if (needle && row.input_text.toLowerCase().indexOf(needle) === -1) {
          return false;
        }
        return true;
      });
    },

    /** Aggregates for the dashboard tiles and charts. */
    stats: function () {
      var rows = this.all();
      var fake = rows.filter(function (r) { return r.prediction === 'FAKE'; }).length;
      var real = rows.length - fake;
      var confidenceSum = rows.reduce(function (sum, r) { return sum + r.confidence; }, 0);
      return {
        total: rows.length,
        fake_count: fake,
        real_count: real,
        avg_confidence: rows.length ? confidenceSum / rows.length : 0,
        fake_percentage: rows.length ? Math.round(fake / rows.length * 1000) / 10 : 0,
        real_percentage: rows.length ? Math.round(real / rows.length * 1000) / 10 : 0,
        recent: rows.slice(0, 5)
      };
    }
  };
})();
