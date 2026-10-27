/* theme.js — loaded synchronously in <head> so the correct theme is applied
 * BEFORE the page paints (no flash of the wrong theme).
 * Order: saved user choice → operating-system preference → light. */
(function () {
  "use strict";
  var KEY = "bookmind-theme";
  var saved = null;
  try { saved = window.localStorage.getItem(KEY); } catch (e) { /* storage blocked */ }
  var systemDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  var theme = saved === "light" || saved === "dark" ? saved : (systemDark ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", theme);
})();
