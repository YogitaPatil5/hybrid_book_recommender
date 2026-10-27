/* main.js — global UI behaviour (no inline scripts: keeps CSP strict). */
(() => {
  "use strict";

  // ── Mobile nav toggle ───────────────────────────────────────────────────
  const navToggle = document.getElementById("navToggle");
  const navLinks = document.querySelector(".nav-links");
  if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
      const open = navLinks.classList.toggle("open");
      navToggle.setAttribute("aria-expanded", String(open));
    });
  }

  // ── Light / dark theme toggle (initial theme is set by theme.js in <head>) ──
  const THEME_KEY = "bookmind-theme";
  const root = document.documentElement;
  const themeToggle = document.getElementById("themeToggle");
  const syncToggle = () => {
    if (!themeToggle) return;
    const dark = root.getAttribute("data-theme") === "dark";
    themeToggle.setAttribute("aria-label", dark ? "Switch to light mode" : "Switch to dark mode");
    themeToggle.setAttribute("aria-pressed", String(dark));
  };
  const applyTheme = (theme, persist) => {
    root.setAttribute("data-theme", theme);
    if (persist) {
      try { window.localStorage.setItem(THEME_KEY, theme); } catch (e) { /* storage blocked */ }
    }
    syncToggle();
  };
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      applyTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark", true);
    });
  }
  // Follow OS changes until the user has made an explicit choice.
  if (window.matchMedia) {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = (e) => {
      let saved = null;
      try { saved = window.localStorage.getItem(THEME_KEY); } catch (err) { /* ignore */ }
      if (!saved) applyTheme(e.matches ? "dark" : "light", false);
    };
    if (mq.addEventListener) mq.addEventListener("change", onChange);
  }
  syncToggle();

  // ── "Show more" on the home page ────────────────────────────────────────
  const showMoreBtn = document.getElementById("showMoreBtn");
  if (showMoreBtn) {
    showMoreBtn.addEventListener("click", () => {
      const hidden = document.querySelectorAll(".book-card:nth-child(n+21):not(.visible)");
      Array.from(hidden).slice(0, 20).forEach((card) => card.classList.add("visible"));
      if (!document.querySelector(".book-card:nth-child(n+21):not(.visible)")) {
        showMoreBtn.style.display = "none";
      }
    });
  }

  // ── Broken cover fallback (local SVG; the old placeholder service is dead) ──
  const placeholder = document.body.dataset.placeholder;
  document.querySelectorAll("img").forEach((img) => {
    img.addEventListener("error", function onError() {
      img.removeEventListener("error", onError); // prevent infinite loops
      if (placeholder) img.src = placeholder;
    });
  });
})();
