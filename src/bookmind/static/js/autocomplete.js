/* autocomplete.js — shared by /search and /recommend.
 *
 * - Focus / click on an EMPTY box  → "Popular books" dropdown (discovery), genre-aware.
 * - Typing (1+ characters)         → matching titles from both catalogues.
 * - Debounced, cancellable fetches; keyboard navigation (↑ ↓ Enter Esc);
 *   results inserted with textContent only (no HTML injection).
 */
(() => {
  "use strict";

  const input = document.getElementById("bookInput");
  const list = document.getElementById("autocompleteList");
  const clearBtn = document.getElementById("clearBtn");
  const form = document.getElementById("searchForm");
  const genreSelect = document.getElementById("genreSelect"); // only on /search
  if (!input || !list) return;

  const MIN_CHARS = 1;
  const DEBOUNCE_MS = 150;
  const LIMIT = 10;
  let activeIdx = -1;
  let timer = null;
  let controller = null;
  let lastKey = null; // what the current dropdown was rendered for

  const options = () => list.querySelectorAll("li.ac-option");

  const close = () => {
    list.innerHTML = "";
    list.classList.remove("open");
    activeIdx = -1;
    lastKey = null;
    input.setAttribute("aria-expanded", "false");
  };

  const toggleClear = () => {
    if (clearBtn) clearBtn.style.display = input.value ? "flex" : "none";
  };

  const select = (title) => {
    input.value = title;
    close();
    toggleClear();
    input.focus();
  };

  const render = (items, label, key) => {
    list.innerHTML = "";
    activeIdx = -1;
    if (!Array.isArray(items) || !items.length) { close(); return; }
    if (label) {
      const header = document.createElement("li");
      header.className = "ac-header";
      header.setAttribute("role", "presentation");
      header.textContent = label;
      list.appendChild(header);
    }
    items.forEach((title, i) => {
      const li = document.createElement("li");
      li.className = "ac-option";
      li.textContent = title;
      li.setAttribute("role", "option");
      li.id = `ac-option-${i}`;
      li.addEventListener("mousedown", (e) => { e.preventDefault(); select(title); });
      list.appendChild(li);
    });
    lastKey = key;
    list.classList.add("open");
    input.setAttribute("aria-expanded", "true");
  };

  const currentGenre = () =>
    genreSelect && genreSelect.value && genreSelect.value !== "All" ? genreSelect.value : "";

  const fetchSuggestions = async (q) => {
    const genre = q ? "" : currentGenre();
    const key = `${q}|${genre}`;
    if (key === lastKey && list.classList.contains("open")) return; // already showing
    if (controller) controller.abort();
    controller = new AbortController();
    const params = new URLSearchParams({ q, limit: String(LIMIT) });
    if (genre) params.set("genre", genre);
    try {
      const res = await fetch(`/api/v1/autocomplete?${params}`, {
        signal: controller.signal,
        headers: { Accept: "application/json" },
      });
      if (!res.ok) { close(); return; }
      const data = await res.json();
      if (input.value.trim() !== q) return; // user kept typing; a newer request wins
      render(data.suggestions || [], data.mode === "popular" ? data.label : null, key);
    } catch (err) {
      if (err.name !== "AbortError") close();
    }
  };

  const refresh = () => {
    const q = input.value.trim();
    clearTimeout(timer);
    if (q.length > 0 && q.length < MIN_CHARS) { close(); return; }
    timer = setTimeout(() => fetchSuggestions(q), q ? DEBOUNCE_MS : 0);
  };

  // Show popular titles as soon as the user focuses / clicks an empty box.
  input.addEventListener("focus", refresh);
  input.addEventListener("click", () => { if (!list.classList.contains("open")) refresh(); });
  input.addEventListener("input", () => { toggleClear(); refresh(); });

  // Changing the genre updates the "Popular in …" list while the box is empty.
  if (genreSelect) {
    genreSelect.addEventListener("change", () => {
      if (!input.value.trim() && document.activeElement === input) refresh();
    });
  }

  input.addEventListener("keydown", (e) => {
    const items = options();
    if (e.key === "ArrowDown" && !items.length) { refresh(); return; }
    if (!items.length) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      activeIdx = e.key === "ArrowDown"
        ? Math.min(activeIdx + 1, items.length - 1)
        : Math.max(activeIdx - 1, 0);
      items.forEach((li, i) => li.classList.toggle("active", i === activeIdx));
      items[activeIdx].scrollIntoView({ block: "nearest" });
      input.setAttribute("aria-activedescendant", items[activeIdx].id);
    } else if (e.key === "Enter" && activeIdx >= 0) {
      e.preventDefault();
      select(items[activeIdx].textContent);
    } else if (e.key === "Escape") {
      close();
    }
  });

  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      input.value = "";
      toggleClear();
      input.focus();
      refresh(); // show popular suggestions again
    });
  }

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".search-bar-wrap")) close();
  });

  if (form) {
    form.addEventListener("submit", (e) => {
      // A query is optional when a genre is selected (genre browsing).
      if (!input.value.trim() && !currentGenre()) { e.preventDefault(); input.focus(); return; }
      close();
      const btn = document.getElementById("submitBtn");
      if (btn) {
        const label = form.dataset.loadingText || "Loading…";
        btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> ${label.replace(/[<>&"]/g, "")}`;
        btn.disabled = true;
      }
    });
  }

  // Re-enable the button when navigating back (bfcache).
  window.addEventListener("pageshow", () => {
    const btn = document.getElementById("submitBtn");
    if (btn) btn.disabled = false;
  });

  toggleClear();
})();
