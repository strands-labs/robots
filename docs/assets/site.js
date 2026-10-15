/* Copy pills, the content fade on instant navigation, and the "On this page" bar. Vanilla, ~40 lines. */
(() => {
  function wire(root) {
    root.querySelectorAll(".sr-copy[data-clipboard-text]").forEach((b) => {
      if (b.__wired) return; b.__wired = true;
      b.addEventListener("click", async () => {
        try { await navigator.clipboard.writeText(b.dataset.clipboardText); } catch { /* no clipboard */ }
        b.classList.add("is-copied"); const t = b.textContent; b.textContent = "copied";
        setTimeout(() => { b.classList.remove("is-copied"); b.textContent = t; }, 1200);
      });
    });
    const inner = root.querySelector(".md-content__inner");
    if (inner) { inner.classList.remove("sr-enter"); void inner.offsetWidth; inner.classList.add("sr-enter"); }
  }
  function toc() {
    const nav = document.querySelector(".md-nav--secondary"); if (!nav) return;
    const a = nav.querySelector(".md-nav__link--active"); if (!a) return;
    const r = a.getBoundingClientRect(), n = nav.getBoundingClientRect();
    nav.style.setProperty("--sr-toc-y", `${r.top - n.top}px`); nav.style.setProperty("--sr-toc-h", `${r.height}px`);
  }
  /* The hero shows a recorded run; if a dashboard answers on the stage's data-live URL, it becomes the live twin. */
  async function live() {
    const stage = document.getElementById("hero-stage"); if (!stage || stage.__live) return;
    const url = new URLSearchParams(location.search).get("live") || stage.dataset.live; if (!url) return;
    /* A v1 dashboard serves its mark as an SVG; anything else answering on the port is not one. */
    const ok = await new Promise((done) => {
      const img = new Image(); const t = setTimeout(() => done(false), 1500);
      img.onload = () => { clearTimeout(t); done(img.naturalWidth > 0); };
      img.onerror = () => { clearTimeout(t); done(false); };
      img.src = url + "mark.svg?probe=" + Date.now();
    });
    if (!ok) return;
    stage.__live = true;
    stage.innerHTML = `<iframe src="${url}" title="strands-robots dashboard, live"></iframe><span class="sr-stage__badge">live · ${url}</span>`;
  }
  function frog() {
    const f = document.querySelector(".sr-frog"); if (!f || f.__wired) return; f.__wired = true;
    const words = ["strands", "agents", "robots"]; let i = 0;  /* every click says the next one */
    f.addEventListener("click", () => {
      f.dataset.say = words[i++ % words.length];
      f.classList.remove("is-jump"); void f.offsetWidth; f.classList.add("is-jump");
      clearTimeout(f.__t); f.__t = setTimeout(() => f.classList.remove("is-jump"), 900);
    });
  }
  const boot = () => { wire(document); toc(); live(); frog(); };
  if (window.document$) window.document$.subscribe(boot); else document.addEventListener("DOMContentLoaded", boot);
  document.addEventListener("scroll", () => requestAnimationFrame(toc), { passive: true });
})();
