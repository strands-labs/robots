/* Copy pills, the content fade on instant navigation, and the "On this page" bar. Vanilla, ~50 lines. */
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
  /* The hero shows a recorded run. If a dashboard runs on this machine it becomes the live page, in an iframe: the
     dashboard's own origin serves it, so this page can show it but never script it or press its buttons (another
     origin), and the dashboard's guard refuses this page's own requests. Candidates: ?dashboard= (or ?live=) if given, else
     the stage's data-live list; loopback only. Each candidate loads in a transparent iframe under the recording (an iframe
     that cannot connect logs nothing here, unlike an image or fetch probe). A dashboard says "sr:hello" when its page
     runs (the badge shows "compiling") and "sr:painted" after its first frame with robots; only then does it replace
     the recording. A candidate that says nothing in 5 s is removed, and older dashboards keep the recording. */
  const LOOPBACK = ["127.0.0.1", "localhost", "[::1]"];
  const scheme = () => (document.body.getAttribute("data-md-color-scheme") === "slate" ? "dark" : "paper");
  function live() {
    const stage = document.getElementById("hero-stage"); if (!stage || stage.__live) return;
    stage.__live = true;
    const q = new URLSearchParams(location.search);
    const asked = [q.get("dashboard"), q.get("live")].filter(Boolean); /* named in the URL: only those */
    const want = asked.length ? asked : (stage.dataset.live || "").split(/\s+/).filter(Boolean);
    const urls = [];
    for (const raw of want) {
      let u; try { u = new URL(raw.includes("//") ? raw : `http://${raw}`, location.href); } catch { continue; }
      if (/^https?:$/.test(u.protocol) && LOOPBACK.includes(u.hostname) && !urls.some((x) => x.origin === u.origin)) urls.push(u);
    }
    const badge = stage.querySelector(".sr-stage__badge");
    const docs = new URL(".", document.querySelector("link[rel=canonical]")?.href || location.href).href;
    const frames = new Map(); /* origin -> {u, frame, timer} */
    let chosen = null;
    const settle = () => { if (!chosen && !frames.size) stage.dataset.state = "recorded"; };
    for (const u of urls) {
      const src = new URL(u.href);
      src.searchParams.set("theme", scheme()); src.searchParams.set("docs", docs);
      const frame = document.createElement("iframe");
      frame.className = "sr-stage__live"; frame.src = src.href; frame.title = "strands-robots dashboard, live";
      frame.tabIndex = -1; frame.setAttribute("aria-hidden", "true");
      frame.setAttribute("allow", "microphone"); /* the dashboard's voice row asks for the mic itself */
      const entry = { u, frame, timer: setTimeout(() => { frame.remove(); frames.delete(u.origin); settle(); }, 5000) };
      frames.set(u.origin, entry);
      stage.appendChild(frame);
    }
    settle();
    addEventListener("message", (e) => {
      const entry = frames.get(e.origin);
      if (!entry || e.source !== entry.frame.contentWindow || !e.data || typeof e.data.type !== "string") return;
      if (chosen && chosen !== entry) return;
      if (e.data.type === "sr:hello") {
        clearTimeout(entry.timer);
        chosen = entry;
        for (const [o, other] of frames) if (other !== entry) { clearTimeout(other.timer); other.frame.remove(); frames.delete(o); }
        stage.dataset.state = "compiling";
        if (badge) badge.textContent = `live · ${entry.u.host} · compiling`;
      } else if (e.data.type === "sr:painted" && chosen === entry) {
        stage.dataset.state = "live";
        entry.frame.removeAttribute("aria-hidden"); entry.frame.tabIndex = 0;
        stage.querySelector("img")?.remove();
        /* the badge would sit on the dashboard's own controls: it becomes a line under the stage */
        badge?.remove();
        const line = document.createElement("p"); line.className = "sr-stage__open";
        const link = document.createElement("a");
        link.className = "sr-stage__badge--live"; link.href = entry.u.href; link.target = "_blank"; link.rel = "noopener";
        link.textContent = `live · ${entry.u.host} · open in a tab`;
        line.appendChild(link); stage.after(line);
        /* Different origins share no storage: tell the frame when the reader flips the palette. */
        new MutationObserver(() => entry.frame.contentWindow?.postMessage({ type: "sr:scheme", scheme: scheme() }, entry.u.origin))
          .observe(document.body, { attributes: true, attributeFilter: ["data-md-color-scheme"] });
      }
    });
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
