// pymdownx.superfences emits mermaid as <pre class="mermaid"><code>SRC</code></pre>.
// Render each block to an SVG with mermaid.render(). Belt-and-suspenders on top
// of Material's native support; only touches blocks not already rendered.
(function () {
  function renderAll() {
    if (typeof window.mermaid === "undefined") return setTimeout(renderAll, 80);
    try { window.mermaid.initialize({ startOnLoad: false, securityLevel: "loose" }); } catch (e) {}
    document.querySelectorAll("pre.mermaid, div.mermaid").forEach(function (el, i) {
      if (el.dataset.rendered === "1" || el.querySelector("svg")) return;
      var code = el.querySelector("code");
      var src = (code ? code.textContent : el.textContent || "").trim();
      if (!src) return;
      var id = "mmd-" + Date.now() + "-" + i;
      try {
        var out = window.mermaid.render(id, src);
        if (out && typeof out.then === "function") {
          out.then(function (r) { el.innerHTML = r.svg; el.dataset.rendered = "1"; }).catch(function(){});
        } else if (typeof out === "string") {
          el.innerHTML = out; el.dataset.rendered = "1";
        } else if (out && out.svg) {
          el.innerHTML = out.svg; el.dataset.rendered = "1";
        }
      } catch (e) {}
    });
  }
  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(function () { setTimeout(renderAll, 0); });
  } else {
    document.addEventListener("DOMContentLoaded", renderAll);
  }
})();
