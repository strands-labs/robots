# Dashboard panels

Every `*.js` file here is an ES module the dashboard imports in server mode (listed by `GET /api/panels`). A module
adds one tab to the side column:

```js
window.SR.registerPanel({
  id: "record",            // lowercase slug, unique ("robots" and "chat" are taken)
  title: "Record",
  icon: "●",               // optional, text
  order: 30,               // tabs sort by order after Robots (default 100)
  mount(el, ctx) {          // once, with an empty <section class="pane">
    el.innerHTML = `<div class="section-title"><span>Record</span></div>`;
    ctx.on("frame", (f) => {});               // also "scene", "mode", "tab", "scheme", "layout", "panel"
    // ctx.api(path, {method, body, confirm}) · ctx.robots · ctx.state · ctx.esc · ctx.log · ctx.chip · ctx.setBadge(n)
  },
});
```

Reuse the page's classes (`.section-title`, `.sr-input`, `.sr-btn`, `.sr-btn.primary`, `.card`, `.tag`, `.muted`) and
`--sr-*` tokens so a panel looks like the rest of the page in both schemes. Never set `X-Confirm` without asking the
operator first (`ctx.api(..., {confirm: true})` only after a `confirm()` naming the real robot).

A row next to something instead of a tab: `window.SR.registerSlot("chat-input" | "bar", mount)`: `chat-input` is a row
right above the chat input (a mic, a mute), `bar` sits in the top bar left of Docs (a REC dot). `mount(el, ctx)` may
return an element to append. `connect.js`, `record.js` and `voice.js` next to `app.js` are imported too.
