### Changed: the dashboard wears the docs design, and its e-stops never cover the chat

The page now uses the docs site's look (paper/ink and the Strands green, light and dark with a toggle, the docs fonts)
in a fixed frame: a top bar, the 3D stage, and a side column with tabs over a docked chat. The floating e-stop tray that
sat over the chat input is gone: **STOP ALL** in the top bar e-stops every robot and mesh device in one click (also
Shift+Esc), and each robot card's header carries that robot's E-STOP and Resume. Cards collapse, sliders are readable,
and a **Docs** link sits in the bar. Front-end modules add side-column tabs with `window.SR.registerPanel({id, title,
mount(el, ctx)})`; the server lists `static/panels/*.js` at `GET /api/panels` and the page imports each in server mode.
