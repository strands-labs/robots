/* Passkeys panel: the passkeys that can sign in to this dashboard (add · rename · remove, never the last one) and
 * sign out. Only shown when sign-in is on. Every name is set with textContent.
 */
import { enrol, logout, status } from "../auth.js";

const state = await status();
if (state && state.enabled !== false && state.signed_in) {
  window.SR.registerPanel({
    id: "passkeys",
    title: "Passkeys",
    icon: "🔑",
    order: 90,
    mount(el, ctx) {
      const list = document.createElement("ul");
      const err = document.createElement("div");
      err.className = "muted"; err.role = "alert";
      const button = (text, onclick) => Object.assign(document.createElement("button"), { type: "button", className: "sr-btn", textContent: text, onclick });
      const guard = (fn) => async () => { err.textContent = ""; try { await fn(); await refresh(); } catch (e) { err.textContent = e.message || String(e); } };
      async function refresh() {
        const { passkeys } = await ctx.api("/api/passkeys");
        list.replaceChildren(...passkeys.map((p) => {
          const li = document.createElement("li");
          const name = document.createElement("span");
          name.textContent = `${p.name} `;
          li.append(name,
            button("rename", guard(async () => { const n = prompt("New name for this passkey", p.name); if (n) await ctx.api(`/api/passkeys/${encodeURIComponent(p.id)}`, { method: "POST", body: { name: n } }); })),
            button("remove", guard(async () => { if (confirm(`Remove passkey "${p.name}"? It can no longer sign in.`)) await ctx.api(`/api/passkeys/${encodeURIComponent(p.id)}`, { method: "DELETE" }); })));
          return li;
        }));
        ctx.setBadge(passkeys.length);
      }
      const title = document.createElement("div");
      title.className = "section-title"; title.textContent = "Passkeys";
      const actions = document.createElement("div");
      actions.append(button("+ add a passkey", guard(() => enrol({ label: prompt("Name for the new passkey", "another device") || "passkey" }))), button("sign out", logout));
      el.append(title, list, actions, err);
      refresh().catch((e) => { err.textContent = e.message || String(e); });
    },
  });
}
