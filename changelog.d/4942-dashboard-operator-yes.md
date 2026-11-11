### Fixed: the dashboard's `X-Confirm` counts only on a signed-in request, and every route that changes state waits for it

- With sign-in off (`auth=False` / `unauthenticated_hardware=True`), `X-Confirm` no longer stands in for the operator:
  a real robot asks its own `confirm` callback.
- `POST /api/robots/{id}/run` on a real robot is refused before the policy is built, so nothing is dialled.
- `POST /api/connect/policy` and every `POST /api/world` verb except `describe`, `pose`, `contacts` and `render` need
  `X-Confirm: yes`; world calls run as a dashboard's (no policy at a network address). The Connect panel asks first.
- `add_object(path=...)` from `/api/world` loads a file only inside `Dashboard(object_dir=...)` or
  `$STRANDS_ROBOTS_DASHBOARD_OBJECTS` (none by default).
