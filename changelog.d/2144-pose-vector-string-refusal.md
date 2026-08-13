### Fixed: a string passed where a pose vector belongs is refused on its type, not its length

`add_camera(target="cube")` reported `'target' must be a 3-element vector, got 4
('cube')`, describing the string's character count as though it were a component
count; `target="box"` reported `elements must be numbers` instead, because a
3-character string reaches the element read rather than the length gate. One
mistake drew three different messages, two of them pointing at the wrong thing to
fix.

The type check now runs before the length probe, so every string draws one
refusal that names the parameter and the type it was given. No verdict changes -
a string was already refused in each case.
