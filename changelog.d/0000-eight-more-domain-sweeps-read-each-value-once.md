### Tests: eight more domain suites read each probe value inside one cell per surface

The serial bus speed, the websocket dial host (policy clients and the Reachy
Mini driver), the remote policy client timeouts, the terrain seed, the MuJoCo
patch-op numeric and vector-width checks and the dashboard mesh knobs gave
every probe value its own cell. Each surface now loops its probe table and
names the failing value in the assertion; facets of one refusal are checked
together. The eight files go from 604 cells to 176 with the same package
lines executed.
