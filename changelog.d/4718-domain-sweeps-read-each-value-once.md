### Tests: five domain suites read every probe value inside one cell per surface

The evaluation loop bounds, the cross-backend gravity domain, the mobile-base
drive command, the Isaac camera readback pixel count and the MuJoCo motion
primitive numeric fields gave every probe value its own cell, and in places
every facet of one refusal its own test. Each surface now loops its probe
table, names the failing value in the assertion, and checks the refusal, its
wording and its "nothing moved" facet together. The five files go from 481
cells to 60 with the same package lines executed.
