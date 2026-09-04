### Docs: the README header nav no longer links to the archived robots-sim repository

`strands-labs/robots-sim` was archived on 2026-08-06 and is read-only, but the
README's header nav still listed it as `Robots Sim` beside the live external
references, steering newcomers to a dead sibling project. The maintained
simulation stack lives in this repository (`strands_robots/simulation/`, MuJoCo
and Isaac backends). The entry and its diamond separator are removed; the
surrounding entries (Strands Docs, MuJoCo, NVIDIA GR00T, LeRobot, Project
Board) are unchanged, and no other `robots-sim` reference existed in the
README.
