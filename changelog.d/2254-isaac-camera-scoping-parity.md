### Quality: pin the Isaac recording camera-scoping matrix to MuJoCo/Newton parity

`IsaacSimulation.start_recording` documents `cameras=` names as either raw
(`arm0/wrist`) or already schema-safe (`arm0__wrist`), and its source comment
claims "parity with MuJoCo/Newton". Both siblings pin the schema-safe spelling;
Isaac pinned only the raw and unknown-name cells, leaving the alias branch
unexecuted. Adds the alias cell, the raw-equals-safe equivalence assertion that
MuJoCo already carries, and the first pin anywhere for the both-spellings
dedup. Tests only; no library behaviour changes.
