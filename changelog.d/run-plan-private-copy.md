### Fixed: an approved agent run loads a private copy of the policy files

An approved agent `run()` on a real robot checked the policy files against `RunPlan.files` after loading them from
their original path, so a file replaced only while the loader was reading it (and put back afterwards) passed the
check. The run now copies the local policy files into a private temporary folder, checks the copy's digest against
the approved one before anything is loaded, binds the copy, and removes it when the run ends. Later writes to the
original path no longer reach the run.
