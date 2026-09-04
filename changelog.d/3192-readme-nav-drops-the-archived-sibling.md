### Docs: the README header nav no longer offers an archived repository

The nav linked `strands-labs/robots-sim`, which was archived at 01:33 UTC on
2026-08-06 and is read-only. A reader following it landed on a repository
presented as a live sibling project that accepts no issue, no pull request and no
push, while the maintained simulation stack the entry implies is elsewhere ships
here in `strands_robots/simulation/` (MuJoCo + Isaac backends, the latter
absorbed from that repository by #1156).

The entry is removed. References to `robots-sim` as *provenance* are untouched
and remain correct: `strands_robots/simulation/isaac/` did arrive from there, and
`tests/simulation/isaac/test_migrated_reference_provenance.py` exists to require
those references be written `robots-sim#N` so they name the repository they
belong to. The distinction is between "this arrived from there" (history, which
stays) and "go there for the live thing" (a destination, wrong once the target is
read-only), so the removal is scoped to the one surface that offered the
repository as a destination.

`tests/test_readme_links_are_not_archived_repositories.py` pins it. Every
`github.com` link in `README.md` is decoded to the repository it names -- across
Markdown, autolink, HTML `href` and bare-URL spellings, since the offending entry
was raw `<a href>` and therefore invisible to Markdown-only link parsing -- and
checked against the set known to be read-only. The population is derived from the
README, so a nav entry or prose link added later is graded on arrival.
