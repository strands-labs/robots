### Fixed: a written version bound never names a floor below the one the manifest declares

Two sites told a reader that an older `strands-agents` was enough. The
doctor's remedy for "strands-agents not importable" named `>=1.0`, and the
`so101_curobo` example's agent + UI install line named `>=0.1` - from before
the 1.x line existed. The declared floor is `>=1.7.0,<2.0.0`.

A bound below the real floor does not make a stale install *unsatisfying*, so
the command reports success and upgrades nothing. Against a virtualenv holding
1.5.0, a release the manifest refuses:

```
$ pip install "strands-agents>=1.0"
Requirement already satisfied: strands-agents>=1.0 ... (1.5.0)
$ pip install "strands-agents>=1.7.0,<2.0.0"
Successfully installed strands-agents-1.54.0
```

Both lines now name the declared requirement. Where `strands-agents` is absent
every spelling resolves the newest release, so what the stale bound cost was the
already-installed case - and, for the doctor, a remedy that could report success
without reaching the floor the package requires.

The rule is now graded rather than left to the next reader: a written bound on a
distribution `[project.dependencies]` declares must not sit below the declared
floor. It joins the audit that already sweeps written install hints for
undeclared extras, over the same scan roots, and reads the manifest rather than
restating a number - so raising a floor reports the hints that no longer match
instead of leaving them to drift silently, which is how both of these got here.

The rule is "not below" and not "equal" on purpose. Of the 16 bounds swept, 14
are `numpy` above the declared `>=1.21.0`: the VERA websocket client requires
`numpy>=1.24` and the Cosmos 3 wire path `numpy>=2`. Those are correct local
requirements, and an equality rule would have reported every one of them.
`[project.optional-dependencies]` is outside the reader, because one
distribution can be declared by several extras at several floors and "the
floor" is then not a single number.
