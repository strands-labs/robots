### Fixed: the dashboard auth store cache serves a hit only while the file still holds its bytes

`auth._load` cached the credential store under `(path, st_mtime_ns, st_size)`.
That tuple names a file, not a version of it. The kernel stamps `st_mtime_ns`
from a coarse clock, so two writes inside one tick are stamped alike, and a
rewrite that keeps the byte count keeps `st_size` -- eight successive same-size
rewrites of a store on ext4 produce one distinct `st_mtime_ns` between them.

Under a stat-only hit the first such edit was the last one seen, and not
transiently: the identity never changes again, so the stale store was served for
the life of the process. An operator retiring a passkey record, or rotating a
`jwt_secret` for one of the same length, is making exactly that same-size edit
-- on the file that decides whether the dashboard is sealed.

The cached entry now carries the bytes it was parsed from, and a hit is served
only while the file still holds them. No stat field can substitute:
`st_ctime_ns` comes from the same tick, and `st_ino` is unchanged by an in-place
rewrite. The read stays and the parse is what the cache saves, which is what the
read licenses skipping. The identity remains the key, the cache remains at one
entry, and the atomic writer re-keys under the payload it just wrote.
