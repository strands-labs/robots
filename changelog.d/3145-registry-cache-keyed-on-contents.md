### Fixed: the registry hot-reload cache is keyed on file contents, not a timestamp

The `robots` registry is the package `robots.json` merged with the user overlay
`user_robots.json`, and the loader cached that merge keyed on both files'
`st_mtime`. The kernel stamps mtime from a coarse clock (4 ms on ext4), so two
writes inside one tick carry the same timestamp - and it never changes again,
so the first write's contents were served for the life of the process. There
was no size in the signature either, so an edit that changed the file's length
was invisible too: 58 of 60 rounds of "write the overlay, read, write it again"
never observed the second edit. `register_robot` from a second process is that
sequence.

The signature is now the bytes of both files, and the overlay is parsed from
the bytes the signature was taken from rather than re-read, so the cached merge
and its key always describe the same overlay. The cache still skips the parse,
the merge and the uniqueness validation for an unchanged registry - two reads
add about 11 us per lookup and avoid about 164 us of work.

`registry.user_registry_mtime()` is replaced by `registry.user_registry_source()`
(the file's bytes) plus `registry.parse_user_robots(source)` (the parse), so one
read feeds both the cache key and the cached value.

A corrupt overlay whose bytes are not valid UTF-8 no longer raises
`UnicodeDecodeError` out of `get_robot()`; undecodable bytes are ignored like
truncated JSON already was.
