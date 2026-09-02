### Fixed

- **mesh**: a cached ACL is served only while the file still holds the bytes it
  was parsed from. The ACL load cache is keyed on the file's identity tuple
  `(path, dev, ino, size, mtime_ns)`, and that tuple also licensed serving an
  entry. It cannot: the kernel stamps `st_mtime_ns` from a coarse clock, so two
  writes inside one tick are stamped alike, and a rewrite that keeps the byte
  count leaves `st_size` alone. An in-place ACL edit of the same length -
  rotating an authorised `cert_common_names` entry is exactly that shape -
  therefore computed the identity the pre-edit contents were cached under, and
  the revoked peer stayed authorised for the life of the process. Each entry now
  carries the bytes it was parsed from, and the single hardened read both
  verifies a hit and feeds the parser on a miss, so the cache still saves the
  JSON5 parse and shape validation.
