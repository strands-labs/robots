### Fixed: a GitHub-hosted model passes its digest check on Windows

`tree_digest` sorted `Path` objects, and Windows compares those without regard to case. A model whose file names differ
in case therefore hashed to something other than the registry's `sha256` and was refused with `AssetMismatch`. Files are
now ordered by their path parts on every platform, which is the order the pinned digests already use on Linux.
