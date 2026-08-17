### Fixed

- **mesh/iot**: the `camera_offload` bucket-ownership threat model cited `CameraOffloader._upload_frame` for the S3 `PutObject` path that omits an `ACL=` kwarg; no such method exists, so a reader auditing that claim could not reach the code it is about. Cite the public `upload_frame`, and widen `test_docstring_xref_roles_resolve` to grade short-form (`Class.member`) roles - the spelling most method cross-references use, and previously outside the guard's fully-qualified-only scope.
