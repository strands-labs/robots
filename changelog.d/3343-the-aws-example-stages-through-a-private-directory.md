### Fixed: the AWS example stages its SSM document and payload through a private directory

`examples/isaac_on_aws/run_smoke.sh` wrote the SSM parameter document to a fixed
path in a world-writable directory - `/tmp/strands_ssm_params.json` - and read it
back to build a command that runs **as root** on the provisioned instance under
the operator's AWS credentials.

Two problems on that one line. On a shared host another local user can pre-create
the path as a symlink, so the `>` redirect clobbers any file the operator can
write, or swap its contents in the window between the write and the
`aws ssm send-command` read - arbitrary command injection into a root-executing
channel (CWE-377). And the document embeds the `aws s3 presign` URL, a one-hour
bearer credential to the whole packed source tree, so at a default-umask path it
outlived the run world-readable.

Both now stage through a single `mktemp -d` directory, created `0700` and removed
by an `EXIT` trap however the script ends. That also corrects the shape the
tarball used: `$(mktemp -t ...).tgz` appends a suffix to the name `mktemp`
*reserved*, so the file actually written is a different, unreserved path - the same
race more weakly - and the reserved one was then leaked for the life of the host.

This matters more than an example usually would: it is the reference deployment
the Isaac docs point operators at, and a shared multi-user development host is
exactly where it runs.
