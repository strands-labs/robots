### Fixed: a policy URL resolves the same however its scheme is capitalised

URL schemes are case-insensitive (RFC 3986 section 3.1), and every stage of
`resolve_policy()` but the first already compared case-insensitively - `MOCK`
resolves to `mock`, `NVIDIA/GR00T-N1.5-3B` routes to `groot`. The URL stage
matched the raw string against the `url_patterns` declared in `policies.json`,
all of them spelled lowercase, so `ZMQ://gpu-box:5555` matched nothing, skipped
that stage and fell through to the HuggingFace fallback as
`lerobot_local(pretrained_name_or_path="ZMQ://gpu-box:5555")` - an attempt to
download a repo by that literal name instead of dialing the sidecar the caller
named, behind a warning rather than a refusal. All five declared schemes (`zmq`,
`ws`/`wss`, `cosmos3`, `vera`, `grpc`) diverged on case. The scheme is now folded
once, where the URL stage begins, so the pattern match and every per-scheme
parser read the same string and a scheme added later inherits the rule. That
also covers the `grpc` branch, which stripped its scheme with
`str.replace("grpc://", "")`: no regex flag reaches a `str.replace`, so matching
the pattern case-insensitively alone would have routed `GRPC://` correctly and
then handed `lerobot_async` the whole URL as its gRPC target. Hostnames, paths
and HuggingFace repo ids keep the caller's exact spelling.
