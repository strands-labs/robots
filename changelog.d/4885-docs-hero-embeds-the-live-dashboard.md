### Added: the docs homepage shows your running dashboard, and `python -m strands_robots.dashboard` starts one

When a dashboard runs on this machine (`?dashboard=host:port`, else 127.0.0.1:8091, then :8090), the docs hero shows
it live in an iframe, in the docs' light or dark scheme; otherwise it shows the recording and the one command that
starts a dashboard: `python -m strands_robots.dashboard [robots...] [--port] [--model]`. The embedding page cannot
script the frame or press its buttons (another origin; the dashboard's guard refuses its requests), and the dashboard
now says who may frame it: `Content-Security-Policy: frame-ancestors` allows loopback pages and the published docs site
only. The dashboard's Docs link points back at the docs that embed it.
