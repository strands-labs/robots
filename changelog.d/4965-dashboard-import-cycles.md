### Internal: the dashboard's routes and helpers type against a Protocol, not the Dashboard class

`http_error` moved to a leaf `dashboard.errors` module and the route and helper modules now type their `dash`
parameter against a `DashboardLike` Protocol (`dashboard.protocols`) instead of importing the `dashboard` module
back for an annotation. The dashboard package no longer has an import cycle: only the class itself, the package
export and the `python -m` entry point name the concrete `Dashboard`. The page's behaviour and public surface
(`strands_robots.dashboard.Dashboard`) are unchanged.
