### Fixed: a bare output filename is written into the sandbox instead of being refused

`validate_output_path` resolved every relative path against the process CWD, so
under confinement the most natural call a caller can make -- `render(output_path="frame.png")`
-- was always refused, and the refusal quoted a CWD-absolute path the caller
never supplied. A one-component name is now anchored to the sandbox root. The
rule is narrow and cannot widen the sandbox: it applies only when confinement is
active and only to a single-component name, `..` is still refused, and the
symlink probe now inspects the anchored destination. Guards-only mode (the
historic video/recording contract) is unchanged.
