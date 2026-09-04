### Fixed

- Every identifier allowlist in the package anchors at the absolute end of the
  string (`\Z`) rather than with `$`, which in non-MULTILINE mode also matches
  immediately before a single trailing newline. An otherwise-valid value carrying
  one no longer passes: `dds_type_name('std_msgs/msg/String\n')` used to return
  `'std_msgs::msg::dds_::String\n_'`, putting one DDS identifier on the graph
  across two lines, and `harness_memory`'s `save_trace` used to write a trace file
  whose name carried the line break. The two mesh validators that already
  anchored this way (`peer_id`, `thing_name`) are unchanged; the ten patterns
  already consulted with `re.fullmatch` change spelling but not behaviour.
- `SagemakerTrainer.validate` enforces the documented 1-32 character bound on
  `base_job_name` for a 33rd character that is a newline. The bound is spent in a
  lookahead, so the end-of-string anchor decided the length as well as the
  character set.
