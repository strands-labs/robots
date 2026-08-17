### Fixed

`processor_overrides` now reaches the pipeline that declares the step. A
checkpoint's preprocessor and postprocessor carry disjoint step keys
(`normalizer_processor` vs `unnormalizer_processor`), and LeRobot refuses an
override key that matches no step in the pipeline it is loading, so passing the
caller's dict to both made every pipeline-specific step unreachable -- only
`device_processor`, present in both, could be applied. Supplying normalizer
`stats` is the documented remedy for a pretraining base checkpoint whose
dataset-prefixed stats leave its declared normalization inert, and that remedy
raised `KeyError` instead of applying. Each override is now routed to the
pipeline that declares it, a key declared by neither is still refused with both
pipelines' step names listed, and the inert-normalization diagnostic names both
steps.
