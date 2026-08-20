### Fixed

- **docs/training**: the dependency table and interpreter callout described a subprocess execution model the trainers do not have, and advertised a `python_executable=` constructor argument none of them declares - every trainer's `**kwargs` absorbed it silently, so following the guidance installed the backend into an interpreter the trainer never consults. The `groot` and `cosmos3` rows now state that the backend is imported in the calling interpreter, and a new guard grades a documented `name=` argument against the class's real parameters.
