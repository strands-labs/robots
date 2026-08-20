### Fixed

- **robot_mesh**: the read-only `peers` / `status` actions now leave an audit row on the Device Connect backend too, not only on the mesh backend. Device Connect is the backend tried first whenever it has devices, so the audited implementations were the fallback: an agent could enumerate every device id and every function name the fleet exposes and leave no forensic trail. The row records the size of the fleet that was read (`devices=N`), matching the mesh rendering's `local=N remote=M`.
