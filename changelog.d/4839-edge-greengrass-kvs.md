### Added: a mesh device runs as a Greengrass v2 component and streams its cameras over Kinesis Video Streams

The `[iot]` mesh gains an optional edge. `mesh.iot.greengrass` turns the process that mounts robots into a
Greengrass v2 component (`recipe()`), the tenant's thing group into the fleet and a deployment into the fleet
update (`deployment()`/`deploy()`); its `main()` mounts the configured robots on one `Mesh` and stays up until
SIGTERM, with robot-to-robot traffic staying on zenoh. `mesh.iot.kvs` names one KVS stream and WebRTC signaling
channel per camera sensor (`camera_entries()`, written into the manifest shadow), builds the measured GStreamer
pipeline (`pipeline()`) and feeds each camera into `kvssink` on its own thread (`publish_cameras()`); without the
`kvssink` plugin the producer refuses by name rather than pretend. `import strands_robots.mesh` still loads no
boto3, awsiot or gi.
