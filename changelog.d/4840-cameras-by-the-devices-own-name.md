### Added: hardware cameras by the device's own name, sized by the policy

`Robot("so101", port=…, cameras={"front": "Logi 4K", "wrist": "USB2.0_CAM1"})`. A string that is not a path or a
number is the device's name (or a unique part of it, case-insensitive), resolved at open against what the OS reports
— AVFoundation on macOS (read through PyAV, the backend that takes names), V4L2 on Linux (`/sys/class/video4linux`,
metadata nodes skipped) — and refused by name when it matches nothing or several devices, listing what is there.
Indices shift whenever a phone or a screen capture joins the machine; names do not. `{"name": …, "width": …}` is the
spec form; a bare index/path and `{index, width, height, fps}` are unchanged. A camera that declared no size takes the
policy's at `run()` (`Cameras.fit` from the policy's requirements, before the first frame — an open camera reopens),
so the dict's keys are the policy's image keys and there is no size or rename map to write. The device keeps its own
frame rate: a camera read at a slower policy rate would wait a whole camera period for a fresh frame.
A named camera on macOS always opens with a size and a rate (`av_options`): ffmpeg's AVFoundation input refuses every
open with `OSError(5, 'Input/output error')` when no `framerate` is given, and a rate without a size asks for the
device's native mode (a hang on one USB camera, the full 4096x2160 sensor on another). A declared or fitted size wins;
an undeclared one opens at 640x480 @ 30.
