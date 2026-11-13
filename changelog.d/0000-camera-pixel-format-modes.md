### Changed: a camera spec chooses its pixel format, and a refused mode lists the camera's modes

A camera read through OpenCV asks for MJPEG by default (`fourcc`, `None` keeps the device's own), so 720p at 30 fps
fits a USB 2.0 camera instead of falling to 10 fps on raw YUYV. A named macOS camera takes `pixel_format` for the
frames AVFoundation hands over (default `uyvy422` as before). A size or rate the camera lacks used to raise a bare
`OSError(5, 'Input/output error')`; it now names the mode asked and lists the modes the camera has.
