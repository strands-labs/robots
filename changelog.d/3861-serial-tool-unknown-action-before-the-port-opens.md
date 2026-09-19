### Fixed: serial_tool refuses an unknown action before it opens the port

A misspelt `serial_tool` action (`feetech_pos`, `bogus`) used to open the serial
port at the requested baud, close it, and only then report `Unknown action`.
Opening a USB-serial port asserts DTR, which resets some controller boards, so a
call the tool was about to refuse still touched the hardware. The action name
is now graded against the tool's action list ahead of the port check, so an
unknown action is refused with the same message and `serial.Serial` is never
called.
