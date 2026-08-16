### Fixed

- Document `STRANDS_GROOT_WIRE_LOG` as the directory the GR00T wire-payload dumps land
  in. The api-reference table offered `1` as the value that enables the diagnostic, but
  the value is passed straight to `os.makedirs`, so following the table wrote pickle
  archives into a directory named `1` under the process working directory. The same row
  described the dumps as raw ZMQ frames although the diagnostic also covers the
  in-process inference path, which opens no socket.
