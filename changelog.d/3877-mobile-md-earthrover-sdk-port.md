### Fixed: `docs/robots/mobile.md` teaches the EarthRover SDK on the port it binds

The EarthRover hardware fence and the page's `port` table spelled every SDK address on `:8001`, a port the earth-rovers-sdk never serves, so copying the fence for a stock SDK produced the same "unreachable" report the old driver default gave. Every address on the page now reads `:8000`, matching `DEFAULT_SDK_URL`, and a grader holds the page to the driver's own port.
