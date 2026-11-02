### Docs: the site and README describe the library as it is

The docs, README and site templates now describe strands-robots as a Python library for building robots, in plain
factual language, with developer-oriented titles and a reorganised navigation (Get started, Guides, Hardware and
fleets, Reference). New pages: Installation (extras table, LeRobot, local docs) and Running a LeRobot policy (split
out of Policies). The device page is now Safety and e-stop (`safety.md`). The build-status page, the design and
schedule references, the AWS IoT planning notes under `docs/iot/` and five unused design drawings are removed; the
IoT quotas that shape the design are a table on the AWS IoT Core page. The dashboard page documents the one-line local
start and the live embed on the docs home page. `tests/test_docs_no_emoji.py` asserts that no page, template or the
README carries an emoji or a design/schedule reference.
