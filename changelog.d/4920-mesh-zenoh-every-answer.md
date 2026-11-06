### Fixed: a zenoh rpc hears every answer, so a second answerer for a device is refused

`invoke` refuses a request that gets two different answers, but on a zenoh mesh it only ever saw one: zenoh's
default query consolidation keeps a single reply per key, and every answerer replies on the same key. Whichever
answered first won, so the check never ran. A zenoh query now asks for every reply (`ConsolidationMode.NONE`), and a
peer that answers for a device it does not hold makes the call raise `RemoteError` instead of being believed.
