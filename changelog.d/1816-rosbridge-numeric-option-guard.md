### Fix: `use_rosbridge` refuses a numeric option it cannot honor

`use_rosbridge` exposes an agent the same `count` / `rate` / `timeout` options as
`use_ros` and `use_rtps` and consumes them the same way, but validated none of
them. `rate=0`, a negative value, `nan` and `inf` all returned
`status="success"` having published every message back-to-back - the requested
pacing discarded, which a mobile base latches as its last command. `count=True`
published one message and reported "published True message(s)". A non-positive
`timeout` returned an empty `echo` result blaming the topic for being silent, and
`timeout=inf` raised `OverflowError` out of a tool documented to return a result
dict.

All three transports now share one owner of the accepted domain instead of
carrying per-tool copies - two copies agreed with each other while the third
transport was written with none, which is how the domain drifted. Each keeps its
own per-action table, so an option an action never reads is still never refused.
