### Changed: the dashboard says how to turn chat and voice on, and the voice controls use the page's buttons

A dashboard started without a model shows "Chat is off" in the chat panel with the command that turns it on
(`--model <bedrock model id>`, or `model=` to `Dashboard`), and disables Send, instead of answering the first message
with an error. New `GET /api/chat` reports whether chat is on. `python -m strands_robots.dashboard` takes
`--voice auto | openai | nova`. The talk and mute buttons use the dashboard's pill buttons in light and dark, a live
conversation shows a pulsing dot, and mute appears only while a conversation is live.
