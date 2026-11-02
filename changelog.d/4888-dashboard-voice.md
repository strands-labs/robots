### Added: the dashboard talks: speech to speech over a strands BidiAgent holding every dashboard tool

`Dashboard(sim, voice="openai" | "nova" | "auto")` serves `WS /ws/voice` and `GET /api/voice`. The page's talk button
streams the microphone (PCM16 mono, 24 kHz for OpenAI, 16 kHz for Nova Sonic) to one `strands.bidi.BidiAgent` per
conversation (`gpt-realtime-2`, or `amazon.nova-2-5-sonic` on Bedrock), and plays its spoken answer; transcripts and
tool calls land in the chat log, with a level meter and a mute. The agent holds the chat agent's tools (`sim.tool` and
every `robot.tool`) plus `dashboard_run` (a policy in the background, as the run button starts it), `dashboard_stop`
and `dashboard_estop`. Every call is an agent's: voice never carries the page's `X-Confirm`, so a real robot refuses
to act or run from it, and a policy at a network address from the model is refused as `robot.tool` refuses it. New
extra `[voice]` (in `[all]`); Nova Sonic also needs `strands-agents[bidi]`.
