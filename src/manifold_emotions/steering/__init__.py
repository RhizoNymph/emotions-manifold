"""Per-request steering through the vLLM fork's OpenAI-compatible API.

Phase 8 needs to push K waypoints' worth of steering vectors at the
model for each emotion pair we're testing, generate N continuations
per waypoint, and aggregate them. The fork supports per-request
steering vectors via ``steering_vectors`` in the chat-completion body
(see docs/steering.md). We use the inline JSON form rather than the
binary wire format here — payload size at K=20 waypoints × hidden=5376
is ~430 KB JSON-encoded, which is fine. Switch to the binary form
later if we hit the API server's event-loop ceiling.
"""
