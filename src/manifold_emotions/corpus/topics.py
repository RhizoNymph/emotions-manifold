"""Story topics used for diff-in-means corpus generation.

Anthropic's protocol uses 100 diverse topics. For the shakedown we use a
single neutral topic per emotion so the emotion drives the content.
"""

from __future__ import annotations

SHAKEDOWN_TOPIC = "a chance encounter with a stranger on a city street"

# Diverse topics for the full run. Picked to span everyday, professional,
# domestic, social, and atmospheric settings so the residual emotional
# signal generalizes across context.
FULL_TOPICS: tuple[str, ...] = (
    "a chance encounter with a stranger on a city street",
    "the last day at a long-held job",
    "an unexpected phone call late at night",
    "a family dinner where an old argument resurfaces",
    "discovering a forgotten letter in an attic",
    "the morning of a difficult medical appointment",
    "rehearsing for a performance the next day",
    "navigating a snowstorm on the highway",
    "moving into an empty apartment",
    "a long train ride through unfamiliar countryside",
)
