"""Optional LLM rephrasing of already-grounded feedback (Anthropic SDK).

This is a *presentation* layer only: it takes the deterministic, evidence-grounded
feedback the scoring model produced and rewrites it as smoother learner-facing prose.
It is forbidden from introducing any claim not already in the grounded text -- the
score and the cited drivers come from the model, not the LLM. Called only when
`settings.anthropic_api_key` is set; `feedback.generate.maybe_rephrase` swallows any
failure and falls back to the template text, so this never breaks the pipeline.

No prompt caching here on purpose: the system prompt is far below the model's minimum
cacheable prefix (~1-4k tokens), so a cache_control breakpoint would silently never
fire. Per-sample calls are short and independent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .generate import Feedback

SYSTEM = (
    "You rewrite automated language-proficiency feedback into warm, concise "
    "learner-facing prose. Rules: (1) Use ONLY the facts in the provided draft and "
    "evidence -- never invent strengths, weaknesses, scores, or numbers. (2) Keep the "
    "overall score exactly as given. (3) 2-3 sentences, second person, encouraging but "
    "honest. (4) No bullet points, no headers, no raw feature names. Return only the "
    "rewritten feedback."
)


def _evidence(fb: Feedback) -> str:
    subs = ", ".join(f"{k}={v:.1f}" for k, v in fb.subscores.items())
    parts = [
        f"Draft: {fb.text}",
        f"Overall score (keep verbatim): {fb.overall:.1f}/10",
        f"Subscores (1-5): {subs}",
    ]
    if fb.strengths:
        parts.append("Strength drivers: " + ", ".join(fb.strengths))
    if fb.growth:
        parts.append("Growth drivers: " + ", ".join(fb.growth))
    return "\n".join(parts)


def rephrase(feedback: Feedback, settings) -> str:
    """Return LLM-rephrased feedback text. Raises on any API/SDK error so the
    caller (maybe_rephrase) can fall back to the grounded template."""
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=300,
        system=SYSTEM,
        messages=[{"role": "user", "content": _evidence(feedback)}],
    )
    text = "".join(block.text for block in response.content if block.type == "text").strip()
    if not text:
        raise ValueError("empty LLM response")
    return text
