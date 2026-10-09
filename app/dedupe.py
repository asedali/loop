"""Near-duplicate detection for Phase 1 idea cards.

Pure text arithmetic. No model call, no embedding, no dependency — deliberately,
because this runs while *rendering* a page. A model asked "are these the same
idea?" would cost a quota unit per pair, can be wrong in a way that loses a card,
and would make a page depend on a model call succeeding. Arithmetic cannot fail
that way.

What this module does NOT do is merge anything. It reports which cards look alike
so the user can decide; the decision to retire a card is theirs, which is
invariant 4 applied to Phase 1.

See specs/10-M1.4-near-duplicate-ideas.md.
"""
import re

# Where to cut, and why here rather than at 0.5.
#
# Calibrated on measured pairs rather than picked by taste. On the table below
# true duplicates score 0.774 and up; distinct ideas score 0.659 and down:
#
#   identical after normalisation .......................... 1.000
#   case / punctuation / whitespace only ................... 1.000
#   the same idea with the words reordered ................. 0.893
#   a synonym swapped (supplies -> provides) .............. 0.876
#   one word changed in a 120-character title ............. 0.836
#   "...for farms" vs "...tool for farms" .................. 0.800
#   "Sell sensors to hospitals" vs "Selling sensors to ..." 0.774
#   "Detect sepsis from BLOOD samples"
#     vs "Detect sepsis from URINE samples" ................ 0.659   <- want differ
#   "...to small research groups at universities"
#     vs "...to large pharmaceutical companies" ........... 0.534   <- want differ
#   "Sell sensors to hospitals" vs "Sell sensors to vets"  0.516   <- want differ
#
# The clean band is 0.68-0.77, and 0.72 sits in the middle of it. That band is
# NARROW — 0.12 wide — and pretending otherwise would be the dishonest part of
# this file. Character-trigram similarity has no notion of which word carries the
# meaning, so "blood samples" and "urine samples" land close together: same
# template, one differentiator, and in research those are entirely different
# tests. A slightly worse phrasings pair could land in the gap either way.
#
# So this is a calibrated heuristic, not a classifier, and the reason the feature
# surfaces duplicates instead of merging them is precisely that. A chip the user
# dismisses costs a glance; a merge the heuristic gets wrong destroys an idea
# nobody can get back, because the app would have thrown away the researcher's
# own work on a similarity score.
NEAR_DUPLICATE_THRESHOLD = 0.72

# Titles are capped at 120 characters by llm.py, so the trigram set is at most
# ~122 entries and a pairwise scan over a page of cards stays trivial. Punctuation
# becomes a space rather than being deleted, so "sensors/to hospitals" and
# "sensors to hospitals" do not collapse into "sensorstohospitals" — deleting it
# would fuse words across the gap and invent matches that are not there.
_NON_WORD = re.compile(r"[^a-z0-9]+")
_SPACES = re.compile(r"\s+")

# Padded so short titles still produce trigrams across their edges, and so two
# titles that share only a word boundary ("ai" vs "aid") are not scored as
# identical on one coincidental trigram.
_PAD = "  "


def normalize(text: str) -> str:
    """Lowercase, drop punctuation, collapse whitespace.

    Case and punctuation are not meaningful differences between two phrasings of
    one idea, and neither is whitespace layout.
    """
    return _SPACES.sub(" ", _NON_WORD.sub(" ", (text or "").lower())).strip()


def trigrams(text: str) -> set:
    """The set of character 3-grams of the normalised text.

    Character n-grams rather than word sets because word sets punish exactly the
    case that matters: "Selling sensors to hospitals" against "Sell sensors to
    hospitals" shares few whole words and almost all of its characters, so a
    word-based measure calls a clear duplicate two different ideas.
    """
    padded = _PAD + normalize(text) + _PAD
    if len(padded) < 3:
        return set()
    return {padded[i:i + 3] for i in range(len(padded) - 2)}


def similarity(left: str, right: str) -> float:
    """Jaccard similarity of two titles, 0.0 to 1.0.

    Zero when either side normalises to nothing, rather than treating two empty
    strings as identical. Not a theoretical case: the padding means `trigrams("")`
    is `{'   '}` — a single three-character trigram of the four padding spaces —
    so two untitled cards would share it completely and score 1.0, making every
    untitled card a duplicate of every other. "Both blank" is evidence of nothing.
    """
    if not normalize(left) or not normalize(right):
        return 0.0
    a, b = trigrams(left), trigrams(right)
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def find_near_duplicates(ideas, threshold: float = None) -> dict:
    """Map each idea id to the ids of the ideas that look like it.

    `ideas` is any iterable of mappings carrying `id` and `title` — RowMappings
    from `db.list_ideas` work directly.

    Symmetric: if two cards are similar enough, each lists the other, because the
    UI names the near-duplicates on the card being looked at rather than only
    declaring a winner.

    Ordering is by descending similarity, then ascending id, so the page renders
    the same way on every load and a test can assert on it.

    Returns `{}` for an empty or single-item input without touching the database.
    """
    threshold = NEAR_DUPLICATE_THRESHOLD if threshold is None else threshold
    rows = [(row["id"], row.get("title") or "") for row in ideas]
    found = {}
    for i, (left_id, left_title) in enumerate(rows):
        matches = []
        for right_id, right_title in rows[i + 1:]:
            score = similarity(left_title, right_title)
            if score >= threshold:
                # Both directions, so each card names its counterpart.
                matches.append((right_id, score))
                found.setdefault(right_id, []).append((left_id, score))
        if matches:
            found.setdefault(left_id, []).extend(matches)
    return {
        idea_id: [other_id for other_id, _ in sorted(
            matches, key=lambda pair: (-pair[1], pair[0]))]
        for idea_id, matches in found.items()
    }