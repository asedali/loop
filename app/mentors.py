"""Founder playbooks: the mentor catalogue (M2.1).

Six sets of documented operating principles, used as *attribution labels*. A
challenge applies one to a venture's real state and returns questions.

What this is not
--------------------------------------------------------------------------------
It is not a simulation of a person. No output from this module, or from the
model that reads it, speaks in a founder's voice, quotes one, or attributes a
statement to them. The name appears only as the reason a *published principle* is
being applied, and every rendered challenge carries a notice saying so.

Why that constraint, rather than a stylistic preference:

  * These are real people, some living. Invented sentences in their voice are a
    reputational and legal problem, not a novelty.
  * This app's thesis is that unevidenced assertion is the thing to strip out
    (TECHNICAL.md invariant 4, and the Phase 3 "a language model can invent
    plausible-sounding ones" warning). A mentor that asserts is the exact defect
    this codebase exists to avoid.
  * A mentor whose only power is "what is your evidence?" is aimed at the failure
    mode that actually hurts researchers: talking themselves past disconfirming
    evidence.

The `evidence` playbook is deliberately not a person. It is this app's own thesis
turned on the researcher, and on the evidence it is the strongest of the six.

See specs/11-M2.1-think-like-a-founder.md.
"""

# Attribution is written the way the page renders it: what the person is cited
# for, with no claim that they said any of it. `source` is the published work the
# principle comes from, so the citation is checkable rather than vibes — the same
# discipline the Phase 3 funding disclaimer insists on for funding programmes.
MENTORS = {
    "focus": {
        "label": "Subtraction",
        "attributed_to": "Steve Jobs",
        "source": "on focus and starting from the customer experience backwards",
        "brief": (
            "Treat focus as subtraction rather than selection, and start from the "
            "customer experience and work backwards to what has to exist."
        ),
        "lens": (
            "Look at what this venture is doing and ask what it should stop doing. "
            "Every block, channel, feature and audience that is not essential is a "
            "candidate for removal. Then ask whether the thing that remains is the "
            "thing the user actually experiences first."
        ),
    },
    "first_principles": {
        "label": "First principles",
        "attributed_to": "Elon Musk",
        "source": "on reasoning from constraints and deleting parts",
        "brief": (
            "Reason from physical and economic constraints rather than from what "
            "is normally done, and delete parts rather than improving them."
        ),
        "lens": (
            "Take the venture's costs, prices or technical assumptions apart into "
            "their components and ask what each one is *actually* constrained by, "
            "as opposed to what convention says it costs. Ask which part could be "
            "removed entirely rather than made better."
        ),
    },
    "demand": {
        "label": "Do things that don't scale",
        "attributed_to": "Paul Graham",
        "source": "on doing things that don't scale early",
        "brief": (
            "Do things manually and inefficiently for a small number of users first; "
            "automation is a sign the idea works, not a way to find out."
        ),
        "lens": (
            "Ask how many people this researcher has personally, individually "
            "spoken to about this idea, and what they did afterwards. Ask whether "
            "any part of the plan depends on reaching people the researcher has "
            "never met."
        ),
    },
    "falsify": {
        "label": "Falsify the riskiest assumption",
        "attributed_to": "Steve Ries",
        "source": "on build-measure-learn and riskiest assumptions",
        "brief": (
            "Find the assumption that, if false, ends the venture, and test that one "
            "before the ones that are merely interesting."
        ),
        "lens": (
            "Name the single assumption this venture has not tested, and ask what it "
            "would take to disprove it this week. Ask why the tested assumptions were "
            "chosen, and whether the untested one is the one that matters."
        ),
    },
    "jobs_to_be_done": {
        "label": "Demand, not features",
        "attributed_to": "Clayton Christensen",
        "source": "on customers hiring products to make progress",
        "brief": (
            "Customers do not buy products; they hire them to make progress in a "
            "situation. A competitor can be a non-consumption or a workaround."
        ),
        "lens": (
            "Ask what progress the customer is hiring this for and what they were "
            "doing before. Ask what they would do if this vanished tomorrow, and "
            "whether that alternative is good enough to survive."
        ),
    },
    "evidence": {
        "label": "The evidence insister",
        "attributed_to": "this app's own method",
        "source": "docs/TECHNICAL.md §10, invariants 3 to 6",
        "brief": (
            "A claim without a stated success criterion and a disconfirming result is "
            "not yet evidence, whatever the confidence behind it."
        ),
        "lens": (
            "Interrogate the evidence itself. Ask what result would have made the "
            "opposite conclusion, whether that result was actually possible, what the "
            "sample size supports, and whether a block marked resolved rests on a "
            "criterion that was never written down."
        ),
    },
}


def get(key: str) -> dict | None:
    """One playbook, or None for an unknown key.

    Returns None rather than raising or defaulting: an unknown key reaching the
    prompt would put a real founder's name next to generated text with no
    principle behind it, which is the failure this module exists to prevent. The
    caller refuses instead.
    """
    entry = MENTORS.get(key)
    if entry is None:
        return None
    return {
        "key": key,
        "label": entry["label"],
        "attributed_to": entry["attributed_to"],
        "source": entry["source"],
        "brief": entry["brief"],
        "lens": entry["lens"],
    }


def all_mentors() -> list:
    """Every playbook, in catalogue order, for the picker."""
    return [get(key) for key in MENTORS]


# Rendered above every challenge, and asserted by the tests. Wording is load-
# bearing: it has to say the output is generated, that the person is not
# involved, and that nothing here is a quotation. Anything vaguer would let a
# reader take a generated sentence for something a real founder said.
NOT_A_QUOTE = (
    "AI-generated. Applying published operating principles as an attribution, "
    "not advice from this person, who is not involved and has not seen this. "
    "Nothing here is a quotation."
)

# Redirect copy after a challenge lands. Says what was done and, importantly, what
# was NOT: nothing on the page changed, because a mentor has no power to change it.
CHALLENGE_READY = (
    "Challenge added below. Nothing on your venture changed — it only asks "
    "questions."
)

# Refusal for a playbook key that is not in the catalogue. Used before any prompt
# is built, so an unknown key can never reach a founder's name.
MENTOR_BAD_KEY = "That isn't a mentor playbook."
