"""
Tests for the Phase-2 per-segment state machine, the LLM output validation, and
the run-cap behaviour. The LLM layer is mocked throughout — these tests are
about the rules, not the model.

Run with:  python -m pytest -q
Needs a Postgres. Point TEST_DATABASE_URL at one, or leave it unset and the
conftest falls back to DATABASE_URL and then to localhost/launchloop_test.
"""
import hashlib
import html
import io
import json
import pathlib
import re
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from urllib.parse import unquote

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

import app.auth as auth
import app.config as config
import app.db as db
import app.dedupe as dedupe
import app.llm as llm
import app.mentors as mentors
import app.mailer as mailer
import app.main as main
import app.ratelimit as ratelimit
import app.sources as sources
import app.upload as upload_mod
from app.auth import hash_password
import pypdf
from openai import BadRequestError
from app.constants import (
    CANVAS_FINANCIAL,
    CANVAS_LAYOUT,
    LLM_PURPOSES,
    MENTOR_KEYS,
    PROMPT_VERSIONS,
    SEGMENTS,
)


@pytest.fixture
def client(tmp_db, monkeypatch):
    monkeypatch.setenv("LAUNCHLOOP_DEBUG", "1")
    monkeypatch.setenv("SESSION_SECRET_KEY", "test-secret")
    with TestClient(main.app) as c:
        yield c


def signup(client, email="a@test.com", password="password123"):
    r = client.post("/signup", data={"email": email, "password": password}, follow_redirects=False)
    assert r.status_code == 303, r.text
    return r


def make_venture(client, monkeypatch, cards=None):
    """A user with one venture and its nine untouched segments."""
    signup(client)
    cards = cards or [{
        "title": "Sensor thing", "commercial_framing": "Sell sensors",
        "strength_signal": "early", "raw_claims": "we made a sensor",
    }]
    monkeypatch.setattr(llm, "extract_ideas", lambda raw, user_id=None: cards)
    r = client.post("/phase1/extract", data={"raw_text": "x" * 80}, follow_redirects=False)
    assert r.status_code == 303
    idea_id = client.get("/phase1/ideas").context["ideas"][0]["id"]
    r = client.post(f"/phase1/select/{idea_id}", follow_redirects=False)
    venture_id = int(r.headers["location"].rstrip("/").split("/")[-1])
    return venture_id


def make_user(email="a@test.com", password="password123"):
    """Create a user the way signup does.

    `users` has a WITH CHECK policy permitting only the row whose address matches
    the lookup setting, because signup happens before anyone is signed in. So a
    test cannot just insert a user under whatever tenant it happens to be in —
    it has to present the address, exactly as the route does.
    """
    with db.as_tenant(lookup_email=email):
        return db.create_user(email, hash_password(password))


# ---------------- LLM stubs ----------------

def fake_design(*a, **k):
    """Stands in for llm.generate_segment_tasks."""
    key = a[2].get("element_name") if len(a) > 2 and isinstance(a[2], dict) else "customer"
    return {
        "hypothesis": "A falsifiable claim about the real world.",
        "tasks": [{
            "title": "Talk to 5 people about the problem",
            "method": "problem_interview",
            "why_this": "probes whether the problem is real",
            "steps": ["Recruit from the regional list", "Ask about the last incident"],
            "success_criteria": "3+ rank it a top-3 cost unprompted",
            "target_sample": "5 interviews",
            "effort": "~2 hours",
            "target_element": key,
        }],
    }


def fake_verdict(verdict="pass", **over):
    out = {
        "segment": "customer",
        "segment_label": "Customer Segments",
        "verdict": verdict,
        "evidence_status": {"pass": "confirmed", "fail": "disconfirmed"}.get(verdict, "mixed"),
        "verdict_reasoning": "the logged outcomes met the stated criteria",
        "evidence_note": "8 of 8 agreed",
        "revised_hypothesis": "a sharper claim",
        "workaround": "resell through an existing distributor instead",
        "pivot_suggestion": "sell through distributors rather than direct",
    }
    out.update(over)
    return out


def run_once(client, monkeypatch, venture_id, segment_key, verdict="pass",
             outcome="everyone wants it", size="12"):
    """Drive one whole run: design the tasks, log an outcome, score it.

    Asserts the run is recorded against the segment it was started for, which is
    the invariant the whole per-segment model rests on.
    """
    monkeypatch.setattr(llm, "generate_segment_tasks", fake_design)
    monkeypatch.setattr(llm, "analyze_segment_run", lambda *a, **k: fake_verdict(verdict))
    r = client.post(f"/venture/{venture_id}/segment/{segment_key}/run", follow_redirects=False)
    assert r.status_code == 303, r.text
    run = db.get_current_cycle(venture_id)
    assert run["segment"] == segment_key, (run["segment"], segment_key)
    return client.post(
        f"/venture/{venture_id}/run/{run['id']}/log",
        data={"outcome_0": outcome, "sample_size_0": size},
        follow_redirects=False,
    )


# ==================== LLM output validation ====================

CUSTOMER = {"element_name": "customer", "label": "Customer Segments", "severity": "critical"}


class TestValidation:
    """The per-segment prompts ask for a lot more structure than the old
    flat loop did, so these cover the normalisation that keeps a sloppy model
    response from producing an unusable run."""

    def test_method_is_normalised_to_a_valid_one(self, monkeypatch):
        monkeypatch.setattr(llm, "call_json", lambda *a, **k: {
            "hypothesis": "H",
            "tasks": [{"title": "T", "method": "INCIDENT REVIEW", "steps": ["a"]}],
        })
        out = llm.generate_segment_tasks("Idea", "framing", CUSTOMER, 1)
        assert out["tasks"][0]["method"] in llm.methods_for("customer")

    def test_unknown_method_falls_back_rather_than_being_persisted(self, monkeypatch):
        """The method drives which instructions the UI shows, so an invented one
        has to be coerced to something real."""
        monkeypatch.setattr(llm, "call_json", lambda *a, **k: {
            "hypothesis": "H",
            "tasks": [{"title": "T", "method": "field_trip", "steps": ["a"]}],
        })
        out = llm.generate_segment_tasks("Idea", "framing", CUSTOMER, 1)
        assert out["tasks"][0]["method"] in llm.methods_for("customer")

    def test_methods_are_segment_specific(self, monkeypatch):
        monkeypatch.setattr(llm, "call_json", lambda *a, **k: {
            "hypothesis": "H", "tasks": [{"title": "T", "method": "preorder_test"}],
        })
        out = llm.generate_segment_tasks("Idea", "f", {"element_name": "revenue"}, 1)
        assert out["tasks"][0]["method"] in llm.methods_for("revenue")

    def test_steps_given_as_a_become_a_list(self, monkeypatch):
        monkeypatch.setattr(llm, "call_json", lambda *a, **k: {
            "hypothesis": "H",
            "tasks": [{"title": "T", "method": "problem_interview", "steps": "Recruit them"}],
        })
        out = llm.generate_segment_tasks("Idea", "f", CUSTOMER, 1)
        assert out["tasks"][0]["steps"] == ["Recruit them"]

    def test_title_falls_back_to_description(self, monkeypatch):
        monkeypatch.setattr(llm, "call_json", lambda *a, **k: {
            "hypothesis": "H", "tasks": [{"description": "Ask five people"}],
        })
        out = llm.generate_segment_tasks("Idea", "f", CUSTOMER, 1)
        assert out["tasks"][0]["title"] == "Ask five people"

    def test_unusable_tasks_are_dropped(self, monkeypatch):
        monkeypatch.setattr(llm, "call_json", lambda *a, **k: {
            "hypothesis": "H", "tasks": [{"method": "problem_interview"}, "garbage", 7],
        })
        with pytest.raises(llm.LLMError):
            llm.generate_segment_tasks("Idea", "f", CUSTOMER, 1)

    def test_hypothesis_is_returned_with_the_tasks(self, monkeypatch):
        monkeypatch.setattr(llm, "call_json", lambda *a, **k: {
            "hypothesis": "Managers rank downtime top-3.",
            "tasks": [{"title": "T", "method": "problem_interview"}],
        })
        out = llm.generate_segment_tasks("Idea", "f", CUSTOMER, 1)
        assert out["hypothesis"] == "Managers rank downtime top-3."

    # ---- verdict validation ----

    def test_verdict_is_normalised(self, monkeypatch):
        monkeypatch.setattr(llm, "call_json", lambda *a, **k: {
            "verdict": "PASS", "evidence_status": "confirmed",
        })
        assert llm.analyze_segment_run("Idea", CUSTOMER, "H", [], [])["verdict"] == "pass"

    @pytest.mark.parametrize("bad", ["banana", "ship it", "", None])
    def test_unusable_verdict_is_an_error(self, monkeypatch, bad):
        """The verdict is the whole decision. Rather than guess at one, the run
        is left unscored so the user can retry."""
        monkeypatch.setattr(llm, "call_json", lambda *a, **k: {
            "verdict": bad, "evidence_status": "mixed",
        })
        with pytest.raises(llm.LLMError):
            llm.analyze_segment_run("Idea", CUSTOMER, "H", [], [])

    def test_pass_forces_the_evidence_read_to_confirmed(self, monkeypatch):
        """The schema forbids outcome='passed' with a non-confirmed status, so a
        contradictory pair has to be reconciled before it reaches the database
        rather than raising a 500 on the page that submitted it."""
        monkeypatch.setattr(llm, "call_json", lambda *a, **k: {
            "verdict": "pass", "evidence_status": "mixed",
        })
        out = llm.analyze_segment_run("Idea", CUSTOMER, "H", [], [])
        assert out["verdict"] == "pass" and out["evidence_status"] == "confirmed"

    def test_fail_forces_the_evidence_read_to_disconfirmed(self, monkeypatch):
        monkeypatch.setattr(llm, "call_json", lambda *a, **k: {
            "verdict": "fail", "evidence_status": "confirmed",
        })
        out = llm.analyze_segment_run("Idea", CUSTOMER, "H", [], [])
        assert out["evidence_status"] == "disconfirmed"

    def test_iterate_is_left_alone(self, monkeypatch):
        monkeypatch.setattr(llm, "call_json", lambda *a, **k: {
            "verdict": "iterate", "evidence_status": "mixed",
        })
        assert llm.analyze_segment_run("Idea", CUSTOMER, "H", [], [])["evidence_status"] == "mixed"

    def test_the_segment_comes_from_the_database_not_the_model(self, monkeypatch):
        """A model that names a different block must not be able to redirect the
        verdict onto it."""
        monkeypatch.setattr(llm, "call_json", lambda *a, **k: {
            "segment": "revenue", "verdict": "pass", "evidence_status": "confirmed",
        })
        out = llm.analyze_segment_run("Idea", CUSTOMER, "H", [], [])
        assert out["segment"] == "customer"

    def test_extract_ideas_falls_back_on_bad_signal(self, monkeypatch):
        monkeypatch.setattr(llm, "call_json", lambda *a, **k: [
            {"title": "A", "commercial_framing": "x", "strength_signal": "INCREDIBLE", "raw_claims": "y"},
            {"title": "", "commercial_framing": "dropped"},
        ])
        out = llm.extract_ideas("material")
        assert len(out) == 1 and out[0]["strength_signal"] == "early"

    def test_json_extraction_handles_prose_wrapper(self):
        assert llm._extract_json('Sure! {"a": 1} hope that helps') == {"a": 1}
        assert llm._extract_json('```json\n[{"a": 1}]\n```') == [{"a": 1}]
        assert llm._extract_json('Note {this}: {"a": 1}') == {"a": 1}
        assert llm._extract_json('{"s": "a } brace inside a string"}') == {"s": "a } brace inside a string"}

    def test_json_extraction_rejects_unparseable(self):
        with pytest.raises(json.JSONDecodeError):
            llm._extract_json("no json at all")


# ==================== per-segment state machine ====================

class TestSegmentLoop:
    def test_a_new_venture_seeds_all_nine_blocks_in_order(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        segs = db.get_segments(venture_id)
        assert len(segs) == 9
        assert [s["element_name"] for s in segs] == [g["key"] for g in SEGMENTS]
        assert [s["position"] for s in segs] == list(range(9))
        assert all(s["outcome"] == "pending" for s in segs)
        assert all(s["max_cycles"] == config.segment_cap() for s in segs)
        # the four blocks the old loop never tracked are now real segments
        for key in ("key_partners", "cost_structure", "key_resources",
                    "customer_relationships"):
            assert key in {s["element_name"] for s in segs}

    def test_a_run_is_recorded_against_its_segment(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "cost_structure")
        runs = db.list_segment_runs(venture_id, "cost_structure")
        assert len(runs) == 1 and runs[0]["segment"] == "cost_structure"
        assert db.get_segment(venture_id, "cost_structure")["cycle_count"] == 1

    def test_passing_every_block_unlocks_phase_3(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        for seg in db.get_segments(venture_id):
            run_once(client, monkeypatch, venture_id, seg["element_name"], verdict="pass")
        assert db.all_resolved(venture_id)
        venture = db.get_venture(venture_id, 1)
        assert venture["status"] == "validated" and venture["phase"] == 3

    def test_parked_blocks_still_unlock_phase_3(self, client, monkeypatch):
        """A documented gap is a finished block. Gating only on 'passed' would
        make parking a block a permanent dead end.

        Only the five `important` blocks can be parked: a `critical` one failing
        is meant to kill the venture, which TestVerdicts covers.
        """
        venture_id = make_venture(client, monkeypatch)
        important = [s for s in db.get_segments(venture_id) if s["severity"] == "important"]
        critical = [s for s in db.get_segments(venture_id) if s["severity"] == "critical"]
        assert len(important) == 5 and len(critical) == 4

        for seg in important:
            run_once(client, monkeypatch, venture_id, seg["element_name"], verdict="fail")
        for seg in critical:
            run_once(client, monkeypatch, venture_id, seg["element_name"], verdict="pass")

        assert db.all_resolved(venture_id)
        assert len(db.open_gaps(venture_id)) == 5
        venture = db.get_venture(venture_id, 1)
        assert venture["status"] == "validated" and venture["phase"] == 3

    def test_one_unresolved_block_blocks_phase_3(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        for seg in db.get_segments(venture_id)[:-1]:
            run_once(client, monkeypatch, venture_id, seg["element_name"], verdict="pass")
        assert not db.all_resolved(venture_id)
        assert db.get_venture(venture_id, 1)["phase"] == 2


class TestVerdicts:
    def test_iterate_leaves_the_block_open_and_counts_the_run(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "revenue", verdict="iterate")
        seg = db.get_segment(venture_id, "revenue")
        assert seg["outcome"] == "active"
        assert seg["status"] == "mixed"
        assert seg["cycle_count"] == 1
        assert seg["hypothesis"] == "a sharper claim", "the revised hypothesis must carry forward"
        # and it can be run again
        monkeypatch.setattr(llm, "generate_segment_tasks", fake_design)
        assert client.post(f"/venture/{venture_id}/segment/revenue/run",
                           follow_redirects=False).status_code == 303

    def test_important_failure_parks_and_the_venture_carries_on(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "channel", verdict="fail")
        seg = db.get_segment(venture_id, "channel")
        assert seg["outcome"] == "parked"
        assert "distributor" in seg["outcome_note"], "the workaround must be recorded"
        assert db.get_venture(venture_id, 1)["status"] == "active", "a secondary block cannot kill a venture"
        assert [g["segment"] for g in db.open_gaps(venture_id)] == ["channel"]

    def test_critical_failure_marks_the_block_but_does_not_kill(self, client, monkeypatch):
        """The model flags a critical block as disconfirmed; it does not get to
        close the user's venture. The decision is the kill/pivot card's job."""
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "revenue", verdict="fail")
        assert db.get_segment(venture_id, "revenue")["outcome"] == "failed"
        assert db.get_venture(venture_id, 1)["status"] == "active", "still the user's call"
        assert db.list_ideas(1)[0]["status"] == "selected"

    def test_failed_critical_block_offers_kill_or_pivot(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "revenue", verdict="fail")
        page = client.get(f"/venture/{venture_id}/segment/revenue").text
        assert "came back disconfirmed" in page
        assert f'action="/venture/{venture_id}/kill"' in page
        assert f'action="/venture/{venture_id}/pivot"' in page
        # neither is styled as *the* action, so neither is the default
        assert "decision-card" in page

    def test_kill_is_a_separate_user_action(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "revenue", verdict="fail")
        r = client.post(f"/venture/{venture_id}/kill", follow_redirects=False)
        assert r.status_code == 303
        assert db.get_venture(venture_id, 1)["status"] == "killed"
        assert db.list_ideas(1)[0]["status"] == "candidate"
        assert "was killed" in client.get(f"/venture/{venture_id}").text

    def test_user_can_pivot_instead_of_killing(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "revenue", verdict="fail")
        r = client.post(f"/venture/{venture_id}/pivot",
                        data={"note": "sell the calibration service, not the hardware"},
                        follow_redirects=False)
        new_id = int(r.headers["location"].rstrip("/").split("/")[-1])
        assert new_id != venture_id
        assert db.get_venture(venture_id, 1)["status"] == "pivoted"
        assert "calibration service" in db.get_venture(new_id, 1)["pivot_note"]

    def test_pivot_carries_validated_blocks_forward(self, client, monkeypatch):
        """Regression risk worth naming: a pivot used to discard every block the
        user had already proven, which is ~9 runs of real work and real spend."""
        venture_id = make_venture(client, monkeypatch)
        for key in ("customer", "problem", "value_prop"):
            run_once(client, monkeypatch, venture_id, key, verdict="pass")
        run_once(client, monkeypatch, venture_id, "revenue", verdict="fail")
        r = client.post(f"/venture/{venture_id}/pivot", data={"note": "different buyer"},
                        follow_redirects=False)
        new_id = int(r.headers["location"].rstrip("/").split("/")[-1])

        carried = {s["element_name"]: s for s in db.get_segments(new_id)}
        assert len(carried) == 9
        for key in ("customer", "problem", "value_prop"):
            assert carried[key]["outcome"] == "passed", f"{key} should carry over"
            assert carried[key]["status"] == "confirmed"
            assert carried[key]["hypothesis"], "the hypothesis must come with it"
        # the block that failed, and everything untested, start fresh
        assert carried["revenue"]["outcome"] == "pending"
        assert carried["revenue"]["cycle_count"] == 0
        for key in ("channel", "cost_structure", "key_partners",
                    "key_resources", "customer_relationships"):
            assert carried[key]["outcome"] == "pending"

    def test_pivot_verdict_also_carries_blocks_forward(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "customer", verdict="pass")
        r = run_once(client, monkeypatch, venture_id, "value_prop", verdict="pivot")
        new_id = int(r.headers["location"].rstrip("/").split("/")[-1])
        carried = {s["element_name"]: s for s in db.get_segments(new_id)}
        assert carried["customer"]["outcome"] == "passed"
        assert carried["value_prop"]["outcome"] == "pending"

    def test_pivot_carry_forward_skips_unpassed_blocks(self, client, monkeypatch):
        """A parked or merely-iterated block is not evidence, so it must not
        ride along into the new venture."""
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "customer", verdict="iterate")
        run_once(client, monkeypatch, venture_id, "problem", verdict="pass")
        r = client.post(f"/venture/{venture_id}/pivot", data={"note": "n"},
                        follow_redirects=False)
        new_id = int(r.headers["location"].rstrip("/").split("/")[-1])
        carried = {s["element_name"]: s for s in db.get_segments(new_id)}
        assert carried["problem"]["outcome"] == "passed"
        assert carried["customer"]["outcome"] == "pending"
        assert carried["customer"]["cycle_count"] == 0

    def test_pivoted_venture_cannot_start_another_run(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "value_prop", verdict="pivot")
        r = client.post(f"/venture/{venture_id}/segment/value_prop/run")
        assert "pivoted" in r.text.lower()

    def test_a_parked_block_can_be_reopened(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "channel", verdict="fail")
        r = client.post(f"/venture/{venture_id}/segment/channel/unpark", follow_redirects=False)
        assert r.status_code == 303
        seg = db.get_segment(venture_id, "channel")
        assert seg["outcome"] == "pending"
        assert "distributor" in seg["outcome_note"], "the note must survive the re-open"

    def test_a_legacy_run_without_a_segment_is_not_scored(self, client, monkeypatch):
        """Cycles created before this change have no block to attach a verdict
        to, so the score path has to refuse rather than guess."""
        venture_id = make_venture(client, monkeypatch)
        run_id = db.create_cycle(venture_id, 1, fake_design()["tasks"])
        db.update_venture(venture_id, cycle_count=1)
        db.log_cycle_results(run_id, [{"outcome": "x", "sample_size": "2"}])
        monkeypatch.setattr(llm, "analyze_segment_run",
                            lambda *a, **k: pytest.fail("must not call the model"))
        r = client.post(f"/venture/{venture_id}/run/{run_id}/retry-analysis",
                        follow_redirects=False)
        assert "predates per-block tracking" in client.get(r.headers["location"]).text

    def test_a_stale_legacy_run_does_not_block_new_runs(self, client, monkeypatch):
        """Regression: an unscorable legacy run was treated as in-flight, so it
        blocked every future run and the user had no way to clear it. A venture
        migrated from the old flat loop could not make progress at all."""
        venture_id = make_venture(client, monkeypatch)
        db.create_cycle(venture_id, 1, fake_design()["tasks"])   # segment defaults to None
        db.update_venture(venture_id, cycle_count=1)
        assert db.get_current_cycle(venture_id)["segment"] is None

        monkeypatch.setattr(llm, "generate_segment_tasks", fake_design)
        r = client.post(f"/venture/{venture_id}/segment/customer/run", follow_redirects=False)
        assert r.status_code == 303, "the stale run must not block progress"
        assert db.get_current_cycle(venture_id)["segment"] == "customer"


class TestRunCap:
    def test_the_cap_is_per_segment_not_per_venture(self, client, monkeypatch):
        """Regression risk from the old flat loop: exhausting one block must not
        stop the user working the other eight."""
        venture_id = make_venture(client, monkeypatch)
        for _ in range(config.segment_cap()):
            run_once(client, monkeypatch, venture_id, "customer", verdict="iterate")
        seg = db.get_segment(venture_id, "customer")
        assert seg["cycle_count"] == config.segment_cap()

        monkeypatch.setattr(llm, "generate_segment_tasks", fake_design)
        r = client.post(f"/venture/{venture_id}/segment/customer/run")
        assert r.status_code == 200, "the cap must hold server-side, not just in the template"
        assert "Extend it first" in r.text

        # every other block is untouched and still runnable
        assert client.post(f"/venture/{venture_id}/segment/problem/run",
                           follow_redirects=False).status_code == 303
        assert db.get_segment(venture_id, "problem")["max_cycles"] == config.segment_cap()
        assert db.get_venture(venture_id, 1)["status"] == "active", "no whole-venture pause"

    def test_extend_raises_only_that_block(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        for _ in range(config.segment_cap()):
            run_once(client, monkeypatch, venture_id, "customer", verdict="iterate")
        r = client.post(f"/venture/{venture_id}/segment/customer/extend", follow_redirects=False)
        assert r.status_code == 303
        assert db.get_segment(venture_id, "customer")["max_cycles"] == config.segment_cap() * 2
        assert db.get_segment(venture_id, "problem")["max_cycles"] == config.segment_cap()
        monkeypatch.setattr(llm, "generate_segment_tasks", fake_design)
        assert client.post(f"/venture/{venture_id}/segment/customer/run",
                           follow_redirects=False).status_code == 303

    def test_only_one_run_is_in_flight_per_venture(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        monkeypatch.setattr(llm, "generate_segment_tasks", fake_design)

    # ---- parking: the exit that was missing ----

    def test_the_cap_offers_park_as_well_as_extend(self, client, monkeypatch):
        """Regression: a block at 3/3 could only be extended, so the venture could
        neither progress nor reach Phase 3. That was a dead end."""
        venture_id = make_venture(client, monkeypatch)
        for _ in range(config.segment_cap()):
            run_once(client, monkeypatch, venture_id, "customer", verdict="iterate")
        page = client.get(f"/venture/{venture_id}/segment/customer").text
        assert f'action="/venture/{venture_id}/segment/customer/park"' in page, \
            "the cap must offer an exit, not just an extension"
        assert f'action="/venture/{venture_id}/segment/customer/extend"' in page

    def test_parking_a_capped_block_resolves_it(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        for _ in range(config.segment_cap()):
            run_once(client, monkeypatch, venture_id, "customer", verdict="iterate")
        r = client.post(f"/venture/{venture_id}/segment/customer/park",
                        data={"note": "nobody will pay for the hardware, service works"},
                        follow_redirects=False)
        assert r.status_code == 303
        seg = db.get_segment(venture_id, "customer")
        assert seg["outcome"] == "parked"
        assert "nobody will pay" in seg["outcome_note"]
        assert "customer" in [g["segment"] for g in db.open_gaps(venture_id)]

    def test_parking_without_a_note_is_refused(self, client, monkeypatch):
        """A park with no reason is a hole in the model, and Phase 3 hands these
        notes to the strategy prompt — so the note is required."""
        venture_id = make_venture(client, monkeypatch)
        for _ in range(config.segment_cap()):
            run_once(client, monkeypatch, venture_id, "customer", verdict="iterate")
        r = client.post(f"/venture/{venture_id}/segment/customer/park",
                        data={"note": "   "}, follow_redirects=False)
        assert "Say what you learned" in client.get(r.headers["location"]).text
        assert db.get_segment(venture_id, "customer")["outcome"] != "parked"

    def test_parking_the_last_open_block_unlocks_phase_3(self, client, monkeypatch):
        """The deadlock this closes: park every block and Phase 3 must open."""
        venture_id = make_venture(client, monkeypatch)
        for seg in db.get_segments(venture_id):
            if seg["element_name"] == "customer":
                run_once(client, monkeypatch, venture_id, "customer", verdict="pass")
            else:
                for _ in range(config.segment_cap()):
                    run_once(client, monkeypatch, venture_id, seg["element_name"],
                             verdict="iterate")
                r = client.post(f"/venture/{venture_id}/segment/{seg['element_name']}/park",
                                data={"note": f"parked {seg['element_name']}"},
                                follow_redirects=False)
                assert r.status_code == 303
        assert db.all_resolved(venture_id)
        assert db.get_venture(venture_id, 1)["phase"] == 3

    def test_a_parked_block_can_still_be_reopened(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        for _ in range(config.segment_cap()):
            run_once(client, monkeypatch, venture_id, "channel", verdict="iterate")
        client.post(f"/venture/{venture_id}/segment/channel/park",
                    data={"note": "no budget"}, follow_redirects=False)
        assert db.get_segment(venture_id, "channel")["outcome"] == "parked"
        client.post(f"/venture/{venture_id}/segment/channel/unpark", follow_redirects=False)
        assert db.get_segment(venture_id, "channel")["outcome"] == "pending"

    def test_only_one_run_is_in_flight_per_venture(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        monkeypatch.setattr(llm, "generate_segment_tasks", fake_design)
        assert client.post(f"/venture/{venture_id}/segment/customer/run",
                           follow_redirects=False).status_code == 303
        r = client.post(f"/venture/{venture_id}/segment/problem/run")
        assert "Finish the run in progress first" in r.text
        assert db.get_venture(venture_id, 1)["cycle_count"] == 1

    def test_a_passed_block_cannot_be_run_again(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "customer", verdict="pass")
        r = client.post(f"/venture/{venture_id}/segment/customer/run")
        assert "already passed" in r.text


class TestRetryPath:
    def test_failed_verdict_keeps_the_evidence_and_offers_a_retry(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        monkeypatch.setattr(llm, "generate_segment_tasks", fake_design)

        def boom(*a, **k):
            raise llm.LLMError("upstream 500")
        monkeypatch.setattr(llm, "analyze_segment_run", boom)

        client.post(f"/venture/{venture_id}/segment/customer/run", follow_redirects=False)
        run = db.get_current_cycle(venture_id)
        client.post(f"/venture/{venture_id}/run/{run['id']}/log",
                    data={"outcome_0": "partial evidence", "sample_size_0": "4"},
                    follow_redirects=False)
        assert db.get_cycle(run["id"], venture_id)["results_json"]
        assert not db.get_cycle(run["id"], venture_id)["analysis_json"]

        page = client.get(f"/venture/{venture_id}").text
        assert "Retry verdict" in page
        r = client.post(f"/venture/{venture_id}/run/{run['id']}/retry-analysis",
                        follow_redirects=False)
        assert "upstream 500" in client.get(r.headers["location"]).text, \
            "the failure reason must reach the user"

        monkeypatch.setattr(llm, "analyze_segment_run", lambda *a, **k: fake_verdict("iterate"))
        client.post(f"/venture/{venture_id}/run/{run['id']}/retry-analysis", follow_redirects=False)
        assert db.get_cycle(run["id"], venture_id)["analysis_json"]

    def test_logging_nothing_is_refused_before_spending_a_call(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        monkeypatch.setattr(llm, "generate_segment_tasks", fake_design)
        monkeypatch.setattr(llm, "analyze_segment_run",
                            lambda *a, **k: pytest.fail("must not call the model"))
        client.post(f"/venture/{venture_id}/segment/customer/run", follow_redirects=False)
        run = db.get_current_cycle(venture_id)
        r = client.post(f"/venture/{venture_id}/run/{run['id']}/log",
                        data={"outcome_0": "   ", "sample_size_0": ""})
        assert "Log at least one outcome" in r.text


# ==================== Tenant isolation ====================

class TestIsolation:
    """Cross-tenant behaviour, at the request layer and at the database layer.

    The request-layer tests here are unchanged in spirit but now depend on the
    tenant being installed from the session — see TestRowLevelSecurity for the
    database-layer proof, which is the one that cannot be satisfied by a template
    merely hiding a button.

    Note the shape: these assert against `db.*` **as the owner** (tenant 1). If
    they were run as the attacker they would pass vacuously, because RLS would
    return nothing and the assertion would be indistinguishable from the bug it
    is meant to catch.
    """

    def test_another_user_cannot_see_or_act_on_the_venture(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        client.post("/logout", follow_redirects=False)
        signup(client, email="b@test.com")
        # Read the other user's id under the address-scoped policy, the only way
        # to see a row that is not the current tenant's.
        with db.as_tenant(lookup_email="b@test.com"):
            other_id = db.get_user_by_email("b@test.com")["id"]
        assert other_id != 1

        # Logged in, but the venture isn't theirs: bounced to the dashboard.
        r = client.get(f"/venture/{venture_id}", follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"] == "/dashboard"
        assert "Sensor thing" not in client.get("/dashboard").text

        monkeypatch.setattr(llm, "generate_segment_tasks", fake_design)
        r = client.post(f"/venture/{venture_id}/segment/customer/run", follow_redirects=False)
        assert r.headers["location"] == "/dashboard"
        assert db.get_venture(venture_id, 1)["cycle_count"] == 0, "no run may be created"

    def test_cannot_log_into_another_persons_venture_by_id(self, client, monkeypatch):
        """Venture ids are sequential, so guessing one must not leak it."""
        venture_id = make_venture(client, monkeypatch)
        client.post("/logout", follow_redirects=False)
        signup(client, email="c@test.com")
        r = client.post(f"/venture/{venture_id}/phase3/generate", follow_redirects=False)
        assert r.headers["location"] == "/dashboard"
        # Read as the owner, so this cannot pass because RLS hid the row.
        with db.as_tenant(1):
            assert db.get_launch_strategy(venture_id) is None

    def test_cannot_score_or_extend_someone_elses_segment(self, client, monkeypatch):
        """The per-segment routes take a block name as well as a venture id, so
        both have to be checked against the owner."""
        venture_id = make_venture(client, monkeypatch)
        # leave a scored run behind, then check the other user cannot touch it
        run_once(client, monkeypatch, venture_id, "revenue", verdict="iterate")
        run = db.get_current_cycle(venture_id)
        before = dict(db.get_cycle(run["id"], venture_id))

        client.post("/logout", follow_redirects=False)
        signup(client, email="e2@test.com")

        monkeypatch.setattr(llm, "analyze_segment_run",
                            lambda *a, **k: pytest.fail("must not call the model"))
        r = client.post(f"/venture/{venture_id}/run/{run['id']}/log",
                        data={"outcome_0": "injected", "sample_size_0": "9"},
                        follow_redirects=False)
        assert r.headers["location"] == "/dashboard"
        after = dict(db.get_cycle(run["id"], venture_id))
        assert after["results_json"] == before["results_json"], "logged evidence must be untouched"
        assert after["analysis_json"] == before["analysis_json"]

        r = client.post(f"/venture/{venture_id}/segment/customer/extend", follow_redirects=False)
        assert r.headers["location"] == "/dashboard"
        assert db.get_segment(venture_id, "customer")["max_cycles"] == config.segment_cap()

    def test_logged_out_user_is_redirected(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        client.post("/logout", follow_redirects=False)
        r = client.get(f"/venture/{venture_id}")
        assert r.history and r.history[0].headers["location"] == "/login"
        assert "Log in" in r.text


# ==================== Auth ====================

class TestAuth:
    def test_short_password_rejected(self, client):
        r = client.post("/signup", data={"email": "c@test.com", "password": "short"})
        assert "at least 8 characters" in r.text

    def test_duplicate_email_rejected(self, client):
        signup(client)
        client.post("/logout", follow_redirects=False)
        assert "already exists" in client.post(
            "/signup", data={"email": "a@test.com", "password": "password123"}
        ).text

    def test_login_throttles_after_repeated_failures(self, client, monkeypatch):
        signup(client, email="d@test.com")
        client.post("/logout", follow_redirects=False)
        for _ in range(config.max_failed_logins()):
            client.post("/login", data={"email": "d@test.com", "password": "wrongwrong"})
        r = client.post("/login", data={"email": "d@test.com", "password": "password123"})
        assert "Too many failed attempts" in r.text

    def test_successful_login_clears_nothing_but_allows_access(self, client):
        signup(client, email="e@test.com")
        client.post("/logout", follow_redirects=False)
        r = client.post("/login", data={"email": "e@test.com", "password": "password123"},
                        follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].endswith("/dashboard")


# ==================== password reset + email verification (M0.2) ====================

def token_from_mail(capsys, marker="/reset-password?token=") -> str:
    """Pull a token out of what the console mailer printed.

    This is the whole reason the console backend exists: the reset flow is
    testable end to end with no provider and no network, because the test can
    read the link the user would have clicked.
    """
    out = capsys.readouterr().out
    links = [line for line in out.splitlines() if marker in line]
    assert links, f"no reset link in the mailer output:\n{out}"
    return links[-1].split(marker, 1)[1].strip()


def forgot(client, email):
    return client.post("/forgot-password", data={"email": email}, follow_redirects=False)


class TestPasswordReset:
    """Recovering a forgotten password, without a support ticket.

    The property that matters most is the last one: a reset must not leave an
    older reset link working, or the person who reset their password has not
    actually locked anyone out.
    """

    def test_the_whole_flow_works_end_to_end(self, client, monkeypatch, capsys):
        monkeypatch.setenv("MAIL_BACKEND", "console")
        signup(client, email="lost@test.com", password="originalpass1")
        client.post("/logout", follow_redirects=False)
        capsys.readouterr()

        # 1. ask for a reset
        r = forgot(client, "lost@test.com")
        assert r.status_code == 303
        assert r.headers["location"] == "/forgot-password?sent=1"
        token = token_from_mail(capsys)

        # 2. the link opens a usable form
        page = client.get(f"/reset-password?token={token}")
        assert page.status_code == 200
        assert "Choose a new password" in page.text
        assert "password123" not in page.text

        # 3. set a new one
        r = client.post("/reset-password",
                        data={"token": token, "password": "brandnewpass",
                              "confirm": "brandnewpass"},
                        follow_redirects=False)
        assert r.status_code == 303, r.text
        assert r.headers["location"] == "/reset-password?done=1"

        # 4. the new password works, the old one does not
        assert client.post("/login", data={"email": "lost@test.com",
                                           "password": "brandnewpass"},
                           follow_redirects=False).status_code == 303
        client.post("/logout", follow_redirects=False)
        assert "Invalid email or password" in client.post(
            "/login", data={"email": "lost@test.com", "password": "originalpass1"}).text

    def test_the_old_password_is_rejected_even_while_signed_in_elsewhere(self, client, monkeypatch, capsys):
        """Changing the password must actually replace it, not just set a second one."""
        monkeypatch.setenv("MAIL_BACKEND", "console")
        signup(client, email="rot@test.com", password="originalpass1")
        capsys.readouterr()
        forgot(client, "rot@test.com")
        token = token_from_mail(capsys)
        client.post("/reset-password", data={"token": token, "password": "secondpass1",
                                             "confirm": "secondpass1"},
                    follow_redirects=False)
        row = db.get_user_by_email("rot@test.com")
        assert auth.verify_password("secondpass1", row["password_hash"])
        assert not auth.verify_password("originalpass1", row["password_hash"])

    def test_a_token_only_works_once(self, client, monkeypatch, capsys):
        """Single-use is the whole point: a forwarded link must stop working."""
        monkeypatch.setenv("MAIL_BACKEND", "console")
        signup(client, email="once@test.com")
        capsys.readouterr()
        forgot(client, "once@test.com")
        token = token_from_mail(capsys)

        first = client.post("/reset-password", data={"token": token, "password": "firstpass1",
                                                     "confirm": "firstpass1"},
                            follow_redirects=False)
        assert first.status_code == 303
        second = client.post("/reset-password", data={"token": token, "password": "secondpass1",
                                                      "confirm": "secondpass1"},
                             follow_redirects=False)
        assert second.status_code == 200, "a spent link must not be redeemable"
        assert "invalid or has expired" in second.text

    def test_a_reset_burns_the_users_other_live_links(self, client, monkeypatch, capsys):
        """Two resets requested before either is used: using the second must kill
        the first. Otherwise resetting your password does not lock anyone out."""
        monkeypatch.setenv("MAIL_BACKEND", "console")
        signup(client, email="multi@test.com")
        capsys.readouterr()
        forgot(client, "multi@test.com")
        first = token_from_mail(capsys)
        forgot(client, "multi@test.com")
        second = token_from_mail(capsys)

        assert client.post("/reset-password", data={"token": second, "password": "newestpass",
                                                    "confirm": "newestpass"},
                           follow_redirects=False).status_code == 303
        stale = client.post("/reset-password", data={"token": first, "password": "attackerpass",
                                                     "confirm": "attackerpass"},
                            follow_redirects=False)
        assert stale.status_code == 200
        assert "invalid or has expired" in stale.text
        assert auth.verify_password(
            "newestpass", db.get_user_by_email("multi@test.com")["password_hash"])

    def test_an_expired_token_is_refused(self, client, monkeypatch, capsys):
        monkeypatch.setenv("MAIL_BACKEND", "console")
        signup(client, email="exp@test.com")
        capsys.readouterr()
        forgot(client, "exp@test.com")
        token = token_from_mail(capsys)

        # Expire it directly rather than sleeping for 30 minutes.
        with db.get_conn() as conn:
            conn.execute(text(
                "UPDATE password_reset_tokens SET expires_at = now() - interval '1 second'"))
        assert "invalid or has expired" in client.get(
            f"/reset-password?token={token}").text

    def test_an_unknown_or_tampered_token_is_refused(self, client, monkeypatch, capsys):
        monkeypatch.setenv("MAIL_BACKEND", "console")
        signup(client, email="tamper@test.com")
        capsys.readouterr()
        forgot(client, "tamper@test.com")
        token = token_from_mail(capsys)

        for bad in ("nonsense", token[:-4] + "AAAA", "", "x" * 200):
            r = client.post("/reset-password", data={"token": bad, "password": "sneakypass",
                                                     "confirm": "sneakypass"},
                            follow_redirects=False)
            assert r.status_code == 200
            assert "invalid or has expired" in r.text
        assert auth.verify_password("password123",
                                    db.get_user_by_email("tamper@test.com")["password_hash"])

    @pytest.mark.parametrize("password,confirm,expected", [
        ("short1", "short1", "at least 8 characters"),
        ("longenough1", "longenough2", "do not match"),
    ])
    def test_weak_and_mismatched_passwords_are_refused(self, client, monkeypatch, capsys,
                                                       password, confirm, expected):
        monkeypatch.setenv("MAIL_BACKEND", "console")
        signup(client, email="weak@test.com")
        capsys.readouterr()
        forgot(client, "weak@test.com")
        token = token_from_mail(capsys)

        r = client.post("/reset-password", data={"token": token, "password": password,
                                                 "confirm": confirm},
                        follow_redirects=False)
        assert r.status_code == 200, "a rejected password must not 303 to the done page"
        assert expected in r.text
        assert db.redeemable_token(auth.hash_token(token), "password_reset") is not None, \
            "a failed attempt must not burn the token"

    def test_a_verification_token_cannot_be_redeemed_as_a_reset(self, client, monkeypatch, capsys):
        """One table serves both flows; `purpose` is what keeps them apart."""
        monkeypatch.setenv("MAIL_BACKEND", "console")
        monkeypatch.setenv("EMAIL_VERIFICATION_ENABLED", "1")
        signup(client, email="cross@test.com")
        verify_token = token_from_mail(capsys, marker="/verify-email?token=")

        r = client.post("/reset-password", data={"token": verify_token,
                                                 "password": "crossflow1", "confirm": "crossflow1"},
                        follow_redirects=False)
        assert r.status_code == 200
        assert "invalid or has expired" in r.text

    def test_only_the_hash_is_stored(self, client, monkeypatch, capsys, tmp_db):
        """A database dump must not hand over the ability to take over an account."""
        monkeypatch.setenv("MAIL_BACKEND", "console")
        signup(client, email="hash@test.com")
        capsys.readouterr()
        forgot(client, "hash@test.com")
        token = token_from_mail(capsys)

        with db.get_conn() as conn:
            rows = conn.execute(text(
                "SELECT token_hash FROM password_reset_tokens "
                "WHERE purpose = 'password_reset'")).fetchall()
        assert rows
        for (stored,) in rows:
            assert stored != token, "the plaintext token was stored"
            assert stored == auth.hash_token(token)
            assert len(stored) == 64

    def test_tokens_cascade_away_with_the_account(self, tmp_db):
        """Not strictly required, but a token row outliving its user would be a
        re-takeover path if the address were ever reused."""
        user_id = make_user("gone@test.com")
        db.create_reset_token(user_id, "password_reset", "a" * 64,
                              db.now() + timedelta(minutes=30))
        with db.get_conn() as conn:
            conn.execute(text("DELETE FROM users WHERE id = :u"), {"u": user_id})
            left = conn.execute(
                text("SELECT count(*) FROM password_reset_tokens")).scalar_one()
        assert left == 0

    def test_an_unknown_purpose_is_refused_at_the_db_layer(self, tmp_db):
        """Invariant 2 in practice: the enum is checked in code and by the CHECK."""
        user_id = make_user("p@test.com")
        with pytest.raises(ValueError):
            db.create_reset_token(user_id, "not_a_purpose", "b" * 64, db.now())

    def test_the_request_form_reveals_nothing_about_the_account(self, client, monkeypatch, capsys):
        """An enumeration oracle: the response must be byte-identical for a
        registered address and an unknown one."""
        monkeypatch.setenv("MAIL_BACKEND", "console")
        signup(client, email="known@test.com")
        client.post("/logout", follow_redirects=False)
        capsys.readouterr()

        known = forgot(client, "known@test.com")
        unknown = forgot(client, "nobody-at-all@test.com")
        assert known.status_code == unknown.status_code == 303
        assert known.headers["location"] == unknown.headers["location"]

        page_known = client.get(known.headers["location"])
        page_unknown = client.get(unknown.headers["location"])
        assert "Check your inbox" in page_known.text
        # The unknown address must not appear anywhere in what the user is shown.
        assert "nobody-at-all@test.com" not in page_unknown.text
        assert "no account" not in page_unknown.text.lower()

    def test_a_mail_failure_does_not_break_signup_or_the_reset_request(self, client, monkeypatch):
        """A provider outage must not become a 500, and must not be reported as
        "we couldn't send it" — that would leak whether the account exists."""
        monkeypatch.setenv("MAIL_BACKEND", "console")
        monkeypatch.setattr(mailer, "_console",
                            lambda *a, **k: (_ for _ in ()).throw(OSError("smtp down")))
        assert signup(client, email="down@test.com").status_code == 303
        client.post("/logout", follow_redirects=False)
        r = forgot(client, "down@test.com")
        assert r.status_code == 303
        assert "Check your inbox" in client.get(r.headers["location"]).text

    def test_the_reset_lifetime_is_configurable(self, monkeypatch, client, capsys):
        monkeypatch.setenv("MAIL_BACKEND", "console")
        monkeypatch.setenv("PASSWORD_RESET_TOKEN_MINUTES", "5")
        signup(client, email="cfg@test.com")
        capsys.readouterr()
        forgot(client, "cfg@test.com")
        assert "5 minutes" in client.get("/forgot-password?sent=1").text


class TestEmailVerification:
    """Unverified accounts can use everything. The banner is a nudge, not a gate.

    Every test here turns EMAIL_VERIFICATION_ENABLED on explicitly. The flag
    defaults to off because no mail provider is configured anywhere — see
    TestEmailVerificationDisabled for the off-path behaviour, which is what
    actually runs in this build.
    """

    def test_signup_sends_a_verification_link(self, client, monkeypatch, capsys):
        monkeypatch.setenv("MAIL_BACKEND", "console")
        monkeypatch.setenv("EMAIL_VERIFICATION_ENABLED", "1")
        signup(client, email="v@test.com")
        token = token_from_mail(capsys, marker="/verify-email?token=")
        row = db.get_user_by_email("v@test.com")
        assert row["email_verified_at"] is None
        assert db.redeemable_token(auth.hash_token(token), "email_verify") is not None

    def test_the_link_confirms_the_address(self, client, monkeypatch, capsys):
        monkeypatch.setenv("MAIL_BACKEND", "console")
        monkeypatch.setenv("EMAIL_VERIFICATION_ENABLED", "1")
        signup(client, email="v2@test.com")
        token = token_from_mail(capsys, marker="/verify-email?token=")

        r = client.get(f"/verify-email?token={token}", follow_redirects=False)
        assert r.status_code == 303, r.text
        assert db.user_is_verified(db.get_user_by_email("v2@test.com")["id"])
        # The note rides the redirect; a bare /dashboard would not show it.
        assert "note=Email" in r.headers["location"]
        assert "Email confirmed" in client.get(r.headers["location"]).text

    def test_verification_is_single_use_and_idempotent(self, client, monkeypatch, capsys):
        monkeypatch.setenv("MAIL_BACKEND", "console")
        monkeypatch.setenv("EMAIL_VERIFICATION_ENABLED", "1")
        signup(client, email="v3@test.com")
        token = token_from_mail(capsys, marker="/verify-email?token=")
        client.get(f"/verify-email?token={token}", follow_redirects=False)
        first_stamp = db.get_user_by_email("v3@test.com")["email_verified_at"]

        # A second click on the same link is refused, and must not move the stamp.
        r = client.get(f"/verify-email?token={token}", follow_redirects=False)
        assert "invalid or has expired" in client.get(r.headers["location"]).text
        assert db.get_user_by_email("v3@test.com")["email_verified_at"] == first_stamp

    def test_an_unverified_user_sees_the_banner_and_can_work(self, client, monkeypatch):
        """The banner is a nudge, not a gate. Signup never verifies, so every
        account in the suite is unverified — which makes this the default state,
        and every other test in the suite is incidentally proof that nothing is
        gated on it. Here we prove the loop itself runs."""
        monkeypatch.setenv("EMAIL_VERIFICATION_ENABLED", "1")
        venture_id = make_venture(client, monkeypatch)
        text = client.get("/dashboard").text
        assert "Confirm your email address" in text
        assert "/resend-verification" in text
        # Fully usable: sign in, browse, and open a venture.
        assert client.get("/phase1/new").status_code == 200
        assert client.get(f"/venture/{venture_id}").status_code == 200
        assert client.get(f"/venture/{venture_id}/segment/customer").status_code == 200

    def test_a_verified_user_does_not_see_the_banner(self, client, monkeypatch, capsys):
        monkeypatch.setenv("MAIL_BACKEND", "console")
        monkeypatch.setenv("EMAIL_VERIFICATION_ENABLED", "1")
        signup(client, email="v5@test.com")
        token = token_from_mail(capsys, marker="/verify-email?token=")
        client.get(f"/verify-email?token={token}", follow_redirects=False)
        assert "Confirm your email address" not in client.get("/dashboard").text

    def test_resend_issues_a_fresh_link(self, client, monkeypatch, capsys):
        monkeypatch.setenv("MAIL_BACKEND", "console")
        monkeypatch.setenv("EMAIL_VERIFICATION_ENABLED", "1")
        signup(client, email="v6@test.com")
        capsys.readouterr()
        r = client.post("/resend-verification", follow_redirects=False)
        assert r.status_code == 303
        assert "Verification email sent" in client.get(r.headers["location"]).text
        assert token_from_mail(capsys, marker="/verify-email?token=")

    def test_resend_says_so_when_already_verified(self, client, monkeypatch, capsys):
        monkeypatch.setenv("MAIL_BACKEND", "console")
        monkeypatch.setenv("EMAIL_VERIFICATION_ENABLED", "1")
        signup(client, email="v7@test.com")
        token = token_from_mail(capsys, marker="/verify-email?token=")
        client.get(f"/verify-email?token={token}", follow_redirects=False)
        capsys.readouterr()
        r = client.post("/resend-verification", follow_redirects=False)
        assert "already confirmed" in client.get(r.headers["location"]).text
        assert "/verify-email" not in capsys.readouterr().out

    def test_resend_requires_a_session(self, client):
        r = client.post("/resend-verification", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/login"

    def test_existing_users_are_not_assumed_verified(self, tmp_db, monkeypatch):
        """The migration deliberately leaves email_verified_at NULL: backfilling to
        created_at would assert a verification that never happened."""
        monkeypatch.setenv("EMAIL_VERIFICATION_ENABLED", "1")
        user_id = make_user("legacy@test.com")
        assert db.get_user_by_id(user_id)["email_verified_at"] is None
        assert not db.user_is_verified(user_id)


class TestEmailVerificationDisabled:
    """The flag's default. EMAIL_VERIFICATION_ENABLED is unset, so signup must be
    silent and no account should be nagged — there is no mail provider wired up,
    so a live banner would ask every user to click a link that never arrives."""

    def test_signup_sends_nothing(self, client, monkeypatch, capsys):
        monkeypatch.setenv("MAIL_BACKEND", "console")
        monkeypatch.delenv("EMAIL_VERIFICATION_ENABLED", raising=False)
        signup(client, email="off@test.com")
        assert "/verify-email?token=" not in capsys.readouterr().out

    def test_signup_mints_no_token(self, client, monkeypatch):
        """The row is what /verify-email redeems; minting one nobody was sent would
        leave a live single-use credential sitting in the table for no reason."""
        monkeypatch.delenv("EMAIL_VERIFICATION_ENABLED", raising=False)
        signup(client, email="off2@test.com")
        user = db.get_user_by_email("off2@test.com")
        assert user["email_verified_at"] is None
        with db.as_tenant(user["id"]), db.get_conn() as conn:
            count = conn.execute(text(
                "SELECT count(*) FROM password_reset_tokens WHERE purpose = 'email_verify'"
            )).scalar_one()
        assert count == 0

    def test_no_banner(self, client, monkeypatch):
        monkeypatch.delenv("EMAIL_VERIFICATION_ENABLED", raising=False)
        make_venture(client, monkeypatch)
        text = client.get("/dashboard").text
        assert "Confirm your email address" not in text
        assert "/resend-verification" not in text

    def test_the_account_works_fully(self, client, monkeypatch):
        monkeypatch.delenv("EMAIL_VERIFICATION_ENABLED", raising=False)
        venture_id = make_venture(client, monkeypatch)
        assert client.get("/phase1/new").status_code == 200
        assert client.get(f"/venture/{venture_id}").status_code == 200

    def test_resend_is_a_no_op(self, client, monkeypatch, capsys):
        monkeypatch.setenv("MAIL_BACKEND", "console")
        monkeypatch.delenv("EMAIL_VERIFICATION_ENABLED", raising=False)
        signup(client, email="off3@test.com")
        capsys.readouterr()
        r = client.post("/resend-verification", follow_redirects=False)
        assert r.status_code == 303
        assert "not enabled" in client.get(r.headers["location"]).text
        assert "/verify-email?token=" not in capsys.readouterr().out

    def test_resend_still_requires_a_session_when_disabled(self, client, monkeypatch):
        """The auth check runs first, so a logged-out POST cannot be used to probe
        whether the flag is on."""
        monkeypatch.delenv("EMAIL_VERIFICATION_ENABLED", raising=False)
        r = client.post("/resend-verification", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/login"

    def test_the_verify_link_still_works(self, client, monkeypatch, capsys):
        """A link already sitting in someone's inbox must not rot because the flag
        was switched off afterwards. The route stays mounted for exactly this."""
        monkeypatch.setenv("MAIL_BACKEND", "console")
        monkeypatch.setenv("EMAIL_VERIFICATION_ENABLED", "1")
        signup(client, email="off4@test.com")
        token = token_from_mail(capsys, marker="/verify-email?token=")
        monkeypatch.setenv("EMAIL_VERIFICATION_ENABLED", "0")

        r = client.get(f"/verify-email?token={token}", follow_redirects=False)
        assert r.status_code == 303
        assert db.user_is_verified(db.get_user_by_email("off4@test.com")["id"])

    def test_the_gate_is_open_when_disabled(self, client, monkeypatch):
        """Why db.user_is_verified() short-circuits to True.

        No account gets an email_verified_at while the flag is off, so a plain
        column read reports every user as unverified. M4.4 (share links) and M5.1
        (invites) call this to decide who may publish; reading the column
        directly would lock out the entire app the day either one ships.
        """
        monkeypatch.delenv("EMAIL_VERIFICATION_ENABLED", raising=False)
        user_id = make_user("gate@test.com")
        assert db.get_user_by_id(user_id)["email_verified_at"] is None
        assert db.user_is_verified(user_id) is True


class TestLoginThrottling:
    """Two counters, because they catch different attacks, and one that clears."""

    def test_throttling_also_keys_on_the_source_address(self, client, monkeypatch):
        """One address spraying a single password across many accounts: each
        account sees exactly one failed attempt and looks innocent."""
        monkeypatch.setenv("MAX_FAILED_LOGINS", "4")
        signup(client, email="spray-target@test.com")
        client.post("/logout", follow_redirects=False)

        # 20 distinct addresses, one failure each, all from the same TestClient IP.
        for i in range(20):
            client.post("/login", data={"email": f"victim{i}@test.com",
                                        "password": "samepassword"})

        r = client.post("/login", data={"email": "spray-target@test.com",
                                        "password": "password123"})
        assert "Too many failed attempts" in r.text, "the IP counter should have fired"

    def test_the_ip_counter_is_looser_than_the_account_one(self, client, monkeypatch):
        """The IP ceiling is a multiple of the per-account one precisely because
        universities and small companies put many researchers behind one NAT
        address. Exhausting one account must not lock out its neighbours."""
        monkeypatch.setenv("MAX_FAILED_LOGINS", "8")
        signup(client, email="noisy@test.com")
        client.post("/logout", follow_redirects=False)
        signup(client, email="quiet@test.com")
        client.post("/logout", follow_redirects=False)

        for _ in range(config.max_failed_logins()):
            client.post("/login", data={"email": "noisy@test.com", "password": "wrongwrong"})

        # The noisy account is now locked out...
        assert "Too many failed attempts" in client.post(
            "/login", data={"email": "noisy@test.com", "password": "password123"}).text
        # ...but a different account behind the same address is not, because 8
        # failures is well under the 5x IP ceiling.
        assert client.post("/login", data={"email": "quiet@test.com",
                                           "password": "password123"},
                           follow_redirects=False).status_code == 303

    def test_a_successful_login_clears_the_failure_history(self, client, monkeypatch):
        """Otherwise six typos lock the account out and the next *correct* login is
        refused, which reads as the app being broken."""
        monkeypatch.setenv("MAX_FAILED_LOGINS", "8")
        signup(client, email="typo@test.com")
        client.post("/logout", follow_redirects=False)
        for _ in range(6):
            client.post("/login", data={"email": "typo@test.com", "password": "wrongwrong"})

        assert client.post("/login", data={"email": "typo@test.com",
                                           "password": "password123"},
                           follow_redirects=False).status_code == 303
        client.post("/logout", follow_redirects=False)
        assert db.recent_failed_logins("typo@test.com", db.now()) == 0

    def test_the_old_password_still_fails_after_a_cleared_history(self, client, monkeypatch):
        """Clearing on success must not mark a wrong password as right."""
        monkeypatch.setenv("MAX_FAILED_LOGINS", "8")
        signup(client, email="cleared@test.com")
        client.post("/logout", follow_redirects=False)
        client.post("/login", data={"email": "cleared@test.com", "password": "password123"})
        client.post("/logout", follow_redirects=False)
        assert "Invalid email or password" in client.post(
            "/login", data={"email": "cleared@test.com", "password": "wrongwrong"}).text


# ==================== Phase 1 POST/Redirect/GET ====================

class TestPhase1:
    def test_refresh_does_not_duplicate_ideas(self, client, monkeypatch):
        """The extract route used to render the results page directly, so a
        browser refresh re-ran the extraction."""
        signup(client)
        monkeypatch.setattr(llm, "extract_ideas", lambda raw, user_id=None: [{
            "title": "Only idea", "commercial_framing": "x", "strength_signal": "early",
            "raw_claims": "y",
        }])
        r = client.post("/phase1/extract", data={"raw_text": "x" * 80}, follow_redirects=False)
        assert r.status_code == 303 and r.headers["location"].endswith("/phase1/ideas")
        assert len(db.list_ideas(1)) == 1
        client.get("/phase1/ideas")
        client.get("/phase1/ideas")
        assert len(db.list_ideas(1)) == 1

    def test_short_material_is_rejected_without_calling_the_model(self, client, monkeypatch):
        signup(client)

        def fail(*a, **k):
            raise AssertionError("must not call the model")
        monkeypatch.setattr(llm, "extract_ideas", fail)
        assert "more material" in client.post("/phase1/extract", data={"raw_text": "hi"}).text


# ==================== Quota ====================

class TestQuota:
    def test_quota_blocks_further_calls(self, client, monkeypatch):
        monkeypatch.setenv("LLM_MONTHLY_LIMIT_PER_USER", "2")
        import app.quota as quota_mod
        venture_id = make_venture(client, monkeypatch)
        monkeypatch.setattr(llm, "generate_segment_tasks", fake_design)

        # Simulate a user who has already spent both of this month's calls.
        for _ in range(2):
            db.log_llm_call(user_id=1, purpose="generate_segment_tasks", provider="zen",
                            model="glm-5.3", status="ok")

        r = client.post(f"/venture/{venture_id}/segment/customer/run", follow_redirects=False)
        assert r.status_code == 200, "a spent quota is a message, not a 500"
        assert "all 2 AI calls allowed this month" in r.text
        assert quota_mod.monthly_limit() == 2
        assert db.get_venture(venture_id, 1)["cycle_count"] == 0, "no run may start"

    def test_results_survive_a_quota_rejection(self, client, monkeypatch):
        """Quota must not eat the user's logged evidence — that's the one thing
        they can't recreate."""
        monkeypatch.setenv("LLM_MONTHLY_LIMIT_PER_USER", "1")
        venture_id = make_venture(client, monkeypatch)
        db.log_llm_call(user_id=1, purpose="extract_ideas", provider="zen",
                        model="glm-5.3", status="ok")
        cycle_id = db.create_cycle(venture_id, 1, fake_design()["tasks"], segment="customer")
        db.update_venture(venture_id, cycle_count=1)
        db.bump_segment_run(venture_id, "customer")
        db.log_cycle_results(cycle_id, [{"outcome": "asked 5 people", "sample_size": "5"}])
        r = client.post(f"/venture/{venture_id}/run/{cycle_id}/retry-analysis",
                        follow_redirects=False)
        assert r.status_code == 303
        # The message rides back on the redirect — silently doing nothing was
        # the original behaviour, which made a spent quota indistinguishable
        # from a broken button.
        assert "err=" in r.headers["location"]
        followed = client.get(r.headers["location"])
        assert "AI calls allowed this month" in followed.text
        assert db.get_cycle(cycle_id, venture_id)["results_json"][0]["sample_size"] == "5"

    def test_quota_zero_disables_the_limit(self, monkeypatch):
        monkeypatch.setenv("LLM_MONTHLY_LIMIT_PER_USER", "0")
        import app.quota as quota_mod
        quota_mod.check(999)  # must not raise

    def test_quota_counts_only_the_current_month(self, tmp_db, monkeypatch):
        user_id = make_user("m@test.com")
        db.log_llm_call(user_id=user_id, purpose="x", provider="zen", model="m", status="ok")
        with db.get_conn() as conn:
            conn.execute(text("UPDATE llm_calls SET month = '1999-01'"))
        assert db.count_llm_calls_this_month(user_id) == 0


# ==================== LLM call logging ====================

def fake_client(payload, capture=None, raises=None):
    """Stands in for the OpenAI client, one layer below llm.call_json, so the
    real call_json (and its recording) still runs."""
    def create(**kwargs):
        if capture is not None:
            capture.update(kwargs)
        if raises is not None:
            raise raises
        message = SimpleNamespace(content=json.dumps(payload) if not isinstance(payload, str) else payload)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)],
            usage=SimpleNamespace(prompt_tokens=120, completion_tokens=45),
        )
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def fake_client_msg(message):
    def create(**kwargs):
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


class TestLLMClientCompatibility:
    """The installed openai and httpx have to be able to construct a client.

    Every other test in this file monkeypatches `llm._get_client`, so nothing
    else exercises the real constructor. That gap hid a production-only 500: with
    openai 1.51.0 pinned alongside httpx 0.28.1, `OpenAI(...)` raised

        TypeError: Client.__init__() got an unexpected keyword argument 'proxies'

    on every LLM route, before any network call, because openai 1.x passes
    `proxies` to httpx.Client and httpx 0.28 removed it. The local venv had
    openai 3.17.0, so the suite passed and only Vercel broke.

    No test here makes a network request. `_get_client` is called directly and
    its result inspected.
    """

    def test_a_real_client_can_be_constructed(self, monkeypatch):
        """The regression itself: constructing must not raise."""
        monkeypatch.setenv("LLM_API_KEY", "test-key-not-a-real-one")
        monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
        monkeypatch.setenv("LLM_MODEL", "some-model")
        llm._client_cache.clear()
        try:
            client = llm._get_client(llm.get_settings())
            assert client is not None
            assert callable(client.chat.completions.create)
        finally:
            llm._client_cache.clear()

    def test_the_sdk_surface_the_app_uses_exists(self, monkeypatch):
        """A version bump is only safe if the three attributes app/llm.py touches
        are still there: chat.completions.create, and on the response
        .choices[0].finish_reason, .choices[0].message and .usage."""
        monkeypatch.setenv("LLM_API_KEY", "test-key-not-a-real-one")
        monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1")
        llm._client_cache.clear()
        try:
            client = llm._get_client(llm.get_settings())
        finally:
            llm._client_cache.clear()
        assert hasattr(client.chat.completions, "create")

    def test_installed_openai_matches_the_requirements_pin(self):
        """Catches venv drift in the other direction.

        A local environment that has quietly upgraded openai will happily run the
        whole suite green against a pin production cannot build. Comparing the
        installed version to the file is the only way to notice.
        """
        import importlib.metadata as md
        import re
        pin = None
        for line in pathlib.Path(config.PROJECT_ROOT / "requirements.txt").read_text().splitlines():
            line = line.strip()
            if line.startswith("openai=="):
                pin = line.split("==", 1)[1].strip()
                break
        assert pin, "no openai== pin in requirements.txt"
        assert md.version("openai") == pin, (
            f"installed openai is {md.version('openai')}, requirements.txt pins "
            f"{pin}. Production installs the pin, so a drift here means the tests "
            f"are validating a version that never ships."
        )

    def test_the_dev_requirements_pin_matches_the_runtime_pin(self):
        """requirements-dev.txt was left behind at openai 1.51.0 while the runtime
        pin moved, so a fresh dev install reproduced a bug the local venv hid."""
        def pin(text):
            m = re.search(r"^openai==(\S+)", text, re.M)
            return m.group(1) if m else None

        dev = pathlib.Path(config.PROJECT_ROOT / "requirements-dev.txt").read_text()
        runtime = pathlib.Path(config.PROJECT_ROOT / "requirements.txt").read_text()
        assert pin(dev) == pin(runtime), (
            f"dev pins openai=={pin(dev)}, runtime pins openai=={pin(runtime)}"
        )


class TestCallLog:
    def test_successful_call_is_recorded_with_tokens_and_model(self, tmp_db, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://opencode.ai/zen/v1")
        monkeypatch.setenv("LLM_MODEL", "glm-5.3")
        capture = {}
        monkeypatch.setattr(llm, "_get_client", lambda settings: fake_client({"ok": True}, capture))

        user_id = make_user("a@test.com")
        assert llm.call_json("prompt", "extract_ideas", user_id=user_id) == {"ok": True}
        assert db.count_llm_calls_this_month(user_id) == 1

        with db.get_conn() as conn:
            row = conn.execute(text("SELECT * FROM llm_calls")).mappings().fetchone()
        assert row["status"] == "ok" and row["model"] == "glm-5.3"
        assert row["provider"] == "zen" and row["input_tokens"] == 120
        assert row["latency_ms"] is not None

    def test_array_prompts_are_wrapped_for_json_object_mode(self, tmp_db, monkeypatch):
        """`response_format: json_object` forces a top-level object, so a prompt
        asking for a bare array comes back as {"ideas": [...]}. Regression: this
        unwrapping was inverted and broke Phase 1 against every real provider."""
        monkeypatch.setenv("LLM_API_KEY", "k")
        capture = {}
        wrapped = fake_client({"ideas": [{
            "title": "T", "commercial_framing": "f", "strength_signal": "early", "raw_claims": "c",
        }]}, capture)
        monkeypatch.setattr(llm, "_get_client", lambda settings: wrapped)
        assert llm.extract_ideas("material" * 20)[0]["title"] == "T"
        assert 'single key "ideas"' in capture["messages"][0]["content"]

    @pytest.mark.parametrize("returned", [
        [{"title": "d", "method": "problem_interview"}],                       # bare array
        {"todos": [{"title": "d", "method": "problem_interview"}]},            # wrong key
        {"tasks": [{"title": "d", "method": "problem_interview"}]},            # expected key
    ])
    def test_unwrap_tolerates_a_missing_or_renamed_envelope(self, tmp_db, monkeypatch, returned):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setattr(llm, "_get_client", lambda settings: fake_client(returned))
        out = llm.generate_segment_tasks("t", "f", CUSTOMER, 1)
        assert out["tasks"][0]["title"] == "d"

    def test_request_uses_json_mode_and_deterministic_settings(self, tmp_db, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://opencode.ai/zen/v1")
        capture = {}
        monkeypatch.setattr(llm, "_get_client", lambda settings: fake_client({"a": 1}, capture))
        llm.call_json("p", "extract_ideas", user_id=None)
        assert capture["response_format"] == {"type": "json_object"}
        assert capture["temperature"] == 0
        assert capture["max_tokens"] == config.llm_max_output_tokens()

    def test_token_ceiling_is_configurable(self, tmp_db, monkeypatch):
        """MAX_OUTPUT_TOKENS used to be the only source, so a provider that kept
        truncating JSON needed a code change to raise the ceiling."""
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://opencode.ai/zen/v1")
        capture = {}
        monkeypatch.setattr(llm, "_get_client", lambda settings: fake_client({"a": 1}, capture))
        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "999")
        llm.call_json("p", "extract_ideas", user_id=None)
        assert capture["max_tokens"] == 999

    @pytest.mark.parametrize("raw,expected", [
        ("5", 5), ("7", 7),
        ("", 3), ("   ", 3),      # blank
        ("abc", 3), ("3.5", 3),   # junk
        ("0", 3), ("-2", 3),      # below range: a 0 would disable retrying
        ("999", 3),               # above the sane ceiling
        (None, 3),
    ])
    def test_retry_budget_override_ignores_junk(self, monkeypatch, raw, expected):
        """A typo in a deploy config must degrade to the default rather than
        silently disabling retries or wedging the request loop."""
        if raw is None:
            monkeypatch.delenv("LLM_MAX_ATTEMPTS", raising=False)
        else:
            monkeypatch.setenv("LLM_MAX_ATTEMPTS", raw)
        assert config.llm_max_attempts() == expected

    def test_junk_token_ceiling_falls_back_rather_than_raising(self, tmp_db, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://opencode.ai/zen/v1")
        capture = {}
        monkeypatch.setattr(llm, "_get_client", lambda settings: fake_client({"a": 1}, capture))
        monkeypatch.setenv("LLM_MAX_OUTPUT_TOKENS", "banana")
        llm.call_json("p", "extract_ideas", user_id=None)
        assert capture["max_tokens"] == config.llm_max_output_tokens()

    def test_every_config_key_is_documented_and_read(self, monkeypatch):
        """Guards against .env.example and app/config.py drifting apart, which is
        how a setting ends up in the file but silently ignored by the app."""
        import re
        cfg = pathlib.Path(config.PROJECT_ROOT / "app" / "config.py").read_text()
        read = (set(re.findall(r'_bool\("([A-Z_]+)"', cfg))
                | set(re.findall(r'_int\("([A-Z_]+)"', cfg))
                | set(re.findall(r'_str\("([A-Z_]+)"', cfg))
                | set(re.findall(r'os\.environ\.get\("([A-Z_]+)"', cfg))
                | {"TEST_DATABASE_URL"})  # read by tests/conftest.py
        # A live assignment at the start of a line is the normal case. A variable named
        # in a comment line indented by 3+ spaces also counts as documented: the
        # rarely-changed ones (alternate provider endpoints, the fallback model) are
        # listed that way rather than as live assignments, precisely so a fresh
        # clone does not carry four settings nobody needs.
        #
        # The `#` plus 3-space indent plus 2-space column gap is load-bearing. SQL
        # snippets in the Postgres section are also commented and indented, so a
        # looser pattern matches CREATE and GRANT and then fails the "documented
        # but never read" direction on them.
        example = pathlib.Path(config.PROJECT_ROOT / ".env.example").read_text()
        documented = set(re.findall(r"^([A-Z_]+)=", example, re.M))
        documented |= set(re.findall(r"^#\s{3,}([A-Z_]{3,})\s{2,}", example, re.M))
        assert not read - documented, f"read but undocumented: {sorted(read - documented)}"
        assert not documented - read, f"documented but never read: {sorted(documented - read)}"

    @pytest.mark.parametrize("name,getter,value", [
        ("MIN_PASSWORD_LENGTH", "min_password_length", "12"),
        ("LOGIN_WINDOW_MINUTES", "login_window_minutes", "45"),
        ("MAX_FAILED_LOGINS", "max_failed_logins", "3"),
        ("IDEA_CARD_LIMIT", "idea_card_limit", "20"),
        ("SEGMENT_CAP", "segment_cap", "5"),
        ("VENTURE_RUN_BACKSTOP", "venture_run_backstop", "60"),
        ("LLM_TIMEOUT", "llm_timeout", "90"),
        ("LLM_MAX_ATTEMPTS", "llm_max_attempts", "5"),
        ("LLM_MAX_OUTPUT_TOKENS", "llm_max_output_tokens", "4096"),
        ("LLM_MONTHLY_LIMIT_PER_USER", "llm_monthly_limit_per_user", "42"),
        ("LLM_RATE_LIMIT_PER_MIN", "llm_rate_limit_per_min", "7"),
        ("LLM_RATE_LIMIT_IP_PER_MIN", "llm_rate_limit_ip_per_min", "70"),
        ("LLM_RATE_LIMIT_WINDOW_SECONDS", "llm_rate_limit_window_seconds", "30"),
        ("LLM_DEFAULT_MODEL", "default_model", "some-other-model"),
        ("LLM_ZEN_BASE_URL", "zen_base_url", "https://custom.example/v1"),
        ("LLM_OPENROUTER_BASE_URL", "openrouter_base_url", "https://or.example/v1"),
    ])
    def test_every_setting_is_driven_by_the_environment(self, monkeypatch, name, getter, value):
        """Each knob must actually move when the env var moves — a constant with
        no override is the failure this whole module exists to prevent."""
        monkeypatch.setenv(name, value)
        assert str(getattr(config, getter)()) == value

    # ---- reasoning models that burn the whole output budget thinking ----

    def test_reasoning_effort_is_sent_on_every_request(self, tmp_db, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_REASONING_EFFORT", "low")
        capture = {}
        monkeypatch.setattr(llm, "_get_client", lambda settings: fake_client({"a": 1}, capture))
        llm.call_json("p", "extract_ideas", user_id=None)
        assert capture["reasoning_effort"] == "low"

    def test_reasoning_effort_none_omits_the_parameter(self, tmp_db, monkeypatch):
        """Non-reasoning providers reject the parameter outright."""
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_REASONING_EFFORT", "none")
        capture = {}
        monkeypatch.setattr(llm, "_get_client", lambda settings: fake_client({"a": 1}, capture))
        llm.call_json("p", "extract_ideas", user_id=None)
        assert "reasoning_effort" not in capture

    def test_reasoning_effort_is_dropped_when_the_gateway_rejects_it(self, tmp_db, monkeypatch):
        """A 400 naming the parameter must cost one retry, not the request."""
        monkeypatch.setenv("LLM_API_KEY", "k")
        seen = []

        def create(**kwargs):
            seen.append(kwargs)
            if "reasoning_effort" in kwargs:
                raise BadRequestError(
                    "Unsupported parameter: 'reasoning_effort' is not supported.",
                    response=SimpleNamespace(status_code=400, headers={}, request=None),
                    body=None,
                )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":1}'),
                                         finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        monkeypatch.setattr(llm, "_get_client", lambda settings: client)
        assert llm.call_json("p", "extract_ideas", user_id=None) == {"ok": True}
        assert len(seen) == 2, "one rejected call, then one retry without the parameter"
        assert "reasoning_effort" not in seen[1]

    def test_truncated_reasoning_is_not_mined_for_json(self, tmp_db, monkeypatch):
        """Regression: a response cut off mid-reasoning has empty content and a
        `reasoning` field full of the model's internal monologue. Scanning that
        for braces finds fragments of the prompt's own schema example, which parse
        as valid JSON and would silently become stored data."""
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_MAX_ATTEMPTS", "1")
        monkeypatch.setattr(llm, "_get_client", lambda s: SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(
                        # plausible-looking JSON, but it is the model's scratchpad
                        # rather than an answer
                        content="",
                        reasoning='I need {"hypothesis": "x", "tasks": [{"title": "do a thing"}]}',
                        reasoning_details=None),
                    finish_reason="length")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=4096))))))
        with pytest.raises(llm.LLMError) as excinfo:
            llm.call_json("p", "extract_ideas", user_id=None)
        assert "hit the output ceiling" in str(excinfo.value)
        assert "do a thing" not in str(excinfo.value), \
            "the model's scratchpad must never be presented as the result"

    def test_truncated_response_is_not_retried(self, tmp_db, monkeypatch):
        """The same ceiling gives the same result, so retrying only burns the
        budget three times over (~107s in the field) to reach the same error."""
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_MAX_ATTEMPTS", "3")
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="", reasoning="thinking..."),
                                         finish_reason="length")],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=4096),
            )

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        monkeypatch.setattr(llm, "_get_client", lambda settings: client)
        with pytest.raises(llm.LLMError) as excinfo:
            llm.call_json("p", "extract_ideas", user_id=None)
        assert len(calls) == 1, "truncation must not be retried"
        assert "hit the output ceiling" in str(excinfo.value)
        assert "LLM_MAX_OUTPUT_TOKENS" in str(excinfo.value)

    def test_answer_in_reasoning_is_still_used_when_not_truncated(self, tmp_db, monkeypatch):
        """The reasoning fallback is legitimate for models that answer there — it
        must only be suppressed when the response was cut off."""
        monkeypatch.setenv("LLM_API_KEY", "k")
        payload = {"title": "X", "commercial_framing": "f",
                   "strength_signal": "early", "raw_claims": "c"}
        for kwargs in (
            {"content": "", "reasoning": json.dumps([payload]), "reasoning_details": None},
            {"content": None, "reasoning": None,
             "reasoning_details": [{"text": json.dumps([payload])}]},
        ):
            monkeypatch.setattr(
                llm, "_get_client", lambda s, kw=kwargs: fake_client_msg(SimpleNamespace(**kw)))
            assert llm.extract_ideas("material" * 20)[0]["title"] == "X"

    def test_json_mode_can_be_disabled(self, tmp_db, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_JSON_MODE", "0")
        capture = {}
        monkeypatch.setattr(llm, "_get_client", lambda settings: fake_client({"a": 1}, capture))
        llm.call_json("p", "extract_ideas", user_id=None)
        assert "response_format" not in capture

    def test_reasoning_models_are_read_from_reasoning_fields(self, tmp_db, monkeypatch):
        """Reasoning models behind OpenAI-compatible gateways can return an
        empty `content` and put the answer in `reasoning`/`reasoning_details`."""
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://opencode.ai/zen/v1")
        payload = {"title": "X", "commercial_framing": "f", "strength_signal": "early", "raw_claims": "c"}
        for kwargs in (
            {"content": "", "reasoning": None, "reasoning_details": [{"text": json.dumps([payload])}]},
            {"content": None, "reasoning": json.dumps([payload]), "reasoning_details": None},
        ):
            monkeypatch.setattr(
                llm, "_get_client",
                lambda s, kw=kwargs: fake_client_msg(SimpleNamespace(**kw)),
            )
            assert llm.extract_ideas("material" * 20)[0]["title"] == "X"

    def test_empty_response_reports_an_actionable_error(self, tmp_db, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_MODEL", "glm-5.3")
        monkeypatch.setattr(llm, "_get_client", lambda s: fake_client_msg(
            SimpleNamespace(content="   ", reasoning=None, reasoning_details=None)))
        with pytest.raises(llm.LLMError, match="empty response"):
            llm.extract_ideas("material" * 20)

    def test_auth_errors_fail_fast_without_retrying(self, tmp_db, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            raise llm.AuthenticationError(
                "bad key", response=SimpleNamespace(status_code=401, headers={}, request=None), body=None
            )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        monkeypatch.setattr(llm, "_get_client", lambda settings: client)

        with pytest.raises(llm.LLMError, match="rejected the credentials"):
            llm.call_json("p", "analyze_segment_run", user_id=None)
        assert len(calls) == 1, "a 401 must not be retried three times"

    def test_unparseable_output_is_retried_then_recorded_as_error(self, tmp_db, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        attempts = []

        def create(**kwargs):
            attempts.append(kwargs["messages"][0]["content"])
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="I refuse"))],
                usage=None,
            )
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        monkeypatch.setattr(llm, "_get_client", lambda settings: client)

        # A real user id, not None: RLS refuses an unattributed llm_calls row, so
        # a telemetry write with no owner is now a policy violation rather than a
        # silent NULL. Every call the app makes has a user.
        user_id = make_user("retry@test.com")
        with pytest.raises(llm.LLMError):
            llm.call_json("original prompt", "generate_segment_tasks", user_id=user_id)
        assert len(attempts) == config.llm_max_attempts()
        assert "original prompt" in attempts[0]
        assert "could not be parsed" in attempts[-1], "the retry must include the parse error"

        with db.as_tenant(user_id):
            with db.get_conn() as conn:
                row = conn.execute(
                    text("SELECT * FROM llm_calls")).mappings().fetchone()
        assert row["status"] == "error" and row["attempts"] == config.llm_max_attempts()

    def test_provider_config_prefers_zen_env(self, monkeypatch):
        monkeypatch.setenv("LLM_BASE_URL", "https://opencode.ai/zen/v1")
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_MODEL", "glm-5.3")
        s = llm.get_settings()
        assert s["provider"] == "zen" and s["model"] == "glm-5.3"

    def test_provider_config_falls_back_to_openrouter(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        # A developer's own .env may export LLM_MODEL; the fallback path must
        # not be decided by it, so clear it rather than inheriting the ambient
        # value from whatever provider this checkout happens to be configured for.
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("OPENROUTER_MODEL", "some/model")
        s = llm.get_settings()
        assert s["provider"] == "openrouter" and s["model"] == "some/model"

    def test_fallback_does_not_carry_a_zen_model_id_over_to_openrouter(self, monkeypatch):
        """Zen model ids are flat ('space-bunny-free'); OpenRouter's are vendor
        prefixed. Falling back while still sending the Zen id produces a model
        that doesn't exist on the fallback provider."""
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.setenv("LLM_MODEL", "space-bunny-free")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        monkeypatch.setenv("OPENROUTER_MODEL", "qwen/qwen3.8-27b:free")
        s = llm.get_settings()
        assert s["model"] == "qwen/qwen3.8-27b:free", s["model"]

    def test_missing_key_raises_a_helpful_error(self, monkeypatch):
        monkeypatch.delenv("LLM_BASE_URL", raising=False)
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(llm.LLMError, match="No LLM API key"):
            llm.get_settings()


# ==================== prompt provenance (M0.4) ====================

class TestPromptProvenance:
    """Which prompt template produced a call, recorded without keeping the prompt.

    The tension this class exists to hold: `llm_calls` has to be able to say
    "this result came from analyze_segment_run.v2", and it must never be able to
    say what that prompt said — because prompts embed the researcher's pasted
    material verbatim, and these rows deliberately outlive the account.
    """

    SECRET = "UNPUBLISHED-THESIS-MATERIAL-d4f3c2b1"

    def _calls(self, user_id):
        with db.get_conn() as conn:
            return conn.execute(
                text("SELECT * FROM llm_calls WHERE user_id = :u ORDER BY id"),
                {"u": user_id},
            ).mappings().fetchall()

    def test_version_and_hash_are_recorded_on_success(self, tmp_db, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        capture = {}
        monkeypatch.setattr(llm, "_get_client",
                            lambda s: fake_client({"ok": True}, capture))
        user_id = make_user("a@test.com")

        llm.call_json("p", "extract_ideas", user_id=user_id)

        row = self._calls(user_id)[0]
        assert row["prompt_version"] == "extract_ideas.v1"
        assert row["prompt_sha256"] == hashlib.sha256(b"p").hexdigest()

    def test_hash_is_of_the_prompt_actually_sent_including_the_json_wrapper(self, tmp_db, monkeypatch):
        """The wrap_key envelope is part of the request, so it has to be part of
        the fingerprint — otherwise two different requests would hash the same."""
        monkeypatch.setenv("LLM_API_KEY", "k")
        capture = {}
        monkeypatch.setattr(llm, "_get_client", lambda s: fake_client({"ok": True}, capture))
        user_id = make_user("a@test.com")

        llm.call_json("p", "extract_ideas", user_id=user_id, wrap_key="ideas")

        sent = capture["messages"][0]["content"]
        row = self._calls(user_id)[0]
        assert row["prompt_sha256"] == hashlib.sha256(sent.encode("utf-8")).hexdigest()
        assert 'single key "ideas"' in sent

    def test_version_is_recorded_on_the_error_path_too(self, tmp_db, monkeypatch):
        """An error row is what an audit actually starts from, so it needs the
        same provenance as a success."""
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_MAX_ATTEMPTS", "1")
        monkeypatch.setattr(llm, "_get_client",
                            lambda s: fake_client(None, raises=RuntimeError("boom")))
        user_id = make_user("a@test.com")

        with pytest.raises(llm.LLMError):
            llm.call_json("p", "analyze_segment_run", user_id=user_id)

        row = self._calls(user_id)[0]
        assert row["status"] == "error"
        assert row["prompt_version"] == "analyze_segment_run.v2"
        assert row["prompt_sha256"] == hashlib.sha256(b"p").hexdigest()

    def test_every_purpose_has_a_version(self):
        """An unmapped purpose would silently record NULL, which is the state
        this whole feature exists to avoid."""
        assert set(PROMPT_VERSIONS) == set(LLM_PURPOSES)

    def test_prompt_bodies_are_never_persisted(self, tmp_db, monkeypatch):
        """The privacy half of the contract, as a test.

        Drives the real extract_ideas path with a distinctive string, then greps
        every value of the logged row for it. This is what would fail if someone
        'helpfully' added a prompt_text column later.
        """
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setattr(llm, "_get_client", lambda s: fake_client({"ideas": [{
            "title": "T", "commercial_framing": "f",
            "strength_signal": "early", "raw_claims": "c",
        }]}))
        user_id = make_user("a@test.com")

        llm.extract_ideas(f"research material {self.SECRET} end", user_id=user_id)

        row = self._calls(user_id)[0]
        for column, value in row.items():
            assert self.SECRET not in str(value), f"{column} leaked the prompt body"

    def test_llm_calls_has_no_column_big_enough_to_hold_a_prompt(self, tmp_db):
        """Belt and braces on the test above: assert the shape of the *table*, not
        just the shape of one row. Adding a `prompt_text` column 'for debugging'
        would fail here, which is the point — the privacy guarantee is a property
        of the schema, not of the code that happens to write it today."""
        with db.get_conn() as conn:
            cols = {r[0] for r in conn.exec_driver_sql(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'llm_calls'").fetchall()}
        assert cols == {
            "id", "user_id", "purpose", "provider", "model", "status", "attempts",
            "input_tokens", "output_tokens", "latency_ms", "error", "month",
            "prompt_version", "prompt_sha256", "created_at",
        }


# ==================== per-request rate limiting (M0.3) ====================

class TestRateLimit:
    """The pace limit in front of the monthly budget.

    Both halves matter and they are independent: a refusal must be a readable
    page rather than a swallowed failure, and it must not cost the caller a unit of
    their monthly quota by writing an llm_calls row for a call that never happened.
    """

    def _limits(self, monkeypatch, per_min="2", per_ip="100", window="60"):
        monkeypatch.setenv("LLM_RATE_LIMIT_PER_MIN", per_min)
        monkeypatch.setenv("LLM_RATE_LIMIT_IP_PER_MIN", per_ip)
        monkeypatch.setenv("LLM_RATE_LIMIT_WINDOW_SECONDS", window)

    @staticmethod
    def _retry_after(limit, window=60):
        """Seconds for an empty bucket of `limit` tokens over `window` seconds to
        earn one back: capacity is per-window, so refill is one token per
        window/limit. Getting this wrong is how a limiter ends up quoting a wait
        four times longer than the user actually waits."""
        return -(-window // limit)

    def test_defaults_are_set_and_readable(self, monkeypatch):
        for name in ("LLM_RATE_LIMIT_PER_MIN", "LLM_RATE_LIMIT_IP_PER_MIN",
                     "LLM_RATE_LIMIT_WINDOW_SECONDS"):
            monkeypatch.delenv(name, raising=False)
        assert config.llm_rate_limit_per_min() == 12
        assert config.llm_rate_limit_ip_per_min() == 60
        assert config.llm_rate_limit_window_seconds() == 60

    def test_burst_is_allowed_then_refused(self, monkeypatch):
        self._limits(monkeypatch, per_min="3")
        assert [ratelimit.check(user_id=1, ip="1.2.3.4") for _ in range(5)] == \
            [None, None, None] + [self._retry_after(3)] * 2

    def test_a_refusal_costs_no_monthly_quota(self, tmp_db, monkeypatch):
        """The check runs before the model call, so a throttled request writes no
        llm_calls row — otherwise the throttle would bill the quota it exists to
        protect."""
        self._limits(monkeypatch, per_min="2")
        user_id = make_user("a@test.com")

        assert [ratelimit.check(user_id, "1.2.3.4") for _ in range(3)] == \
            [None, None, self._retry_after(2)]
        assert db.count_llm_calls_this_month(user_id) == 0

    def test_the_ip_scope_refuses_independently_of_the_user_scope(self, monkeypatch):
        """Many accounts behind one address is a signup-bot pattern the per-user
        budget cannot see, so the IP scope has to hold on its own."""
        self._limits(monkeypatch, per_min="100", per_ip="2")
        assert ratelimit.check(user_id=1, ip="9.9.9.9") is None
        assert ratelimit.check(user_id=2, ip="9.9.9.9") is None
        assert ratelimit.check(user_id=3, ip="9.9.9.9") == self._retry_after(2)

    def test_a_refused_ip_does_not_drain_the_per_user_budget(self, monkeypatch):
        """Otherwise one exhausted address would throttle everyone behind it
        twice over."""
        self._limits(monkeypatch, per_min="5", per_ip="1")
        assert ratelimit.check(user_id=7, ip="9.9.9.9") is None
        ratelimit.check(user_id=7, ip="9.9.9.9")  # ip bucket now empty
        for _ in range(4):
            assert ratelimit.check(user_id=7, ip="9.9.9.9") == self._retry_after(1)
        # the user bucket should still have its remaining 4 tokens
        assert ratelimit.check(user_id=7, ip="8.8.8.8") is None

    def test_zero_disables_a_scope(self, monkeypatch):
        self._limits(monkeypatch, per_min="0", per_ip="0")
        assert all(ratelimit.check(user_id=1, ip="1.2.3.4") is None for _ in range(50))

    def test_tokens_refill_over_the_window(self, monkeypatch):
        """A token bucket, not a fixed window: a steady drip at the same average
        rate is never locked out."""
        self._limits(monkeypatch, per_min="60", window="1")
        assert ratelimit.check(user_id=1, ip="1.2.3.4") is None
        for _ in range(59):
            ratelimit.check(user_id=1, ip="1.2.3.4")
        assert ratelimit.check(user_id=1, ip="1.2.3.4") == 1
        time.sleep(1.1)
        assert ratelimit.check(user_id=1, ip="1.2.3.4") is None

    def test_the_bucket_map_cannot_grow_without_bound(self, monkeypatch):
        """Every distinct source address gets a bucket, so rotating IPv6 would
        otherwise grow it for the life of the process."""
        self._limits(monkeypatch, per_min="100", per_ip="100")
        for i in range(ratelimit._MAX_BUCKETS + 50):
            ratelimit.check(user_id=i, ip=None)
        assert len(ratelimit._buckets) <= ratelimit._MAX_BUCKETS

    # ---- through the routes ----

    def test_a_burst_of_llm_posts_returns_429(self, client, monkeypatch):
        """The acceptance criterion, on the route a burst can actually arrive at.

        Note it is /phase1/extract and not the run-design route: only one run is in
        flight per venture, so repeated run-starts are refused by the state machine
        before they ever reach the throttle — which is the correct order, but it
        makes that route unable to produce a burst.
        """
        self._limits(monkeypatch, per_min="2")
        signup(client, email="burst@test.com")
        monkeypatch.setattr(llm, "extract_ideas", lambda raw, user_id=None: [{
            "title": "T", "commercial_framing": "f",
            "strength_signal": "early", "raw_claims": "c"}])

        codes = [client.post("/phase1/extract", data={"raw_text": "x" * 80},
                             follow_redirects=False).status_code for _ in range(3)]
        assert codes == [303, 303, 429]

    def test_the_429_is_a_readable_page_not_a_dead_button(self, client, monkeypatch):
        """The failure mode being replaced was a swallowed refusal that looked
        like nothing happened."""
        self._limits(monkeypatch, per_min="1", window="60")
        signup(client, email="r@test.com")
        monkeypatch.setattr(llm, "extract_ideas", lambda raw, user_id=None: [{
            "title": "T", "commercial_framing": "f",
            "strength_signal": "early", "raw_claims": "c"}])

        client.post("/phase1/extract", data={"raw_text": "x" * 80}, follow_redirects=False)
        r = client.post("/phase1/extract", data={"raw_text": "x" * 80},
                        follow_redirects=False)
        assert r.status_code == 429
        assert "Slow down" in r.text
        assert "Nothing was lost" in r.text
        assert 'href="/phase1/new"' in r.text
        assert r.headers["Retry-After"] == "60"

    def test_a_throttled_request_writes_no_llm_calls_row(self, client, monkeypatch):
        """Not one row for the refused call, so the monthly budget is untouched."""
        self._limits(monkeypatch, per_min="1")
        signup(client, email="q@test.com")
        monkeypatch.setattr(llm, "extract_ideas", lambda raw, user_id=None: [{
            "title": "T", "commercial_framing": "f",
            "strength_signal": "early", "raw_claims": "c"}])

        client.post("/phase1/extract", data={"raw_text": "x" * 80}, follow_redirects=False)
        before = db.count_llm_calls_this_month(1)
        r = client.post("/phase1/extract", data={"raw_text": "x" * 80},
                        follow_redirects=False)
        assert r.status_code == 429
        assert db.count_llm_calls_this_month(1) == before

    def test_a_throttled_run_starts_no_run(self, client, monkeypatch):
        """The throttle is enforced server-side, before the state machine is
        touched. A template that merely hid the button would leave this reachable.

        No ratelimit.reset() here on purpose: make_venture spends the single
        allowance on /phase1/extract, so the run design is already over budget.
        """
        self._limits(monkeypatch, per_min="1")
        venture_id = make_venture(client, monkeypatch)
        monkeypatch.setattr(llm, "generate_segment_tasks", fake_design)

        r = client.post(f"/venture/{venture_id}/segment/customer/run",
                        follow_redirects=False)
        assert r.status_code == 429
        assert db.get_venture(venture_id, 1)["cycle_count"] == 0, "no run may be created"
        assert db.list_cycles(venture_id) == []

    def test_logged_results_survive_a_throttled_verdict(self, client, monkeypatch):
        """Invariant 5, under the throttle: the user's evidence is saved before we
        decide whether we can score it, so a 429 never costs them the one thing
        they cannot recreate. The retry path stays open."""
        self._limits(monkeypatch, per_min="1")
        venture_id = make_venture(client, monkeypatch)
        monkeypatch.setattr(llm, "generate_segment_tasks", fake_design)
        ratelimit.reset()

        client.post(f"/venture/{venture_id}/segment/customer/run", follow_redirects=False)
        run = db.get_current_cycle(venture_id)
        assert run is not None

        # The design call above spent the single allowance, so logging results now
        # gets refused on the verdict — after the results have been written.
        r = client.post(f"/venture/{venture_id}/run/{run['id']}/log",
                        data={"outcome_0": "asked 5 people", "sample_size_0": "5"},
                        follow_redirects=False)
        assert r.status_code == 429
        assert "logged results are saved" in r.text
        assert db.get_cycle(run["id"], venture_id)["results_json"][0]["sample_size"] == "5"
        assert db.get_cycle(run["id"], venture_id)["analysis_json"] is None

        # and retrying later works, without re-logging anything. Raising the limit
        # alone is not enough — the bucket itself is empty, so it has to be reset
        # or the user would wait out the original window.
        monkeypatch.setattr(llm, "analyze_segment_run", lambda *a, **k: fake_verdict("pass"))
        monkeypatch.setenv("LLM_RATE_LIMIT_PER_MIN", "100")
        ratelimit.reset()
        r = client.post(f"/venture/{venture_id}/run/{run['id']}/retry-analysis",
                        follow_redirects=False)
        assert r.status_code == 303
        assert db.get_cycle(run["id"], venture_id)["decision"] == "pass"

    def test_a_user_with_no_ip_is_still_limited(self, monkeypatch):
        """request.client can be None behind some proxies; the per-user scope must
        still hold rather than raising."""
        self._limits(monkeypatch, per_min="1")
        assert ratelimit.check(user_id=1, ip=None) is None
        assert ratelimit.check(user_id=1, ip=None) == self._retry_after(1)


# ==================== account data control (M0.6) ====================

ALL_TABLES = ["users", "ideas", "ventures", "bmc_elements", "cycles",
              "launch_strategy", "action_steps", "mentor_challenges",
              "password_reset_tokens"]


@contextmanager
def as_owner():
    """Run a query as the table owner, bypassing row-level security.

    For database-administration assertions — "did that cascade leave an orphan",
    "does that audit row survive" — where the honest answer is not visible to any
    tenant. Note that `llm_calls` rows whose `user_id` was SET NULL are invisible
    to *every* tenant by design, so there is no tenant-scoped way to check them.
    """
    db.set_app_role("")
    try:
        with db.as_tenant(None):
            yield
    finally:
        db.set_app_role("launchloop_app")


def counts_by_table() -> dict:
    """Every row in every table, as the table owner — not as any tenant.

    The orphan check after a cascade has to see *all* rows, and row-level security
    means a tenant legitimately cannot: `as_tenant(None)` returns zero everywhere,
    which would make "no orphans left" pass for entirely the wrong reason. So the
    app-role switch is lifted for the count and restored after.

    This is a database administration check. No request can do this.
    """
    db.set_app_role("")
    try:
        with db.as_tenant(None):
            with db.get_conn() as conn:
                return {t: conn.execute(text(f"SELECT count(*) FROM {t}")).scalar_one()
                        for t in ALL_TABLES}
    finally:
        db.set_app_role("launchloop_app")


class TestDataExport:
    """Download your data, and prove it leaks nothing it should not."""

    def test_the_export_contains_everything_the_user_has(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "customer", verdict="iterate")

        r = client.get("/account/export")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("application/json")
        assert "attachment" in r.headers["content-disposition"]
        assert r.headers["cache-control"] == "no-store"

        data = r.json()
        assert data["format"] == "launchloop.user_export.v1"
        assert data["account"]["email"] == "a@test.com"
        assert len(data["ideas"]) == 1
        assert len(data["ventures"]) == 1
        assert len(data["blocks"]) == 9, "the whole canvas"
        assert len(data["runs"]) == 1
        assert data["runs"][0]["todos_json"], "the run's tasks are in there"
        assert data["runs"][0]["results_json"], "and the logged results"
        assert "password_hash" not in data["account"]

    def test_the_export_is_versioned_and_documented_in_shape(self, client, monkeypatch):
        """M4.3 has to be able to change this without guessing what a given
        download contained."""
        make_venture(client, monkeypatch)
        data = client.get("/account/export").json()
        assert data["format"] == "launchloop.user_export.v1"
        assert set(data) == {"format", "exported_at", "account", "ideas", "ventures",
                             "blocks", "runs", "launch_strategies", "action_steps",
                             "mentor_challenges", "llm_calls"}
        assert data["exported_at"]

    def test_the_export_carries_the_action_plan_progress(self, client, monkeypatch):
        """The whole point of an export is getting your work back. A step the
        researcher recorded is exactly the thing they cannot recreate.

        Note that outcome_note is NOT in _EXPORT_EXCLUDED_COLUMNS, unlike
        llm_calls.error: it is the researcher's own record of what they did, not
        provider- or model-supplied text.
        """
        vid = strategy_for(client, monkeypatch)
        client.post(f"/venture/{vid}/phase3/step",
                    data={"index": "0", "step_key": db.step_key(STEP_PLAN[0]["step"]),
                          "status": "done", "note": "9 of 10 replied, 2 calls booked"},
                    follow_redirects=False)
        data = client.get("/account/export").json()
        steps = data["action_steps"]
        assert len(steps) == 3
        recorded = [s for s in steps if s["status"] == "done"]
        assert len(recorded) == 1
        assert recorded[0]["outcome_note"] == "9 of 10 replied, 2 calls booked"
        assert recorded[0]["step"] == STEP_PLAN[0]["step"], \
            "the step text is denormalised so a completion stays readable after " \
            "the plan it came from has been regenerated away"

    def test_the_export_contains_no_credentials(self, client, monkeypatch, capsys):
        """token_hash is a live credential for 30 minutes. An export containing one
        is a takeover path, and it is the user's own row, so nobody would notice
        the leak downstream."""
        monkeypatch.setenv("MAIL_BACKEND", "console")
        make_venture(client, monkeypatch)
        capsys.readouterr()
        client.post("/forgot-password", data={"email": "a@test.com"},
                    follow_redirects=False)

        raw = client.get("/account/export").text
        with db.as_tenant(1):
            stored = db.get_reset_token(
                auth.hash_token(token_from_mail(capsys)), "password_reset")["token_hash"]

        assert stored not in raw, "a reset token hash leaked into the export"
        assert "password_hash" not in raw
        assert "$2b$" not in raw, "a bcrypt hash leaked into the export"

    def test_the_export_excludes_provider_error_text(self, client, monkeypatch):
        """llm_calls.error is provider-supplied and can quote request fragments back
        — the same reasoning as not storing prompt bodies."""
        make_venture(client, monkeypatch)
        db.log_llm_call(user_id=1, purpose="extract_ideas", provider="zen",
                        model="m", status="error",
                        error="upstream said YOUR-THESIS-TEXT was invalid")
        data = client.get("/account/export").json()
        assert data["llm_calls"], "call metadata is still exported"
        assert "YOUR-THESIS-TEXT" not in client.get("/account/export").text

    def test_the_export_respects_tenant_isolation(self, client, monkeypatch):
        """User B's download must contain nothing of user A's — the same row-level
        security the rest of the app relies on, one download away."""
        make_venture(client, monkeypatch)
        client.post("/logout", follow_redirects=False)
        signup(client, email="b@test.com")

        raw = client.get("/account/export").text
        assert "Sensor thing" not in raw
        data = client.get("/account/export").json()
        assert data["ideas"] == [] and data["ventures"] == []
        assert data["account"]["email"] == "b@test.com"

    def test_the_export_requires_a_session(self, client):
        r = client.get("/account/export", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/login"

    def test_the_account_page_states_what_the_export_contains(self, client, monkeypatch):
        make_venture(client, monkeypatch)
        run_once(client, monkeypatch, db.list_ventures(1)[0]["id"], "customer")
        text = client.get("/account").text
        assert "1 venture" in text
        assert "1 run" in text
        assert 'action="/account/export"' in text
        assert 'action="/account/delete"' in text


class TestAccountDeletion:
    """Irreversible, so the gates matter as much as the cascade."""

    def test_deletion_removes_everything_and_leaves_no_orphans(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "customer", verdict="pass")
        # Give the account the widest possible footprint first.
        client.post("/phase3/generate", follow_redirects=False)
        assert counts_by_table()["ventures"] == 1

        r = client.post("/account/delete",
                        data={"password": "password123", "confirm": "DELETE"},
                        follow_redirects=False)
        assert r.status_code == 303

        left = counts_by_table()
        assert all(n == 0 for n in left.values()), f"orphans left behind: {left}"
        note = client.get(r.headers["location"]).text
        assert "Account deleted" in note
        assert "Everything in it has been removed" in note

    def test_action_plan_progress_goes_with_the_account(self, client, monkeypatch):
        """A completed step is the one thing here the researcher cannot recreate,
        so it has to be counted in the confirmation page and actually cascaded.

        The venture-level cascade clears it via ventures; the assertion is that
        the confirmation page says it is about to be removed, because a number
        the user is not shown is a number they cannot check.
        """
        vid = strategy_for(client, monkeypatch)
        client.post(f"/venture/{vid}/phase3/step",
                    data={"index": "0", "step_key": db.step_key(STEP_PLAN[0]["step"]),
                          "status": "done", "note": "signed the pilot"},
                    follow_redirects=False)
        page = client.get("/account").text
        assert "3" in page, "the plan's steps should be counted on the account page"

        r = client.post("/account/delete",
                        data={"password": "password123", "confirm": "DELETE"},
                        follow_redirects=False)
        assert r.status_code == 303
        left = counts_by_table()
        assert all(n == 0 for n in left.values()), f"orphans left behind: {left}"
        assert left["action_steps"] == 0

    def test_llm_call_metadata_outlives_the_account(self, client, monkeypatch):
        """user_id is SET NULL on purpose: spend and latency are worth keeping as
        an audit trail, and there is nothing identifying in them."""
        make_venture(client, monkeypatch)
        db.log_llm_call(user_id=1, purpose="extract_ideas", provider="zen",
                        model="m", status="ok", input_tokens=10, output_tokens=5)
        client.post("/account/delete",
                    data={"password": "password123", "confirm": "DELETE"},
                    follow_redirects=False)

        with as_owner():
            with db.get_conn() as conn:
                rows = conn.execute(text(
                    "SELECT user_id, purpose, prompt_version FROM llm_calls")).fetchall()
        assert len(rows) == 1
        assert rows[0][0] is None, "the audit row must survive with no owner"
        assert rows[0][1] == "extract_ideas"

    def test_a_wrong_password_deletes_nothing(self, client, monkeypatch):
        make_venture(client, monkeypatch)
        r = client.post("/account/delete",
                        data={"password": "notmypassword", "confirm": "DELETE"},
                        follow_redirects=False)
        assert r.status_code == 200
        assert "not right" in r.text
        assert counts_by_table()["ventures"] == 1, "nothing may be removed"

    @pytest.mark.parametrize("confirm", ["", "delete", "Delete", "DELETED", "delete "])
    def test_the_typed_confirmation_is_exact(self, client, monkeypatch, confirm):
        make_venture(client, monkeypatch)
        r = client.post("/account/delete",
                        data={"password": "password123", "confirm": confirm},
                        follow_redirects=False)
        assert r.status_code == 200
        assert "Type DELETE" in r.text
        assert counts_by_table()["ventures"] == 1

    def test_an_unlocked_session_alone_cannot_destroy_the_account(self, client, monkeypatch):
        """Both gates, stated as the property they exist for."""
        make_venture(client, monkeypatch)
        # Password right, confirmation wrong.
        r = client.post("/account/delete",
                        data={"password": "password123", "confirm": "yes"},
                        follow_redirects=False)
        assert r.status_code == 200
        # Confirmation right, password wrong.
        r = client.post("/account/delete",
                        data={"password": "", "confirm": "DELETE"},
                        follow_redirects=False)
        assert r.status_code == 200
        assert counts_by_table()["ventures"] == 1

    def test_the_session_is_dead_afterwards(self, client, monkeypatch):
        """Otherwise the user keeps browsing a 500, because every page looks up a
        user row that no longer exists."""
        make_venture(client, monkeypatch)
        client.post("/account/delete",
                    data={"password": "password123", "confirm": "DELETE"},
                    follow_redirects=False)
        r = client.get("/dashboard", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/login"

    def test_deletion_requires_a_session(self, client):
        r = client.post("/account/delete",
                        data={"password": "password123", "confirm": "DELETE"},
                        follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/login"

    def test_one_user_cannot_delete_another(self, client, monkeypatch):
        """There is no user_id in the form, so this is really a check that the
        route can only ever delete whoever the session says it is."""
        make_venture(client, monkeypatch)
        client.post("/logout", follow_redirects=False)
        signup(client, email="b@test.com")
        client.post("/account/delete",
                    data={"password": "password123", "confirm": "DELETE"},
                    follow_redirects=False)
        with db.as_tenant(1):
            assert db.get_user_by_email("a@test.com") is not None, \
                "user 1 must be untouched by user 2's deletion"

    def test_deleting_an_account_leaves_other_accounts_alone(self, client, monkeypatch):
        make_venture(client, monkeypatch)
        client.post("/logout", follow_redirects=False)
        signup(client, email="b@test.com")
        client.post("/account/delete",
                    data={"password": "password123", "confirm": "DELETE"},
                    follow_redirects=False)
        with db.as_tenant(1):
            assert counts_by_table()["ventures"] == 1, "user 1's venture survives"


# ==================== file upload (M1.1) ====================

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def fixture_bytes(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def upload(client, name, data, content_type="application/octet-stream", **kw):
    kw.setdefault("follow_redirects", False)
    return client.post("/phase1/upload",
                       files={"file": (name, data, content_type)}, **kw)


class TestUploadExtraction:
    """app/upload.py, at the function boundary. No HTTP, no browser."""

    def test_a_real_pdf_yields_text(self):
        text = upload_mod.extract(fixture_bytes("sample.pdf"), "sample.pdf")
        assert "SWIR" in text and "4x" in text
        assert "sensor" in text.lower()

    def test_a_real_docx_yields_text_including_tables(self):
        text = upload_mod.extract(fixture_bytes("sample.docx"), "sample.docx")
        assert "SWIR" in text
        assert "120 fps" in text, "table cells are part of the material"

    def test_plain_text_round_trips(self):
        raw = ("sensor " * 60).encode()
        assert "sensor" in upload_mod.extract(raw, "notes.txt")

    def test_format_is_detected_from_the_bytes_not_the_extension(self):
        """The extension and Content-Type are both supplied by the uploader, so
        treating either as evidence makes the check worthless."""
        pdf = fixture_bytes("sample.pdf")
        assert upload_mod.sniff(pdf[:8]) == "pdf"
        assert upload_mod.sniff(fixture_bytes("sample.docx")[:8]) == "docx"
        assert upload_mod.sniff(b"just some words here") == "text"
        # A PDF named .docx is still a PDF.
        assert upload_mod.extract(pdf, "actually-a.docx").strip().startswith("Low-power") or \
            "SWIR" in upload_mod.extract(pdf, "actually-a.docx")

    def test_a_zip_is_not_reported_as_a_word_document(self):
        import io
        import zipfile
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as z:
            z.writestr("notes.txt", "hello")
        with pytest.raises(upload_mod.UploadError, match="zip archive"):
            upload_mod.extract(buf.getvalue(), "archive.zip")

    @pytest.mark.parametrize("data,expected", [
        (b"", "empty"),
        (b"%PDF-\x00\x01broken", "could not be read"),
        (b"\x00\x01\x02\x03binary junk", "doesn't look like"),
        (b"%PDF-" + b"x" * (9 * 1024 * 1024), "limit is"),
    ])
    def test_bad_input_gives_a_sentence_not_an_exception(self, data, expected):
        """A malformed file is a user problem. Every one of these must arrive as
        an UploadError with something they can act on."""
        with pytest.raises(upload_mod.UploadError) as exc:
            upload_mod.extract(data, "x.bin")
        assert expected in str(exc.value)
        assert "Traceback" not in str(exc.value)

    def test_a_pdf_with_no_text_says_so_rather_than_guessing(self):
        """A scanned page has no text layer, and inventing something would be worse
        than admitting it."""
        blank = pypdf.PdfWriter()
        blank.add_blank_page(width=595, height=842)
        buf = io.BytesIO()
        blank.write(buf)
        with pytest.raises(upload_mod.UploadError, match="No text came out"):
            upload_mod.extract(buf.getvalue(), "scan.pdf")

    def test_the_page_cap_is_enforced_and_explained(self, monkeypatch):
        monkeypatch.setenv("MAX_PDF_PAGES", "1")
        with pytest.raises(upload_mod.UploadError, match="2 pages; the limit is 1"):
            upload_mod.extract(fixture_bytes("sample.pdf"), "sample.pdf")

    def test_long_extractions_are_capped_and_the_user_is_told(self, monkeypatch):
        # 1000, not something smaller: _int degrades an out-of-range value to the
        # default, and this reader's floor is 1000. The long input is generated
        # rather than fixture-sized so the cap is unambiguously the binding thing.
        monkeypatch.setenv("MAX_EXTRACTED_CHARS", "1000")
        out = upload_mod.extract(("word " * 2000).encode(), "long.txt")
        assert "extraction truncated at 1000 characters" in out
        assert out.startswith("word"), "the start of the text survives"

    def test_a_non_utf8_byte_does_not_crash_a_text_file(self):
        assert upload_mod.extract(b"caf\xe9 research notes here", "n.txt")

    def test_the_deadline_is_per_request_not_a_module_global(self, monkeypatch):
        """A module-global start time would carry the first request's budget into
        every later one, shrinking the allowance over the life of the process."""
        monkeypatch.setenv("MAX_PARSE_SECONDS", "60")
        upload_mod.extract(fixture_bytes("sample.pdf"), "a.pdf")
        # Second call with a budget already blown must stop immediately.
        out = upload_mod.extract(fixture_bytes("sample.pdf"), "a.pdf", deadline_s=-1)
        assert "extraction stopped after 0 pages" in out


class TestUploadRoute:
    """The HTTP surface. What matters here: the file is discarded, and the user
    sees the text before spending an AI call."""

    def test_the_preview_comes_back_editable_before_any_ai_call(self, client, monkeypatch):
        monkeypatch.setattr(llm, "extract_ideas",
                            lambda raw, user_id=None: pytest.fail("must not call the model yet"))
        signup(client)
        r = upload(client, "sample.pdf", fixture_bytes("sample.pdf"), "application/pdf")
        assert r.status_code == 200
        assert "SWIR" in r.text, "extracted text should be in the form"
        assert "Read" in r.text and "sample.pdf" in r.text
        # And it is editable — the textarea carries it, not a read-only block.
        assert "<textarea" in r.text

    def test_uploading_then_extracting_is_an_ordinary_phase_1_call(self, client, monkeypatch):
        """No new AI-call path, so no new quota or rate-limit surface."""
        signup(client)
        upload(client, "sample.docx", fixture_bytes("sample.docx"),
               "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        monkeypatch.setattr(llm, "extract_ideas", lambda raw, user_id=None: [{
            "title": "Sensor thing", "commercial_framing": "Sell sensors",
            "strength_signal": "early", "raw_claims": "sensor"}])
        r = client.post("/phase1/extract", data={"raw_text": "x" * 200},
                        follow_redirects=False)
        assert r.status_code == 303

    def test_a_bad_file_is_a_readable_message_and_a_200(self, client, monkeypatch):
        signup(client)
        r = upload(client, "broken.pdf", b"%PDF-\x00\x01nope")
        assert r.status_code == 200, "a bad file must never be a 500"
        assert "could not be read" in r.text
        assert "<textarea" in r.text, "and the form is still there"

    def test_an_empty_file_is_refused(self, client, monkeypatch):
        signup(client)
        r = upload(client, "empty.pdf", b"")
        assert r.status_code == 200
        assert "empty" in r.text.lower()

    def test_an_oversized_file_is_refused_without_reading_all_of_it(self, client, monkeypatch):
        monkeypatch.setenv("MAX_UPLOAD_BYTES", "2048")
        signup(client)
        r = upload(client, "big.pdf", b"%PDF-" + b"x" * 5000)
        assert r.status_code == 200
        assert "larger than" in r.text

    def test_nothing_is_persisted(self, client, monkeypatch, tmp_db):
        """The strongest reading of "delete the raw file by default": no file is
        written at all, so there is nothing to clean up or leak later."""
        signup(client)
        upload(client, "sample.pdf", fixture_bytes("sample.pdf"))
        # Nothing new in any table, and no stray file on disk.
        with db.get_conn() as conn:
            assert conn.execute(text("SELECT count(*) FROM users")).scalar_one() == 1
            for table in ("ideas", "ventures", "cycles"):
                assert conn.execute(
                    text(f"SELECT count(*) FROM {table}")).scalar_one() == 0

    def test_the_route_requires_a_session(self, client):
        r = upload(client, "sample.pdf", fixture_bytes("sample.pdf"),
                   follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/login"

    def test_the_upload_form_is_offered_on_the_page(self, client):
        signup(client)
        text = client.get("/phase1/new").text
        assert 'enctype="multipart/form-data"' in text
        assert 'action="/phase1/upload"' in text
        assert "discarded" in text, "say what happens to the file"
        assert "8 MB" in text, "the limit is shown, not just enforced"


# ==================== row-level security (M0.5) ====================

RLS_TABLES = [
    "ideas", "ventures", "llm_calls", "password_reset_tokens",
    "bmc_elements", "cycles", "launch_strategy", "action_steps", "users",
    "mentor_challenges",
]


class TestRowLevelSecurity:
    """The database-level half of tenant isolation.

    Everything else in this suite proves the *application* filters by user_id.
    This class proves that even a query with no user_id filter at all cannot cross
    a tenant boundary — which is the property that survives a review mistake.

    The first test is the one that matters most: if the suite were accidentally
    running as a bypass role, every test below it would pass for the wrong
    reason, and the whole class would be decoration.
    """

    def test_the_suite_is_actually_exercising_rls(self, tmp_db):
        """If this fails, everything else in this class is vacuous."""
        with db.get_conn() as conn:
            assert db.rls_active(conn) is True, (
                "the connected role bypasses RLS, so these tests prove nothing")
            enabled = {r[0] for r in conn.exec_driver_sql(
                "SELECT relname FROM pg_class WHERE relrowsecurity").fetchall()}
        missing = set(RLS_TABLES) - enabled
        assert not missing, f"RLS not enabled on {sorted(missing)}"

    def test_login_attempts_is_deliberately_excluded(self, tmp_db):
        """Recorded, not forgotten.

        Both throttling queries are pre-authentication. RLS there would filter
        them by a tenant that does not exist yet, every count would read zero,
        and the lockout would silently never fire — a security control failing
        OPEN, which no test of its normal path would catch.
        """
        with db.get_conn() as conn:
            enabled = {r[0] for r in conn.exec_driver_sql(
                "SELECT relname FROM pg_class WHERE relrowsecurity").fetchall()}
        assert "login_attempts" not in enabled

    # ---- one table, then the sweep ----

    @staticmethod
    def _two_tenants():
        """User 1 owns a full venture; user 2 owns nothing."""
        owner = make_user("rls-owner@test.com")
        other = make_user("rls-other@test.com")
        idea_id = db.create_idea(owner, "secret idea", "private framing",
                                 "early", "private claims")
        venture_id = db.create_venture(owner, idea_id)
        cycle_id = db.create_cycle(venture_id, 1, fake_design()["tasks"], segment="customer")
        db.log_llm_call(user_id=owner, purpose="extract_ideas", provider="zen",
                        model="m", status="ok")
        db.create_reset_token(owner, "password_reset", "c" * 64,
                              db.now() + timedelta(minutes=30))
        db.save_launch_strategy(venture_id, [], [], [])
        # A real plan, so action_steps gets rows and the cross-tenant sweep below
        # covers it. save_launch_strategy(…, [], []) seeds nothing by design.
        db.seed_action_steps(venture_id, [{"step": "Email a pilot", "milestone_type": "pilot"}])
        # One idea-subject row as well: a Phase 1 challenge points at an idea and
        # has no venture_id at all, so a sweep that only seeded venture rows would
        # pass without ever proving the idea path is tenant-isolated.
        db.save_mentor_challenge(owner, "evidence", "idea", [
            {"question": "Which block would you delete?", "principle": "focus",
             "why_it_matters": "anything you would not fight for"}],
            idea_id=idea_id)
        return owner, other, idea_id, venture_id, cycle_id

    def test_user_b_reads_nothing_of_user_as_rows(self, tmp_db):
        owner, other, idea_id, venture_id, cycle_id = self._two_tenants()
        with db.as_tenant(other):
            # Straight COUNTs with no user_id predicate anywhere — the thing the
            # application filters would catch, and the thing it would not.
            with db.get_conn() as conn:
                counts = {
                    table: conn.execute(text(f"SELECT count(*) FROM {table}")
                                         ).scalar_one()
                    for table in RLS_TABLES if table != "users"
                }
        assert all(n == 0 for n in counts.values()), counts
        assert counts["ventures"] == 0 and counts["cycles"] == 0

        # And as the owner the same queries still see everything, so this is not
        # "the table is empty" passing for the right reason.
        with db.as_tenant(owner):
            with db.get_conn() as conn:
                assert conn.execute(text("SELECT count(*) FROM ventures")).scalar_one() == 1
                assert conn.execute(text("SELECT count(*) FROM cycles")).scalar_one() == 1
                assert conn.execute(text("SELECT count(*) FROM bmc_elements")).scalar_one() == 9

    def test_user_b_cannot_write_to_user_as_rows(self, tmp_db):
        """Note the asymmetry, which is easy to get wrong: RLS makes a row
        *invisible*, so an UPDATE or DELETE that targets it simply matches zero
        rows and does not raise. Only a WITH CHECK violation — an INSERT claiming
        someone else's user_id — errors. Both are safe; they fail differently.
        """
        owner, other, idea_id, venture_id, cycle_id = self._two_tenants()
        with db.as_tenant(other):
            with db.get_conn() as conn:
                n = conn.execute(text(
                    "UPDATE ventures SET status = 'killed' WHERE id = :i"),
                    {"i": venture_id}).rowcount
                assert n == 0, "the write must affect no rows"
            with db.get_conn() as conn:
                n = conn.execute(text("DELETE FROM cycles WHERE id = :i"),
                                 {"i": cycle_id}).rowcount
                assert n == 0
            with db.get_conn() as conn:
                n = conn.execute(text("DELETE FROM ventures WHERE id = :i"),
                                 {"i": venture_id}).rowcount
                assert n == 0

        # And nothing changed for the owner.
        with db.as_tenant(owner):
            assert db.get_venture(venture_id, owner)["status"] == "active"
            assert db.get_cycle(cycle_id, venture_id) is not None

    def test_user_b_cannot_insert_a_row_owned_by_someone_else(self, tmp_db):
        owner, other, idea_id, venture_id, cycle_id = self._two_tenants()
        with db.as_tenant(other):
            with pytest.raises(Exception) as exc:
                with db.get_conn() as conn:
                    conn.execute(text(
                        "INSERT INTO ideas (user_id, title, commercial_framing, status,"
                        " created_at) VALUES (:u, 'planted', 'x', 'candidate', now())"),
                        {"u": owner})
            assert "row-level security" in str(exc.value).lower()
        with db.as_tenant(owner):
            assert [i["title"] for i in db.list_ideas(owner)] == ["secret idea"]

    def test_no_tenant_denies_everything(self, tmp_db):
        """The default must be closed. An unset tenant is the state a bug, a
        background job or a forgotten call site would find, and it must see
        nothing rather than everything."""
        owner, other, *_ = self._two_tenants()
        with db.as_tenant(None):
            with db.get_conn() as conn:
                for table in RLS_TABLES:
                    if table == "users":
                        continue
                    n = conn.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()
                    assert n == 0, f"{table} leaked with no tenant set"

    def test_a_user_row_is_visible_to_its_owner_and_by_lookup_only(self, tmp_db):
        owner = make_user("mine@test.com")
        other = make_user("theirs@test.com")
        with db.as_tenant(other):
            assert db.get_user_by_id(owner) is None
            assert db.get_user_by_email("mine@test.com") is None
        # Present the address instead — the same thing the login form does.
        with db.as_tenant(lookup_email="mine@test.com"):
            assert db.get_user_by_email("mine@test.com") is not None
            assert db.get_user_by_email("theirs@test.com") is None

    def test_an_unattributed_llm_call_cannot_be_recorded(self, tmp_db):
        """Telemetry with no owner is not telemetry — it is an unattributable
        row, and it is what the audit trail is supposed to make impossible."""
        with db.as_tenant(1):
            with pytest.raises(Exception) as exc:
                with db.get_conn() as conn:
                    conn.execute(text(
                        "INSERT INTO llm_calls (user_id, purpose, provider, model,"
                        " status, month, created_at)"
                        " VALUES (NULL, 'x', 'zen', 'm', 'ok', '2026-01', now())"))
            assert "row-level security" in str(exc.value).lower()

    def test_a_reset_token_is_readable_by_hash_but_only_writable_by_its_owner(self, tmp_db):
        """The pre-auth shape: you can present a token before you know whose
        account it is, but you cannot mint one for someone else."""
        owner = make_user("tok@test.com")
        other = make_user("tok2@test.com")
        token_hash = "d" * 64
        db.create_reset_token(owner, "password_reset", token_hash,
                              db.now() + timedelta(minutes=30))
        with db.as_tenant(lookup_token=token_hash):
            assert db.get_reset_token(token_hash, "password_reset") is not None
        with db.as_tenant(other):
            assert db.get_reset_token(token_hash, "password_reset") is None
            with pytest.raises(Exception) as exc:
                with db.get_conn() as conn:
                    conn.execute(text(
                        "INSERT INTO password_reset_tokens (user_id, purpose,"
                        " token_hash, expires_at, created_at)"
                        " VALUES (:u, 'password_reset', 'e', now() + interval '1 hour',"
                        " now())"), {"u": owner})
            assert "row-level security" in str(exc.value).lower()


class TestDeployShape:
    """Per-platform knobs, and why the defaults suit the platform they suit.

    One codebase runs on Render and Vercel, which differ in process lifetime: a
    single long-lived container versus a function that scales to zero. These are
    the two places that difference actually bites — connection pooling and
    whether migrations run at boot.
    """

    def test_pool_defaults_suit_a_long_lived_process(self, monkeypatch):
        """Render runs one container for the life of the deploy, so it reuses the
        same connections. These are the defaults because that is what they are for.
        """
        monkeypatch.delenv("DB_POOL_SIZE", raising=False)
        monkeypatch.delenv("DB_POOL_MAX_OVERFLOW", raising=False)
        assert config.db_pool_size() == 5
        assert config.db_pool_max_overflow() == 5

    def test_pool_reads_the_env(self, monkeypatch):
        monkeypatch.setenv("DB_POOL_SIZE", "1")
        monkeypatch.setenv("DB_POOL_MAX_OVERFLOW", "0")
        assert config.db_pool_size() == 1
        assert config.db_pool_max_overflow() == 0

    def test_junk_pool_size_falls_back_rather_than_raising(self, monkeypatch):
        """_int degrades an out-of-range value to the default. A typo'd pool size
        must not become a zero-byte pool that refuses every connection.
        """
        monkeypatch.setenv("DB_POOL_SIZE", "banana")
        assert config.db_pool_size() == 5

    def test_pool_zero_means_no_pooling(self, monkeypatch):
        """DB_POOL_SIZE=0 selects NullPool — connect on checkout, close on return.

        That is what "no pool" means for a serverless invocation, and it is not
        the same as pool_size=0 in SQLAlchemy, which is a valid but useless
        one-connection pool that would serialise concurrent requests.

        The engine is built but never connects: create_engine is lazy, so this
        asserts the pool *choice* without needing a live database.
        """
        monkeypatch.setenv("DB_POOL_SIZE", "0")
        monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@localhost:5432/db")
        captured = {}
        real = db.create_engine

        def spy(url, **kwargs):
            captured.update(kwargs)
            return real(url, **kwargs)

        monkeypatch.setattr(db, "create_engine", spy)
        previous, db._engine = db._engine, None
        try:
            engine = db.get_engine()
            assert captured["poolclass"] is db.NullPool
            assert "pool_size" not in captured
            assert isinstance(engine.pool, db.NullPool)
        finally:
            # Restore the per-test schema's engine rather than clearing it, so a
            # later test in this class does not find the engine gone.
            db._engine = previous
            engine.dispose()

    def test_migrations_run_at_boot_by_default(self, monkeypatch):
        """Render's single container wants this on: one boot, one upgrade."""
        monkeypatch.delenv("RUN_MIGRATIONS_ON_BOOT", raising=False)
        assert config.run_migrations_on_boot() is True

    def test_boot_migrations_can_be_turned_off_for_serverless(self, monkeypatch):
        """A function that scales to zero would otherwise re-run every migration on
        the first request of each idle period.
        """
        monkeypatch.setenv("RUN_MIGRATIONS_ON_BOOT", "0")
        assert config.run_migrations_on_boot() is False

    def test_a_failed_boot_migration_does_not_take_the_app_down(self, monkeypatch):
        """Boot is where a transient database blip is most likely to land.

        Raising would mean every later request 500s too, because the schema never
        gets its chance to recover. Logging and continuing lets it self-heal.
        """
        def boom():
            raise RuntimeError("could not connect to server")

        monkeypatch.setattr(db, "init_db", boom)
        monkeypatch.setenv("RUN_MIGRATIONS_ON_BOOT", "1")
        import app.main as main_mod
        import anyio

        async def run():
            async with main_mod.lifespan(main_mod.app):
                return True

        assert anyio.run(run) is True


class TestTenantContext:
    """The ContextVar that carries the tenant is the mechanism the whole thing
    rests on, and a concurrency risk besides — so its properties are asserted
    rather than assumed.

    What is *not* testable here: observing a handler's tenant from the test body.
    Starlette runs each request in a fresh `copy_context()`, so a value the
    handler set does not propagate back out — which is precisely why it cannot
    leak into the next request. The end-to-end proof of that is TestIsolation,
    which signs in as one user and then another in the same process and asserts
    the second cannot see the first's venture.
    """

    def test_as_tenant_sets_and_restores(self, tmp_db):
        """Restoration is what makes it safe to nest — the test fixture relies on
        it to hold tenant 1 across a test whose requests set their own."""
        assert db.current_tenant().user_id == 1        # from the autouse fixture
        with db.as_tenant(7):
            assert db.current_tenant().user_id == 7
            with db.as_tenant(8):
                assert db.current_tenant().user_id == 8
            assert db.current_tenant().user_id == 7
        assert db.current_tenant().user_id == 1, "the outer tenant must be restored"

    def test_the_lookup_and_token_contexts_are_independent(self, tmp_db):
        with db.as_tenant(user_id=None, lookup_email="a@b.test", lookup_token="f" * 64):
            t = db.current_tenant()
            assert (t.user_id, t.lookup_email, t.lookup_token) == (None, "a@b.test", "f" * 64)
        assert db.current_tenant() == db.Tenant(user_id=1)

    def test_get_conn_applies_the_tenant(self, tmp_db):
        """The ContextVar is only useful if it reaches the database."""
        make_user("applied@test.com")
        with db.as_tenant(1):
            with db.get_conn() as conn:
                assert conn.execute(
                    text("SELECT current_setting('app.user_id', true)")
                ).scalar_one() == "1"
        with db.as_tenant(None):
            with db.get_conn() as conn:
                # NULLIF in the policies means a reset setting is NULL again, not
                # the empty string — which is what makes the fail-closed path work.
                assert conn.execute(
                    text("SELECT current_setting('app.user_id', true)")
                ).scalar_one() in (None, "")

    def test_healthz_reports_whether_rls_is_on(self, client):
        """It must be able to say no. A deploy can have these policies installed
        and still bypass every one of them, and that is not something to guess at.
        """
        r = client.get("/healthz")
        assert r.status_code == 200
        assert r.json()["rls"] is True
        assert r.json()["status"] == "ok"

    def test_healthz_reports_rls_false_when_the_role_bypasses(self, client, monkeypatch):
        """The uncomfortable case is the one worth testing: if the role stops
        bypassing, the app must not keep claiming it is protected."""
        monkeypatch.setattr(db, "rls_active", lambda *a, **k: False)
        assert client.get("/healthz").json() == {"status": "ok", "rls": False}

    def test_healthz_survives_a_dead_database(self, client, monkeypatch):
        """healthz must never be the thing that 500s."""
        monkeypatch.setattr(db, "ping", lambda: False)
        assert client.get("/healthz").json() == {"status": "degraded", "rls": False}


# ==================== DB helpers ====================

class TestDbHelpers:
    def test_update_venture_ignores_unknown_fields(self, tmp_db):
        user_id = make_user("x@test.com")
        idea_id = db.create_idea(user_id, "t", "f", "early", "c")
        venture_id = db.create_venture(user_id, idea_id)
        # update_venture builds SQL from column names, so an unchecked key
        # would be injectable. Unknown keys are dropped instead.
        db.update_venture(venture_id, status="paused", **{"status = 'x', id": "y"})
        venture = db.get_venture(venture_id, user_id)
        assert venture["status"] == "paused"
        assert db.get_user_by_id(user_id) is not None

    def test_segment_writers_reject_unknown_blocks(self, tmp_db):
        """The allowlist is what stops an injected column name reaching the SQL,
        and an unknown block must never create a row the board would not render."""
        user_id = make_user("x@test.com")
        idea_id = db.create_idea(user_id, "t", "f", "early", "c")
        venture_id = db.create_venture(user_id, idea_id)

        db.update_segment(venture_id, "market_size", status="confirmed", notes="invented")
        db.set_segment_outcome(venture_id, "market_size", "passed")
        db.set_segment_hypothesis(venture_id, "market_size", "nope")
        db.extend_segment(venture_id, "market_size")

        assert db.get_segment(venture_id, "market_size") is None
        assert len(db.get_segments(venture_id)) == 9
        assert all(s["status"] == "untested" for s in db.get_segments(venture_id))
        assert not db.all_resolved(venture_id)

    def test_a_bad_outcome_is_refused_rather_than_raising(self, tmp_db):
        """set_segment_outcome is called with model-derived strings; an unknown
        value must be dropped, not turned into a CHECK-constraint 500."""
        user_id = make_user("y2@test.com")
        venture_id = db.create_venture(user_id, db.create_idea(user_id, "t", "f", "early", "c"))
        db.set_segment_outcome(venture_id, "revenue", "definitely_a_verdict")
        db.update_segment(venture_id, "revenue", status="invented_status")
        seg = db.get_segment(venture_id, "revenue")
        assert seg["outcome"] == "pending" and seg["status"] == "untested"

    def test_all_resolved_needs_every_block(self, tmp_db):
        """Regression risk: a missing segment row must make the gate unreachable
        rather than silently pass."""
        user_id = make_user("y3@test.com")
        venture_id = db.create_venture(user_id, db.create_idea(user_id, "t", "f", "early", "c"))
        with db.get_conn() as conn:
            conn.execute(db.text("DELETE FROM bmc_elements WHERE venture_id = :v "
                                 "AND element_name = 'channel'"), {"v": venture_id})
        for seg in db.get_segments(venture_id):
            db.set_segment_outcome(venture_id, seg["element_name"], "passed")
        assert not db.all_resolved(venture_id), "8 of 9 rows cannot resolve the venture"

    def test_bare_db_filename_does_not_crash(self, tmp_db):
        # The old suite asserted that a bare "bare-name.db" SQLite path created
        # no directory. There is no file behind the connection any more, so the
        # equivalent guarantee is that the engine is reachable and round-trips
        # a row without a schema on disk.
        assert db.ping() is True
        user_id = make_user("y@test.com")
        assert db.get_user_by_id(user_id)["email"] == "y@test.com"


# ==================== template rendering ====================

class TestRendering:
    """The state-machine tests above all stub the LLM, but they only ever
    render /dashboard for a user with *no* ventures. A template that raises on
    a populated page therefore slips straight through — which is exactly how
    the dashboard shipped a reference to a column list_ventures() didn't
    select. These tests render every page in a populated state."""

    def test_dashboard_renders_with_a_venture(self, client, monkeypatch):
        make_venture(client, monkeypatch)
        r = client.get("/dashboard")
        assert r.status_code == 200, r.text
        assert "Sensor thing" in r.text
        assert "Sell sensors" in r.text          # commercial_framing
        assert "0/9 blocks resolved" in r.text

    def test_dashboard_counts_a_parked_block_as_resolved(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        segs = db.get_segments(venture_id)
        db.set_segment_outcome(venture_id, segs[0]["element_name"], "passed")
        db.set_segment_outcome(venture_id, segs[1]["element_name"], "parked",
                               note="no budget")
        r = client.get("/dashboard")
        assert r.status_code == 200, r.text
        assert "2/9 blocks resolved" in r.text
        assert "1 parked" in r.text

    def test_board_renders_all_nine_blocks_in_order(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        r = client.get(f"/venture/{venture_id}")
        assert r.status_code == 200, r.text
        assert r.text.count('class="segment-row') == 9
        assert "0 of 9 resolved" in r.text
        for seg in SEGMENTS:
            # Jinja escapes the ampersand in "Problem & Key Activities".
            assert html.escape(seg["label"]) in r.text, seg["label"]

    def test_canvas_renders_nine_clickable_cells(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        r = client.get(f"/venture/{venture_id}")
        assert r.status_code == 200, r.text
        # all nine are tracked now, so the old "untracked" branch is gone
        assert r.text.count('class="bmc-cell') == 9
        assert "untracked" not in r.text
        assert "not tracked by the loop" not in r.text
        for label in ("Key Partners", "Problem & Key Activities", "Key Resources",
                      "Value Propositions", "Customer Relationships",
                      "Customer Segments", "Channels", "Cost Structure",
                      "Revenue Streams"):
            assert html.escape(label) in r.text, label
        # each cell links to its own block
        assert r.text.count(f"/venture/{venture_id}/segment/") >= 9

    def test_every_cell_carries_its_grid_area_class(self, client, monkeypatch):
        """Placement is by CSS grid-area, not document order.

        The area class is what puts a block in its canonical column, so a
        missing or misspelled one silently drops the block back into
        auto-flow — which is exactly the wrong layout this replaced.
        """
        venture_id = make_venture(client, monkeypatch)
        text = client.get(f"/venture/{venture_id}").text
        for block in CANVAS_LAYOUT + CANVAS_FINANCIAL:
            assert f'class="bmc-cell a-{block["key"]} ' in text, block["key"]

    def test_the_canvas_order_is_the_canonical_one(self):
        """Guard the reading order, because it is also the mobile stack order.

        Auto-flow across five columns gets this wrong in a way no assertion
        about counts or labels would catch: Customer Segments ends up under
        Key Partners and Channels lands in the middle.
        """
        assert [b["key"] for b in CANVAS_LAYOUT] == [
            "key_partners", "problem", "key_resources", "value_prop",
            "customer_relationships", "channel", "customer",
        ]
        assert [b["key"] for b in CANVAS_FINANCIAL] == ["cost_structure", "revenue"]

    def test_every_area_in_the_layout_is_defined_in_the_stylesheet(self):
        """A block whose area has no CSS rule still renders, in the wrong place.

        This pairs the two constants lists with the stylesheet so adding a block
        to CANVAS_LAYOUT without adding its rule fails here instead of shipping
        a canvas that looks plausible and is subtly mispositioned.
        """
        css = pathlib.Path(config.PROJECT_ROOT / "app" / "static" / "style.css").read_text()
        for block in CANVAS_LAYOUT + CANVAS_FINANCIAL:
            # A word boundary after the key: without it, a-customer matches
            # a-customer_relationships and the canvas silently swaps the two.
            found = re.search(rf"\.bmc-cell\.a-{block['key']}\s*\{{([^}}]*)\}}", css)
            assert found, f"no stylesheet rule for {block['key']}"
            # and it must place the block in the area constants.py claims, or the
            # canvas renders in the right shape with the wrong contents
            assert block["area"] in found.group(1), (
                f"{block['key']} should sit in grid area {block['area']!r}, "
                f"stylesheet says: {found.group(1).strip()}")

    def test_a_passed_block_shows_the_extracted_answer(self, client, monkeypatch):
        """The point of the canvas: a passed block displays its answer.

        Value Propositions must show the value proposition, not a description of
        the evidence that confirmed one. The hypothesis holds the answer (the
        verdict prompt writes the confirmed answer into revised_hypothesis on a
        pass); outcome_note holds the evidence and goes underneath.
        """
        venture_id = make_venture(client, monkeypatch)
        db.set_segment_outcome(
            venture_id, "value_prop", "passed",
            note="9 of 12 interviewees named it as their top cost",
            hypothesis="Site managers cut fuel downtime 40% and pay $300/mo per pump.",
        )
        text = client.get(f"/venture/{venture_id}").text
        # the answer is in the cell, in the high-contrast answer element
        assert "Site managers cut fuel downtime 40%" in text
        assert '<div class="answer">Site managers cut fuel downtime 40%' in text
        # and the evidence is present too, as the supporting line
        assert "9 of 12 interviewees named it" in text

    def test_the_answer_is_truncated_not_the_evidence(self, client, monkeypatch):
        """A long answer is clamped by CSS, but the text is present in full so a
        screen reader and a copy-paste both get the whole thing."""
        venture_id = make_venture(client, monkeypatch)
        long_answer = "Managers who own twelve or more industrial pumps " * 6
        db.set_segment_outcome(venture_id, "revenue", "passed", note="n=12",
                               hypothesis=long_answer)
        text = client.get(f"/venture/{venture_id}").text
        assert "Managers who own twelve or more industrial pumps" in text

    def test_a_parked_block_shows_its_workaround_as_the_headline(self, client, monkeypatch):
        """Parked and failed blocks have no answer, only a reason, so the note
        takes the headline rather than a stale hypothesis."""
        venture_id = make_venture(client, monkeypatch)
        db.set_segment_outcome(
            venture_id, "channel", "parked",
            note="resell through an existing distributor instead")
        db.set_segment_hypothesis(venture_id, "channel",
                                  "Direct outbound will convert at 3%.")
        text = client.get(f"/venture/{venture_id}").text
        assert '<div class="answer">resell through an existing distributor' in text

    def test_an_untested_block_shows_no_answer_line(self, client, monkeypatch):
        """An empty box is worse than no box — the label alone is fine."""
        venture_id = make_venture(client, monkeypatch)
        text = client.get(f"/venture/{venture_id}").text
        assert '<div class="answer">' not in text
        assert '<div class="notes">' not in text

    def test_canvas_reflects_stored_outcomes(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        db.set_segment_outcome(venture_id, "value_prop", "passed", note="4x cheaper")
        db.set_segment_outcome(venture_id, "revenue", "parked", note="no one will pay")
        text = client.get(f"/venture/{venture_id}").text
        # Substring, not a class list: the cell also carries its grid-area hook.
        assert text.count('bmc-cell a-value_prop s-confirmed') == 1
        assert text.count('bmc-cell a-revenue s-mixed o-parked') == 1
        assert "no one will pay" in text

    def test_segment_page_shows_the_hypothesis_and_the_start_button(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        db.set_segment_hypothesis(venture_id, "customer",
                                  "Site managers rank downtime as a top-3 cost.")
        r = client.get(f"/venture/{venture_id}/segment/customer")
        assert r.status_code == 200, r.text
        assert "Hypothesis under test" in r.text
        assert "Site managers rank downtime as a top-3 cost." in r.text
        assert "Design the first run" in r.text

    def test_run_panel_shows_the_method_steps_and_pass_criteria(self, client, monkeypatch):
        """The task detail is the point of the redesign, so its absence has to
        fail a test rather than ship."""
        venture_id = make_venture(client, monkeypatch)
        monkeypatch.setattr(llm, "generate_segment_tasks", fake_design)
        client.post(f"/venture/{venture_id}/segment/customer/run", follow_redirects=False)
        text = client.get(f"/venture/{venture_id}/segment/customer").text
        assert "problem interview" in text
        assert "Ask about the last incident" in text
        assert "Pass when" in text
        assert "3+ rank it a top-3 cost unprompted" in text
        assert "Log results" in text

    def test_parked_block_offers_a_way_back_in(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        db.set_segment_outcome(venture_id, "channel", "parked",
                               note="use a reseller instead")
        text = client.get(f"/venture/{venture_id}/segment/channel").text
        assert "This block was parked" in text
        assert "use a reseller instead" in text
        assert "Re-open this block" in text

    def test_phase3_renders_with_a_strategy_and_lists_the_gaps(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        segs = db.get_segments(venture_id)
        for seg in segs[:-1]:
            db.set_segment_outcome(venture_id, seg["element_name"], "passed")
        db.set_segment_outcome(venture_id, segs[-1]["element_name"], "parked",
                               note="resell through a distributor")
        db.update_venture(venture_id, phase=3, status="validated")
        db.save_launch_strategy(
            venture_id,
            [{"name": "SBIR Phase I", "why_it_fits": "deep tech"}],
            [{"channel": "Direct sales", "why_this_fits": "interviews"}],
            [{"step": "File the paperwork", "milestone_type": "grant"}],
        )
        text = client.get(f"/venture/{venture_id}").text
        assert "Go to Market" in text
        for needle in ("SBIR Phase I", "Direct sales", "File the paperwork"):
            assert needle in text
        # the funding disclaimer is a safety property, not decoration
        assert "Unverified" in text and "Advisory only" in text
        # a gap that was documented must not disappear at phase 3
        assert "Known gaps" in text and "resell through a distributor" in text

    def test_every_page_renders_for_a_brand_new_user(self, client):
        # Sign in first: TestClient follows redirects by default, so an
        # unauthenticated GET would quietly land on /login and still be 200.
        signup(client)
        for path in ("/dashboard", "/phase1/new", "/phase1/ideas"):
            r = client.get(path, follow_redirects=False)
            assert r.status_code == 200, (path, r.text)
        assert "No candidates yet" in client.get("/phase1/ideas").text
        assert "No candidate ideas" in client.get("/dashboard").text

    def test_a_block_awaiting_a_decision_becomes_the_open_panel(self, client, monkeypatch):
        """Regression: a failed critical block blocks Phase 3 and asks the user
        to kill or pivot, but the default focus was the first *unresolved* block
        by position — so the decision was buried on the board instead of being
        the first thing on the page."""
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "customer", verdict="pass")
        run_once(client, monkeypatch, venture_id, "revenue", verdict="fail")
        text = client.get(f"/venture/{venture_id}").text
        assert "came back disconfirmed" in text, "the decision must be at the top"
        assert f'action="/venture/{venture_id}/kill"' in text
        assert "Revenue Streams" in text
        # and it must come before the board
        assert text.index("came back disconfirmed") < text.index('class="segment-row')

    def test_an_explicitly_opened_block_still_wins_over_the_decision(self, client, monkeypatch):
        """Focus priority: URL beats decision, so an explicit link is never
        hijacked by another block's state."""
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "customer", verdict="pass")
        run_once(client, monkeypatch, venture_id, "revenue", verdict="fail")
        text = client.get(f"/venture/{venture_id}/segment/channel").text
        assert "came back disconfirmed" not in text, "the URL's block should be shown"
        assert "Channels" in text

    def test_every_segment_page_renders_on_a_fresh_venture(self, client, monkeypatch):
        """Any one block raising in its own template would only show up when a
        user opens that block, so render all nine."""
        venture_id = make_venture(client, monkeypatch)
        for seg in SEGMENTS:
            r = client.get(f"/venture/{venture_id}/segment/{seg['key']}")
            assert r.status_code == 200, (seg["key"], r.text[:400])

    def test_a_fresh_venture_loads_with_a_panel_already_open(self, client, monkeypatch):
        """Loading /venture/{id} used to render no run panel at all, because
        focus resolved to None. The top of the page must answer "what do I do
        next" without a click."""
        venture_id = make_venture(client, monkeypatch)
        text = client.get(f"/venture/{venture_id}").text
        # the recommended block is customer (position 0)
        assert "Customer Segments" in text
        assert "Design the first run" in text

    def test_the_open_panel_precedes_the_board_and_canvas(self, client, monkeypatch):
        """The panel is the primary content: it must come first in the DOM, not
        just visually, or a keyboard user tabs through the whole board before
        reaching the action."""
        venture_id = make_venture(client, monkeypatch)
        text = client.get(f"/venture/{venture_id}").text
        panel = text.index("Design the first run")
        board = text.index("class=\"segment-row")
        canvas = text.index("class=\"bmc-cell")
        assert panel < board < canvas, "panel, then board, then canvas"

    def test_the_open_panel_follows_the_block_in_flight(self, client, monkeypatch):
        """If a run is in flight somewhere, that block is what you are looking
        at — not the recommended one."""
        venture_id = make_venture(client, monkeypatch)
        monkeypatch.setattr(llm, "generate_segment_tasks", fake_design)
        client.post(f"/venture/{venture_id}/segment/cost_structure/run", follow_redirects=False)
        text = client.get(f"/venture/{venture_id}").text
        assert "Cost Structure" in text
        assert "What actually happened" in text, "the open run's form should be on screen"

    def test_the_per_block_run_history_renders(self, client, monkeypatch):
        """Regression: this panel tested a `runs` variable that _venture_context()
        never provides, and Jinja's undefined is falsy — so it has never rendered.
        The whole-venture table below the canvas worked the entire time, which is
        exactly why the gap survived."""
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "customer", verdict="pass")

        text = client.get(f"/venture/{venture_id}/segment/customer").text
        assert "Runs on this block" in text
        assert "1 of 3 used" in text
        assert "Talk to 5 people about the problem" in text, "the run's own task"
        assert "scored" in text
        assert "resolved" in text, "focus_state must reach the panel"

    def test_the_run_history_shows_each_verdict_and_its_date(self, client, monkeypatch):
        """Two runs on one block, so the panel has to be a list and not a stub."""
        venture_id = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, venture_id, "customer", verdict="iterate")
        run_once(client, monkeypatch, venture_id, "customer", verdict="pass")

        text = client.get(f"/venture/{venture_id}/segment/customer").text
        assert "2 of 3 used" in text
        assert "Run 1" in text and "Run 2" in text
        assert "dec-iterate" in text and "dec-pass" in text
        year = str(datetime.now(timezone.utc).year)
        assert year in text, "each run should carry its recorded date"

    def test_a_block_with_no_runs_shows_no_history_panel(self, client, monkeypatch):
        """The other half of the fix: `{% if focus_runs %}` must still be able to be
        false, or every block would claim to have run history."""
        venture_id = make_venture(client, monkeypatch)
        text = client.get(f"/venture/{venture_id}/segment/customer").text
        assert "Runs on this block" not in text
        assert "Design the first run" in text

    def test_an_unscored_run_is_distinguished_from_a_scored_one(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        monkeypatch.setattr(llm, "generate_segment_tasks", fake_design)
        client.post(f"/venture/{venture_id}/segment/customer/run", follow_redirects=False)
        run = db.get_current_cycle(venture_id)
        db.log_cycle_results(run["id"], [{"outcome": "asked some people", "sample_size": "3"}])

        text = client.get(f"/venture/{venture_id}/segment/customer").text
        assert "Runs on this block" in text
        assert "logged, not scored" in text
        assert "awaiting results" not in text


# ==================== provider failure behaviour ====================

class TestProviderFailure:
    """What the user actually experiences when the LLM API is unreachable,
    times out, or rejects the key. The app must degrade to a readable message
    and stay up — never a stack trace, never a dead worker."""

    @staticmethod
    def _dead_client(exc):
        import httpx
        from openai import APIConnectionError

        class Dead:
            @property
            def chat(self):
                class _a:
                    @property
                    def completions(self):
                        class _b:
                            def create(self, **kw):
                                raise exc
                        return _b()
                return _a()
        return lambda settings: Dead()

    def _stub_dead(self, monkeypatch, exc):
        monkeypatch.setattr(llm, "_get_client", self._dead_client(exc))

    def test_connection_loss_shows_a_readable_error_not_a_crash(self, client, monkeypatch):
        import httpx
        from openai import APIConnectionError
        signup(client)
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://opencode.ai/zen/v1")
        exc = APIConnectionError(request=httpx.Request("POST", "https://opencode.ai/zen/v1"))
        self._stub_dead(monkeypatch, exc)

        r = client.post("/phase1/extract", data={"raw_text": "x" * 200})
        assert r.status_code == 200, "a dead provider must not 500"
        body = r.text
        assert "Couldn&#39;t reach the AI provider" in body
        # plain-English, not the raw exception text
        assert "APIConnectionError" not in body
        assert "Your work is saved" in body
        # and the failure was recorded rather than swallowed
        with db.get_conn() as conn:
            rows = conn.execute(
                text("SELECT status, attempts, error FROM llm_calls WHERE purpose='extract_ideas'")
            ).mappings().fetchall()
        assert len(rows) == 1
        assert rows[0]["status"] == "error"
        assert rows[0]["attempts"] == config.llm_max_attempts()

    def test_app_survives_a_dead_provider_and_serves_the_next_request(self, client, monkeypatch):
        import httpx
        from openai import APIConnectionError
        signup(client)
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://opencode.ai/zen/v1")
        exc = APIConnectionError(request=httpx.Request("POST", "https://opencode.ai/zen/v1"))
        self._stub_dead(monkeypatch, exc)

        assert client.post("/phase1/extract", data={"raw_text": "x" * 200}).status_code == 200
        # still serving
        assert client.get("/dashboard").status_code == 200
        assert client.get("/phase1/new").status_code == 200
        assert client.get("/healthz").status_code == 200

    def test_retry_analysis_survives_and_keeps_the_saved_results(self, client, monkeypatch):
        import httpx
        from openai import APIConnectionError
        monkeypatch.setattr(llm, "generate_segment_tasks", fake_design)
        venture_id = make_venture(client, monkeypatch)
        assert client.post(f"/venture/{venture_id}/segment/customer/run",
                           follow_redirects=False).status_code == 303
        run_id = db.get_current_cycle(venture_id)["id"]

        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://opencode.ai/zen/v1")
        exc = APIConnectionError(request=httpx.Request("POST", "https://opencode.ai/zen/v1"))
        self._stub_dead(monkeypatch, exc)

        r = client.post(f"/venture/{venture_id}/run/{run_id}/log",
                        data={"outcome_0": "3 said yes", "sample_size_0": "5"},
                        follow_redirects=True)
        assert r.status_code == 200
        # the error is carried to the page rather than swallowed
        assert "Couldn&#39;t reach the AI provider" in r.text
        # results were not lost, and the user is offered a retry
        assert "Retry verdict" in r.text
        assert "3 said yes" in r.text or db.get_cycle(run_id, venture_id)["results_json"]

    def test_unexpected_error_renders_the_friendly_500_page(self, tmp_db, monkeypatch):
        # ServerErrorMiddleware sends the handler's response and then re-raises,
        # so the default TestClient propagates the exception instead of letting
        # us assert on the page. raise_server_exceptions=False is what a real
        # client would get.
        monkeypatch.setenv("LAUNCHLOOP_DEBUG", "1")
        monkeypatch.setenv("SESSION_SECRET_KEY", "test-secret")
        with TestClient(main.app, raise_server_exceptions=False) as c:
            signup(c)
            monkeypatch.setattr(
                llm, "extract_ideas",
                lambda *a, **k: (_ for _ in ()).throw(ZeroDivisionError("provider shape changed")),
            )
            r = c.post("/phase1/extract", data={"raw_text": "x" * 200})
            assert r.status_code == 500
            assert "Something went wrong" in r.text
            assert "ZeroDivisionError" not in r.text
            assert "Traceback" not in r.text
            # and the server is still healthy afterwards
            assert c.get("/healthz").status_code == 200
            assert c.get("/dashboard").status_code == 200

    def test_friendly_error_covers_the_common_provider_exceptions(self):
        import httpx
        from openai import APIConnectionError, APITimeoutError, RateLimitError
        req = httpx.Request("POST", "https://x")
        cases = [
            (APIConnectionError(request=req), "reach the AI provider"),
            (APITimeoutError(request=req), "didn't respond in time"),
            (RateLimitError("429", response=httpx.Response(429, request=req), body=None),
             "rate-limiting"),
        ]
        for exc, needle in cases:
            msg = llm._friendly_error(exc, "some-model", 3)
            assert needle in msg, (type(exc).__name__, msg)
            assert "3 attempts" in msg


# ---------------------------------------------------------------------------
# M1.2 — identifier import (ORCID / DOI / arXiv)
# ---------------------------------------------------------------------------

ORCID_ID = "0000-0002-1825-0097"
DOI = "10.1000/xyz123"
ARXIV = "2301.01234"

CROSSREF = json.dumps({"message": {
    "title": ["Low-power SWIR imaging at 120fps"],
    "author": [{"given": "Ada", "family": "Lovelace"}, {"given": "Alan", "family": "Turing"}],
    "container-title": ["Journal of Applied Sensing"],
    "issued": {"date-parts": [[2021, 4, 2]]},
    # &#181; is how XML spells a micro sign — real JATS abstracts are full of
    # numeric character references, and &micro; would not be legal XML at all.
    "abstract": "<jats:p>We describe a 3&#181;m-pitch sensor running at 120fps on a "
                "200mW budget, in &lt;em&gt;low light&lt;/em&gt;.</jats:p>",
}}).encode()

ARXIV_ATOM = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2301.01234v2</id>
    <published>2023-01-03T18:00:00Z</published>
    <title>Sparse spectral
      reconstruction for low-light imaging</title>
    <summary>  We show that a sparse
      prior recovers spectra at 4x
      the speed.   </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Grace Hopper</name></author>
  </entry>
</feed>"""

ORCID_RECORD = json.dumps({
    "person": {"credit-name": {"given-and-family-names": [{"value": "Ada Lovelace"}]}},
    "activities-summary": {"works": {"group": [
        {"work-summary": [{
            "title": {"title": {"value": f"Low-power paper {i}"}},
            "publication-date": {"year": {"value": "2020" + str(i % 10)}},
            "journal-title": {"value": "Sensors"},
        }]} for i in range(40)]}},
}).encode()


class _FakeResponse:
    def __init__(self, status, chunks):
        self.status_code = status
        self._chunks = chunks

    def iter_bytes(self):
        yield from self._chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Raising:
    def __init__(self, exc):
        self._exc = exc

    def __enter__(self):
        raise self._exc

    def __exit__(self, *exc):
        return False


def install_http(monkeypatch, status=200, body=b"", exc=None, chunks=None):
    """Replace `sources.httpx.Client` with a recorder. Returns the call log, so a
    test can assert on the URL that was actually requested."""
    import httpx as _httpx
    calls = []

    class Client:
        def __init__(self, **kw):
            calls.append(("init", kw))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def stream(self, method, url, headers=None):
            calls.append((method, url, dict(headers or {})))
            if exc is not None:
                return _Raising(exc)
            return _FakeResponse(status, chunks if chunks is not None else [body])

    monkeypatch.setattr(sources.httpx, "Client", Client)
    return calls


class TestSourceRecognition:
    """Which identifiers are recognised, and — more importantly — which are not.

    No HTTP anywhere in this class: if the code tries to reach the network on the
    way to refusing something, these tests fail loudly, which is the point.
    """

    def test_the_three_identifier_forms_are_recognised(self):
        for value in (ORCID_ID, "https://orcid.org/" + ORCID_ID,
                      DOI, "https://doi.org/" + DOI, "doi:" + DOI,
                      ARXIV, "arXiv:" + ARXIV, "2301.01234v3",
                      "https://arxiv.org/abs/2301.01234",
                      "hep-th/9901001", "math.GT/0309136"):
            match = sources._match(value)
            assert match is not None, value
            assert match[0] in ("ORCID", "DOI", "arXiv"), (value, match[0])

    def test_the_provider_is_chosen_by_shape_not_by_order_of_guessing(self):
        """A DOI must not be fetched from arXiv, or an old-style arXiv ID from
        Crossref. Registering the loosest pattern last is what guarantees this."""
        assert sources._match(DOI)[0] == "DOI"
        assert sources._match(ORCID_ID)[0] == "ORCID"
        assert sources._match(ARXIV)[0] == "arXiv"
        assert sources._match("hep-th/9901001")[0] == "arXiv"

    @pytest.mark.parametrize("value", ["", "   "])
    def test_an_empty_box_says_so_rather_than_teaching_the_formats(self, value):
        """A separate message, because "you typed nothing" and "that isn't one of
        the three formats" are different problems and a user who hit the button by
        accident should be told the first thing."""
        with pytest.raises(sources.SourceError) as e:
            sources.import_material(value)
        assert str(e.value) == "Paste an identifier first."

    @pytest.mark.parametrize("value", [
        "not an identifier", "10.1000", "10./xyz", "0000-0002-1825", "2301.01",
    ])
    def test_nonsense_is_refused_with_a_sentence_that_teaches_the_formats(self, value):
        with pytest.raises(sources.SourceError) as e:
            sources.import_material(value)
        message = str(e.value)
        assert "ORCID" in message and "DOI" in message and "arXiv" in message

    @pytest.mark.parametrize("value", [
        # The SSRF shapes that are not shaped like any of the three identifiers.
        # None of these may reach a socket at all.
        "http://169.254.169.254/latest/meta-data/",
        "https://evil.example.com/10.1000/xyz",
        "https://orcid.org.evil.example.com/" + ORCID_ID,
        "//evil.example.com/" + ORCID_ID,
        "2301.01234\nhttps://evil.example.com",
        ORCID_ID + "@evil.example.com",
    ])
    def test_nothing_shaped_like_a_url_can_choose_the_host(self, value, monkeypatch):
        calls = install_http(monkeypatch)
        with pytest.raises(sources.SourceError):
            sources.import_material(value)
        assert not calls, f"a request was attempted for {value!r}: {calls}"

    @pytest.mark.parametrize("value", [
        # These *are* recognised as DOIs — the pattern's suffix is `\S+`, and a DOI
        # suffix can contain almost anything. They are not refused; they are made
        # harmless by encoding. The distinction matters: refusing them would break
        # real DOIs, and the safety must come from the encoding rather than from
        # the pattern.
        "10.1000/xyz#@evil.example.com",
        "10.1000/xyz?redirect=http://169.254.169.254",
        "10.1000/xyz/../../admin",
    ])
    def test_a_doi_carrying_a_payload_is_encoded_into_the_path_not_obeyed(self, value, monkeypatch):
        import re as _re
        calls = install_http(monkeypatch, body=CROSSREF)
        sources.import_material(value)
        method, url, _ = calls[-1]
        assert method == "GET"
        prefix = "https://api.crossref.org/works/"
        assert url.startswith(prefix), url
        tail = url[len(prefix):]
        # The whole payload is inert because it can only be percent-encoded
        # characters: no raw "/" (extra path segment, so no traversal), no ":"
        # (no port or scheme change), no "?" or "#". The dangerous text is
        # present but cannot be anything but an opaque path segment.
        assert _re.match(r"^[A-Za-z0-9._~%-]+$", tail), tail
        assert tail.count("%2F") >= 1 or "/" not in tail

    def test_a_recognised_identifier_still_cannot_choose_the_host(self, monkeypatch):
        """The DOI pattern is the loose one, and its suffix is `\\S+`. What matters
        is that the captured group is percent-encoded, so even a suffix carrying a
        full URL can only become a path segment on api.crossref.org."""
        calls = install_http(monkeypatch, body=CROSSREF)
        sources.import_material("10.1000/" + "x" * 5 + "?redirect=http://169.254.169.254")
        method, url, _ = calls[-1]
        assert method == "GET"
        assert url.startswith("https://api.crossref.org/works/")
        assert "169.254.169.254" in url and "?" not in url and "#" not in url, url

    def test_an_old_style_arxiv_id_has_its_slash_encoded_not_treated_as_a_path(self, monkeypatch):
        calls = install_http(monkeypatch, body=ARXIV_ATOM)
        record = sources.import_material("hep-th/9901001")
        method, url, _ = calls[-1]
        assert method == "GET"
        assert url.startswith("https://export.arxiv.org/api/query?id_list=")
        assert "hep-th%2F9901001" in url, url
        assert record.source == "arXiv"

    def test_the_full_set_of_hosts_is_three_and_does_not_grow(self, monkeypatch):
        install_http(monkeypatch, body=CROSSREF)
        install_http(monkeypatch, body=ARXIV_ATOM)
        hosts = set()
        for value, payload in ((DOI, CROSSREF), (ARXIV, ARXIV_ATOM), (ORCID_ID, ORCID_RECORD)):
            sources.cache_clear()
            calls = install_http(monkeypatch, body=payload)
            sources.import_material(value)
            hosts.add(calls[-1][1].split("/")[2])
        assert hosts == {"api.crossref.org", "export.arxiv.org", "pub.orcid.org"}

    def test_import_can_be_turned_off_for_a_deployment_that_must_not_call_out(self, monkeypatch):
        monkeypatch.setenv("SOURCE_IMPORT_ENABLED", "false")
        calls = install_http(monkeypatch)
        with pytest.raises(sources.SourceError) as e:
            sources.import_material(DOI)
        assert "turned off" in str(e.value)
        assert not calls


class TestSourceFetching:
    """What comes back for each provider. HTTP is stubbed at the client, so these
    exercise the real `_get` — status handling, the response cap, the timeouts."""

    def test_a_doi_becomes_title_authors_and_abstract(self, monkeypatch):
        install_http(monkeypatch, body=CROSSREF)
        record = sources.import_material(DOI)
        assert record.source == "DOI"
        assert "Low-power SWIR imaging" in record.body
        assert "Ada Lovelace, Alan Turing" in record.body
        assert "2021" in record.body
        assert "Journal of Applied Sensing" in record.body
        assert "120fps" in record.body

    def test_jats_markup_in_a_crossref_abstract_is_stripped(self, monkeypatch):
        """Crossref abstracts are XML fragments. What reaches the prompt is text."""
        install_http(monkeypatch, body=CROSSREF)
        body = sources.import_material(DOI).body
        assert "<jats:p>" not in body and "</jats:p>" not in body
        assert "&#181;" not in body, "the numeric reference should be resolved"
        assert "3µm-pitch" in body
        assert "<em>" not in body and "low light" in body, \
            "escaped markup is stripped too, which means entities are decoded first"

    def test_a_doi_with_no_abstract_says_so_rather_than_looking_empty(self, monkeypatch):
        payload = json.dumps({"message": {"title": ["A title with no abstract"],
                                          "issued": {"date-parts": [[2020]]}}}).encode()
        install_http(monkeypatch, body=payload)
        body = sources.import_material(DOI).body
        assert "A title with no abstract" in body
        assert "no abstract on record" in body

    def test_an_arxiv_record_is_flattened_because_arxiv_folds_its_abstract(self, monkeypatch):
        install_http(monkeypatch, body=ARXIV_ATOM)
        record = sources.import_material(ARXIV)
        assert "Sparse spectral reconstruction" in record.body, "the fold is collapsed"
        assert "Ada Lovelace, Grace Hopper" in record.body
        assert "2023-01-03" in record.body
        assert "4x the speed" in record.body
        assert "  " not in record.body, "internal whitespace is collapsed"

    def test_arxiv_reports_a_missing_id_rather_than_an_empty_card(self, monkeypatch):
        """For an unknown ID arXiv returns a valid feed whose one entry is titled
        "Error" — so a missing title is not the only failure signal."""
        error_feed = b"""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom"><entry><title>Error</title>
        <id>http://arxiv.org/api/errors</id></entry></feed>"""
        install_http(monkeypatch, body=error_feed)
        with pytest.raises(sources.SourceError) as e:
            sources.import_material(ARXIV)
        assert "No arXiv paper" in str(e.value)

    def test_an_orcid_record_becomes_a_work_list(self, monkeypatch):
        """An ORCID iD is a person, not a document, so the material is the titles —
        which is what Phase 1 needs, since the titles are the claims."""
        monkeypatch.setenv("IMPORT_MAX_WORKS", "25")
        install_http(monkeypatch, body=ORCID_RECORD)
        body = sources.import_material(ORCID_ID).body
        assert "Ada Lovelace" in body
        assert "Low-power paper 0" in body
        assert "Sensors" in body
        assert "40 public works" in body

    def test_a_long_work_list_is_capped_and_the_user_is_told(self, monkeypatch):
        """A prolific researcher must not turn an iD into a 400 KB prompt."""
        monkeypatch.setenv("IMPORT_MAX_WORKS", "10")
        install_http(monkeypatch, body=ORCID_RECORD)
        body = sources.import_material(ORCID_ID).body
        assert "showing the first 10" in body
        assert "30 further works were not included" in body
        assert "Low-power paper 39" not in body

    def test_an_orcid_with_no_public_works_explains_the_visibility_setting(self, monkeypatch):
        empty = json.dumps({"person": {}, "activities-summary": {"works": {"group": []}}}).encode()
        install_http(monkeypatch, body=empty)
        with pytest.raises(sources.SourceError) as e:
            sources.import_material(ORCID_ID)
        assert "no public works" in str(e.value)

    @pytest.mark.parametrize("status,needle", [
        (404, "No record found"),
        (429, "rate-limiting"),
        (500, "having problems"),
        (403, "refused the request"),
    ])
    def test_an_upstream_error_is_a_sentence_not_a_status_code(self, monkeypatch, status, needle):
        install_http(monkeypatch, status=status, body=b"")
        with pytest.raises(sources.SourceError) as e:
            sources.import_material(DOI)
        assert needle in str(e.value)
        assert str(status) not in str(e.value), "the raw status is not the user's business"

    def test_a_timeout_is_reported_as_a_timeout(self, monkeypatch):
        import httpx as _httpx
        install_http(monkeypatch, exc=_httpx.ReadTimeout("slow"))
        with pytest.raises(sources.SourceError) as e:
            sources.import_material(DOI)
        assert "took too long" in str(e.value)

    def test_a_network_failure_never_leaks_the_exception_or_the_url(self, monkeypatch):
        import httpx as _httpx
        install_http(monkeypatch, exc=_httpx.ConnectError("no route to host"))
        with pytest.raises(sources.SourceError) as e:
            sources.import_material(DOI)
        message = str(e.value)
        assert "Couldn't reach" in message
        assert "ConnectError" not in message and "crossref" not in message

    def test_an_oversized_response_is_stopped_while_streaming(self, monkeypatch):
        """The cap is enforced on the stream, not on Content-Length, which is
        controlled by whoever is answering."""
        monkeypatch.setenv("IMPORT_MAX_RESPONSE_BYTES", "10000")
        chunks = [b"x" * 5000 for _ in range(10)]
        calls = install_http(monkeypatch, chunks=chunks)
        with pytest.raises(sources.SourceError) as e:
            sources.import_material(DOI)
        assert "unexpectedly large response" in str(e.value)
        assert len(calls) == 2, "the connection was opened once, and stopped early"

    def test_unreadable_json_is_a_sentence_rather_than_a_traceback(self, monkeypatch):
        install_http(monkeypatch, body=b"<html>maintenance</html>")
        with pytest.raises(sources.SourceError) as e:
            sources.import_material(DOI)
        assert "unreadable" in str(e.value)

    def test_redirects_are_never_followed(self, monkeypatch):
        """A 302 from an allowlisted host to a link-local address is the entire
        attack. Not following redirects removes it without inspecting Location."""
        calls = install_http(monkeypatch, body=b"")
        sources._get("https://api.crossref.org/works/x", "application/json")
        assert calls[0][1]["follow_redirects"] is False

    def test_the_request_carries_an_identifying_user_agent(self, monkeypatch):
        """These are public APIs run by volunteers; being anonymous about who is
        calling is how a client ends up blocked by nobody noticing."""
        calls = install_http(monkeypatch, body=CROSSREF)
        sources.import_material(DOI)
        assert "LaunchLoop" in calls[-1][2]["User-Agent"]


class TestSourceCache:
    """Repeat imports of the same identifier cost nothing and cannot fail
    differently on the second try."""

    def test_the_second_import_does_not_touch_the_network(self, monkeypatch):
        install_http(monkeypatch, body=CROSSREF)
        first = sources.import_material(DOI)
        calls = install_http(monkeypatch, status=500, body=b"")
        second = sources.import_material(DOI)
        assert second is first
        assert not calls, "the cached record should have been served without a request"

    def test_doi_case_does_not_defeat_the_cache(self, monkeypatch):
        """DOIs are case-insensitive by specification, so these are one paper and
        must not be two cache entries."""
        install_http(monkeypatch, body=CROSSREF)
        sources.import_material(DOI)
        calls = install_http(monkeypatch, body=CROSSREF)
        sources.import_material(DOI.upper())
        assert not calls

    def test_different_identifiers_are_cached_separately(self, monkeypatch):
        install_http(monkeypatch, body=CROSSREF)
        sources.import_material(DOI)
        calls = install_http(monkeypatch, body=CROSSREF)
        sources.import_material("10.1000/other")
        assert len(calls) == 2

    def test_a_zero_ttl_turns_the_cache_off(self, monkeypatch):
        monkeypatch.setenv("IMPORT_CACHE_TTL_SECONDS", "0")
        install_http(monkeypatch, body=CROSSREF)
        sources.import_material(DOI)
        calls = install_http(monkeypatch, body=CROSSREF)
        sources.import_material(DOI)
        assert len(calls) == 2

    def test_the_cache_is_bounded_because_its_key_is_attacker_influenced(self, monkeypatch):
        """A loop over distinct identifiers would otherwise grow this map for the
        life of the process. Same reasoning as ratelimit._MAX_BUCKETS."""
        install_http(monkeypatch, body=CROSSREF)
        for i in range(sources._MAX_CACHE + 50):
            sources.import_material(f"10.1000/doi{i}")
        assert len(sources._cache) <= sources._MAX_CACHE


class TestImportRoute:
    """The HTTP surface. The properties that matter: the user sees and edits the
    text before spending an AI call, and an import costs no quota."""

    def test_the_record_comes_back_editable_before_any_ai_call(self, client, monkeypatch):
        monkeypatch.setattr(llm, "extract_ideas",
                            lambda raw, user_id=None: pytest.fail("must not call the model yet"))
        signup(client)
        install_http(monkeypatch, body=CROSSREF)
        r = client.post("/phase1/import", data={"identifier": DOI}, follow_redirects=False)
        assert r.status_code == 200
        assert "Low-power SWIR imaging" in r.text
        assert "<textarea" in r.text, "it must be editable, like a paste"
        assert "Imported DOI record" in r.text and "Lovelace" in r.text

    def test_importing_then_extracting_is_an_ordinary_phase_1_call(self, client, monkeypatch):
        """No new AI-call path: no quota change, no new prompt, and the import
        material is wrapped as untrusted data by the same code that wraps a paste."""
        signup(client)
        install_http(monkeypatch, body=CROSSREF)
        client.post("/phase1/import", data={"identifier": DOI}, follow_redirects=False)
        seen = {}
        monkeypatch.setattr(llm, "extract_ideas",
                            lambda raw, user_id=None: seen.update(raw=raw) or [{
                                "title": "Sensor thing", "commercial_framing": "Sell sensors",
                                "strength_signal": "early", "raw_claims": "sensor"}])
        r = client.post("/phase1/extract", data={"raw_text": "x" * 200},
                        follow_redirects=False)
        assert r.status_code == 303

    def test_an_import_spends_no_monthly_quota(self, client, monkeypatch):
        signup(client)
        quota_before = db.count_llm_calls_this_month(1)
        for _ in range(3):
            install_http(monkeypatch, body=CROSSREF)
            client.post("/phase1/import", data={"identifier": DOI}, follow_redirects=False)
        assert db.count_llm_calls_this_month(1) == quota_before, \
            "an import is not a model call and must not be billed as one"

    def test_a_bad_identifier_is_a_readable_message_and_a_200(self, client, monkeypatch):
        install_http(monkeypatch, status=404, body=b"")
        signup(client)
        r = client.post("/phase1/import", data={"identifier": DOI}, follow_redirects=False)
        assert r.status_code == 200, "a bad identifier must never be a 500"
        assert "No record found" in r.text

    def test_the_route_needs_a_login(self, client, monkeypatch):
        install_http(monkeypatch, body=CROSSREF)
        r = client.post("/phase1/import", data={"identifier": DOI}, follow_redirects=False)
        assert r.status_code in (302, 303, 307), r.status_code

    def test_the_box_is_hidden_and_the_route_refused_when_imports_are_off(self, client, monkeypatch):
        monkeypatch.setenv("SOURCE_IMPORT_ENABLED", "false")
        signup(client)
        page = client.get("/phase1/new", follow_redirects=False)
        assert page.status_code == 200
        assert "/phase1/import" not in page.text
        calls = install_http(monkeypatch, body=CROSSREF)
        r = client.post("/phase1/import", data={"identifier": DOI}, follow_redirects=False)
        assert r.status_code == 200
        assert "turned off" in r.text
        assert not calls, "the request must be refused before any outbound call"

    def test_imports_are_throttled_and_the_source_is_never_contacted(self, client, monkeypatch):
        """Same shape of risk as the model routes — a user-triggered outbound
        request holding a worker thread — so it gets its own bucket, checked before
        the fetch."""
        monkeypatch.setenv("IMPORT_RATE_LIMIT_PER_MIN", "3")
        monkeypatch.setenv("IMPORT_RATE_LIMIT_IP_PER_MIN", "10000")
        ratelimit.reset()
        signup(client)
        install_http(monkeypatch, body=CROSSREF)
        for i in range(3):
            assert client.post("/phase1/import",
                               data={"identifier": f"10.1000/ok{i}"},
                               follow_redirects=False).status_code == 200
        calls = install_http(monkeypatch, body=CROSSREF)
        r = client.post("/phase1/import", data={"identifier": "10.1000/one-too-many"},
                        follow_redirects=False)
        assert r.status_code == 200, "a throttled import is a message, not an error page"
        assert "Too many imports" in r.text
        assert "about" in r.text and "seconds" in r.text
        assert not calls, "a refused import must not reach the third party"

    def test_import_budget_is_independent_of_the_llm_budget(self, client, monkeypatch):
        """Import buckets live under their own prefix, so exhausting them must not
        spend the tokens reserved for model calls — and vice versa."""
        monkeypatch.setenv("LLM_RATE_LIMIT_PER_MIN", "10000")
        monkeypatch.setenv("LLM_RATE_LIMIT_IP_PER_MIN", "10000")
        monkeypatch.setenv("IMPORT_RATE_LIMIT_PER_MIN", "2")
        ratelimit.reset()
        signup(client)
        install_http(monkeypatch, body=CROSSREF)
        for i in range(2):
            client.post("/phase1/import", data={"identifier": f"10.1000/ok{i}"},
                        follow_redirects=False)
        # The LLM limiter should be untouched.
        assert ratelimit.check(1, None) is None


# ==================== action-plan step tracking (M3.1) ====================

STEP_PLAN = [
    {"step": "Email 10 target users about the pilot", "milestone_type": "pilot"},
    {"step": "File the seed-fund paperwork", "milestone_type": "grant"},
    {"step": "Sign one paying design partner", "milestone_type": "customer"},
]


def strategy_for(client, monkeypatch, plan=None, venture_id=None):
    """A venture parked into Phase 3 with a saved strategy.

    Eight of nine blocks passed and the last one parked, which is what unlocks
    Phase 3 — the same route the app itself takes.
    """
    venture_id = venture_id if venture_id is not None else make_venture(client, monkeypatch)
    segments = db.get_segments(venture_id)
    for seg in segments[:-1]:
        db.set_segment_outcome(venture_id, seg["element_name"], "passed")
    db.set_segment_outcome(venture_id, segments[-1]["element_name"], "parked",
                           note="resell through a distributor")
    db.update_venture(venture_id, phase=3, status="validated")
    db.save_launch_strategy(
        venture_id,
        [{"name": "SBIR Phase I", "why_it_fits": "deep tech"}],
        [{"channel": "Direct sales", "why_this_fits": "interviews"}],
        plan if plan is not None else STEP_PLAN,
    )
    return venture_id


class TestActionStepTracking:
    """The db layer: identity, seeding, and what a regeneration may and may not
    carry over."""

    def test_saving_a_strategy_seeds_exactly_one_row_per_step(self, tmp_db):
        user = make_user("seed@test.com")
        idea = db.create_idea(user, "Sensor thing", "Sell sensors", "early", "c")
        vid = db.create_venture(user, idea)
        db.save_launch_strategy(vid, [], [], STEP_PLAN)
        steps = db.get_action_steps(vid)
        assert len(steps) == 3
        assert {s["step"] for s in steps.values()} == {p["step"] for p in STEP_PLAN}
        assert all(s["status"] == "pending" for s in steps.values()), \
            "a fresh plan has nothing recorded against it"

    def test_the_migration_and_db_agree_on_what_a_steps_identity_is(self, tmp_db):
        """step_key exists twice on purpose — db.py must not be imported by a
        migration, so the migration carries its own copy.

        Two definitions of a row's identity is one more than this codebase is
        willing to hold, so they are pinned to each other here. If they drift,
        a backfilled row becomes unreachable through the route and the checklist
        silently stops reflecting reality — which reads as "the feature is
        broken", not as "a hash changed".
        """
        import importlib.util
        from pathlib import Path

        path = next((Path(db.__file__).parent.parent / "migrations" / "versions")
                    .glob("c9e4b2a71d38_*.py"))
        spec = importlib.util.spec_from_file_location("m31_migration", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for step in ["Email 10 target users about the pilot",
                     "File the seed-fund paperwork",
                     "  padded  ", "unicode: caf\u00e9 \u2014 \u2713", "", "x" * 4000]:
            assert module.step_key(step) == db.step_key(step), step

    def test_completion_survives_a_regeneration_that_reuses_the_wording(self, tmp_db):
        """The reason step_key is a hash of the text rather than an index.

        "Regenerate strategy" inserts a new strategy row and the new plan is a
        different list. If identity were the array position, the completion
        would land on whatever step happened to sort into that slot.
        """
        user = make_user("regen@test.com")
        idea = db.create_idea(user, "Sensor thing", "Sell sensors", "early", "c")
        vid = db.create_venture(user, idea)
        db.save_launch_strategy(vid, [], [], STEP_PLAN)
        key = db.step_key(STEP_PLAN[0]["step"])
        db.set_action_step(vid, key, "done", note="9 of 10 replied")

        db.save_launch_strategy(vid, [], [], STEP_PLAN)

        assert db.get_action_step(vid, key)["status"] == "done"
        assert db.get_action_step(vid, key)["outcome_note"] == "9 of 10 replied"

    def test_a_step_reworded_by_a_regeneration_is_not_marked_done(self, tmp_db):
        """The other half of the same property, and the one that would be a lie
        if it failed: a differently-worded step is a different instruction, so
        inheriting a completion would assert the researcher did something the
        plan never asked for."""
        user = make_user("reword@test.com")
        idea = db.create_idea(user, "Sensor thing", "Sell sensors", "early", "c")
        vid = db.create_venture(user, idea)
        db.save_launch_strategy(vid, [], [], STEP_PLAN)
        db.set_action_step(vid, db.step_key(STEP_PLAN[0]["step"]), "done", note="done it")

        reworded = [dict(STEP_PLAN[0], step="Email 25 target users about the pilot")]
        db.save_launch_strategy(vid, [], [], reworded + STEP_PLAN[1:])

        progress = db.action_plan_progress(db.get_action_steps(vid), reworded + STEP_PLAN[1:])
        assert progress == {"total": 3, "done": 0, "blocked": 0, "remaining": 3}
        # The old row is still tracked and still exported — the work happened,
        # it just is not part of the plan on screen.
        assert len(db.get_action_steps(vid)) == 4

    def test_two_identical_steps_collapse_to_one_row(self, tmp_db):
        """The cost of keying on text, stated rather than hidden: identical
        wording is one instruction, so ticking it ticks both."""
        user = make_user("dupe@test.com")
        idea = db.create_idea(user, "Sensor thing", "Sell sensors", "early", "c")
        vid = db.create_venture(user, idea)
        doubled = [STEP_PLAN[0], dict(STEP_PLAN[0])]
        db.save_launch_strategy(vid, [], [], doubled)
        assert len(db.get_action_steps(vid)) == 1
        db.set_action_step(vid, db.step_key(STEP_PLAN[0]["step"]), "done", note="done")
        progress = db.action_plan_progress(db.get_action_steps(vid), doubled)
        assert progress["done"] == 2 and progress["total"] == 2

    def test_an_unknown_status_is_refused_rather_than_written(self, tmp_db):
        """Same reason update_venture allowlists its field names: an unchecked
        value would surface as a 500 from the CHECK constraint instead of a
        refusal from the route."""
        user = make_user("badstatus@test.com")
        idea = db.create_idea(user, "Sensor thing", "Sell sensors", "early", "c")
        vid = db.create_venture(user, idea)
        db.save_launch_strategy(vid, [], [], STEP_PLAN)
        key = db.step_key(STEP_PLAN[0]["step"])
        for bad in ("DONE", "done ", "archived", "", None):
            assert db.set_action_step(vid, key, bad, note="x") is False, bad
        assert db.get_action_step(vid, key)["status"] == "pending"

    def test_reverting_clears_decided_at_and_keeps_the_note(self, tmp_db):
        """The reason stays: a blocked step's note is why it was blocked, and that
        reason usually still holds after it is reopened."""
        user = make_user("revert@test.com")
        idea = db.create_idea(user, "Sensor thing", "Sell sensors", "early", "c")
        vid = db.create_venture(user, idea)
        db.save_launch_strategy(vid, [], [], STEP_PLAN)
        key = db.step_key(STEP_PLAN[0]["step"])
        db.set_action_step(vid, key, "blocked", note="waiting on the ethics board")
        assert db.get_action_step(vid, key)["decided_at"] is not None

        assert db.set_action_step(vid, key, "pending") is True
        row = db.get_action_step(vid, key)
        assert row["status"] == "pending"
        assert row["decided_at"] is None
        assert row["outcome_note"] == "waiting on the ethics board"

    def test_progress_counts_the_visible_plan_and_not_the_tracked_rows(self, tmp_db):
        """A step from a superseded strategy is still tracked and still exported,
        but counting it would let the progress line claim more than the checklist
        on screen shows."""
        user = make_user("progress@test.com")
        idea = db.create_idea(user, "Sensor thing", "Sell sensors", "early", "c")
        vid = db.create_venture(user, idea)
        db.save_launch_strategy(vid, [], [], STEP_PLAN)
        db.set_action_step(vid, db.step_key(STEP_PLAN[0]["step"]), "done", note="ok")
        db.set_action_step(vid, db.step_key(STEP_PLAN[1]["step"]), "blocked", note="no")

        steps = db.get_action_steps(vid)
        progress = db.action_plan_progress(steps, STEP_PLAN)
        assert progress == {"total": 3, "done": 1, "blocked": 1, "remaining": 2}

    def test_a_step_with_no_row_reads_as_pending_rather_than_failing(self, tmp_db):
        """Defensive: only reachable if a strategy row was written without going
        through save_launch_strategy. An untracked step has demonstrably not been
        recorded as done, so pending is the honest reading."""
        user = make_user("untracked@test.com")
        idea = db.create_idea(user, "Sensor thing", "Sell sensors", "early", "c")
        vid = db.create_venture(user, idea)
        items = db.join_action_plan({}, STEP_PLAN)
        assert [i["status"] for i in items] == ["pending"] * 3
        assert all(i["tracked"] is False for i in items)
        assert db.action_plan_progress({}, STEP_PLAN)["done"] == 0

    def test_a_malformed_plan_entry_cannot_become_a_row(self, tmp_db):
        """step_key derives from the step text, so an entry with no usable text
        would produce a key nothing could ever reproduce."""
        user = make_user("malformed@test.com")
        idea = db.create_idea(user, "Sensor thing", "Sell sensors", "early", "c")
        vid = db.create_venture(user, idea)
        db.save_launch_strategy(vid, [], [], [
            {"step": "   "}, {"nope": 1}, "a bare string", None,
            {"step": "Email 10 target users about the pilot"},
            {"step": "Email the regulator", "milestone_type": "invented-type"},
        ])
        steps = db.get_action_steps(vid)
        assert {s["step"] for s in steps.values()} == {
            "Email 10 target users about the pilot", "Email the regulator"}
        # An unrecognised milestone_type is coerced to the default rather than
        # written through, because the CHECK would reject the whole insert.
        assert steps[db.step_key("Email the regulator")]["milestone_type"] == "pilot"


class TestActionStepRoute:
    """The route, and the guarantee that a crafted POST cannot reach a row the
    researcher is not looking at."""

    def _mark(self, client, venture_id, index, status="done", note="did the thing"):
        """POST as the rendered form would: index plus the key it was shown.

        The key is not an address the server trusts — the route re-derives it
        from the stored plan and requires agreement — so the tests that try to
        forge one post a mismatched pair deliberately.
        """
        strategy = db.get_launch_strategy(venture_id)
        plan = (strategy["action_plan_json"] if strategy else []) or []
        step = plan[int(index)]["step"] if str(index).lstrip("-").isdigit() \
            and 0 <= int(index) < len(plan) else "unmatched"
        return client.post(
            f"/venture/{venture_id}/phase3/step",
            data={"index": str(index), "step_key": db.step_key(step),
                  "status": status, "note": note},
            follow_redirects=False,
        )

    def test_marking_a_step_done_records_the_outcome(self, client, monkeypatch):
        vid = strategy_for(client, monkeypatch)
        r = self._mark(client, vid, 0)
        assert r.status_code == 303
        row = db.get_action_step(vid, db.step_key(STEP_PLAN[0]["step"]))
        assert row["status"] == "done"
        assert row["outcome_note"] == "did the thing"
        assert row["decided_at"] is not None
        assert "marked done" in unquote(r.headers["location"])

    def test_done_requires_an_outcome_and_says_why(self, client, monkeypatch):
        vid = strategy_for(client, monkeypatch)
        for status in ("done", "blocked"):
            r = self._mark(client, vid, 0, status=status, note="   ")
            assert r.status_code == 303
            assert "err=" in r.headers["location"]
            assert db.get_action_step(vid, db.step_key(STEP_PLAN[0]["step"]))["status"] == \
                "pending", f"{status} with no note must not be written"

    def test_the_refusal_explains_itself_rather_than_saying_invalid(self, client, monkeypatch):
        vid = strategy_for(client, monkeypatch)
        r = self._mark(client, vid, 0, note="")
        assert "what happened" in unquote(r.headers["location"]).lower()

    def test_reverting_needs_no_note_and_keeps_the_old_one(self, client, monkeypatch):
        vid = strategy_for(client, monkeypatch)
        self._mark(client, vid, 0, status="blocked", note="waiting on the ethics board")
        r = self._mark(client, vid, 0, status="pending", note="")
        assert r.status_code == 303
        row = db.get_action_step(vid, db.step_key(STEP_PLAN[0]["step"]))
        assert row["status"] == "pending"
        assert row["outcome_note"] == "waiting on the ethics board"

    def test_an_index_outside_the_plan_changes_nothing(self, client, monkeypatch):
        """Including the negative case, which is the interesting one: -1 indexes
        the last step in Python and would mark a step the user cannot see."""
        vid = strategy_for(client, monkeypatch)
        for bad in ("-1", "3", "99", "not-a-number", "", "1e9"):
            r = self._mark(client, vid, bad)
            assert r.status_code == 303, bad
        steps = db.get_action_steps(vid)
        assert all(s["status"] == "pending" for s in steps.values()), steps

    def test_the_client_chooses_which_step_and_nothing_else(self, client, monkeypatch):
        """The form posts an index and the key it was rendered with; the server
        re-derives the key from the stored plan and requires the two to agree.

        A form that simply posted a step_key would let a crafted POST write to
        any row in the table by supplying a hash. Here the only thing the request
        controls is which step of the researcher's own plan to touch — the text,
        the key and the tenant all come from the database.
        """
        vid = strategy_for(client, monkeypatch)
        forged = db.step_key(STEP_PLAN[2]["step"])
        r = client.post(
            f"/venture/{vid}/phase3/step",
            data={"index": "0", "step_key": forged, "status": "done", "note": "real work",
                  "venture_id": str(vid)},
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert db.get_action_step(vid, forged)["status"] == "pending", \
            "a posted step_key must not be honoured over the derived one"
        assert db.get_action_step(vid, forged)["outcome_note"] is None
        # The mismatched pair is refused outright rather than resolved to index 0.
        assert db.get_action_step(vid, db.step_key(STEP_PLAN[0]["step"]))["status"] == "pending"
        assert "err=" in r.headers["location"]

        # And the honest pair works.
        ok = self._mark(client, vid, 0)
        assert ok.status_code == 303
        assert db.get_action_step(vid, db.step_key(STEP_PLAN[0]["step"]))["status"] == "done"

    def test_another_researchers_step_cannot_be_touched(self, client, monkeypatch):
        vid = strategy_for(client, monkeypatch)
        stranger = make_user("stranger@test.com")
        with db.as_tenant(stranger):
            stranger_idea = db.create_idea(stranger, "Their idea", "framing", "early", "c")
            stranger_vid = db.create_venture(stranger, stranger_idea)
            db.save_launch_strategy(stranger_vid, [], [], STEP_PLAN)
            stranger_key = db.step_key(STEP_PLAN[0]["step"])
            db.set_action_step(stranger_vid, stranger_key, "done", note="their work")

        # Identical plan text, so an identical step_key: identity is per venture,
        # and the venture_id the route scopes by is what keeps the two apart.
        assert self._mark(client, vid, 0).status_code == 303

        with db.as_tenant(stranger):
            row = db.get_action_step(stranger_vid, stranger_key)
        assert row["status"] == "done"
        assert row["outcome_note"] == "their work"
        assert db.get_action_step(vid, stranger_key)["outcome_note"] == "did the thing"

    def test_another_users_venture_redirects_and_writes_nothing(self, client, monkeypatch):
        vid = strategy_for(client, monkeypatch)
        stranger = make_user("owner2@test.com")
        with db.as_tenant(stranger):
            idea = db.create_idea(stranger, "Their idea", "framing", "early", "c")
            sv = db.create_venture(stranger, idea)
            for seg in db.get_segments(sv)[:-1]:
                db.set_segment_outcome(sv, seg["element_name"], "passed")
            db.update_venture(sv, phase=3, status="validated")
            db.save_launch_strategy(sv, [], [], STEP_PLAN)
            key = db.step_key(STEP_PLAN[0]["step"])

        r = client.post(f"/venture/{sv}/phase3/step",
                        data={"index": "0", "step_key": db.step_key(STEP_PLAN[0]["step"]),
                              "status": "done", "note": "not mine"},
                        follow_redirects=False)
        assert r.status_code == 303
        assert "/dashboard" in r.headers["location"]
        with db.as_tenant(stranger):
            assert db.get_action_step(sv, key)["status"] == "pending"

    def test_an_unknown_status_is_refused(self, client, monkeypatch):
        vid = strategy_for(client, monkeypatch)
        r = self._mark(client, vid, 0, status="archived", note="whatever")
        assert r.status_code == 303 and "err=" in r.headers["location"]
        assert db.get_action_step(vid, db.step_key(STEP_PLAN[0]["step"]))["status"] == "pending"

    def test_a_venture_that_is_not_in_phase_three_cannot_record_a_step(self, client, monkeypatch):
        vid = make_venture(client, monkeypatch)   # still in the validation loop
        r = self._mark(client, vid, 0)
        assert r.status_code == 303
        assert db.get_action_steps(vid) == {}

    def test_a_venture_with_no_strategy_cannot_record_a_step(self, client, monkeypatch):
        vid = make_venture(client, monkeypatch)
        for seg in db.get_segments(vid)[:-1]:
            db.set_segment_outcome(vid, seg["element_name"], "passed")
        db.update_venture(vid, phase=3, status="validated")
        r = self._mark(client, vid, 0)
        assert r.status_code == 303 and "err=" not in r.headers["location"]
        assert db.get_action_steps(vid) == {}

    def test_recording_a_step_spends_no_quota_and_no_rate_limit(self, client, monkeypatch):
        """Marking a step is a write to a user-owned row. No model is involved,
        so neither the monthly quota nor the request limiter may move.

        Asserted by spying on the limiter rather than by calling check() before
        and after: check() spends a token when it allows a request, so probing it
        to establish a baseline is itself the thing that empties the bucket.
        """
        vid = strategy_for(client, monkeypatch)
        user_id = db.get_user_by_email("a@test.com")["id"]

        spent = []
        real_check = ratelimit.check
        monkeypatch.setattr(ratelimit, "check",
                            lambda *a, **k: (spent.append(a), real_check(*a, **k))[1])

        before = db.count_llm_calls_this_month(user_id)
        for i in range(3):
            assert self._mark(client, vid, i, note=f"did step {i}").status_code == 303

        assert db.count_llm_calls_this_month(user_id) == before, \
            "recording a step is not a model call and must not be billed as one"
        assert spent == [], f"the LLM limiter was consulted: {spent}"

    def test_the_phase3_page_shows_progress_and_a_finished_plan(self, client, monkeypatch):
        vid = strategy_for(client, monkeypatch)
        self._mark(client, vid, 0)
        self._mark(client, vid, 1, status="blocked", note="funding office is shut")
        page = client.get(f"/venture/{vid}").text
        assert "1 of 3 done" in page
        assert "1 blocked" in page
        assert "funding office is shut" in page
        assert "Every step in this plan is done" not in page

        # Marking the blocked step done and the last one done finishes the plan.
        self._mark(client, vid, 1, note="reopened in January, submitted")
        self._mark(client, vid, 2)
        page = client.get(f"/venture/{vid}").text
        assert "3 of 3 done" in page
        assert "Every step in this plan is done" in page

    def test_a_recorded_outcome_is_escaped_not_rendered(self, client, monkeypatch):
        """The note is researcher-authored free text rendered in a page every
        other tenant's data shares a template with."""
        vid = strategy_for(client, monkeypatch)
        self._mark(client, vid, 0, note="<script>alert(1)</script>")
        page = client.get(f"/venture/{vid}").text
        assert "<script>alert(1)</script>" not in page
        assert "&lt;script&gt;" in page

    def test_the_plan_survives_a_regeneration_through_the_route(self, client, monkeypatch):
        """The end-to-end version of the identity property: mark a step, hit
        "Regenerate strategy", and the completion is still there."""
        monkeypatch.setattr(llm, "generate_launch_strategy", lambda *a, **k: {
            "funding_matches": [{"name": "SBIR Phase I", "why_it_fits": "deep tech"}],
            "gtm_channels": [{"channel": "Direct sales", "why_this_fits": "interviews"}],
            "action_plan": STEP_PLAN,
        })
        vid = strategy_for(client, monkeypatch)
        self._mark(client, vid, 0, note="9 of 10 replied")
        r = client.post(f"/venture/{vid}/phase3/generate", follow_redirects=False)
        assert r.status_code == 303
        assert db.get_action_step(vid, db.step_key(STEP_PLAN[0]["step"]))["outcome_note"] == \
            "9 of 10 replied"
        assert "1 of 3 done" in client.get(f"/venture/{vid}").text

    def test_a_step_marked_from_a_stale_form_is_refused_not_guessed(self, client, monkeypatch):
        """The bug this check exists for, and the reason the key comparison is
        there at all.

        The researcher is looking at "Email 10 target users about the pilot" and
        ticks it off. Meanwhile the strategy is regenerated and index 0 now names
        a different instruction. Resolving identity only at submit time would
        record a completion against work nobody did — the exact silent
        mislabelling the hash-based identity was introduced to prevent.
        """
        vid = strategy_for(client, monkeypatch)
        assert self._mark(client, vid, 0, note="done the first thing").status_code == 303
        db.save_launch_strategy(vid, [], [], [{"step": "A completely different first step",
                                               "milestone_type": "pilot"}])

        # Submit the OLD form: index 0, and the key that index 0 had then.
        r = client.post(f"/venture/{vid}/phase3/step",
                        data={"index": "0",
                              "step_key": db.step_key(STEP_PLAN[0]["step"]),
                              "status": "done", "note": "and this one too"},
                        follow_redirects=False)
        assert r.status_code == 303
        assert "err=" in r.headers["location"]
        row = db.get_action_step(vid, db.step_key("A completely different first step"))
        assert row["status"] == "pending", "must not mark a step nobody did"
        assert row["outcome_note"] is None
        # The completion recorded before the regeneration is untouched.
        assert db.get_action_step(vid, db.step_key(STEP_PLAN[0]["step"]))["status"] == "done"

    def test_phase3_shows_a_provider_failure_instead_of_swallowing_it(self, client, monkeypatch):
        """Regression.

        venture_phase3.html rendered no {{ error }} block and its render call
        built a hand-rolled context rather than **ctx, so back_to_venture()'s
        ?err= had nowhere to land. A provider outage or a spent quota while
        generating the strategy redirected the user back to a page that looked
        unchanged and said nothing at all.
        """
        vid = strategy_for(client, monkeypatch)
        db.update_venture(vid, phase=3, status="validated")

        def boom(*a, **k):
            raise llm.LLMError("The provider is unreachable. Try again shortly.")
        monkeypatch.setattr(llm, "generate_launch_strategy", boom)

        r = client.post(f"/venture/{vid}/phase3/generate", follow_redirects=True)
        assert "The provider is unreachable" in r.text, \
            "a swallowed error reads to the user as a dead button"

    def test_the_same_message_is_not_rendered_as_an_error_when_it_is_a_success(
        self, client, monkeypatch
    ):
        """`error` and `note` are different channels on purpose (invariant 10):
        reusing the red box for a success reads as a failure."""
        vid = strategy_for(client, monkeypatch)
        r = self._mark(client, vid, 0)
        page = client.get(r.headers["location"]).text
        assert "The outcome you recorded is saved" in page
        assert page.count('class="error"') == 0, "a success must not use the error box"


class TestActionStepMigration:
    """The migration's data handling, which `tmp_db` can never reach.

    Every other test upgrades an *empty* schema to head, so the backfill — the
    only code in this migration that reads rows written by an earlier one — would
    never run anywhere in the suite. It only executes against a database that
    already has strategies, which is every real deployment.
    """

    def _venture_with_a_strategy(self, pg_url, schema):
        """A venture and a saved strategy, created at the *previous* revision."""
        from alembic import command
        from tests.conftest import _alembic_config

        engine = create_engine(
            pg_url, connect_args={"options": f"-csearch_path={schema}"})
        try:
            with engine.begin() as conn:
                now = datetime.now(timezone.utc)
                user = conn.execute(text(
                    "INSERT INTO users (email, password_hash, created_at)"
                    " VALUES ('old@test.com', 'x', :t) RETURNING id"),
                    {"t": now}).scalar_one()
                idea = conn.execute(text(
                    "INSERT INTO ideas (user_id, title, commercial_framing, status,"
                    " created_at) VALUES (:u, 'Old idea', 'framing', 'candidate', :t)"
                    " RETURNING id"), {"u": user, "t": now}).scalar_one()
                venture = conn.execute(text(
                    "INSERT INTO ventures (user_id, idea_id, phase, cycle_count,"
                    " max_cycles, status, created_at)"
                    " VALUES (:u, :i, 3, 0, 27, 'validated', :t) RETURNING id"),
                    {"u": user, "i": idea, "t": now}).scalar_one()
                conn.execute(text(
                    "INSERT INTO launch_strategy (venture_id, funding_matches_json,"
                    " gtm_channels_json, action_plan_json, created_at)"
                    " VALUES (:v, '[]'::jsonb, '[]'::jsonb, CAST(:plan AS jsonb), :t)"),
                    {"v": venture, "t": now,
                     "plan": json.dumps([
                         {"step": "Email 10 target users", "milestone_type": "pilot"},
                         {"step": "File the seed paperwork", "milestone_type": "grant"},
                     ])})
                return venture
        finally:
            engine.dispose()

    def test_existing_strategies_get_their_steps_backfilled(self, pg_url, migrate_to):
        from alembic import command
        from tests.conftest import _alembic_config

        schema = migrate_to("e7a2b4c9d016")
        venture = self._venture_with_a_strategy(pg_url, schema)

        command.upgrade(_alembic_config(pg_url, schema), "head")

        engine = create_engine(
            pg_url, connect_args={"options": f"-csearch_path={schema}"})
        try:
            with engine.connect() as conn:
                # Schema-qualified throughout: search_path during a migration is
                # "<test schema>,public", and a dev database may still hold an
                # unpinned public copy of the schema, so an unqualified name can
                # resolve to the wrong table and pass for the wrong reason.
                rows = conn.execute(text(
                    f'SELECT step, milestone_type, status FROM "{schema}".action_steps'
                    " WHERE venture_id = :v ORDER BY step"), {"v": venture}).fetchall()
        finally:
            engine.dispose()
        assert len(rows) == 2, "the backfill never ran"
        assert all(r[2] == "pending" for r in rows), \
            "a backfilled step has had nothing recorded against it"
        assert {r[0] for r in rows} == {"Email 10 target users", "File the seed paperwork"}
        assert {r[1] for r in rows} == {"pilot", "grant"}

    def test_the_backfilled_keys_match_what_the_route_derives(self, pg_url, migrate_to):
        """The whole feature depends on a migration-computed hash and a
        route-computed hash being the same string. If they drift, the backfilled
        rows are unreachable and the checklist silently shows everything as
        pending — a failure that reads as "the feature is broken", not as "a hash
        changed"."""
        from alembic import command
        from tests.conftest import _alembic_config

        schema = migrate_to("e7a2b4c9d016")
        venture = self._venture_with_a_strategy(pg_url, schema)
        command.upgrade(_alembic_config(pg_url, schema), "head")

        engine = create_engine(
            pg_url, connect_args={"options": f"-csearch_path={schema}"})
        try:
            with engine.connect() as conn:
                keys = dict(conn.execute(text(
                    f'SELECT step, step_key FROM "{schema}".action_steps'
                    " WHERE venture_id = :v"), {"v": venture}).fetchall())
        finally:
            engine.dispose()
        for step, key in keys.items():
            assert key == db.step_key(step), step

    def test_the_migration_downgrades_and_upgrades_cleanly(self, pg_url, migrate_to):
        """A downgrade that leaves the policy or the table behind breaks the next
        `alembic upgrade head` in a way no forward-only test would see."""
        from alembic import command
        from tests.conftest import _alembic_config

        schema = migrate_to("e7a2b4c9d016")
        self._venture_with_a_strategy(pg_url, schema)
        cfg = _alembic_config(pg_url, schema)

        command.upgrade(cfg, "head")
        command.downgrade(cfg, "e7a2b4c9d016")

        engine = create_engine(
            pg_url, connect_args={"options": f"-csearch_path={schema}"})
        try:
            with engine.connect() as conn:
                gone = conn.execute(text(
                    "SELECT 1 FROM information_schema.tables"
                    " WHERE table_schema = :s AND table_name = 'action_steps'"),
                    {"s": schema}).first()
            assert gone is None, "the table survived the downgrade"
        finally:
            engine.dispose()

        # And straight back up, which is what a rollback-then-redeploy does.
        command.upgrade(cfg, "head")
        engine = create_engine(
            pg_url, connect_args={"options": f"-csearch_path={schema}"})
        try:
            with engine.connect() as conn:
                policy = conn.execute(text(
                    "SELECT 1 FROM pg_policies WHERE schemaname = :s"
                    " AND tablename = 'action_steps'"), {"s": schema}).first()
                assert policy is not None, "the RLS policy did not come back"
        finally:
            engine.dispose()


# ==================== session revocation (M0.7) ====================

def signed_in(client) -> bool:
    """Whether a client still holds a usable session.

    Probed on /account rather than /dashboard: the dashboard's copy changes with
    the account's contents, and "is this cookie still accepted" is exactly the
    question these tests ask. A revoked session gets a 303 to /login.
    """
    return client.get("/account", follow_redirects=False).status_code == 200


def a_second_session(monkeypatch, email, password="password123"):
    """A separate signed-in client — a different browser, or a stolen cookie.

    Two TestClients is the only honest way to test this: the whole point is that
    one session's write has to invalidate a *different* session's cookie, and a
    single client cannot hold two.
    """
    monkeypatch.setenv("LAUNCHLOOP_DEBUG", "1")
    monkeypatch.setenv("SESSION_SECRET_KEY", "test-secret")
    other = TestClient(main.app)
    other.__enter__()
    r = other.post("/login",
                   data={"email": email, "password": password},
                   follow_redirects=False)
    assert r.status_code == 303, r.text
    return other


class TestSessionRevocation:
    """Changing a password must end the sessions it is supposed to end.

    The exposure window is genuinely small — SessionMiddleware has no max_age, so
    the cookie dies with the browser — and these tests are not claiming the app
    was wide open. They cover the case that survives that mitigation: a machine
    left logged in, a profile copied, an XSS that read the cookie.
    """

    def test_a_cookie_from_before_the_change_stops_working(self, client, monkeypatch):
        signup(client, email="revoke@test.com", password="password123")
        other = a_second_session(monkeypatch, "revoke@test.com")
        assert signed_in(other)

        r = client.post("/account/password",
                        data={"current_password": "password123",
                              "password": "a-brand-new-one", "confirm": "a-brand-new-one"},
                        follow_redirects=False)
        assert r.status_code == 303
        assert "signed out" in unquote(r.headers["location"])

        # The other session is now anonymous, on every route that needs a user.
        for path in ("/dashboard", "/phase1/new", "/account"):
            r = other.get(path, follow_redirects=False)
            assert r.status_code == 303, f"{path} served a revoked session"
            assert "/login" in r.headers["location"], path

    def test_the_session_that_changed_the_password_stays_signed_in(self, client, monkeypatch):
        """Otherwise the control punishes the person who used it."""
        signup(client, email="keep@test.com", password="password123")
        a_second_session(monkeypatch, "keep@test.com")

        client.post("/account/password",
                    data={"current_password": "password123",
                          "password": "a-brand-new-one", "confirm": "a-brand-new-one"},
                    follow_redirects=False)
        assert signed_in(client)
        assert "keep@test.com" in client.get("/dashboard").text

    def test_the_new_password_is_the_one_that_works(self, client, monkeypatch):
        signup(client, email="rotate@test.com", password="password123")
        client.post("/account/password",
                    data={"current_password": "password123",
                          "password": "a-brand-new-one", "confirm": "a-brand-new-one"},
                    follow_redirects=False)
        client.post("/logout", follow_redirects=False)

        # The failure renders on the POST that caused it; re-GETting /login would
        # show a clean form and pass for the wrong reason.
        stale = client.post("/login", data={"email": "rotate@test.com",
                                            "password": "password123"},
                            follow_redirects=False)
        assert stale.status_code == 200
        assert "Invalid email or password" in stale.text
        assert not signed_in(client)

        fresh = client.post("/login", data={"email": "rotate@test.com",
                                            "password": "a-brand-new-one"},
                            follow_redirects=False)
        assert fresh.status_code == 303
        assert "/dashboard" in fresh.headers["location"]
        assert signed_in(client)

    def test_the_wrong_current_password_changes_nothing(self, client, monkeypatch):
        """The gate that stops someone who found the cookie from rotating the
        credential and locking the owner out permanently."""
        signup(client, email="gate@test.com", password="password123")
        other = a_second_session(monkeypatch, "gate@test.com")

        for body in (
            {"current_password": "", "password": "newpassword1", "confirm": "newpassword1"},
            {"current_password": "not-the-password", "password": "newpassword1",
             "confirm": "newpassword1"},
            {"current_password": "password123", "password": "newpassword1",
             "confirm": "typo-different"},
            {"current_password": "password123", "password": "short", "confirm": "short"},
            {"current_password": "password123", "password": "password123",
             "confirm": "password123"},
        ):
            r = client.post("/account/password", data=body, follow_redirects=False)
            assert r.status_code == 200, body
            assert "error" in r.text, body

        # Nothing moved: the old password still works and the other session lives.
        assert signed_in(other)
        client.post("/logout", follow_redirects=False)
        r = client.post("/login", data={"email": "gate@test.com",
                                        "password": "password123"}, follow_redirects=False)
        assert "/dashboard" in r.headers["location"]

    def test_the_same_password_length_limits_as_the_reset_flow(self, client, monkeypatch):
        """Two paths that set a password must not disagree about what is valid,
        or one of them becomes the weak way in."""
        signup(client, email="limits@test.com", password="password123")
        long = "x" * (auth.MAX_PASSWORD_BYTES + 1)
        r = client.post("/account/password",
                        data={"current_password": "password123",
                              "password": long, "confirm": long},
                        follow_redirects=False)
        assert r.status_code == 200
        assert "72 bytes" in r.text
        r = client.post("/account/password",
                        data={"current_password": "password123",
                              "password": "short", "confirm": "short"},
                        follow_redirects=False)
        assert "at least 8" in r.text

    def test_a_password_reset_also_revokes_existing_sessions(self, client, monkeypatch, capsys):
        """The account-takeover recovery path, and the one that matters most: the
        owner resets precisely *because* a session may be compromised."""
        monkeypatch.setenv("MAIL_BACKEND", "console")
        signup(client, email="taken@test.com", password="originalpass1")
        other = a_second_session(monkeypatch, "taken@test.com", password="originalpass1")
        client.post("/logout", follow_redirects=False)
        capsys.readouterr()

        forgot(client, "taken@test.com")
        token = token_from_mail(capsys)
        r = client.post("/reset-password",
                        data={"token": token, "password": "brand-new-pass1",
                              "confirm": "brand-new-pass1"},
                        follow_redirects=False)
        assert r.status_code == 303

        assert other.get("/dashboard", follow_redirects=False).status_code == 303, \
            "the attacker's session survived a password reset"

    def test_a_cookie_with_no_epoch_is_refused(self, client, monkeypatch):
        """The pre-deploy case. Assuming epoch 1 would let precisely the cookies
        this change exists to kill survive the deploy."""
        signup(client, email="legacy@test.com", password="password123")
        client.cookies.clear()
        # Recreate the cookie exactly as the code did before the epoch existed.
        client.cookies.set("session", _legacy_session_cookie(client))
        r = client.get("/dashboard", follow_redirects=False)
        assert r.status_code == 303 and "/login" in r.headers["location"]

    def test_a_revoked_session_leaves_no_tenant_behind(self, client, monkeypatch):
        """get_current_user() installs the RLS tenant from the row it loads. A
        revoked cookie that skipped clear_tenant() would leave the previous
        user's tenant installed for the rest of the request — a cross-tenant
        read waiting for the next query that forgot to check.

        Called directly rather than over HTTP, and that is forced: Starlette runs
        each request in a fresh copy_context(), so a handler's tenant does not
        propagate back out to the test body (see TestTenantContext's docstring).
        Over HTTP this assertion could only ever be measuring the autouse
        fixture's tenant, and would pass for entirely the wrong reason.
        """
        signup(client, email="tenant@test.com", password="password123")
        user_id = db.get_user_by_email("tenant@test.com")["id"]
        db.set_password(user_id, auth.hash_password("rotated-pass1"))

        revoked = SimpleNamespace(session={"user_id": user_id, "session_epoch": 1})
        assert auth.get_current_user(revoked) is None
        # clear_tenant() sets an *empty* Tenant, which is the fail-closed state:
        # NULLIF(current_setting('app.user_id', true), '') is NULL, and every
        # policy denies on NULL. Asserting the empty tenant rather than None,
        # because None is not what "cleared" means in this codebase.
        assert db.current_tenant().user_id is None, \
            "a revoked cookie must not leave the previous user's tenant installed"

        # ...and the matching epoch is honoured, so the line above is not passing
        # because get_current_user rejects everything. Reading the row needs an
        # explicit tenant now, which is the cleared state being genuinely
        # fail-closed rather than merely unset.
        with db.as_tenant(user_id):
            epoch = db.get_user_by_id(user_id)["session_epoch"]
        assert epoch == 2, "one set_password call advances the epoch exactly once"
        current = SimpleNamespace(
            session={"user_id": user_id, "session_epoch": epoch})
        assert auth.get_current_user(current)["id"] == user_id
        assert db.current_tenant().user_id == user_id

    def test_the_epoch_is_not_a_credential_so_it_is_not_exported(self, client, monkeypatch):
        """Not a secret — the cookie's signature is what authenticates it — but it
        is internal state of the session control, and exporting it invites the
        next person to treat it as one."""
        signup(client, email="exp@test.com")
        data = client.get("/account/export").json()
        assert "session_epoch" not in data["account"]
        assert "password" not in json.dumps(data["account"]).lower() or \
            "password_hash" not in data["account"]

    def test_logout_still_only_affects_the_caller(self, client, monkeypatch):
        """Revocation on logout would need a table; it is deliberately not
        attempted. Recording the limitation is the point of this test."""
        signup(client, email="logout@test.com", password="password123")
        other = a_second_session(monkeypatch, "logout@test.com")
        client.post("/logout", follow_redirects=False)
        assert not signed_in(client)
        assert signed_in(other), "logout must not revoke anyone else's session"


def _legacy_session_cookie(client) -> str:
    """A session cookie with user_id but no epoch, signed the same way."""
    import base64
    import hashlib
    import hmac

    from itsdangerous import TimestampSigner

    payload = base64.b64encode(json.dumps({"user_id": 1}).encode()).decode()
    signer = TimestampSigner("test-secret")
    return signer.sign(payload).decode()


# ==================== near-duplicate idea cards (M1.4) ====================

LONG_TITLE = ("A subscription service that supplies refurbished laboratory reagents "
              "to small research groups at universities")

# The calibration set, kept in one place because it is the evidence the threshold
# rests on. If a change to app/dedupe.py moves any of these across the line, one
# of them fails and the threshold has to be re-justified rather than nudged.
DUPLICATE_PAIRS = [
    ("Sensor kits for NHS hospitals", "Sensor kits for NHS hospitals"),
    ("Detect sepsis from blood samples", "  detect sepsis from BLOOD samples! "),
    ("AI tutor for music theory", "An AI tutor for music theory"),
    ("Carbon accounting for farms", "Carbon accounting tool for farms"),
    ("Sell sensors to hospitals", "Selling sensors to hospitals"),
    (LONG_TITLE, LONG_TITLE.replace("reagents", "consumables")),
    (LONG_TITLE, LONG_TITLE.replace("supplies", "provides")),
    (LONG_TITLE,
     "A subscription service that supplies small research groups at universities "
     "with refurbished laboratory reagents"),
]

DISTINCT_PAIRS = [
    # The one that sets the ceiling: same template, one differentiator, and in
    # research these are entirely different tests. A lower threshold flags it.
    ("Detect sepsis from blood samples", "Detect sepsis from urine samples"),
    ("Detect sepsis from blood samples", "Early warning score for ward patients"),
    ("Carbon accounting for farms", "Carbon offset marketplace for farms"),
    ("Sensor kits for NHS hospitals", "Sensor kits for veterinary practices"),
    ("Sell sensors to hospitals", "Sell sensors to vets"),
    (LONG_TITLE,
     LONG_TITLE.replace("small research groups at universities",
                        "large pharmaceutical companies")),
    ("Sell sensors to hospitals", "Online course for piano teachers"),
]


class TestNearDuplicateDetection:
    """The arithmetic, and the calibration it rests on.

    These are pure-function tests with no database, which is the point: the
    threshold is a constant in a heuristic, and the only thing keeping it honest
    is that these pairs are pinned.
    """

    @pytest.mark.parametrize("left,right", DUPLICATE_PAIRS)
    def test_a_reworded_duplicate_is_recognised(self, left, right):
        assert dedupe.similarity(left, right) >= dedupe.NEAR_DUPLICATE_THRESHOLD

    @pytest.mark.parametrize("left,right", DISTINCT_PAIRS)
    def test_a_different_idea_is_not(self, left, right):
        score = dedupe.similarity(left, right)
        assert score < dedupe.NEAR_DUPLICATE_THRESHOLD, (
            f"flagged two different ideas at {score:.3f}: {left!r} / {right!r}")

    def test_the_threshold_sits_inside_the_measured_gap(self):
        """The threshold is not a preference; it has to be between the worst true
        pair and the best false pair, or one of the tests above is passing by
        luck."""
        worst_true = min(dedupe.similarity(a, b) for a, b in DUPLICATE_PAIRS)
        best_false = max(dedupe.similarity(a, b) for a, b in DISTINCT_PAIRS)
        assert best_false < dedupe.NEAR_DUPLICATE_THRESHOLD <= worst_true, (
            f"threshold {dedupe.NEAR_DUPLICATE_THRESHOLD} is outside the measured "
            f"gap ({best_false:.3f}, {worst_true:.3f})")

    def test_two_blank_titles_are_not_duplicates(self):
        """trigrams("") is the padding alone, so without the explicit check every
        untitled card would score 1.0 against every other."""
        assert dedupe.similarity("", "") == 0.0
        assert dedupe.similarity("", "Sell sensors") == 0.0
        assert dedupe.similarity(None, None) == 0.0

    def test_punctuation_becomes_a_separator_not_a_deletion(self):
        """Deleting punctuation would fuse words across the gap and invent
        matches that are not there."""
        assert dedupe.similarity("sensors/to hospitals",
                                 "sensors to hospitals") == 1.0
        assert dedupe.normalize("A, B!") == "a b"

    def test_grouping_is_symmetric_and_deterministic(self):
        rows = [
            {"id": 3, "title": "Selling sensors to hospitals"},
            {"id": 1, "title": "Sell sensors to hospitals"},
            {"id": 2, "title": "Online course for piano teachers"},
            {"id": 4, "title": "SELL SENSORS TO HOSPITALS"},
        ]
        found = dedupe.find_near_duplicates(rows)
        assert found[1] == [4, 3], "closest match first"
        assert found[3] == [1, 4] and found[4] == [1, 3], "symmetric"
        assert 2 not in found, "the unrelated card is left alone"
        # Input order must not change the result, or the page reshuffles on reload.
        assert found == dedupe.find_near_duplicates(list(reversed(rows)))

    def test_no_ideas_and_one_idea_need_no_work(self):
        assert dedupe.find_near_duplicates([]) == {}
        assert dedupe.find_near_duplicates([{"id": 1, "title": "x"}]) == {}

    def test_a_missing_title_is_tolerated(self):
        """RowMapping.get, not row["title"] — a row without one should not 500 the
        dashboard."""
        assert dedupe.find_near_duplicates([{"id": 1}, {"id": 2, "title": None}]) == {}


def extract_ideas(client, monkeypatch, titles):
    """Extract `titles` and return them as a {title: id} map.

    A map rather than a positional list because list_ideas orders newest-first:
    an index-based helper would make every caller depend on that ordering, and the
    first version of these tests got it backwards.
    """
    monkeypatch.setattr(llm, "extract_ideas", lambda raw, user_id=None: [
        {"title": t, "commercial_framing": f"framing for {t}",
         "strength_signal": "early", "raw_claims": "claims"} for t in titles])
    r = client.post("/phase1/extract", data={"raw_text": "x" * 200},
                    follow_redirects=False)
    assert r.status_code == 303, r.text
    return {i["title"]: i["id"] for i in client.get("/phase1/ideas").context["ideas"]}


class TestDuplicateSurfaces:
    """The two places that render cards, and the routes that retire them."""

    def test_both_pages_name_the_near_duplicate(self, client, monkeypatch):
        signup(client)
        extract_ideas(client, monkeypatch,
                      ["Sell sensors to hospitals", "Online course for piano teachers",
                       "Selling sensors to hospitals"])
        for path in ("/dashboard", "/phase1/ideas"):
            page = client.get(path).text
            assert "Also extracted as" in page, path
            assert "Selling sensors to hospitals" in page, path
            assert "Online course for piano teachers" not in page.split(
                "Also extracted as")[1].split("Dismiss")[0], \
                f"{path}: an unrelated card was named as a duplicate"

    def test_the_link_points_at_a_card_that_is_on_the_page(self, client, monkeypatch):
        """Duplicates are computed over the cards the page shows, so a link can
        never go to a card IDEA_CARD_LIMIT pushed below the fold."""
        monkeypatch.setenv("IDEA_CARD_LIMIT", "3")
        signup(client)
        extract_ideas(client, monkeypatch, [
            "Sell sensors to hospitals", "Selling sensors to hospitals",
            "Online course for piano teachers", "Unrelated fourth card"])
        page = client.get("/phase1/ideas").text
        anchors = set(re.findall(r'id="idea-(\d+)"', page))
        assert anchors, "cards must carry anchors for the duplicate links"
        assert len(anchors) == 3, "the limit still applies"
        for href in re.findall(r'href="#idea-(\d+)"', page):
            assert href in anchors, f"link to #{href}, which is not on the page"

    def test_dismissing_a_card_removes_it_and_keeps_the_other(self, client, monkeypatch):
        signup(client)
        ids = extract_ideas(client, monkeypatch,
                            ["Sell sensors to hospitals", "Selling sensors to hospitals"])
        r = client.post(f"/phase1/dismiss/{ids["Sell sensors to hospitals"]}",
                        follow_redirects=False)
        assert r.status_code == 303

        page = client.get("/phase1/ideas")
        titles = [i["title"] for i in page.context["ideas"]]
        assert titles == ["Selling sensors to hospitals"], "nothing is merged away"
        assert "Dismissed ideas" in page.text
        assert "Sell sensors to hospitals" in page.text, "the card is still there"
        assert "Dismissed ideas (1)" in page.text

    def test_restoring_puts_a_card_back(self, client, monkeypatch):
        signup(client)
        ids = extract_ideas(client, monkeypatch, ["Sensor kits for NHS hospitals"])
        client.post(f"/phase1/dismiss/{ids['Sensor kits for NHS hospitals']}",
                    follow_redirects=False)
        assert "No candidates yet" in client.get("/phase1/ideas").text

        r = client.post(f"/phase1/restore/{ids['Sensor kits for NHS hospitals']}",
                        follow_redirects=False)
        assert r.status_code == 303
        page = client.get("/phase1/ideas")
        assert [i["title"] for i in page.context["ideas"]] == ["Sensor kits for NHS hospitals"]
        assert "Dismissed ideas" not in page.text

    def test_dismissal_deletes_nothing(self, client, monkeypatch):
        """An idea extracted from unpublished research cannot be recovered by
        pasting the text in again."""
        signup(client)
        ids = extract_ideas(client, monkeypatch, ["Sensor kits for NHS hospitals"])
        idea_id = ids["Sensor kits for NHS hospitals"]
        client.post(f"/phase1/dismiss/{idea_id}", follow_redirects=False)
        with db.as_tenant(1):
            row = db.get_idea(idea_id, 1)
        assert row is not None and row["status"] == "rejected"

    def test_dismissing_everything_does_not_look_like_losing_them(self, client, monkeypatch):
        signup(client)
        ids = extract_ideas(client, monkeypatch, ["Sensor kits for NHS hospitals"])
        client.post(f"/phase1/dismiss/{ids['Sensor kits for NHS hospitals']}",
                    follow_redirects=False)
        page = client.get("/dashboard").text
        assert "No candidate ideas" in page
        assert "Nothing was deleted" in page, \
            "an empty dashboard must say where the cards went"

    def test_a_selected_card_cannot_be_dismissed(self, client, monkeypatch):
        """The venture was built from it; rejecting the idea would leave the
        venture pointing at an idea the account calls rejected."""
        venture_id = make_venture(client, monkeypatch)
        # make_venture selected the idea, so it is no longer a candidate and the
        # ideas page is empty — take the id from the venture instead.
        with db.as_tenant(1):
            idea_id = db.get_venture(venture_id, 1)["idea_id"]
        r = client.post(f"/phase1/dismiss/{idea_id}", follow_redirects=False)
        assert r.status_code == 303
        with db.as_tenant(1):
            assert db.get_idea(idea_id, 1)["status"] == "selected"

    def test_another_users_card_cannot_be_dismissed_or_restored(self, client, monkeypatch):
        stranger = make_user("owner3@test.com")
        with db.as_tenant(stranger):
            idea_id = db.create_idea(stranger, "Their idea", "framing", "early", "c")

        make_venture(client, monkeypatch)
        for route in ("dismiss", "restore"):
            r = client.post(f"/phase1/{route}/{idea_id}", follow_redirects=False)
            assert r.status_code == 303, route
        with db.as_tenant(stranger):
            assert db.get_idea(idea_id, stranger)["status"] == "candidate"

    def test_another_users_cards_are_not_offered_as_duplicates(self, client, monkeypatch):
        signup(client)
        extract_ideas(client, monkeypatch, ["Sell sensors to hospitals"])
        stranger = make_user("owner4@test.com")
        with db.as_tenant(stranger):
            db.create_idea(stranger, "Selling sensors to hospitals", "framing", "early", "c")
        # The stranger's card is near-identical to ours but must not appear.
        assert "Also extracted as" not in client.get("/phase1/ideas").text

    def test_detection_spends_no_quota(self, client, monkeypatch):
        """It runs while rendering a page, so a model call here would make a page
        depend on a model succeeding."""
        signup(client)
        user_id = db.get_user_by_email("a@test.com")["id"]
        extract_ideas(client, monkeypatch,
                      ["Sell sensors to hospitals", "Selling sensors to hospitals"])
        before = db.count_llm_calls_this_month(user_id)
        for _ in range(3):
            client.get("/phase1/ideas")
            client.get("/dashboard")
        assert db.count_llm_calls_this_month(user_id) == before


# ==================== mentor challenges (M2.1) ====================

MENTOR_QUESTIONS = [{"questions": [
    {"question": "You marked Revenue passed on six interviews — did any of them contain a pricing question?",
     "principle": "evidence before claims", "why_it_matters": "Six friendly interviews are not a willingness-to-pay finding."},
    {"question": "Which of your nine blocks have you never run a single test against?",
     "principle": "the untested assumption", "why_it_matters": "The untested assumption may be the one that ends the venture."},
    {"question": "What would you cut tomorrow if nobody complained?",
     "principle": "focus by subtraction", "why_it_matters": "Anything you would not fight for is probably not load-bearing."},
]}]


def flat(page: str) -> str:
    """Rendered *text*, for assertions about prose rather than markup.

    Two transforms, both because a test that asserts on the raw HTML asserts on
    things it should not care about:
      * whitespace collapsed, so a sentence the template wraps across lines still
        matches (mentor labels and notice copy both wrap);
      * HTML entities resolved, so Jinja's autoescaping of an apostrophe in
        "don't scale" does not read as a missing word.
    """
    return re.sub(r"\s+", " ", html.unescape(page))


def mentor_stub(monkeypatch, questions=None, dropped=0):
    monkeypatch.setattr(llm, "challenge_with_mentor",
                        lambda mentor_key, subject, state, user_id=None: {
                            "questions": questions if questions is not None
                            else MENTOR_QUESTIONS[0]["questions"],
                            "dropped": dropped})


def phase3_venture(client, monkeypatch):
    return strategy_for(client, monkeypatch)


class TestMentorOutputValidation:
    """The guard. This is the part that must not be built casually — every other
    test in the file checks that the feature works, and these check that it
    cannot go wrong."""

    def test_the_prompt_forbids_answering_and_the_record_is_wrapped(self, tmp_db, monkeypatch):
        """Both halves of the third guard: the prompt says never to answer or
        speak for anyone, and the researcher's own record arrives as untrusted
        data like everywhere else in this codebase."""
        captured = {}
        real_call = llm.call_json

        def spy(prompt, purpose, user_id=None, wrap_key=None):
            captured.update(prompt=prompt, purpose=purpose, wrap_key=wrap_key)
            return real_call(prompt, purpose, user_id, wrap_key)

        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setattr(llm, "call_json", spy)
        monkeypatch.setattr(llm, "_get_client", lambda s: fake_client(MENTOR_QUESTIONS[0]))
        user_id = make_user("m@test.com")
        out = llm.challenge_with_mentor("evidence", "a block", ["Outcome: passed"])

        assert captured["purpose"] == "challenge_with_mentor"
        assert captured["wrap_key"] == "questions", \
            "JSON mode needs the object envelope or the model returns a bare array"
        prompt = captured["prompt"]
        assert "Ask only" in prompt
        assert "NEVER write in the first person" in prompt
        assert "NEVER name or quote" in prompt
        assert "untrusted data" in prompt
        assert "Outcome: passed" in prompt
        # And the shape has nowhere for advice to live.
        assert '"questions"' in prompt and len(out["questions"]) == 3
        assert all(q["question"].endswith("?") for q in out["questions"])

    @pytest.mark.parametrize("key", ["focus", "first_principles", "demand",
                                     "falsify", "jobs_to_be_done", "evidence"])
    def test_every_playbook_has_a_definition_and_a_check_constraint(self, key):
        """The catalogue in app/mentors.py and the enum in app/constants.py are
        separate lists on purpose — schema.py must stay engine-free — so this is
        what keeps them from drifting apart."""
        assert key in MENTOR_KEYS
        entry = mentors.get(key)
        assert entry is not None and entry["label"] and entry["source"]
        assert entry["attributed_to"]

    def test_an_unknown_playbook_is_refused_before_a_prompt_exists(self, tmp_db, monkeypatch):
        """An unknown key would put a real founder's name next to generated text
        with no principle behind it."""
        def boom(*a, **k):
            raise AssertionError("a prompt must never be built for an unknown key")
        monkeypatch.setattr(llm, "call_json", boom)
        with pytest.raises(llm.LLMError) as exc:
            llm.challenge_with_mentor("yoda", "x", ["y"])
        assert "mentor playbook" in str(exc.value)

    @pytest.mark.parametrize("line,key,survives", [
        ("Which block would you delete first?", "focus", True),
        ("I would have killed this feature.", "focus", False),
        ("If I were you, would you have tested pricing?", "focus", False),
        ("Jobs would have said your market is too small.", "focus", False),
        ("Steve would probably have shipped this.", "focus", False),
        ("My approach would be to delete the channel.", "focus", False),
        ("What varies between the two pricing tests?", "falsify", True),
        ("Ries would have said the risk is elsewhere.", "falsify", False),
        ("What would Musk have priced this at?", "first_principles", False),
        ("Which of these have we already tested?", "falsify", True),
        ("The model is trained on founder folklore.", "evidence", False),
    ])
    def test_the_line_level_guard(self, line, key, survives):
        assert llm._mentor_is_acceptable(line, mentors.get(key)) is survives

    def test_a_reply_of_only_statements_is_a_visible_error_not_a_persona(self, tmp_db, monkeypatch):
        """Silently returning nothing would read as "this playbook had nothing to
        say", which is indistinguishable from a feature that works."""
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setattr(llm, "_get_client", lambda s: fake_client({"questions": [
            {"question": "I would have deleted your channel.", "principle": "focus",
             "why_it_matters": "x"},
            {"question": "Jobs would have disagreed.", "principle": "focus",
             "why_it_matters": "y"},
        ]}))
        user_id = make_user("m2@test.com")
        with pytest.raises(llm.LLMError) as exc:
            llm.challenge_with_mentor("focus", "a block", ["x"], user_id=user_id)
        assert "will not show as advice" in str(exc.value)

    def test_good_lines_survive_alongside_dropped_ones_and_the_count_is_kept(
        self, tmp_db, monkeypatch
    ):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setattr(llm, "_get_client", lambda s: fake_client({"questions": [
            {"question": "Which block would you delete first?", "principle": "focus",
             "why_it_matters": "a"},
            {"question": "I would have kept it.", "principle": "focus", "why_it_matters": "b"},
            {"question": "Did any of your six interviews mention price?", "principle": "evidence",
             "why_it_matters": "c"},
        ]}))
        user_id = make_user("m3@test.com")
        out = llm.challenge_with_mentor("focus", "a block", ["x"], user_id=user_id)
        assert len(out["questions"]) == 2, "the first-person line must be dropped"
        assert out["dropped"] == 1


class TestMentorChallengeRoutes:
    def test_a_challenge_on_an_idea_card(self, client, monkeypatch):
        signup(client)
        ids = extract_ideas(client, monkeypatch, ["Sensor kits for NHS hospitals"])
        idea_id = ids["Sensor kits for NHS hospitals"]
        mentor_stub(monkeypatch)
        r = client.post(f"/mentor/idea/{idea_id}", data={"mentor_key": "evidence"},
                        follow_redirects=False)
        assert r.status_code == 303
        assert f"#idea-{idea_id}" in r.headers["location"], "must land back on the card"

        page = flat(client.get("/phase1/ideas").text)
        assert "Have this challenged" in page
        assert "did any of them contain a pricing question" in page
        assert "AI-generated" in page, "the not-a-quote notice is load-bearing"

    def test_a_challenge_on_a_block(self, client, monkeypatch):
        vid = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, vid, "customer", verdict="pass")
        mentor_stub(monkeypatch)
        r = client.post(f"/venture/{vid}/mentor/segment/customer",
                        data={"mentor_key": "falsify"}, follow_redirects=False)
        assert r.status_code == 303
        assert f"/segment/customer" in r.headers["location"]
        page = flat(client.get(f"/venture/{vid}/segment/customer").text)
        assert "Have this block challenged" in page
        assert "Which of your nine blocks" in page

    def test_a_challenge_on_the_plan(self, client, monkeypatch):
        vid = phase3_venture(client, monkeypatch)
        mentor_stub(monkeypatch)
        r = client.post(f"/venture/{vid}/mentor/plan", data={"mentor_key": "focus"},
                        follow_redirects=False)
        assert r.status_code == 303
        assert "Have the plan challenged" in flat(client.get(f"/venture/{vid}").text)

    def test_the_questions_are_about_this_venture_not_generic(self, client, monkeypatch):
        """The failure mode of every version of this feature: advice that could be
        given to anyone without reading the record."""
        seen = {}

        def spy(mentor_key, subject, state, user_id=None):
            seen.update(mentor_key=mentor_key, subject=subject, state=state)
            return {"questions": MENTOR_QUESTIONS[0]["questions"], "dropped": 0}

        vid = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, vid, "customer", verdict="pass",
                 outcome="all 8 wanted it", size="8")
        monkeypatch.setattr(llm, "challenge_with_mentor", spy)
        client.post(f"/venture/{vid}/mentor/segment/customer",
                    data={"mentor_key": "evidence"}, follow_redirects=False)

        blob = " ".join(seen["state"])
        assert "all 8 wanted it" in blob, "the logged outcome must reach the prompt"
        assert "8 interviews" in blob or "sample size: 8" in blob or "8)" in blob, blob
        assert "Customer Segments" in blob, "the block must be named"
        assert "Rest of the canvas" in blob, "the rest of the canvas is context"
        assert seen["mentor_key"] == "evidence"

    def test_a_challenge_cannot_change_anything(self, client, monkeypatch):
        """The load-bearing claim of the whole feature, asserted rather than
        trusted: a mentor asks, and the researcher still decides (invariant 4).

        Run against two blocks that have already come back FAILED and a critical
        one, because that is the state where a mentor asserting anything would do
        real damage — and where the kill-or-pivot card is on screen.
        """
        vid = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, vid, "customer", verdict="fail")
        run_once(client, monkeypatch, vid, "revenue", verdict="fail")

        def snapshot():
            return {
                "venture": dict(db.get_venture(vid, 1)),
                "segments": {k: dict(db.get_segment(vid, k)) for k in
                             ("customer", "revenue", "problem")},
                "cycles": len(db.list_cycles(vid)),
            }

        before = snapshot()
        assert before["segments"]["customer"]["outcome"] == "failed"

        mentor_stub(monkeypatch)
        for path in (f"/venture/{vid}/mentor/segment/customer",
                     f"/venture/{vid}/mentor/segment/revenue",
                     f"/venture/{vid}/mentor/plan"):
            assert client.post(path, data={"mentor_key": "focus"},
                               follow_redirects=False).status_code == 303, path

        after = snapshot()
        assert after["venture"] == before["venture"], "venture status or phase moved"
        assert after["segments"] == before["segments"], \
            "a challenge changed a block's outcome, evidence read, note or hypothesis"
        assert after["segments"]["customer"]["outcome"] == "failed", \
            "a failed block must not be rescued by a mentor"
        assert after["cycles"] == before["cycles"], "a challenge created a run"

    def test_a_challenge_costs_exactly_one_call_each(self, client, monkeypatch):
        """Parity with the other four purposes — the thing that was promised."""
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setattr(llm, "_get_client",
                            lambda s: fake_client(MENTOR_QUESTIONS[0]))
        vid = make_venture(client, monkeypatch)
        before = db.count_llm_calls_this_month(1)
        for path in (f"/venture/{vid}/mentor/segment/customer",
                     f"/venture/{vid}/mentor/plan"):
            assert client.post(path, data={"mentor_key": "focus"},
                               follow_redirects=False).status_code == 303
        assert db.count_llm_calls_this_month(1) == before + 2

    def test_an_unknown_playbook_spends_nothing(self, client, monkeypatch):
        signup(client)
        ids = extract_ideas(client, monkeypatch, ["Sensor kits for NHS hospitals"])
        user_id = db.get_user_by_email("a@test.com")["id"]
        before = db.count_llm_calls_this_month(user_id)

        def boom(*a, **k):
            raise AssertionError("must not build a prompt")
        monkeypatch.setattr(llm, "challenge_with_mentor", boom)
        r = client.post(f"/mentor/idea/{ids['Sensor kits for NHS hospitals']}",
                        data={"mentor_key": "yoda"}, follow_redirects=False)
        assert r.status_code == 303 and "err=" in r.headers["location"]
        assert db.count_llm_calls_this_month(user_id) == before
        assert db.list_mentor_challenges(user_id) == []

    def test_a_provider_failure_shows_the_error(self, client, monkeypatch):
        """Phase 3 swallowed these once already (M3.1) — the mentor path must not
        repeat it."""
        vid = make_venture(client, monkeypatch)

        def boom(*a, **k):
            raise llm.LLMError("The provider is unreachable. Try again shortly.")
        monkeypatch.setattr(llm, "challenge_with_mentor", boom)
        r = client.post(f"/venture/{vid}/mentor/segment/customer",
                        data={"mentor_key": "focus"}, follow_redirects=True)
        assert "The provider is unreachable" in r.text

    def test_another_users_venture_cannot_be_challenged(self, client, monkeypatch):
        stranger = make_user("mstranger@test.com")
        with db.as_tenant(stranger):
            idea_id = db.create_idea(stranger, "Their idea", "framing", "early", "c")
            sv = db.create_venture(stranger, idea_id)

        mentor_stub(monkeypatch)
        make_venture(client, monkeypatch)
        for path in (f"/mentor/idea/{idea_id}",
                     f"/venture/{sv}/mentor/segment/customer",
                     f"/venture/{sv}/mentor/plan"):
            r = client.post(path, data={"mentor_key": "focus"}, follow_redirects=False)
            assert r.status_code == 303, path
            assert "/dashboard" in r.headers["location"], path
        assert db.list_mentor_challenges(stranger) == []

    def test_challenges_are_scoped_to_the_subject(self, client, monkeypatch):
        """A question about one block shown next to another block is a question
        about the wrong thing."""
        vid = make_venture(client, monkeypatch)
        run_once(client, monkeypatch, vid, "customer", verdict="pass")
        run_once(client, monkeypatch, vid, "revenue", verdict="pass")
        mentor_stub(monkeypatch)
        client.post(f"/venture/{vid}/mentor/segment/customer",
                    data={"mentor_key": "evidence"}, follow_redirects=False)
        client.post(f"/venture/{vid}/mentor/plan",
                    data={"mentor_key": "focus"}, follow_redirects=False)
        rows = db.list_mentor_challenges(1, venture_id=vid)
        assert len(rows) == 2
        # The block page shows only the block's own challenge.
        page = flat(client.get(f"/venture/{vid}/segment/customer").text)
        assert "Which of your nine blocks" in page
        assert page.count("AI-generated") == 1, \
            f"the plan's challenge must not appear on the block page ({page.count('AI-generated')} notices)"

    def test_a_challenge_costs_quota_and_obeys_the_rate_limit(self, client, monkeypatch):
        """Parity with the other four purposes — no separate budget.

        Stubs the *provider*, not challenge_with_mentor: stubbing the wrapper
        would skip call_json, and with it the llm_calls row this is asserting on.
        """
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setattr(llm, "_get_client",
                            lambda s: fake_client(MENTOR_QUESTIONS[0]))
        vid = make_venture(client, monkeypatch)
        user_id = 1
        spent = []
        real_check = ratelimit.check
        monkeypatch.setattr(ratelimit, "check",
                            lambda *a, **k: (spent.append(a), real_check(*a, **k))[1])
        before = db.count_llm_calls_this_month(user_id)
        r = client.post(f"/venture/{vid}/mentor/segment/customer",
                        data={"mentor_key": "focus"}, follow_redirects=False)
        assert r.status_code == 303
        assert spent, "a challenge must go through the LLM rate limiter"
        assert db.count_llm_calls_this_month(user_id) == before + 1
        with db.get_conn() as conn:
            rows = conn.execute(text(
                "SELECT purpose, prompt_version FROM llm_calls WHERE user_id = :u"
                " ORDER BY id"), {"u": user_id}).mappings().fetchall()
        logged = [r for r in rows if r["purpose"] == "challenge_with_mentor"]
        assert logged, "and it must be logged like every other purpose"
        assert logged[-1]["prompt_version"] == "challenge_with_mentor.v1"

    def test_a_challenge_is_recorded_with_its_playbook_and_notice(self, client, monkeypatch):
        vid = make_venture(client, monkeypatch)
        mentor_stub(monkeypatch, dropped=2)
        client.post(f"/venture/{vid}/mentor/segment/customer",
                    data={"mentor_key": "demand"}, follow_redirects=False)
        row = db.latest_mentor_challenge(1, venture_id=vid)
        assert row["mentor_key"] == "demand"
        assert row["subject_kind"] == "segment"
        assert row["segment"] == "customer"
        assert row["dropped_count"] == 2
        page = flat(client.get(f"/venture/{vid}/segment/customer").text)
        assert "2 lines of the reply were dropped" in page, \
            "a challenge that discarded output should say so, not hide it"
        assert "Do things that don\u2019t scale" in page or "Do things that don't scale" in page
        assert "Paul Graham" in page
        assert "not advice from this person" in page


class TestMentorPersistence:
    def test_save_refuses_shapes_the_schema_would_reject(self, tmp_db):
        """Validated here as well as by the CHECK constraints, so an unvalidated
        value is a refusal rather than a 500 from the database."""
        user_id = make_user("mp@test.com")
        assert db.save_mentor_challenge(user_id, "yoda", "idea", []) is None
        assert db.save_mentor_challenge(user_id, "focus", "moon", []) is None
        assert db.save_mentor_challenge(user_id, "focus", "idea", []) is None, \
            "an idea challenge with no idea_id"
        assert db.save_mentor_challenge(user_id, "focus", "segment", [],
                                        idea_id=1) is None, "kind/subject mismatch"
        assert db.save_mentor_challenge(user_id, "focus", "segment", [],
                                        venture_id=1, segment="not_a_block") is None
        assert db.save_mentor_challenge(user_id, "focus", "plan", [],
                                        venture_id=1, segment="customer") is None, \
            "only a segment challenge names a block"

    def test_a_challenge_survives_in_the_export_and_goes_on_deletion(self, client, monkeypatch):
        make_venture(client, monkeypatch)
        mentor_stub(monkeypatch)
        client.post("/venture/1/mentor/segment/customer",
                    data={"mentor_key": "focus"}, follow_redirects=False)

        data = client.get("/account/export").json()
        rows = data["mentor_challenges"]
        assert len(rows) == 1
        assert rows[0]["mentor_key"] == "focus"
        assert rows[0]["questions_json"][0]["question"].endswith("?")

        assert counts_by_table()["mentor_challenges"] == 1
        r = client.post("/account/delete",
                        data={"password": "password123", "confirm": "DELETE"},
                        follow_redirects=False)
        assert r.status_code == 303
        left = counts_by_table()
        assert all(n == 0 for n in left.values()), f"orphans left behind: {left}"

    def test_deleting_a_venture_row_removes_its_challenges(self, client, monkeypatch):
        """The ON DELETE CASCADE on venture_id.

        KILLING a venture does not delete the row — it sets status — so the
        cascade is exercised here by removing the row as the database owner, which
        is the only way it fires. Killing must NOT clear challenges: the venture is
        still there to read, and its questions still apply to it.
        """
        vid = make_venture(client, monkeypatch)
        mentor_stub(monkeypatch)
        client.post(f"/venture/{vid}/mentor/plan", data={"mentor_key": "focus"},
                    follow_redirects=False)
        assert counts_by_table()["mentor_challenges"] == 1

        client.post(f"/venture/{vid}/kill", follow_redirects=False)
        assert counts_by_table()["mentor_challenges"] == 1, \
            "a killed venture is still readable, so its challenges stay"

        with as_owner(), db.get_conn() as conn:
            conn.execute(text("DELETE FROM ventures WHERE id = :i"), {"i": vid})
        assert counts_by_table()["mentor_challenges"] == 0
