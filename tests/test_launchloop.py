"""
Tests for the Phase-2 per-segment state machine, the LLM output validation, and
the run-cap behaviour. The LLM layer is mocked throughout — these tests are
about the rules, not the model.

Run with:  python -m pytest -q
Needs a Postgres. Point TEST_DATABASE_URL at one, or leave it unset and the
conftest falls back to DATABASE_URL and then to localhost/launchloop_test.
"""
import html
import json
import pathlib
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

import app.config as config
import app.db as db
import app.llm as llm
import app.main as main
from app.auth import hash_password
from openai import BadRequestError
from app.constants import SEGMENTS


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
    def test_another_user_cannot_see_or_act_on_the_venture(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        client.post("/logout", follow_redirects=False)
        signup(client, email="b@test.com")
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
        user_id = db.create_user("m@test.com", hash_password("password123"))
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


class TestCallLog:
    def test_successful_call_is_recorded_with_tokens_and_model(self, tmp_db, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "k")
        monkeypatch.setenv("LLM_BASE_URL", "https://opencode.ai/zen/v1")
        monkeypatch.setenv("LLM_MODEL", "glm-5.3")
        capture = {}
        monkeypatch.setattr(llm, "_get_client", lambda settings: fake_client({"ok": True}, capture))

        user_id = db.create_user("a@test.com", hash_password("password123"))
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
        documented = set(re.findall(
            r"^([A-Z_]+)=",
            pathlib.Path(config.PROJECT_ROOT / ".env.example").read_text(), re.M))
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

        with pytest.raises(llm.LLMError):
            llm.call_json("original prompt", "generate_segment_tasks", user_id=None)
        assert len(attempts) == config.llm_max_attempts()
        assert "original prompt" in attempts[0]
        assert "could not be parsed" in attempts[-1], "the retry must include the parse error"

        with db.get_conn() as conn:
            row = conn.execute(text("SELECT * FROM llm_calls")).mappings().fetchone()
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


# ==================== DB helpers ====================

class TestDbHelpers:
    def test_update_venture_ignores_unknown_fields(self, tmp_db):
        user_id = db.create_user("x@test.com", hash_password("password123"))
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
        user_id = db.create_user("x@test.com", hash_password("password123"))
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
        user_id = db.create_user("y2@test.com", hash_password("password123"))
        venture_id = db.create_venture(user_id, db.create_idea(user_id, "t", "f", "early", "c"))
        db.set_segment_outcome(venture_id, "revenue", "definitely_a_verdict")
        db.update_segment(venture_id, "revenue", status="invented_status")
        seg = db.get_segment(venture_id, "revenue")
        assert seg["outcome"] == "pending" and seg["status"] == "untested"

    def test_all_resolved_needs_every_block(self, tmp_db):
        """Regression risk: a missing segment row must make the gate unreachable
        rather than silently pass."""
        user_id = db.create_user("y3@test.com", hash_password("password123"))
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
        user_id = db.create_user("y@test.com", hash_password("password123"))
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

    def test_canvas_reflects_stored_outcomes(self, client, monkeypatch):
        venture_id = make_venture(client, monkeypatch)
        db.set_segment_outcome(venture_id, "value_prop", "passed", note="4x cheaper")
        db.set_segment_outcome(venture_id, "revenue", "parked", note="no one will pay")
        text = client.get(f"/venture/{venture_id}").text
        assert text.count('class="bmc-cell s-confirmed') == 1
        assert text.count('class="bmc-cell s-mixed o-parked') == 1
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
