"""What a worker is told, and what it is recorded as having said.

Four findings from the audit of 2026-09-24, all of the same kind: the durable
record and the thing that actually happened described each other inaccurately.

* Output with no `FINDING:` line was published as a finding at confidence
  0.5 -- a number nobody stated, attached to a claim nobody made.
* `CONFIDENCE:` with nothing after it raised `IndexError`, so a truncated but
  usable result became a worker failure and spent a retry.
* Board provenance the read receipt froze as "what was shown" was left out of
  the rendering, so it was not shown.
* A profile binding `top_p: 0.9` generated at the service default of 0.95,
  while the record said 0.9.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from amoeba.errors import ResourceExhausted
from amoeba.neuocyte import Neuocyte, _parse_finding
from amoeba.roles import _is_context_pressure
from amoeba.rpc import RpcError


# ---------------------------------------------------------------------------
# The parser
# ---------------------------------------------------------------------------
def test_a_confidence_field_with_nothing_in_it_does_not_crash():
    """A truncated result is still a result; it used to cost an attempt."""
    parsed = _parse_finding("FINDING: A useful result\nCONFIDENCE:\nEVIDENCE: observed")
    assert parsed["finding"] == "A useful result"
    assert parsed["confidence"] is None
    assert parsed["parsed"] is True


@pytest.mark.parametrize("raw", ["CONFIDENCE: high", "CONFIDENCE: ", "CONFIDENCE:"])
def test_a_confidence_nobody_stated_is_not_invented(raw):
    parsed = _parse_finding(f"FINDING: something\n{raw}")
    assert parsed["confidence"] is None, "a number appeared that nobody wrote"


def test_a_stated_confidence_is_kept():
    assert _parse_finding("FINDING: x\nCONFIDENCE: 0.8")["confidence"] == 0.8
    assert _parse_finding("FINDING: x\nCONFIDENCE: 7")["confidence"] == 1.0


def test_output_that_is_not_a_finding_is_not_reported_as_one():
    """A refusal to answer used to become a half-confident claim."""
    parsed = _parse_finding("I cannot determine that from the context given.")
    assert parsed["parsed"] is False
    assert parsed["confidence"] is None
    assert "cannot determine" in parsed["finding"], "the text is still kept"


def test_unshaped_output_is_posted_as_a_note_not_a_finding():
    posted = {}

    def call(verb, **kw):
        posted.update(kw)
        return {"post_id": "post_1"}

    worker = SimpleNamespace(
        sup=SimpleNamespace(call=call), neuocyte_id="nc_1",
        log=logging.getLogger("t"), model_generation="gen", snapshot_id=None)
    item = {"board_access": "read_write", "work_id": "work_1", "objective": "look"}

    Neuocyte._publish_finding(worker, item, _parse_finding("no format here"), {})
    assert posted["post_type"] == "note"
    assert posted["confidence"] is None

    posted.clear()
    Neuocyte._publish_finding(worker, item, _parse_finding("FINDING: a\nCONFIDENCE: 0.6"), {})
    assert posted["post_type"] == "finding"
    assert posted["confidence"] == 0.6


# ---------------------------------------------------------------------------
# What the worker is shown
# ---------------------------------------------------------------------------
def _worker(read_result):
    return SimpleNamespace(
        sup=SimpleNamespace(call=lambda verb, **kw: read_result),
        neuocyte_id="nc_1", log=logging.getLogger("t"))


def test_a_post_is_shown_with_what_became_of_its_attempt():
    """The read receipt froze this as shown; the rendering left it out."""
    block, ids = Neuocyte._board_context(
        _worker({"posts": [{"post_id": "p1", "post_type": "finding",
                            "author": "nc_0", "body": "the cache is cold",
                            "attempt_fate": "fenced",
                            "work_note": "superseded by a later attempt"}],
                 "silent_attempts": {"count": 0}}),
        {"board_access": "read_write", "work_id": "w1"})

    assert "fenced" in block, "the post's fate was not shown"
    assert "superseded" in block
    assert ids == ["p1"]


def test_attempts_that_posted_nothing_are_still_reported():
    """Silence distinguishes "nobody has looked" from "three have, and died"."""
    block, ids = Neuocyte._board_context(
        _worker({"posts": [], "silent_attempts": {"count": 3}}),
        {"board_access": "read_write", "work_id": "w1"})

    assert "3 earlier attempt" in block
    assert ids == []


def test_a_board_naive_worker_stays_board_naive():
    """Independence is the point; provenance does not leak past it."""
    block, ids = Neuocyte._board_context(
        _worker({"posts": [{"post_id": "p1"}], "silent_attempts": {"count": 9}}),
        {"board_access": "none", "work_id": "w1"})

    assert "9" not in block and ids == []


def test_a_retry_is_told_it_is_one():
    """Durable, and previously never said to the worker doing the retrying."""
    said = Neuocyte._attempt_context(
        SimpleNamespace(), {"attempt": 3, "failure": "the file was locked"})
    assert "attempt 3" in said
    assert "file was locked" in said

    assert Neuocyte._attempt_context(SimpleNamespace(), {"attempt": 1}) == ""


# ---------------------------------------------------------------------------
# What reaches the model
# ---------------------------------------------------------------------------
class _Inf:
    def __init__(self):
        self.generate_calls = []

    def call(self, method, **params):
        if method == "generate":
            self.generate_calls.append(params)
            return {"text": "ok", "finish_reason": "stop", "completion_tokens": 1}
        return {}


def _role_with(settings):
    from amoeba.roles import RoleProcess

    role = RoleProcess.__new__(RoleProcess)
    role.role = "id"
    role.inf = _Inf()
    role.session_id = "sess"
    role.turns = 0
    role.profile_settings = settings
    return role


def test_the_sampling_a_profile_binds_reaches_the_call():
    """Asserted at the call boundary. Comparing key sets could not see this."""
    role = _role_with({"top_p": 0.9, "top_k": 20, "temperature": 0.3,
                       "max_tokens": 64, "seed": 7})
    role._infer("think about it", skip_input=True)

    call = role.inf.generate_calls[0]
    assert call["top_p"] == 0.9, "the profile's top_p never reached inference"
    assert call["top_k"] == 20
    assert call["temperature"] == 0.3
    assert call["seed"] == 7


def test_a_setting_the_profile_does_not_bind_is_not_sent():
    """Sending a default as though it were bound is the same lie backwards."""
    role = _role_with({"max_tokens": 64})
    role._infer("think", skip_input=True)
    call = role.inf.generate_calls[0]
    assert "top_p" not in call and "top_k" not in call


# ---------------------------------------------------------------------------
# What a refusal means
# ---------------------------------------------------------------------------
def test_pressure_is_recognised_by_what_the_refusal_says_it_is():
    """Reworded, it used to stop being pressure and become a role failure."""
    reworded = RpcError("the session has no room left for this generation",
                        remote_code="resource_exhausted",
                        pressure="context_pressure")
    assert _is_context_pressure(reworded)


def test_a_different_shortage_is_not_pressure():
    """A generic exhaustion code also describes running out of sessions."""
    other = RpcError("no free sequence slots", remote_code="resource_exhausted")
    assert not _is_context_pressure(other)


def test_the_old_wording_is_still_understood():
    """Refusals raised by code that does not carry the marker still say so."""
    assert _is_context_pressure(
        ResourceExhausted("prompt plus its generation allowance exceeds the "
                          "configured context budget"))


def test_the_arbiter_declares_its_own_refusal():
    from amoeba.arbiter import Arbiter
    from amoeba.config import ArbiterConfig

    arbiter = Arbiter(ArbiterConfig())
    with pytest.raises(ResourceExhausted) as caught:
        arbiter.clamp_inference(prompt_tokens=10 ** 9, max_tokens=16,
                                deadline=None, budget_tokens=1024,
                                budget_basis="total")
    assert caught.value.details.get("pressure") == "context_pressure"
    assert _is_context_pressure(caught.value)
