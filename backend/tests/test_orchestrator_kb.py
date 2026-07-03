"""Tests for the orchestrator's knowledge-base resolution behavior.

Focus (per the requested change):
  * New tickets fetch and apply a KB solution when the issue exists in the KB.
  * Duplicate tickets whose original is NOT resolved yet also fetch a KB solution
    and share it, instead of being left unresolved.
  * Duplicate tickets whose original IS resolved reuse that resolution.
  * When no KB solution exists, behavior degrades gracefully (link/escalate).

The orchestrator's agents and repositories are replaced with lightweight fakes so
these are fast, deterministic unit tests with no DB / network / embeddings.
"""
from __future__ import annotations

import json

import pytest

from app.agents.base import AgentResult
from app.agents.orchestrator import AgentOrchestrator
from app.models import Ticket
from app.models.ticket import TicketStatus


# --- Test doubles -----------------------------------------------------------

class _FakeAudit:
    def record(self, *args, **kwargs) -> None:  # noqa: D401
        pass

    def event(self, *args, **kwargs) -> None:
        pass


class _FakeAgent:
    """Returns a preset AgentResult; records the calls it received."""

    def __init__(self, result: AgentResult) -> None:
        self._result = result
        self.calls = 0

    def run(self, *args, **kwargs) -> AgentResult:
        self.calls += 1
        return self._result


class _ExplodingAgent:
    """Fails the test if run() is ever called (asserts a branch is skipped)."""

    def run(self, *args, **kwargs):  # noqa: D401
        raise AssertionError("agent.run() should not have been called")


class _FakeTicketRepo:
    def __init__(self, originals: dict[int, Ticket] | None = None) -> None:
        self._originals = originals or {}

    def get(self, ticket_id: int) -> Ticket | None:
        return self._originals.get(ticket_id)


class _FakeUserRepo:
    def pick_available_agent(self, department):  # noqa: D401
        return None


class _FakeArticleRepo:
    def __init__(self) -> None:
        self.incremented: list[int] = []

    def increment_retrieval(self, article_id: int) -> None:
        self.incremented.append(article_id)


# --- Builders ---------------------------------------------------------------

def _rag_result(*, answer: str, retrieval: float, confidence: float,
                article_id: int | None = 5) -> AgentResult:
    sources = []
    if article_id is not None:
        sources = [{"article_id": article_id, "title": "VPN Guide",
                    "snippet": "steps", "score": retrieval}]
    return AgentResult(
        "RAGAgent", "rag_generated",
        {"answer": answer, "sources": sources, "retrieval_strength": retrieval},
        confidence=confidence, model_version="test",
    )


def _dup_result(*, is_duplicate: bool, duplicate_of_id: int | None = None,
                confidence: float = 0.0) -> AgentResult:
    return AgentResult(
        "DuplicateAgent", "duplicate_checked",
        {"is_duplicate": is_duplicate, "duplicate_of_id": duplicate_of_id,
         "suggestions": []},
        confidence=confidence, model_version="test",
    )


def _build_orchestrator(*, dup: AgentResult, rag, originals=None) -> AgentOrchestrator:
    """Construct an orchestrator with all collaborators faked. `rag` may be an
    AgentResult (wrapped in a fake) or an already-built agent (e.g. exploding)."""
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch.db = None
    orch.audit = _FakeAudit()
    orch.user_repo = _FakeUserRepo()
    orch.article_repo = _FakeArticleRepo()
    orch.tickets = _FakeTicketRepo(originals)
    orch.ocr = _FakeAgent(AgentResult("OCRAgent", "ocr_done", {"text": ""}))
    orch.intent = _FakeAgent(AgentResult(
        "IntentAgent", "intent_done", {"category": "IT", "intent": "vpn_issue"}))
    orch.priority = _FakeAgent(AgentResult(
        "PriorityAgent", "priority_done", {"priority": "medium", "priority_score": 0.5}))
    orch.duplicate = _FakeAgent(dup)
    orch.rag = rag if hasattr(rag, "run") else _FakeAgent(rag)
    orch.routing = _FakeAgent(AgentResult(
        "RoutingAgent", "routed", {"department": "IT"}))
    return orch


def _new_ticket(tid: int = 1) -> Ticket:
    t = Ticket(employee_id=1, title="VPN broken",
               description="Cannot connect to the corporate VPN")
    t.id = tid
    t.status = TicketStatus.OPEN
    return t


@pytest.fixture(autouse=True)
def _no_vector_writes(monkeypatch):
    """Silence the vector-store upserts the orchestrator performs."""
    monkeypatch.setattr("app.agents.orchestrator.upsert_ticket_embedding",
                        lambda *a, **k: None)


# --- Tests ------------------------------------------------------------------

def test_new_ticket_fetches_kb_solution_when_issue_exists():
    """New ticket + confident KB match -> auto-resolved from the knowledge base."""
    orch = _build_orchestrator(
        dup=_dup_result(is_duplicate=False),
        rag=_rag_result(answer="Restart the VPN client. [1]", retrieval=0.76, confidence=0.88),
    )
    ticket = _new_ticket()

    orch.process(ticket)

    assert ticket.status == TicketStatus.RESOLVED
    assert ticket.resolution == "Restart the VPN client. [1]"
    assert ticket.resolution_source == "auto"
    assert json.loads(ticket.kb_sources)[0]["article_id"] == 5
    assert orch.article_repo.incremented == [5]


def test_duplicate_with_unresolved_original_shares_kb_solution():
    """The requested fix: a duplicate whose original has no resolution yet still
    fetches the KB solution and shares it instead of staying unresolved."""
    original = _new_ticket(tid=99)
    original.status = TicketStatus.IN_PROGRESS
    original.resolution = None  # not resolved yet

    orch = _build_orchestrator(
        dup=_dup_result(is_duplicate=True, duplicate_of_id=99, confidence=0.93),
        rag=_rag_result(answer="Reset your VPN profile. [1]", retrieval=0.74, confidence=0.85),
        originals={99: original},
    )
    ticket = _new_ticket(tid=2)

    orch.process(ticket)

    assert ticket.status == TicketStatus.RESOLVED
    assert ticket.resolution == "Reset your VPN profile. [1]"
    assert ticket.resolution_source == "auto"
    assert ticket.duplicate_of_id == 99          # duplicate link preserved
    assert "#99" in (ticket.routing_reason or "")  # explains the shared solution


def test_duplicate_with_resolved_original_reuses_its_resolution():
    """A resolved original short-circuits RAG: reuse its recorded resolution."""
    original = _new_ticket(tid=99)
    original.status = TicketStatus.RESOLVED
    original.resolution = "Original team fix"
    original.kb_sources = json.dumps([{"article_id": 7, "title": "X"}])

    orch = _build_orchestrator(
        dup=_dup_result(is_duplicate=True, duplicate_of_id=99, confidence=0.95),
        rag=_ExplodingAgent(),  # RAG must NOT run when the original is resolved
        originals={99: original},
    )
    ticket = _new_ticket(tid=3)

    orch.process(ticket)

    assert ticket.status == TicketStatus.RESOLVED
    assert ticket.resolution == "Original team fix"
    assert ticket.resolution_source == "duplicate_match"
    assert ticket.duplicate_of_id == 99


def test_duplicate_without_kb_solution_links_as_duplicate():
    """Duplicate of an unresolved original AND no KB solution -> plain link."""
    original = _new_ticket(tid=99)
    original.status = TicketStatus.IN_PROGRESS
    original.resolution = None

    orch = _build_orchestrator(
        dup=_dup_result(is_duplicate=True, duplicate_of_id=99, confidence=0.92),
        rag=_rag_result(answer="", retrieval=0.2, confidence=0.1, article_id=None),
        originals={99: original},
    )
    ticket = _new_ticket(tid=4)

    orch.process(ticket)

    assert ticket.status == TicketStatus.DUPLICATE
    assert ticket.duplicate_of_id == 99
    assert ticket.resolution is None
    assert ticket.routing_reason  # a plain-language duplicate explanation


def test_new_ticket_without_kb_match_escalates_to_l2():
    """Regression: no KB match but a recognized category -> escalate to L2."""
    orch = _build_orchestrator(
        dup=_dup_result(is_duplicate=False),
        rag=_rag_result(answer="", retrieval=0.3, confidence=0.2, article_id=None),
    )
    ticket = _new_ticket(tid=5)

    orch.process(ticket)

    assert ticket.status == TicketStatus.ESCALATED
    assert ticket.escalation_target == "L2"
    assert ticket.department == "IT"
