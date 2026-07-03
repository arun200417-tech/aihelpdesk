"""Agent Orchestrator — runs the ticket triage lifecycle (design doc Section 9.1).

Pipeline (each step is audited via the Audit Agent for AI governance, FR-12):
  1. OCR         — extract text from an attached screenshot (if any).
  2. Intent      — classify category + intent.
  3. Priority    — score urgency.
  4. Duplicate   — vector-similarity vs open tickets:
                     * merge tier, original resolved   -> reuse its resolution.
                     * merge tier, original unresolved -> fetch a KB solution and
                       share it with the duplicate (auto-resolve) if one exists;
                       otherwise just link them so the team handles them together.
                     * suggest tier -> keep suggestions, continue to RAG.
  5. RAG         — KB retrieval + LLM -> grounded answer + confidence.
  6. Decision    — confidence >= threshold -> auto-resolve with citations;
                   otherwise route to the right department and assign an
                   available agent (humans resolve/escalate manually from there).

It mutates the Ticket in place and returns the pipeline trace + duplicate
suggestions. VisionAgent and EmbeddingAgent were removed: their outputs were
recorded but never influenced ticket outcome (Vision only produced image
metadata nothing consumed; EmbeddingAgent only reported a vector dimension —
the real vector is stored via upsert_ticket_embedding).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from app.agents.audit_agent import AuditAgent
from app.agents.base import AgentResult
from app.agents.duplicate_agent import DuplicateAgent
from app.agents.intent_agent import IntentAgent
from app.agents.ocr_agent import OCRAgent
from app.agents.priority_agent import PriorityAgent
from app.agents.rag_agent import RAGAgent
from app.agents.routing_agent import RoutingAgent
from app.core.config import get_settings
from app.core.vectorstore import upsert_ticket_embedding
from app.models import Ticket
from app.models.ticket import TicketStatus
from app.repositories.article_repo import ArticleRepository
from app.repositories.log_repo import LogRepository
from app.repositories.ticket_repo import TicketRepository
from app.repositories.user_repo import UserRepository

settings = get_settings()


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AgentOrchestrator:
    def __init__(self, db) -> None:
        self.db = db
        self.audit = AuditAgent(LogRepository(db))
        self.user_repo = UserRepository(db)
        self.article_repo = ArticleRepository(db)
        self.tickets = TicketRepository(db)
        self.ocr = OCRAgent()
        self.intent = IntentAgent()
        self.priority = PriorityAgent()
        self.duplicate = DuplicateAgent()
        self.rag = RAGAgent()
        self.routing = RoutingAgent()

    def _trace(self, steps: list[dict], result: AgentResult) -> None:
        steps.append({
            "agent": result.agent_name,
            "event": result.event,
            "detail": result.detail,
            "confidence": result.confidence,
        })

    def process(self, ticket: Ticket) -> tuple[list[dict], list[dict]]:
        steps: list[dict] = []
        steps_text = f"{ticket.title}\n{ticket.description}".strip()

        # --- Step 1: OCR on any attachment ---
        ocr_res = self.ocr.run(ticket.attachment_path)
        self.audit.record(ocr_res, ticket.id)
        self._trace(steps, ocr_res)
        ticket.ocr_text = ocr_res.output.get("text", "")
        if ticket.ocr_text:
            steps_text += "\n" + ticket.ocr_text

        # --- Step 2: Intent / category classification ---
        intent_res = self.intent.run(steps_text)
        self.audit.record(intent_res, ticket.id, inputs={"text": steps_text[:500]})
        self._trace(steps, intent_res)
        ticket.category = intent_res.output.get("category")
        ticket.intent = intent_res.output.get("intent")

        # --- Step 3: Priority prediction ---
        prio_res = self.priority.run(steps_text, ticket.category)
        self.audit.record(prio_res, ticket.id)
        self._trace(steps, prio_res)
        ticket.priority = prio_res.output.get("priority")
        ticket.priority_score = prio_res.output.get("priority_score", 0.0)

        # --- Step 4: Duplicate detection (against other open tickets) ---
        dup_res = self.duplicate.run(steps_text, exclude_ticket_id=ticket.id)
        self.audit.record(dup_res, ticket.id)
        self._trace(steps, dup_res)
        suggestions = dup_res.output.get("suggestions", [])

        if dup_res.output.get("is_duplicate"):
            # Store this ticket's vector before branching so future dedup works.
            upsert_ticket_embedding(ticket.id, steps_text, ticket.status)
            dup_id = dup_res.output.get("duplicate_of_id")
            original = self.tickets.get(dup_id) if dup_id else None
            ticket.duplicate_of_id = dup_id
            # (a) The matched original already has a resolution -> reuse it.
            if self._resolve_as_duplicate(ticket, dup_res, original, steps):
                upsert_ticket_embedding(ticket.id, steps_text, ticket.status)
                return steps, suggestions
            # (b) The original has no resolution YET, but the issue may already be
            #     answered in the knowledge base. Fetch that solution and share it
            #     with this duplicate ticket instead of leaving it unresolved.
            if self._share_kb_solution(ticket, steps_text, steps, original):
                upsert_ticket_embedding(ticket.id, steps_text, ticket.status)
                return steps, suggestions
            # (c) No KB solution either -> link as duplicate with a clear reason.
            ticket.status = TicketStatus.DUPLICATE
            ticket.first_response_at = ticket.first_response_at or _now()
            ticket.routing_reason = self._duplicate_reason(ticket, original, resolved=False)
            self.audit.event(event="linked_duplicate", ticket_id=ticket.id, detail=ticket.routing_reason)
            self._trace(steps, AgentResult(
                "DuplicateResolver", "linked_duplicate",
                detail=ticket.routing_reason, confidence=dup_res.confidence))
            upsert_ticket_embedding(ticket.id, steps_text, ticket.status)
            return steps, suggestions

        # Index this ticket so later submissions can detect it as a duplicate.
        upsert_ticket_embedding(ticket.id, steps_text, ticket.status)

        # --- Steps 5 & 6: RAG knowledge search + resolution decision ---
        self._resolve_new_ticket(ticket, steps_text, steps)

        # Refresh stored embedding with the final status.
        upsert_ticket_embedding(ticket.id, steps_text, ticket.status)
        return steps, suggestions

    def _run_rag(self, ticket: Ticket, steps_text: str,
                 steps: list[dict]) -> tuple[bool, str | None, float]:
        """Step 5 — KB retrieval + grounded answer. Records citations on the
        ticket, bumps article retrieval counts, and returns
        (has_kb_match, answer, confidence) for the caller's decision."""
        rag_res = self.rag.run(steps_text)
        self.audit.record(rag_res, ticket.id)
        self._trace(steps, rag_res)
        sources = rag_res.output.get("sources", [])
        ticket.confidence = rag_res.confidence or 0.0
        for src in sources:
            if src.get("article_id"):
                self.article_repo.increment_retrieval(src["article_id"])
        ticket.kb_sources = json.dumps(sources)  # keep KB citations for context
        retrieval = rag_res.output.get("retrieval_strength", 0.0) or 0.0
        has_kb_match = retrieval >= settings.kb_match_min_score
        return has_kb_match, rag_res.output.get("answer"), ticket.confidence

    def _auto_resolve_from_kb(self, ticket: Ticket, answer: str,
                              steps: list[dict], note: str) -> None:
        """Apply a knowledge-base solution to the ticket (auto-resolution)."""
        ticket.status = TicketStatus.RESOLVED
        ticket.resolution = answer
        ticket.resolution_source = "auto"
        ticket.resolved_at = _now()
        ticket.routing_reason = note
        self.audit.event(event="auto_resolved", ticket_id=ticket.id, detail=note)
        self._trace(steps, AgentResult(
            "ConfidenceCheck", "auto_resolved", detail=note, confidence=ticket.confidence))

    def _resolve_new_ticket(self, ticket: Ticket, steps_text: str, steps: list[dict]) -> None:
        """Step 6 — resolution decision for a non-duplicate ticket.
          Rule 1: KB match AND confidence >= threshold  -> auto-resolve.
          Rule 2: KB match but confidence < threshold    -> escalate to L2.
          Rule 3: no KB match but recognized category     -> escalate to L2.
          Rule 4: no KB match and no recognized category  -> leave In Progress."""
        threshold = settings.ai_confidence_threshold
        has_kb_match, answer, confidence = self._run_rag(ticket, steps_text, steps)
        recognized_category = bool(ticket.category) and ticket.category != "Other"
        ticket.first_response_at = _now()

        if has_kb_match and confidence >= threshold and answer:
            self._auto_resolve_from_kb(
                ticket, answer, steps,
                note=f"Applied KB solution (confidence {confidence:.0%} >= {threshold:.0%}).")
        elif has_kb_match:
            self._escalate_to_l2(
                ticket, steps,
                note=(f"KB match found but AI confidence {confidence:.0%} is below "
                      f"{threshold:.0%} (ambiguous). Escalated to L2 for manual review."))
        elif recognized_category:
            self._escalate_to_l2(
                ticket, steps,
                note=(f"No confident KB solution for this {ticket.category} issue. "
                      f"Escalated to L2 support."))
        else:
            ticket.status = TicketStatus.IN_PROGRESS
            ticket.routing_reason = "Awaiting further classification or KB entry."
            self.audit.event(event="awaiting_classification", ticket_id=ticket.id,
                             detail=ticket.routing_reason)
            self._trace(steps, AgentResult(
                "ConfidenceCheck", "awaiting_classification",
                detail=ticket.routing_reason, confidence=confidence))

    def _share_kb_solution(self, ticket: Ticket, steps_text: str, steps: list[dict],
                           original: Ticket | None) -> bool:
        """A duplicate whose original is not resolved yet: if the knowledge base
        already contains a confident solution for this issue, share it with the
        ticket (auto-resolve while keeping the duplicate link). Returns True if a
        KB solution was applied, False to fall back to plain duplicate linking."""
        threshold = settings.ai_confidence_threshold
        has_kb_match, answer, confidence = self._run_rag(ticket, steps_text, steps)
        ticket.first_response_at = _now()
        if has_kb_match and confidence >= threshold and answer:
            oid = original.id if original else ticket.duplicate_of_id
            note = (f"This is similar to ticket #{oid}, which isn't resolved yet — but the "
                    f"knowledge base already had a solution, so we applied it for you "
                    f"(confidence {confidence:.0%}).")
            self._auto_resolve_from_kb(ticket, answer, steps, note=note)
            return True
        return False

    def _resolve_as_duplicate(self, ticket: Ticket, dup_res: AgentResult,
                              original: Ticket | None, steps: list[dict]) -> bool:
        """If the matched original ticket already has a resolution, reuse it to
        auto-resolve this ticket (source=duplicate_match). Returns True if resolved."""
        if not (original and original.resolution):
            return False

        ticket.status = TicketStatus.RESOLVED
        ticket.resolution = original.resolution
        ticket.resolution_source = "duplicate_match"
        ticket.kb_sources = original.kb_sources
        ticket.duplicate_of_id = original.id
        ticket.confidence = dup_res.confidence or 0.0
        ticket.first_response_at = _now()
        ticket.resolved_at = _now()
        ticket.routing_reason = self._duplicate_reason(ticket, original, resolved=True)
        detail = f"Auto-resolved as duplicate of #{original.id} (reused its resolution)."
        self.audit.event(event="auto_resolved_duplicate", ticket_id=ticket.id, detail=detail)
        self._trace(steps, AgentResult(
            "DuplicateResolver", "auto_resolved_duplicate",
            detail=detail, confidence=dup_res.confidence,
        ))
        return True

    def _escalate_to_l2(self, ticket: Ticket, steps: list[dict], note: str) -> None:
        """Route to the right department and escalate to L2 support (Rules 2 & 3)."""
        route_res = self.routing.run(ticket.category, ticket.confidence)
        self.audit.record(route_res, ticket.id)
        ticket.department = route_res.output.get("department")
        ticket.status = TicketStatus.ESCALATED
        ticket.escalation_target = "L2"
        ticket.routing_reason = note
        agent = self.user_repo.pick_available_agent(ticket.department)
        if agent:
            ticket.assigned_agent_id = agent.id
        self.audit.event(event="escalated_l2", ticket_id=ticket.id, detail=note)
        self._trace(steps, AgentResult(
            "EscalationAgent", "escalated_l2", detail=note, confidence=ticket.confidence))

    @staticmethod
    def _duplicate_reason(ticket: Ticket, original: Ticket | None, resolved: bool) -> str:
        """Plain-language explanation shown to the employee (no scores/agent names)."""
        oid = original.id if original else "?"
        otitle = original.title if (original and original.title) else "an earlier ticket"
        theme = ticket.category or "the same topic"
        if resolved:
            return (f'This is the same as your earlier ticket #{oid} ("{otitle}"), which was '
                    f"already resolved — so we applied that solution for you automatically.")
        return (f'This looks similar to ticket #{oid} ("{otitle}") because both are about '
                f"{theme}. We've linked them so the same team handles them together.")
