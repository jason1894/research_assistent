"""SQLAlchemy-backed conversation history database."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger(__name__)


class _Base(DeclarativeBase):
    pass


class Conversation(_Base):
    """ORM model for a single conversation turn."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(128), nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    role = Column(String(32), nullable=False)  # "user" | "assistant" | "system"
    content = Column(Text, nullable=False)
    domain = Column(String(64), nullable=True)
    feedback = Column(String(16), nullable=True)  # "positive" | "negative" | None


class ConversationDB:
    """Stores and retrieves conversation history using SQLite.

    Args:
        db_path: Path to the SQLite database file.  Use ``":memory:"`` for
            an in-memory database.
        max_history: Maximum number of turns returned by
            :meth:`get_history`.
    """

    def __init__(
        self,
        db_path: str = "./data/memory.db",
        max_history: int = 100,
    ) -> None:
        self.db_path = db_path
        self.max_history = max_history

        db_url = f"sqlite:///{db_path}" if db_path != ":memory:" else "sqlite://"
        self._engine = create_engine(db_url, connect_args={"check_same_thread": False})
        _Base.metadata.create_all(self._engine)
        self._Session = sessionmaker(bind=self._engine)
        logger.info("ConversationDB initialised at: %s", db_path)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        domain: Optional[str] = None,
        feedback: Optional[str] = None,
    ) -> int:
        """Persist a single conversation turn.

        Args:
            session_id: Identifier for the conversation session.
            role: Speaker role (``"user"``, ``"assistant"``, or ``"system"``).
            content: Message text.
            domain: Optional research domain tag.
            feedback: Optional quality signal (``"positive"`` or
                ``"negative"``).

        Returns:
            The auto-assigned database row ID.
        """
        with Session(self._engine) as session:
            msg = Conversation(
                session_id=session_id,
                role=role,
                content=content,
                domain=domain,
                feedback=feedback,
            )
            session.add(msg)
            session.commit()
            session.refresh(msg)
            return msg.id  # type: ignore[return-value]

    def save_feedback(self, message_id: int, feedback: str) -> bool:
        """Update the feedback field for an existing message.

        Args:
            message_id: The row ID of the message.
            feedback: ``"positive"`` or ``"negative"``.

        Returns:
            True if the row was updated.
        """
        with Session(self._engine) as session:
            msg = session.get(Conversation, message_id)
            if msg is None:
                return False
            msg.feedback = feedback  # type: ignore[assignment]
            session.commit()
            return True

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Return conversation history for a session.

        Args:
            session_id: Session identifier.
            limit: Maximum number of turns to return.  Falls back to
                ``self.max_history``.

        Returns:
            List of turn dicts ordered oldest-first.
        """
        effective_limit = limit or self.max_history
        with Session(self._engine) as session:
            rows = (
                session.query(Conversation)
                .filter(Conversation.session_id == session_id)
                .order_by(Conversation.timestamp.asc())
                .limit(effective_limit)
                .all()
            )
        return [self._row_to_dict(r) for r in rows]

    def get_all_sessions(self) -> List[str]:
        """Return a sorted list of all known session IDs."""
        with Session(self._engine) as session:
            rows = (
                session.query(Conversation.session_id)
                .distinct()
                .order_by(Conversation.session_id)
                .all()
            )
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_session(self, session_id: str) -> int:
        """Delete all messages belonging to a session.

        Args:
            session_id: Session to delete.

        Returns:
            Number of rows deleted.
        """
        with Session(self._engine) as session:
            count = (
                session.query(Conversation)
                .filter(Conversation.session_id == session_id)
                .delete()
            )
            session.commit()
        logger.info("Deleted %d messages for session '%s'.", count, session_id)
        return count

    # ------------------------------------------------------------------
    # Export for LoRA training
    # ------------------------------------------------------------------

    def export_for_training(
        self,
        min_feedback: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Export conversation pairs formatted for LoRA fine-tuning.

        Pairs consecutive ``user`` + ``assistant`` turns from the same
        session into ``{"instruction": ..., "response": ...}`` dicts.

        Args:
            min_feedback: If ``"positive"``, only export pairs where the
                assistant turn has positive feedback.

        Returns:
            List of training example dicts.
        """
        examples: List[Dict[str, str]] = []

        with Session(self._engine) as session:
            sessions = [r[0] for r in session.query(Conversation.session_id).distinct()]

        for sid in sessions:
            history = self.get_history(sid, limit=None)
            i = 0
            while i < len(history) - 1:
                turn = history[i]
                next_turn = history[i + 1]
                if turn["role"] == "user" and next_turn["role"] == "assistant":
                    if min_feedback == "positive" and next_turn.get("feedback") != "positive":
                        i += 2
                        continue
                    examples.append(
                        {
                            "instruction": turn["content"],
                            "response": next_turn["content"],
                            "domain": turn.get("domain", ""),
                        }
                    )
                    i += 2
                else:
                    i += 1

        logger.info("Exported %d training examples from conversation history.", len(examples))
        return examples

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(row: Conversation) -> Dict[str, Any]:
        return {
            "id": row.id,
            "session_id": row.session_id,
            "timestamp": row.timestamp.isoformat() if row.timestamp else None,
            "role": row.role,
            "content": row.content,
            "domain": row.domain,
            "feedback": row.feedback,
        }
