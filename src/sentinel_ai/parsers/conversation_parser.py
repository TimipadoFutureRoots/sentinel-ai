"""Multi-format conversation parser with auto-detection.

Supported formats:
    - plain   : User:/AI: (or user:/ai:) prefixed lines
    - chatgpt : ChatGPT JSON export (mapping dict with message nodes)
    - json    : sentinel-ai native (sessions with turns)
    - claude  : Claude JSON (messages with sender: human/assistant)

Returns standardised list of Session objects.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..models import Role, Session, Turn

logger = logging.getLogger(__name__)

FormatType = Literal["auto", "plain", "chatgpt", "claude", "json"]


@dataclass
class TurnPair:
    """A paired user turn and system (assistant) turn."""

    user_turn: str
    system_turn: str


class ConversationParser:
    """Parse conversation files in multiple formats into Session objects."""

    def __init__(self, session_turn_limit: int = 20) -> None:
        self.session_turn_limit = session_turn_limit

    def parse_file(
        self,
        filepath: str | Path,
        fmt: FormatType = "auto",
    ) -> list[Session]:
        """Parse a conversation file and return a list of Sessions."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        raw = filepath.read_text(encoding="utf-8")
        if not raw.strip():
            raise ValueError(f"File is empty: {filepath}")

        if fmt == "auto":
            fmt = self._detect_format(raw)
            logger.info("Auto-detected format: %s", fmt)

        if fmt == "plain":
            return self._parse_plain(raw)
        elif fmt == "chatgpt":
            return self._parse_chatgpt(raw)
        elif fmt == "claude":
            return self._parse_claude(raw)
        elif fmt == "json":
            return self._parse_json(raw)
        else:
            raise ValueError(f"Unknown format: {fmt}")

    def _detect_format(self, raw: str) -> FormatType:
        """Auto-detect the conversation format."""
        stripped = raw.strip()

        # Try JSON first
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                return "plain"

            # sentinel-ai native: has "sessions" key with turns
            if isinstance(data, dict) and "sessions" in data:
                return "json"

            # ChatGPT export: has "mapping" key
            if isinstance(data, dict) and "mapping" in data:
                return "chatgpt"

            # Claude export: list of dicts with "sender" key
            if isinstance(data, list) and data and isinstance(data[0], dict):
                if "sender" in data[0]:
                    return "claude"

            # Claude export: dict with chat_messages containing sender
            if isinstance(data, dict) and "chat_messages" in data:
                msgs = data["chat_messages"]
                if msgs and isinstance(msgs[0], dict) and "sender" in msgs[0]:
                    return "claude"

            # Fallback: list of sessions or single session dict
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, dict) and "turns" in first:
                    return "json"
            if isinstance(data, dict) and "turns" in data:
                return "json"

            # Unknown JSON structure, try json parser
            return "json"

        # Plain text detection: lines starting with user:/ai:
        if re.search(r"(?mi)^(user|human)\s*:", stripped):
            return "plain"

        return "plain"

    # -- Plain text parser ---------------------------------------------------

    def _parse_plain(self, raw: str) -> list[Session]:
        """Parse plain text with User:/AI: prefixes."""
        turns: list[Turn] = []
        current_role: Role | None = None
        current_lines: list[str] = []

        for line in raw.splitlines():
            # Match role prefix (case-insensitive)
            m = re.match(
                r"^(user|human|User|Human)\s*:\s*(.*)",
                line,
                re.IGNORECASE,
            )
            if m:
                if current_role is not None and current_lines:
                    turns.append(Turn(
                        role=current_role,
                        content="\n".join(current_lines).strip(),
                    ))
                current_role = Role.USER
                current_lines = [m.group(2)] if m.group(2).strip() else []
                continue

            m = re.match(
                r"^(ai|assistant|AI|Assistant|system|System)\s*:\s*(.*)",
                line,
                re.IGNORECASE,
            )
            if m:
                if current_role is not None and current_lines:
                    turns.append(Turn(
                        role=current_role,
                        content="\n".join(current_lines).strip(),
                    ))
                current_role = Role.ASSISTANT
                current_lines = [m.group(2)] if m.group(2).strip() else []
                continue

            # Continuation line
            if current_role is not None:
                current_lines.append(line)

        # Flush last turn
        if current_role is not None and current_lines:
            turns.append(Turn(
                role=current_role,
                content="\n".join(current_lines).strip(),
            ))

        if not turns:
            raise ValueError("No turns found in plain text input")

        return self._split_into_sessions(turns)

    # -- ChatGPT JSON parser -------------------------------------------------

    def _parse_chatgpt(self, raw: str) -> list[Session]:
        """Parse ChatGPT JSON export with mapping dict."""
        data = json.loads(raw)
        mapping = data.get("mapping", {})

        # Build ordered list of messages by following parent links
        nodes = []
        for node_id, node in mapping.items():
            msg = node.get("message")
            if msg is None:
                continue
            author_role = msg.get("author", {}).get("role", "")
            parts = msg.get("content", {}).get("parts", [])
            content = " ".join(str(p) for p in parts if isinstance(p, str)).strip()
            if not content:
                continue
            if author_role in ("user", "assistant"):
                nodes.append({
                    "id": node_id,
                    "role": Role.USER if author_role == "user" else Role.ASSISTANT,
                    "content": content,
                    "parent": node.get("parent"),
                    "create_time": msg.get("create_time", 0),
                })

        # Sort by create_time
        nodes.sort(key=lambda n: n.get("create_time", 0) or 0)

        turns = [Turn(role=n["role"], content=n["content"]) for n in nodes]
        if not turns:
            raise ValueError("No user/assistant messages found in ChatGPT export")

        return self._split_into_sessions(turns)

    # -- Claude JSON parser --------------------------------------------------

    def _parse_claude(self, raw: str) -> list[Session]:
        """Parse Claude JSON export (sender: human/assistant)."""
        data = json.loads(raw)

        # Handle both list format and dict with chat_messages
        if isinstance(data, dict) and "chat_messages" in data:
            messages = data["chat_messages"]
        elif isinstance(data, list):
            messages = data
        else:
            raise ValueError("Unrecognised Claude JSON structure")

        turns: list[Turn] = []
        for msg in messages:
            sender = msg.get("sender", "").lower()
            content = msg.get("text", "") or msg.get("content", "")
            if isinstance(content, list):
                # Claude sometimes has content as list of text blocks
                content = " ".join(
                    b.get("text", "") if isinstance(b, dict) else str(b)
                    for b in content
                ).strip()
            if not content:
                continue

            if sender == "human":
                turns.append(Turn(role=Role.USER, content=content))
            elif sender == "assistant":
                turns.append(Turn(role=Role.ASSISTANT, content=content))

        if not turns:
            raise ValueError("No human/assistant messages found in Claude export")

        return self._split_into_sessions(turns)

    # -- Sentinel-AI native JSON parser --------------------------------------

    def _parse_json(self, raw: str) -> list[Session]:
        """Parse sentinel-ai native JSON (sessions with turns)."""
        data = json.loads(raw)

        # Direct sessions list
        if isinstance(data, dict) and "sessions" in data:
            sessions = []
            for s in data["sessions"]:
                # Coerce session_id to string (golden files use integers)
                if isinstance(s, dict) and "session_id" in s:
                    s["session_id"] = str(s["session_id"])
                session = Session.model_validate(s)
                sessions.append(session)
            return sessions if sessions else [self._empty_session()]

        # List of sessions
        if isinstance(data, list):
            sessions = []
            for item in data:
                if isinstance(item, dict) and "turns" in item:
                    sessions.append(Session.model_validate(item))
            if sessions:
                return sessions

        # Single session dict
        if isinstance(data, dict) and "turns" in data:
            return [Session.model_validate(data)]

        raise ValueError("Cannot parse JSON as sentinel-ai format")

    # -- Session splitting ---------------------------------------------------

    def _split_into_sessions(
        self, turns: list[Turn], prefix: str = "session",
    ) -> list[Session]:
        """Split a flat list of turns into sessions of configurable size."""
        if not turns:
            return [self._empty_session()]

        sessions: list[Session] = []
        chunk: list[Turn] = []

        for turn in turns:
            chunk.append(turn)
            # Count turn pairs (a user + assistant pair = 2 turns)
            if len(chunk) >= self.session_turn_limit:
                sessions.append(Session(
                    session_id=f"{prefix}_{len(sessions) + 1}",
                    turns=chunk,
                ))
                chunk = []

        if chunk:
            sessions.append(Session(
                session_id=f"{prefix}_{len(sessions) + 1}",
                turns=chunk,
            ))

        return sessions

    @staticmethod
    def _empty_session() -> Session:
        return Session(session_id="empty", turns=[])
