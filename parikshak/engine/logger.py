"""Hash-chained run log - the downlinkable flight record.

Every record carries the SHA-256 of the previous record, so the file proves its
own integrity: change one byte anywhere and every hash after it fails to
verify. That is what makes a 40 KB log a substitute for 1.35 GB of video rather
than a summary of it.

The header pins what produced the run - procedure hash, belief schema version,
model versions, the alert operating point - because a log that does not say
which procedure and which weights produced it cannot be audited six weeks later,
and "we think it was the November build" is not an answer for a PI.

Format: JSONL, one record per line, header first. Same reasoning as traces - a
flight record nobody can read with `head` is a flight record nobody checks.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOG_FORMAT_VERSION = "1.0"

#: The chain's anchor. A fixed, published value so the first record's prev_hash
#: is not a magic empty string that could be confused with a missing field.
GENESIS = "0" * 64


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical(record: dict[str, Any]) -> str:
    """Byte-stable serialisation for hashing.

    Sorted keys and fixed separators, so a record hashes identically on the
    Jetson and on a reviewer's laptop. Without this the chain verifies only on
    the machine that wrote it, which is the same as not verifying.
    """
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass
class RunLogger:
    """Appends hash-chained records. Writes JSONL and an optional text mirror."""

    path: Path
    procedure_id: str
    procedure_sha256: str
    run_id: str
    belief_schema: str
    operating_point: str = ""
    model_versions: dict[str, str] = field(default_factory=dict)
    mirror_text: bool = True

    records: list[dict[str, Any]] = field(default_factory=list, repr=False)
    prev_hash: str = GENESIS

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.append("run_start", {
            "log_format": LOG_FORMAT_VERSION,
            "run_id": self.run_id,
            "procedure_id": self.procedure_id,
            "procedure_sha256": self.procedure_sha256,
            "belief_schema": self.belief_schema,
            "operating_point": self.operating_point,
            "models": dict(self.model_versions),
        }, t=0.0)

    # ------------------------------------------------------------------
    def append(self, kind: str, payload: dict[str, Any], t: float) -> dict[str, Any]:
        """Add one record and extend the chain.

        `seq` and `prev_hash` are inside the hashed body on purpose: without
        them, records could be reordered or removed and each individual hash
        would still check out.
        """
        body = {
            "seq": len(self.records),
            "t_mono": round(t, 3),
            "t_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "kind": kind,
            "prev_hash": self.prev_hash,
            "payload": payload,
        }
        body["hash"] = _sha256(_canonical(body))
        self.prev_hash = body["hash"]
        self.records.append(body)
        return body

    # -- typed helpers ---------------------------------------------------
    def step_event(self, t: float, step_id: str, status: str, reason: str = "",
                   confidence: float | None = None) -> None:
        self.append("step", {"step_id": step_id, "status": status,
                             "reason": reason, "confidence": confidence}, t)

    def deviation(self, dev) -> None:
        self.append("deviation", {
            "kind": dev.kind.value, "step_id": dev.step_id,
            "severity": dev.severity, "confidence": round(dev.confidence, 4),
            "reason": dev.reason, "entities": list(dev.entities),
        }, dev.t)

    def alert(self, alert) -> None:
        self.append("alert", {
            "kind": alert.kind.value, "step_id": alert.step_id,
            "severity": alert.severity.label, "channels": list(alert.channels),
            "text": alert.text, "reason": alert.reason,
            "confidence": round(alert.confidence, 4),
        }, alert.t)

    def notice(self, notice) -> None:
        """UNVERIFIED is logged as its own kind, never as a deviation. The
        distinction has to survive into the record, or the eval cannot separate
        "we declined to guess" from "we were wrong"."""
        self.append("unverified", {"step_id": notice.step_id,
                                   "zones": list(notice.zones),
                                   "text": notice.text}, notice.t)

    def override(self, t: float, step_id: str, token: str) -> None:
        self.append("operator_override", {"step_id": step_id, "token": token}, t)

    def close(self, t: float, summary: dict[str, Any]) -> None:
        self.append("run_end", summary, t)
        self.write()

    # ------------------------------------------------------------------
    def write(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8", newline="\n") as fh:
            for rec in self.records:
                fh.write(_canonical(rec) + "\n")
        if self.mirror_text:
            self.text_path.write_text(self.render_text(), encoding="utf-8", newline="\n")
        return self.path

    @property
    def text_path(self) -> Path:
        return self.path.with_suffix(".txt")

    def render_text(self) -> str:
        """The human mirror. A PI reading this six weeks later should not have
        to parse JSON to find out what the crew did."""
        lines = [
            f"PARIKSHAK run log - {self.run_id}",
            f"procedure {self.procedure_id}  sha256:{self.procedure_sha256[:16]}",
            f"belief schema v{self.belief_schema}",
            f"operating point: {self.operating_point}",
            "-" * 78,
        ]
        for rec in self.records:
            p = rec["payload"]
            t = rec["t_mono"]
            kind = rec["kind"]
            if kind == "step":
                lines.append(f"{t:8.1f}s  {p['status']:<11} {p['step_id']:<10} {p['reason']}")
            elif kind == "deviation":
                lines.append(f"{t:8.1f}s  DEVIATION   {p['step_id']:<10} "
                             f"{p['kind']} ({p['severity']}) - {p['reason']}")
            elif kind == "alert":
                lines.append(f"{t:8.1f}s  ALERT       {p['step_id']:<10} "
                             f"[{p['severity']}] {p['text']}")
            elif kind == "unverified":
                lines.append(f"{t:8.1f}s  UNVERIFIED  {p['step_id']:<10} {p['text']}")
            elif kind == "operator_override":
                lines.append(f"{t:8.1f}s  OVERRIDE    {p['step_id']:<10} token={p['token']!r}")
            elif kind == "copilot":
                lines.append(f"{t:8.1f}s  COPILOT     {p.get('step_id', ''):<10} "
                             f"[{p.get('source', '')}] {p.get('text', '')}")
            elif kind == "stall":
                lines.append(f"{t:8.1f}s  STALLED     {p.get('step_id', ''):<10} "
                             f"no progress {p.get('idle_s', '?')} s; missing: {', '.join(p.get('unmet', []))}")
            elif kind == "run_end":
                lines.append("-" * 78)
                for key, value in p.items():
                    lines.append(f"  {key}: {value}")
        lines.append("-" * 78)
        lines.append(f"{len(self.records)} records, chain head {self.prev_hash[:16]}")
        return "\n".join(lines) + "\n"

    @property
    def size_bytes(self) -> int:
        return sum(len(_canonical(r)) + 1 for r in self.records)


# --------------------------------------------------------------------------
def verify_chain(records: Iterable[dict[str, Any]]) -> tuple[bool, str]:
    """Recompute every hash. Returns (ok, message naming the first bad record).

    Pointing at the FIRST divergence matters: in a chain, one edit invalidates
    everything after it, so a verifier that only says "invalid" tells a reviewer
    nothing about where the record was altered.
    """
    prev = GENESIS
    for i, rec in enumerate(records):
        stated = rec.get("hash")
        if stated is None:
            return False, f"record {i} has no hash"
        if rec.get("prev_hash") != prev:
            return False, (f"record {i} (seq {rec.get('seq')}, kind {rec.get('kind')!r}) "
                           f"does not chain: prev_hash {rec.get('prev_hash', '')[:12]} "
                           f"!= {prev[:12]}")
        body = {k: v for k, v in rec.items() if k != "hash"}
        if _sha256(_canonical(body)) != stated:
            return False, (f"record {i} (seq {rec.get('seq')}, kind {rec.get('kind')!r}) "
                           f"content does not match its hash - altered after writing")
        prev = stated
    return True, f"chain intact, {prev[:16]} at head"


def read_log(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def verify_log_file(path: str | Path) -> tuple[bool, str]:
    return verify_chain(read_log(path))
