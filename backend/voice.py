"""Voice command grammar shared by the browser (Web Speech) and the server
(Groq Whisper push-to-talk). A fixed grammar + fuzzy phrase matching keeps
commands accurate: only short utterances can trigger, and the match has to be
close to a known phrase."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

COMMANDS: dict[str, list[str]] = {
    "next": ["next", "help", "next step", "what's next", "what is next", "what do i do", "what should i do now",
             "guide me", "next instruction", "tell me the next step", "convey next step", "what now"],
    "review": ["review", "check", "review step", "review my step", "check my step", "check step", "am i doing it right",
               "is this correct", "is this right", "how am i doing", "audit step", "verify step"],
    "repeat": ["repeat", "say again", "repeat instruction", "say that again", "repeat that"],
    "status": ["status", "progress", "where am i", "which step", "current step"],
    "reset": ["reset procedure", "restart procedure", "reset experiment", "restart experiment", "start over"],
    "start_camera": ["start camera", "camera on", "start experiment", "begin experiment", "start tracking"],
    "stop_camera": ["stop camera", "camera off", "power off camera", "turn off camera", "stop tracking"],
    "record": ["start recording", "record video", "begin recording"],
    "stop_record": ["stop recording", "end recording", "save recording"],
    "mesh_on": ["show mesh", "mesh on"],
    "mesh_off": ["hide mesh", "mesh off"],
    "open_3d": ["open 3d", "show 3d", "open 3d view", "3d view", "open rack world"],
    "close_3d": ["close 3d", "hide 3d", "close 3d view"],
    "mute": ["mute voice", "be quiet", "stop talking", "silence"],
    "unmute": ["unmute", "unmute voice", "speak again"],
}

LABELS = {
    "next": "Convey next step", "review": "Review current step", "repeat": "Repeat last instruction",
    "status": "Say status", "reset": "Reset procedure", "start_camera": "Start camera", "stop_camera": "Stop camera",
    "record": "Start recording", "stop_record": "Stop recording", "mesh_on": "Show mesh", "mesh_off": "Hide mesh",
    "open_3d": "Open 3D rack world", "close_3d": "Close 3D", "mute": "Mute voice", "unmute": "Unmute voice",
}

_NORM = [(r"\b(three|3)[ -]?d\b", "3d"), (r"\bwhat's\b", "what's"), (r"[^a-z0-9' ]+", " "), (r"\s+", " ")]


def normalise(text: str) -> str:
    t = (text or "").lower().strip()
    for pat, rep in _NORM:
        t = re.sub(pat, rep, t)
    return t.strip()


def match(text: str, threshold: float = 0.82) -> dict:
    t = normalise(text)
    words = t.split()
    best = {"command": None, "score": 0.0, "phrase": None, "text": t}
    if not words:
        return best
    for cmd, phrases in COMMANDS.items():
        for ph in phrases:
            n_ph = len(ph.split())
            if n_ph == 1 and len(words) > 2:
                continue  # one-word commands only count when said on their own
            if re.search(rf"\b{re.escape(ph)}\b", t):
                score = 1.0 if len(words) <= n_ph + 3 else 0.9
            elif len(words) <= 6:
                score = SequenceMatcher(None, t, ph).ratio()
            else:
                score = 0.0
            if score > best["score"]:
                best = {"command": cmd, "score": round(score, 3), "phrase": ph, "text": t}
    if best["score"] < threshold or len(words) > 10:
        best["command"] = None
    return best


def whisper_prompt() -> str:
    return "Voice commands: " + ", ".join(sorted({p for ps in COMMANDS.values() for p in ps})) + "."
