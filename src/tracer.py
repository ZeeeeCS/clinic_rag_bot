"""tracer.py — request tracing (per-step timing/logging to logs/traces.jsonl)."""
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import logging
import os
logging.basicConfig(level=logging.INFO)
trace_file = os.getenv("TRACE_FILE", "logs/traces.jsonl")
class TraceStep:
    """Class representing a single step in the trace."""
    def __init__(self, step_name: str, start_time: float):
        self.step_name = step_name
        self.start_time = start_time
        self.end_time: Optional[float] = None
        self.duration: Optional[float] = None
        self.details: Dict[str, Any] = {}

    def end_step(self):
        """Mark the end of the step and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert the trace step to a dictionary."""
        return {
            "step_name": self.step_name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "details": self.details,
        }