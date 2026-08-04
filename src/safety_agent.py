"""safety_agent.py — SafetyAgent (hard-rule + LLM judgment layers)."""
import json
import re
from dataclasses import dataclass

HARD_TRIGGERS = [
    "chest pain", "can't breathe", "difficulty breathing",
    "bleeding heavily", "won't stop bleeding",
    "suicidal", "want to die", "kill myself", "end my life",
    "overdose", "took too many pills",
    "baby not moving", "no fetal movement",
]
class SafetyAgent:
    """SafetyAgent class for handling safety checks."""
    def __init__(self):
        self.hard_triggers = HARD_TRIGGERS

    def check_hard_triggers(self, user_input: str) -> bool:
        """Check if the user input contains any hard triggers."""
        for trigger in self.hard_triggers:
            if re.search(trigger, user_input, re.IGNORECASE):
                return True
        return False

    def evaluate_safety(self, user_input: str) -> str:
        """Evaluate the safety of the user input."""
        if self.check_hard_triggers(user_input):
            return "Immediate attention required. Please contact emergency services."
        else:
            return "No immediate safety concerns detected."
    def handle_user_input(self, user_input: str) -> str:
        """Handle user input and provide safety evaluation."""
        safety_evaluation = self.evaluate_safety(user_input)
        return safety_evaluation
    def parse_json(self, json_string: str) -> dict:
        """Parse a JSON string into a dictionary."""
        try:
            return json.loads(json_string)
        except Exception as e:
            return {"error": str(e)}
    def run(self, user_input: str) -> str:
        """Run the safety agent on user input."""
        return self.handle_user_input(user_input)
    def get_hard_triggers(self) -> list:
        """Get the list of hard triggers."""
        return self.hard_triggers
