"""generator.py — AnswerGenerator."""
import os
from typing import Any, Dict, List, Optional
import groq
from config_loader import ConfigLoader

class AnswerGenerator:
    def __init__(self, config_loader: Optional[ConfigLoader] = None) -> None:
        self.config_loader = config_loader or ConfigLoader()
        # Strict system prompt for RAG behavior
        self.system_prompt = (
            "You are a professional medical RAG assistant. "
            "STRICT RULE: Use ONLY the provided evidence to answer. "
            "If no evidence is provided, you must start your response by stating: "
            "'I could not find specific documents in my medical database regarding this.' "
            "Then you may provide general educational information."
        )

    def generate(self, query: str, evidence: Optional[List[Dict[str, Any]]] = None) -> str:
        api_key = os.getenv("GROQ_API_KEY")
        
        if not evidence:
            full_user_prompt = (
                f"Question: {query}\n\n"
                "Note: No medical evidence was retrieved from the database. "
                "Acknowledge this lack of data and provide a very brief general response."
            )
        else:
            context_text = "\n\n".join([f"Source: {item.get('metadata', {}).get('source', 'Unknown')}\nContent: {item['text']}" for item in evidence])
            full_user_prompt = (
                f"Context from database:\n{context_text}\n\n"
                f"User Question: {query}\n\n"
                "Answer strictly using the context above."
            )

        if not api_key:
            return "Configuration Error: GROQ_API_KEY is missing."

        try:
            client = groq.Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": full_user_prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"I encountered an error generating a response: {str(e)}"