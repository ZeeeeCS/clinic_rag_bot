# """generator.py — AnswerGenerator (LLM call wrapper)."""
# from __future__ import annotations

# import os
# from typing import Any, Dict, List, Optional

# import groq  # type: ignore

# from config_loader import ConfigLoader
# from dotenv import load_dotenv
# load_dotenv()

# class AnswerGenerator:
#     """Generate a final answer from routed context and retrieved evidence."""

#     def __init__(self, config_loader: Optional[ConfigLoader] = None) -> None:
#         self.config_loader = config_loader or ConfigLoader()
#         self.system_prompt = (
#             "You are a helpful medical information assistant. "
#             "Provide concise, non-diagnostic guidance and encourage urgent care for emergencies."
#         )
#         self.model_config = self._get_model_config()

#     def _get_model_config(self) -> Dict[str, Any]:
#         model_settings = self.config_loader.get_model_setting("generation")
#         generator_settings = self.config_loader.get("generator", {}) or {}
#         models_settings = self.config_loader.get("models", {}) or {}

#         if not isinstance(model_settings, dict):
#             model_settings = {}

#         model_name = (
#             model_settings.get("model")
#             or models_settings.get("generation")
#             or generator_settings.get("primary_model")
#             or generator_settings.get("fallback_model")
#         )

#         provider = str(
#             model_settings.get("provider")
#             or models_settings.get("generation_provider")
#             or generator_settings.get("provider")
#             or "groq"
#         ).lower()

#         if not provider:
#             provider = "groq"
#             os.getenv("GROQ_API_KEY")

#         return {"model": model_name, "provider": provider}

#     def _build_prompt(self, query: str, evidence: Optional[List[Dict[str, Any]]] = None) -> str:
#         if not evidence:
#             return query

#         context = "\n".join(
#             f"- {item.get('source') or 'source'}: {str(item.get('text', ''))[:220]}"
#             for item in evidence[:3]
#         )
#         return f"Question: {query}\n\nEvidence:\n{context}"

#     def _call_groq(self, query: str, evidence: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
#         if groq is None:
#             return None

#         api_key = os.getenv("GROQ_API_KEY")
#         if not api_key:
#             return None

#         model_name = self.model_config.get("model") or "llama-3.1-8b-instant"
#         try:
#             client = groq.Groq(api_key=api_key)
#             response = client.chat.completions.create(
#                 model=model_name,
#                 messages=[
#                     {"role": "system", "content": self.system_prompt},
#                     {"role": "user", "content": self._build_prompt(query, evidence)},
#                 ],
#                 temperature=0.2,
#             )
#             choices = getattr(response, "choices", None) or []
#             if not choices:
#                 return None
#             content = getattr(choices[0].message, "content", "")
#             if isinstance(content, str) and content.strip():
#                 return content.strip()
#         except Exception:
#             return None

#         return None

#     def generate(self, query: str, evidence: Optional[List[Dict[str, Any]]] = None) -> str:
#         if not query or not isinstance(query, str):
#             return "Please provide a health-related question."

#         if not evidence:
#             return (
#                 "I can help with general medical information. "
#                 "Please share more detail about your symptoms or question."
#             )

#         groq_api_key = os.getenv("GROQ_API_KEY")
#         should_use_groq = bool(groq_api_key) and self.model_config.get("provider") == "groq"

#         if should_use_groq:
#             groq_answer = self._call_groq(query, evidence)
#             if groq_answer:
#                 return groq_answer

#         context = "\n".join(
#             f"- {item.get('source') or 'source'}: {str(item.get('text', ''))[:220]}"
#             for item in evidence[:3]
#         )
#         return (
#             f"Based on the available evidence, here is a concise response to your question: \n{context}"
#         )

#     def __call__(self, query: str, evidence: Optional[List[Dict[str, Any]]] = None) -> str:
#         return self.generate(query, evidence)

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