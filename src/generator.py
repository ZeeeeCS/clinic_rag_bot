"""src/generator.py — AnswerGenerator with Groq / Local Qwen toggle."""
import os
from typing import Any, Dict, List, Optional

from config_loader import ConfigLoader


class AnswerGenerator:
    def __init__(self, config_loader: Optional[ConfigLoader] = None) -> None:
        self.config_loader = config_loader or ConfigLoader()

        self.system_prompt = (
            "You are a professional medical RAG assistant. "
            "STRICT RULE: Use ONLY the provided evidence to answer. "
            "If no evidence is provided, you must start your response by stating: "
            "'I could not find specific documents in my medical database regarding this.' "
            "Then you may provide general educational information."
        )

        # ═══ TOGGLE: set USE_LOCAL_LLM=true in .env to use free Qwen ═══
        self.use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

        if self.use_local:
            from local_llm import LocalQwen
            self.local_llm = LocalQwen()
        else:
            import groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY is missing. Set USE_LOCAL_LLM=true for free local mode.")
            self.groq_client = groq.Groq(api_key=api_key)

    def generate(self, query: str, evidence: Optional[List[Dict[str, Any]]] = None) -> str:
        # ═══ LIMIT CONTEXT TO AVOID 413 / TPM ERRORS ═══
        MAX_CHUNK_LEN = 800
        MAX_TOTAL_LEN = 1600

        if not evidence:
            full_user_prompt = (
                f"Question: {query}\n\n"
                "Note: No medical evidence was retrieved from the database. "
                "Acknowledge this lack of data and provide a very brief general response."
            )
        else:
            context_parts = []
            total_len = 0
            for item in evidence[:2]:          # Only top 2 chunks
                text = item.get("text", "")[:MAX_CHUNK_LEN]
                source = item.get("source") or item.get("metadata", {}).get("source", "Unknown")
                chunk = f"Source: {source}\nContent: {text}"
                if total_len + len(chunk) > MAX_TOTAL_LEN:
                    break
                context_parts.append(chunk)
                total_len += len(chunk)

            context_text = "\n\n".join(context_parts)
            full_user_prompt = (
                f"Context from database:\n{context_text}\n\n"
                f"User Question: {query}\n\n"
                "Answer strictly using the context above."
            )

        try:
            if self.use_local:
                return self.local_llm.complete(full_user_prompt)
            else:
                response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",  # 12K TPM, fits better
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": full_user_prompt}
                    ],
                    temperature=0.3
                )
                return response.choices[0].message.content

        except Exception as e:
            return f"I encountered an error generating a response: {str(e)}"