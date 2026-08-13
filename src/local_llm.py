"""src/local_llm.py — Free local Qwen2.5-0.5B-Instruct wrapper."""
import os
import warnings

# Suppress transformers verbosity on first load
warnings.filterwarnings("ignore", category=UserWarning)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalQwen:
    """Runs Qwen2.5-0.5B-Instruct locally."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        print(f"Loading local model: {model_name} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,      
            trust_remote_code=True
        )
        print("Local Qwen model ready.")

    def complete(self, prompt: str, max_new_tokens: int = 256) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional medical assistant. "
                    "Use ONLY the provided context to answer. "
                    "If the answer is not in the context, say you don't have enough information."
                )
            },
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.5,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's reply
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()

        return response