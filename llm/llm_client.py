"""
LLM Client for open-source local models using Hugging Face transformers.

This replaces direct OpenAI chat completions with a generic interface.
"""
from typing import List, Dict

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

import config


class LLMClient:
    """
    Simple wrapper around a local/open-source causal LLM.

    It accepts a chat-style list of messages and returns generated text.
    """

    _pipeline = None

    @classmethod
    def _get_pipeline(cls):
        if cls._pipeline is None:
            model_name = config.LLM_MODEL
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            )
            cls._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
        return cls._pipeline

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """
        Generate text from a list of chat messages.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature (0.0 = greedy)
        """
        # Simple chat-to-prompt conversion
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"[SYSTEM]\n{content}\n")
            elif role == "user":
                prompt_parts.append(f"[USER]\n{content}\n")
            else:
                prompt_parts.append(f"[ASSISTANT]\n{content}\n")
        prompt_parts.append("[ASSISTANT]\n")
        full_prompt = "\n".join(prompt_parts)

        pipe = self._get_pipeline()
        outputs = pipe(
            full_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=max(temperature, 1e-5),
            pad_token_id=pipe.tokenizer.eos_token_id,
        )
        generated = outputs[0]["generated_text"]
        # Heuristic: return everything after the last [ASSISTANT] marker
        if "[ASSISTANT]" in generated:
            return generated.split("[ASSISTANT]")[-1].strip()
        return generated.strip()


