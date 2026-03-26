import os
import re
import json
import time
import logging

from openai import OpenAI
from dotenv import load_dotenv

from rag import KnowledgeBase
from agents.guardrails import check_code_safety, get_restricted_builtins


class BaseAgent:
    def __init__(self, name, kb=None):
        load_dotenv()
        self.name = name
        self.client = OpenAI(api_key=os.getenv("API_KEY"), base_url=os.getenv("BASE_URL"))
        self.model = os.getenv("MODEL")
        self.kb = kb or KnowledgeBase()
        self.log = []
        self.logger = logging.getLogger(name)

    def call_llm(self, system, user, temperature=0.3):
        return self.call_llm_multi([
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], temperature)

    def call_llm_multi(self, messages, temperature=0.3):
        start = time.time()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
            )
            text = response.choices[0].message.content
            self._log_call(time.time() - start, True,
                           sum(len(m["content"]) for m in messages), len(text))
            return text
        except Exception as e:
            self._log_call(time.time() - start, False, error=str(e))
            return ""

    def _log_call(self, latency, success, prompt_len=0, response_len=0, error=None):
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "agent": self.name,
            "latency": round(latency, 2),
            "success": success,
        }
        if success:
            entry.update(prompt_len=prompt_len, response_len=response_len)
            self.logger.info(f"LLM call: {latency:.1f}s, {response_len} chars")
        else:
            entry["error"] = error
            self.logger.error(f"LLM call failed: {error}")
        self.log.append(entry)

    def parse_json(self, text, fallback=None):
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        for attempt in [
            lambda: json.loads(text),
            lambda: json.loads(re.search(r"```(?:json)?\s*([\s\S]*?)```", text).group(1)),
            lambda: json.loads(re.search(r"\{[\s\S]*\}", text).group(0)),
        ]:
            try:
                return attempt()
            except (json.JSONDecodeError, AttributeError):
                continue
        self.logger.warning("Failed to parse JSON from LLM response")
        return fallback or {}

    def extract_code(self, text):
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        match = re.search(r"```(?:python)?\s*([\s\S]*?)```", text)
        return match.group(1).strip() if match else None

    def execute_code(self, code, namespace, max_retries=5, context=""):
        is_safe, violations = check_code_safety(code)
        if not is_safe:
            self.logger.warning(f"Unsafe code detected: {violations}. Requesting fix.")
            fix = self.call_llm(
                "The code contains disallowed operations: "
                + ", ".join(violations) + ". "
                "Rewrite without these operations. Output only code in a ```python block.",
                f"Requirements:\n{context}\n\nCode:\n```python\n{code}\n```",
            )
            code = self.extract_code(fix) or code
            is_safe, violations = check_code_safety(code)
            if not is_safe:
                raise RuntimeError(f"Code still unsafe after fix: {violations}")

        if "__builtins__" not in namespace:
            namespace["__builtins__"] = get_restricted_builtins()

        last_error = None
        for attempt in range(max_retries):
            try:
                exec(code, namespace)
                return namespace
            except Exception as e:
                last_error = e
                self.logger.warning(f"Code exec failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    fix = self.call_llm(
                        "Fix the Python code. Keep ALL original requirements. "
                        "Output only corrected code in a ```python block.",
                        f"Requirements:\n{context}\n\nCode:\n```python\n{code}\n```\n\nError:\n{e}",
                    )
                    code = self.extract_code(fix) or code
        raise last_error
