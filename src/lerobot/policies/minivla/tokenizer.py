"""
tokenizer.py

Official Qwen2.5 tokenizer wrapper for MiniVLA.
Mirrors teach_code/MiniVLA/prismatic/models/backbones/llm/qwen25.py and
prismatic/models/backbones/llm/prompting/qwen_prompter.py.

Key design:
  - Load Qwen/Qwen2.5-0.5B tokenizer
  - Add 256 extra tokens: "<|extra_0|>" .. "<|extra_255|>"
  - resize_token_embeddings(len(tokenizer), pad_to_multiple_of=64)
  - padding_side="right", sync pad_token_id
  - QwenPromptBuilder: system/user/assistant wrapping with <|im_start|>, <|im_end|>, <|endoftext|>
"""

from __future__ import annotations

from typing import Optional

from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Official system prompts
# ---------------------------------------------------------------------------
SYS_PROMPTS = {
    "prismatic": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "openvla": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
}


# ---------------------------------------------------------------------------
# QwenPromptBuilder (character-for-character match with official)
# ---------------------------------------------------------------------------
class QwenPromptBuilder:
    """
    Official Qwen prompt builder for MiniVLA.
    Wraps messages in <|im_start|>system/user/assistant<|im_end|> format.
    """

    def __init__(self, model_family: str = "openvla", system_prompt: Optional[str] = None):
        self.model_family = model_family
        self.system_prompt = (
            SYS_PROMPTS[model_family] if system_prompt is None else system_prompt
        ).strip()

        self.bos = self.start = "<|im_start|>"
        self.eos = "<|endoftext|>"
        self.end = "<|im_end|>"

        self.wrap_system = lambda msg: f"{self.start}system\n{msg}{self.end}\n"
        self.wrap_human = lambda msg: f"{self.start}user\n{msg}{self.end}\n{self.start}assistant\n"
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.end}\n"

        self.prompt = ""
        self.turn_count = 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        if self.turn_count == 0 and self.system_prompt is not None:
            self.prompt += self.wrap_system(self.system_prompt)

        if self.turn_count % 2 == 0:
            wrapped = self.wrap_human(message)
        else:
            wrapped = self.wrap_gpt(message)

        self.prompt += wrapped
        self.turn_count += 1
        return wrapped

    def get_prompt(self) -> str:
        if self.turn_count % 2 == 0:
            assert self.prompt[-1] == "\n", f"malformed prompt missing newline before EOS: {self.prompt!r}"
            return self.prompt[:-1] + self.eos
        return self.prompt


# ---------------------------------------------------------------------------
# VLATokenizerWrapper: loads Qwen tokenizer + adds extra tokens
# ---------------------------------------------------------------------------
class VLATokenizerWrapper:
    """
    Wraps the Qwen2.5 tokenizer and adds 256 extra action tokens.
    """

    def __init__(self, base_vlm_checkpoint: str = "Qwen/Qwen2.5-0.5B", num_extra_tokens: int = 256):
        self.base_vlm_checkpoint = base_vlm_checkpoint
        self.num_extra_tokens = num_extra_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_vlm_checkpoint,
            trust_remote_code=True,
            padding_side="right",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if num_extra_tokens > 0:
            extra_tokens = [f"<|extra_{i}|>" for i in range(num_extra_tokens)]
            added = self.tokenizer.add_tokens(extra_tokens)
            assert added == num_extra_tokens, (
                f"Added {added} of {num_extra_tokens} extra tokens to tokenizer!"
            )

        self.tokenizer_len = len(self.tokenizer)
        self.tokenizer.resize_token_embeddings(self.tokenizer_len, pad_to_multiple_of=64)
        self.tokenizer.model_max_length = 32768

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    def build_prompt(self, instruction: str, action_text: Optional[str] = None) -> str:
        """
        Build the official MiniVLA prompt.
        Training: includes human turn + assistant response with action tokens.
        Inference: only the human turn, ending at assistant start.
        """
        builder = QwenPromptBuilder("openvla")
        builder.add_turn("human", f"What action should the robot take to {instruction.lower()}?")
        if action_text is not None:
            builder.add_turn("gpt", action_text)
        return builder.get_prompt()

    def build_inference_prompt(self, instruction: str) -> str:
        """Build prompt for inference (no assistant response)."""
        builder = QwenPromptBuilder("openvla")
        builder.add_turn("human", f"What action should the robot take to {instruction.lower()}?")
        return builder.get_prompt()