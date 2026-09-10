"""
tokenizer.py

Official Qwen2.5 tokenizer wrapper for MiniVLA.
Mirrors teach_code/MiniVLA/prismatic/models/backbones/llm/qwen25.py and
prismatic/models/backbones/llm/prompting/qwen_prompter.py.

Key design:
  - Load Qwen/Qwen2.5-0.5B tokenizer
  - Add 256 extra tokens: "<|extra_0|>" .. "<|extra_255|>"
  - Do NOT force pad_token = eos_token; use tokenizer's original pad_token_id
  - padding_side="right"
  - QwenPromptBuilder: system/user/assistant wrapping with <|im_start|> and <|im_end|>
  - resize_token_embeddings is called on the LLM model, NOT on the tokenizer
"""

from __future__ import annotations

from typing import Optional

from transformers import AutoTokenizer


SYS_PROMPTS = {
    "prismatic": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "openvla": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
}


class QwenPromptBuilder:
    """
    Mirrors teach_code/MiniVLA/prismatic/models/backbones/llm/prompting/qwen_prompter.py.
    Builds prompts with Qwen's special token format.
    """

    def __init__(self, model_family: str, system_prompt: Optional[str] = None):
        self.system_prompt = (
            SYS_PROMPTS[model_family] if system_prompt is None else system_prompt
        ).strip()

        self.bos = self.start = "<|im_start|>"
        self.eos = "<|endoftext|>"
        self.end = "<|im_end|>"

        self.wrap_system = lambda msg: f"{self.start}system\n{msg}{self.end}\n"
        self.wrap_human = lambda msg: f"{self.start}user\n{msg}{self.end}\n{self.start}assistant\n"
        self.wrap_gpt = lambda msg: f"{msg if msg != '' else ' '}{self.end}\n"

        self.prompt, self.turn_count = "", 0

    def add_turn(self, role: str, message: str) -> str:
        assert (role == "human") if (self.turn_count % 2 == 0) else (role == "gpt")
        message = message.replace("<image>", "").strip()

        if self.turn_count == 0 and self.system_prompt is not None:
            self.prompt += self.wrap_system(self.system_prompt)

        if (self.turn_count % 2) == 0:
            human_message = self.wrap_human(message)
            wrapped_message = human_message
        else:
            gpt_message = self.wrap_gpt(message)
            wrapped_message = gpt_message

        self.prompt += wrapped_message
        self.turn_count += 1
        return wrapped_message

    def get_potential_prompt(self, message: str) -> str:
        prompt_copy = str(self.prompt)
        human_message = self.wrap_human(message)
        prompt_copy += human_message
        return prompt_copy

    def get_prompt(self) -> str:
        if self.turn_count % 2 == 0:
            assert self.prompt[-1] == "\n", f"malformed prompt ({self.prompt}) missing newline before EOS append!"
            return self.prompt[:-1] + self.eos
        return self.prompt


class VLATokenizerWrapper:
    """
    Wraps the Qwen2 tokenizer with extra action tokens.
    Mirrors teach_code/MiniVLA/prismatic/models/backbones/llm/qwen25.py initialization:
      1. Load Qwen tokenizer
      2. Add <|extra_0|>..<|extra_255|>
      3. Use original pad_token_id (do NOT force pad_token = eos_token)
      4. Sync pad_token_id to llm.config.pad_token_id (done in vla_backbone.py)
    
    IMPORTANT: Do NOT call tokenizer.resize_token_embeddings() here.
    resize_token_embeddings must be called on the LLM model, not the tokenizer.
    """

    def __init__(
        self,
        base_vlm_checkpoint: str = "Qwen/Qwen2.5-0.5B",
        num_extra_tokens: int = 256,
    ):
        self.base_vlm_checkpoint = base_vlm_checkpoint
        self.num_extra_tokens = num_extra_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_vlm_checkpoint,
            trust_remote_code=True,
            padding_side="right",
        )

        self.original_pad_token_id = self.tokenizer.pad_token_id

        if num_extra_tokens > 0:
            added = self.tokenizer.add_tokens(
                [f"<|extra_{i}|>" for i in range(num_extra_tokens)]
            )
            assert added == num_extra_tokens, (
                f"Added {added} of {num_extra_tokens} extra tokens to tokenizer!"
            )

        self.tokenizer_len = len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        return self.original_pad_token_id

    def build_prompt(self, instruction: str, action_text: str, state_text: str | None = None) -> str:
        """
        Build training prompt with instruction and action response.
        Mirrors RLDSBatchTransform conversation building.
        If state_text is provided, appends proprioceptive state to the instruction.
        """
        if state_text:
            full_instruction = f"What action should the robot take to {instruction.lower()}? The robot state is: {state_text}"
        else:
            full_instruction = f"What action should the robot take to {instruction.lower()}?"
        prompt_builder = QwenPromptBuilder("openvla")
        prompt_builder.add_turn("human", full_instruction)
        prompt_builder.add_turn("gpt", action_text)
        return prompt_builder.get_prompt()

    def build_inference_prompt(self, instruction: str, state_text: str | None = None) -> str:
        """
        Build inference prompt (no action response).
        If state_text is provided, appends proprioceptive state to the instruction.
        """
        if state_text:
            full_instruction = f"What action should the robot take to {instruction.lower()}? The robot state is: {state_text}"
        else:
            full_instruction = f"What action should the robot take to {instruction.lower()}?"
        prompt_builder = QwenPromptBuilder("openvla")
        prompt_builder.add_turn("human", full_instruction)
        return prompt_builder.get_prompt()