"""The checkpoint's own tokenizer and chat template, as rendered by `transformers.apply_chat_template`.

Any object with the same four members works in place of a `transformers` tokenizer: `apply_chat_template`,
`encode`, `decode` and `bos_token_id`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class ChatTokenizer(Protocol):
    bos_token_id: int

    def apply_chat_template(self, conversation: list[dict[str, str]], **kwargs: Any) -> Any: ...

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = ...) -> str: ...


def load_tokenizer(checkpoint: Path) -> ChatTokenizer:
    """Tokenizer, special tokens and chat template from the checkpoint directory."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(checkpoint))


def render_user_turn(tokenizer: ChatTokenizer, user_text: str, **template_kwargs: Any) -> tuple[str, list[int]]:
    """One user message with the generation prompt appended, as text and token ids.

    The template writes the start token itself, so the text is encoded without adding special tokens.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True, **template_kwargs
    )
    return text, tokenizer.encode(text, add_special_tokens=False)


def decode(tokenizer: ChatTokenizer, ids: list[int], keep_special: bool = True) -> str:
    return tokenizer.decode(ids, skip_special_tokens=not keep_special)
