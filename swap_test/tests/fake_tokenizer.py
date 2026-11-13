"""A byte-level tokenizer that renders the checkpoint's chat template like transformers, for tests.

Kept apart from `helpers` so that processes that only need the tokenizer do not import torch.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, ClassVar

import jinja2
from jinja2 import nodes
from jinja2.ext import Extension
from jinja2.sandbox import ImmutableSandboxedEnvironment

FIXTURES = Path(__file__).parent / "fixtures"
SPECIALS = {
    "<|begin_of_text|>": 256,
    "<|end_of_text|>": 257,
    "<|start_header_id|>": 258,
    "<|end_header_id|>": 259,
    "<|eot_id|>": 260,
    "<|start_think|>": 261,
    "<|end_think|>": 262,
}


class _GenerationTag(Extension):
    """`{% generation %}` ... `{% endgeneration %}` as a pass-through block, as transformers renders it."""

    tags: ClassVar[set[str]] = {"generation"}

    def parse(self, parser: jinja2.parser.Parser) -> nodes.Node:
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
        return nodes.CallBlock(self.call_method("_render"), [], [], body).set_lineno(lineno)

    def _render(self, caller: Any) -> str:
        return caller()


class FakeTokenizer:
    """Bytes as ids 0-255 and the chat special tokens as ids 256-262; renders a chat template like transformers."""

    bos_token_id = SPECIALS["<|begin_of_text|>"]
    _split = re.compile("(" + "|".join(re.escape(s) for s in SPECIALS) + ")")

    def __init__(self, template_path: Path = FIXTURES / "chat_template.jinja") -> None:
        env = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=[_GenerationTag, "jinja2.ext.loopcontrols"]
        )
        self._template = env.from_string(template_path.read_text(encoding="utf-8"))

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **kwargs: Any,
    ) -> str:
        assert not tokenize
        return self._template.render(
            messages=conversation,
            add_generation_prompt=add_generation_prompt,
            bos_token="<|begin_of_text|>",
            eos_token="<|end_of_text|>",
            **kwargs,
        )

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = [self.bos_token_id] if add_special_tokens else []
        for piece in self._split.split(text):
            ids += [SPECIALS[piece]] if piece in SPECIALS else list(piece.encode("utf-8"))
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        names = {v: k for k, v in SPECIALS.items()}
        out, buf = [], bytearray()
        for i in ids:
            if i < 256:
                buf.append(i)
                continue
            out.append(buf.decode("utf-8", errors="replace"))
            buf = bytearray()
            if not skip_special_tokens:
                out.append(names.get(i, f"<|id{i}|>"))
        out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)
