"""Show, without a model, that a continued assistant message loses its trailing newline, and check the prompt file.

`assistant_prefix_token_ids` of marin (vendor/skyrl/skyrl-train/skyrl_train/inference_engines/chat_template.py at
80004cc06) renders an unfinished assistant message with
`tokenizer.apply_chat_template(messages, add_generation_prompt=False, continue_final_message=True)`. The chat
template trims assistant content; that removes the trailing space of the tag Hugging Face appends to find the end
of the message, and Hugging Face then right-strips the text before the tag. So a prefix that ends in a newline is
rendered without it.

The script prints one example and then checks every record of the prompt file: an arm-T prompt must equal that
rendering of its messages, and an arm-U prompt must equal the same text with the removed whitespace restored and
tokenized as a whole.

Usage: python render_check.py --tokenizer DIR_OR_REPO [--revision REV] [--prompts prompts.jsonl]
The probe used the tokenizer of open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21 at revision
fe666478454e79fe6027a0bab0388d9abd68f51a (tokenizer.json, tokenizer_config.json, chat_template.jinja).
"""

import argparse
import collections
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
EXAMPLE_PREFIX = '<tool_call>\n{"name": "lookup", "arguments": {"id": 7}}\n'


def render_on_main(tokenizer, messages: list[dict]) -> list[int]:
    return tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False, add_generation_prompt=False, continue_final_message=True
    )


def render_with_whitespace(tokenizer, messages: list[dict]) -> list[int]:
    """The same rendering with the prefix's trailing whitespace restored and the whole text tokenized."""
    prefix = messages[-1]["content"]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, continue_final_message=True
    )
    if not text.endswith(prefix):
        text += prefix[len(prefix.rstrip()) :]
    return tokenizer.encode(text, add_special_tokens=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="a directory or a Hugging Face repository")
    parser.add_argument("--revision")
    parser.add_argument("--prompts", type=Path, default=HERE / "prompts.jsonl")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision)

    messages = [{"role": "user", "content": "Call lookup for id 7."}, {"role": "assistant", "content": EXAMPLE_PREFIX}]
    on_main, restored = render_on_main(tokenizer, messages), render_with_whitespace(tokenizer, messages)
    print("example prefix:", repr(EXAMPLE_PREFIX))
    print("  rendered on main ends with the prefix:", tokenizer.decode(on_main).endswith(EXAMPLE_PREFIX))
    print("  last token on main:", on_main[-1], repr(tokenizer.decode(on_main[-1:])))
    print("  last token with the newline restored:", restored[-1], repr(tokenizer.decode(restored[-1:])))

    counts = collections.Counter()
    with open(args.prompts, encoding="utf-8") as handle:
        for line in handle:
            prompt = json.loads(line)
            arm = prompt["probe_arm"]
            render = render_on_main if arm == "T" else render_with_whitespace
            counts[f"arm {arm} records"] += 1
            counts[f"arm {arm} records reproduced"] += render(tokenizer, prompt["messages"]) == prompt["prompt_ids"]
    for key, value in sorted(counts.items()):
        print(f"{value:5d}  {key}")
    reproduced = sum(v for k, v in counts.items() if k.endswith("reproduced"))
    total = sum(v for k, v in counts.items() if k.endswith("records"))
    return 0 if reproduced == total else 1


if __name__ == "__main__":
    sys.exit(main())
