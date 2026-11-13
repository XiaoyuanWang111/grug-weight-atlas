"""Prompt builder on three made-up GSM8K rows, as JSONL and as parquet, with the real chat template."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from helpers import FakeTokenizer

from swaptest import prompts as prompts_module
from swaptest.prompts import build_prompts, read_gsm8k, select_indices
from swaptest.records import read_jsonl

ROWS = [
    {
        "question": "Ann has 3 apples and buys 4 more. How many apples does she have?",
        "answer": "She has 3 + 4 = <<3+4=7>>7 apples.\n#### 7",
    },
    {"question": "A pen costs $2. How much do 5 pens cost?", "answer": "5 * 2 = <<5*2=10>>10\n#### 10"},
    {
        "question": "Tom ran 1,200 meters on Monday and twice that on Tuesday. How far did he run on Tuesday?",
        "answer": "1200 * 2 = <<1200*2=2400>>2400\n#### 2,400",
    },
]
EXPECTED_TEXT = (
    "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nReasoning: /think<|eot_id|>"
    "<|start_header_id|>user<|end_header_id|>\nQuestion: {q}\nAnswer:<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
)


@pytest.fixture(params=["jsonl", "parquet"])
def gsm8k_file(request, tmp_path):
    if request.param == "jsonl":
        path = tmp_path / "test.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    else:
        path = tmp_path / "test-00000-of-00001.parquet"
        pq.write_table(pa.Table.from_pylist(ROWS), path)
    return path


def test_prompts_render_the_template_with_one_start_token(gsm8k_file, tmp_path):
    out = tmp_path / "prompts.jsonl"
    prompts_module.main(
        [
            "--gsm8k",
            str(gsm8k_file),
            "--checkpoint",
            str(tmp_path),
            "--out",
            str(out),
            "--count",
            "3",
            "--seed",
            "9984",
        ],
        tokenizer=FakeTokenizer(),
    )
    prompts = read_jsonl(out)
    assert [p["idx"] for p in prompts] == [0, 1, 2]
    for p in prompts:
        assert p["prompt_text"] == EXPECTED_TEXT.format(q=ROWS[p["idx"]]["question"])
        assert p["prompt_ids"].count(FakeTokenizer.bos_token_id) == 1 and p["prompt_ids"][0] == 256
        assert FakeTokenizer().decode(p["prompt_ids"]) == p["prompt_text"]
    assert [p["gold"] for p in prompts] == ["7", "10", "2,400"]


def test_a_second_start_token_is_rejected():
    class AddsStartToken(FakeTokenizer):
        def encode(self, text, add_special_tokens=True):
            return super().encode(text, add_special_tokens=True)

    with pytest.raises(ValueError, match="exactly one start token"):
        build_prompts(ROWS, [0], AddsStartToken())


def test_selection_is_seeded_and_extensions_do_not_overlap():
    first = select_indices(1319, 48, 0, 9984)
    assert first == select_indices(1319, 48, 0, 9984) and len(set(first)) == 48
    assert first != select_indices(1319, 48, 0, 9985)
    extension = select_indices(1319, 48, 48, 9984)
    assert not set(first) & set(extension)
    with pytest.raises(ValueError):
        select_indices(3, 2, 2, 9984)


def test_rows_without_question_or_answer_are_rejected(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"question": "q"}) + "\n")
    with pytest.raises(ValueError, match="lacks"):
        read_gsm8k(path)
