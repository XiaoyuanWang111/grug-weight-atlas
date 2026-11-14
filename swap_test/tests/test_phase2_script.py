"""Smoke test of the phase-2 flow: run_phase2.sh from the merge check to the capability analysis on tiny models.

B, E and E2 are tiny float32 checkpoints whose router biases follow the 67B layout (float32 in B, bfloat16 in the
experts, frozen during fine-tuning); Merge-TA0.5 is their half-strength merge with B's router biases kept, so the
first merge check must fail on the router biases alone and the pre-registered rebuild with router-bias
coefficients 0 must pass. Every byte token is a stop token, so no arm runs away: P is not reproduced as a
runaway (V2 fails). With 40 GSM8K prompts, 0 of 40 against 0 of 40 is "no effect" (the upper end of the Newcombe
interval is 0.088; with 4 prompts it is 0.49, "inconclusive"), so N, G and P are eligible, N is the candidate and P
is scored too; the made-up replies cannot be correct, so V3 fails at the end (exit code 43). The script is run
three times, stopping at the two pauses.

run_phase2c.sh then runs on the same data directory with arm N as the scored arm (phase 2 has no task replies of
G here): stop S7 with one correct quick-look reply required; with none required, the raised cap (the limit is set
to 0 replies at the cap) and the stop after V3 fails on arm A, before the scored arm is sampled. To reach the last
two steps the test then rewrites the decision of step c5 to "passed", which the script itself never does.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from helpers import CRUX_ROWS, SPECIALS, FakeTokenizer, mmlu_rows, random_weights, tiny_config, write_checkpoint

from swaptest import prompts as gsm8k_prompts
from swaptest.config import ModelConfig
from swaptest.merge import merge_values
from swaptest.records import read_jsonl

HARNESS = Path(__file__).resolve().parent.parent
CFG = ModelConfig.from_dict(tiny_config())
STOPS = [*range(256), SPECIALS["<|end_of_text|>"], SPECIALS["<|eot_id|>"]]


def find_bash() -> str | None:
    if os.name != "nt":
        return shutil.which("bash")
    git = shutil.which("git")  # Git's bash; the System32 bash.exe starts a different system
    for folder in Path(git).resolve().parents if git else ():
        if (folder / "bin" / "bash.exe").is_file():
            return str(folder / "bin" / "bash.exe")
    return None


BASH = find_bash()


def make_data(root: Path) -> None:
    g = torch.Generator().manual_seed(11)
    base = random_weights(CFG, seed=12)
    experts = []
    for k in range(2):
        tuned = {}
        for name, t in base.items():
            if name.endswith(".mlp.router.bias"):
                tuned[name] = t.to(torch.bfloat16)  # frozen bias, exported in bfloat16
            else:
                tuned[name] = t + 0.05 * (k + 1) * torch.randn(t.shape, generator=g)
        experts.append(tuned)
    half = {}
    for name, t in base.items():
        donors = [e[name].to(t.dtype) for e in experts]
        half[name] = t.clone() if name.endswith(".mlp.router.bias") else merge_values(t, donors, (0.5, 0.5), name)
    for label, weights in (("b", base), ("e", experts[0]), ("e2", experts[1]), ("merge-ta0.5", half)):
        write_checkpoint(root / label, tiny_config(), weights)
        (root / label / "generation_config.json").write_text(json.dumps({"eos_token_id": STOPS}))
    (root / "cruxeval").mkdir()
    (root / "cruxeval" / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in CRUX_ROWS))
    (root / "cruxeval" / "REVISION").write_text("0" * 40 + "\n")
    (root / "mmlu_pro" / "data").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(mmlu_rows(8, False)), root / "mmlu_pro/data/test-00000-of-00001.parquet")
    pq.write_table(
        pa.Table.from_pylist(mmlu_rows(6, True, 1)), root / "mmlu_pro/data/validation-00000-of-00001.parquet"
    )
    (root / "mmlu_pro" / "REVISION").write_text("1" * 40 + "\n")
    (root / "markers").mkdir()
    for item in ("merge", "merge_meta", "b", "e", "e2", "task_data"):
        (root / "markers" / f"{item}.verified").write_text("verified\n")

    phase1 = root / "phase1"
    (phase1 / "main").mkdir(parents=True)
    gsm8k = root / "gsm8k.jsonl"
    gsm8k.write_text(
        "".join(json.dumps({"question": f"What is {i} + 1?", "answer": f"#### {i + 1}"}) + "\n" for i in range(40))
    )
    gsm8k_prompts.main(
        [
            "--gsm8k",
            str(gsm8k),
            "--checkpoint",
            str(root / "merge-ta0.5"),
            "--out",
            str(phase1 / "prompts.jsonl"),
            "--count",
            "40",
        ],
        tokenizer=FakeTokenizer(),
    )
    replies = []
    for p in read_jsonl(phase1 / "prompts.jsonl"):
        text = f"The answer is {p['gold']}."
        ids = [*FakeTokenizer().encode(text, add_special_tokens=False), SPECIALS["<|eot_id|>"]]
        replies.append(
            {
                "arm": "A",
                "prompt_id": p["idx"],
                "seed": 9984,
                "temperature": 1.0,
                "prompt_tokens": 9,
                "n_new": len(ids),
                "stop_reason": ids[-1],
                "token_ids": ids,
                "text": text,
                "wall_s": 1.0,
            }
        )
    (phase1 / "main" / "replies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in replies))
    (phase1 / "analysis.json").write_text(json.dumps({"v1": {"passed": True}}))
    (phase1 / "prereg_hashes.txt").write_text("phase 1\n")


@pytest.mark.skipif(BASH is None, reason="needs bash")
def test_run_phase2_flow(tmp_path):
    data = tmp_path / "data"
    make_data(data)
    wrapper = tmp_path / "python"
    launcher = (HARNESS / "tests" / "fake_tokenizer_python.py").as_posix()
    wrapper.write_text(
        f'#!/usr/bin/env bash\nexec "{Path(sys.executable).as_posix()}" "{launcher}" "$@"\n', newline="\n"
    )
    wrapper.chmod(0o755)
    env = os.environ | {
        "PYTHONPATH": str(HARNESS),
        "DATA_DIR": data.as_posix(),
        "PY": wrapper.as_posix(),
        "THREADS": "2",
        "CAP": "12",
        "BATCH_SIZE": "12",
        "N_TASK": "3",
        "TASK_CAP": "10",
        "TASK_BATCH_SIZE": "8",
        "QUICK_PER_TASK": "2",
        "QUICK_OFFSET": "3",
        "QUICK_MIN_CORRECT": "0",
        "QUICK_MAX_CAPPED": "6",
        "QUICK_MAX_MEDIAN": "100",
        "MIN_FREE_GIB": "0",
        "KV_HEADROOM_GIB": "0",
        "LONG_LEN": "40",
        "LONG_PREFILL": "30",
        "PREFILL_CHUNK": "8",
        "MIXED_STEPS": "4",
        "TIME_BATCHES": "1,3",
        "TIME_STEPS": "2",
        "BOOT_DRAWS": "200",
        "GRADE_WORKERS": "2",
    }

    def run(script: str = "run_phase2.sh", **extra: str) -> int:
        done = subprocess.run(
            [BASH, (HARNESS / "scripts" / script).as_posix()],
            env=env | extra,
            capture_output=True,
            text=True,
            timeout=1500,
        )
        assert "ERROR" not in done.stdout, done.stdout + done.stderr
        return done.returncode

    out = data / "phase2"
    assert run() == 35, (out / "logs" / "run_phase2.log").read_text()  # pause after the quick look
    check = json.loads((out / "merge_check.json").read_text())
    assert not check["equal"] and check["router_bias_kept_at_anchor"]
    assert json.loads((out / "merge_check_bias_at_b.json").read_text())["equal"]
    assert "router biases at B's values" in (out / "merge_check.txt").read_text()
    tasks = read_jsonl(out / "task_prompts.jsonl")
    assert len(tasks) == 9 and not {p["idx"] for p in tasks} & {
        p["idx"] for p in read_jsonl(out / "quick_prompts.jsonl")
    }

    assert run(TASKS_APPROVED="1") == 44  # pause before the capability runs
    prereg = (out / "prereg_phase2.md").read_text()
    assert "____" not in prereg and "(merge_spec_sha256)" in prereg
    assert len(read_jsonl(out / "runaway" / "replies.jsonl")) == 120
    assert len(read_jsonl(out / "teacher_forced.jsonl")) == 120
    decisions = dict(line.split(" ", 1) for line in (out / "runaway_decisions.txt").read_text().splitlines())
    assert decisions["v1"] == "passed" and decisions["v2"] == "fail" and decisions["task_arms"] == "N,P"

    assert run(TASKS_APPROVED="1", CAPABILITY_APPROVED="1") == 43  # V3 fails on made-up replies
    assert len(read_jsonl(out / "tasks_A" / "replies.jsonl")) == 9
    assert sorted({r["arm"] for r in read_jsonl(out / "tasks_merge" / "replies.jsonl")}) == ["N", "P"]
    capability = json.loads((out / "capability.json").read_text())
    assert not capability["v3"]["passed"] and set(capability["comparisons"]) == {"N", "P"}
    assert (out / "SHA256SUMS").is_file()
    assert run(TASKS_APPROVED="1", CAPABILITY_APPROVED="1") == 43  # resumes with every step done

    out_c = data / "phase2c"
    knobs = {"SCORED": "N", "QUICK_N": "2", "QUICK_CAP_LIMIT": "0", "CAP_RAISED": "12"}
    assert run("run_phase2c.sh", **knobs, QUICK_MIN_CORRECT="1") == 51, (out_c / "logs" / "run_phase2c.log").read_text()
    assert not (out_c / "tasks_A").exists()  # stop S7 comes before the main runs
    prereg = (out_c / "prereg_phase2c.md").read_text()
    assert "____" not in prereg and "(phase2_replies_g_sha256)" in prereg
    tasks_c = read_jsonl(out_c / "task_prompts.jsonl")
    assert [p for p in tasks_c if p["task"] != "mmlu_pro"] == [p for p in tasks if p["task"] != "mmlu_pro"]
    zero_shot = [p for p in tasks_c if p["task"] == "mmlu_pro"]
    assert len(zero_shot) == 3 and all(p["idx"] >= 400000 and p["shots"] == 0 for p in zero_shot)
    assert sorted(p["idx"] for p in read_jsonl(out_c / "run_prompts.jsonl")) == sorted(p["idx"] for p in zero_shot)
    assert len(read_jsonl(out_c / "quick_prompts.jsonl")) == 2

    assert run("run_phase2c.sh", **knobs, QUICK_MIN_CORRECT="0") == 43  # V3 fails on arm A
    assert (out_c / "cap.txt").read_text().strip() == "12"
    assert json.loads((out_c / "tasks_A" / "run_config.json").read_text())["cap"] == 12
    assert len(read_jsonl(out_c / "tasks_A" / "replies.jsonl")) == 3
    assert not (out_c / "tasks_merge").exists() and not (out_c / "capability.json").exists()
    # arm A's three replies to phase 2's MMLU-Pro prompts with worked examples do not enter the analysis
    assert "3 replies to prompts outside the prompt file are left out" in (out_c / "v3_arm_A.txt").read_text()

    (out_c / "v3_arm_A_decisions.txt").write_text("v3 passed" + chr(10))
    assert run("run_phase2c.sh", **knobs, QUICK_MIN_CORRECT="0") == 43  # V3 fails again in the full analysis
    assert sorted({r["arm"] for r in read_jsonl(out_c / "tasks_merge" / "replies.jsonl")}) == ["N"]
    capability = json.loads((out_c / "capability.json").read_text())
    assert set(capability["comparisons"]) == {"N"} and capability["comparisons"]["N"]["pairs"] == 9
    assert capability["scores"]["A"]["mmlu_pro"]["n"] == 3 and (out_c / "SHA256SUMS").is_file()
    (data / "phase2" / "merge_spec.json").write_text("{}")
    assert run("run_phase2c.sh", **knobs, QUICK_MIN_CORRECT="0") == 2  # an input changed after the pre-registration
