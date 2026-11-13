"""Stand-in Python for the script smoke test: `python fake_tokenizer_python.py -m MODULE ARGS ...` runs the module
with the test tokenizer (tests/fake_tokenizer.py) in place of the checkpoint's, which needs `transformers`."""

import runpy
import sys

from fake_tokenizer import FakeTokenizer

import swaptest.chat

if len(sys.argv) < 3 or sys.argv[1] != "-m":
    sys.exit("usage: fake_tokenizer_python.py -m MODULE [ARGS ...]")
swaptest.chat.load_tokenizer = lambda checkpoint: FakeTokenizer()
module = sys.argv[2]
sys.argv = [module, *sys.argv[3:]]
runpy.run_module(module, run_name="__main__", alter_sys=True)
