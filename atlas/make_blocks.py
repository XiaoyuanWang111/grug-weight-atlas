"""Write the `tensor_coefficients` blocks that build the two missing cells of marin#9984's 2 x 2 as new merges.

#9984 built B + a (E - B) + b (E2 - B) with the #9974 tools at a = b = 0.5 and a = b = 1.0. The attention GatedNorm
projections and the attention gate (three tensor families, 78 tensors) moved 1.33% to 3.45% of their norm in E and
E2, the weight matrices 0.04% to 0.07%. The two missing cells hold these families at one strength and everything
else at the other:

  cell C  gate_half.json   default coefficients [1.0, 1.0], the three families at [0.5, 0.5]
  cell D  gate_full.json   default coefficients [0.5, 0.5], the three families at [1.0, 1.0]

Each output file is the value of the recipe key "tensor_coefficients" (tensor name -> coefficients over the
donors, in the donor order E, E2); the rest of the recipe is the one already used for the two existing merges.
gate_patch.py builds cell D without a new merge.

Usage: python atlas/make_blocks.py [<out dir, default recipes_2x2 in the repository root>]

Network: none.
"""

import json
import sys
from pathlib import Path

import st_index

LAYERS = 26
FAMILIES = ("attn_gated_norm.down_proj.weight", "attn_gated_norm.up_proj.weight", "self_attn.attn_gate.weight")


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else st_index.ROOT / "recipes_2x2"
    names = [f"model.layers.{layer}.{family}" for layer in range(LAYERS) for family in FAMILIES]
    out_dir.mkdir(exist_ok=True)
    for file_name, coefficients in (("gate_half.json", [0.5, 0.5]), ("gate_full.json", [1.0, 1.0])):
        text = json.dumps({name: coefficients for name in names}, indent=2) + "\n"
        (out_dir / file_name).write_text(text, encoding="utf-8")
        print(file_name, len(names), "tensors at", coefficients)


if __name__ == "__main__":
    main()
