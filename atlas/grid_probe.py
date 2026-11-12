"""How do the stored BF16 values of one tensor differ between a parent and its children, by size of the parent weight?

For each binade of |parent| (values in [2^e, 2^(e+1)), where one BF16 step is 2^(e-7)) the script prints, per child:
the share of values that changed, the mean number of grid steps among changed values, the share of changed values
that moved exactly one step, and the share of changes that point away from zero.

A float32 master copy rounded at save time predicts: change probability about |update| / step, so it halves from
one binade to the next larger one, and nearly all changes are single steps. Updates applied directly to BF16
parameters predict a sharp cut: binades whose half step exceeds the update never change.

Usage: python atlas/grid_probe.py [--slice START:COUNT] <tensor name> <parent repo> <child repo> [<child repo> ...]
       (--slice reads only COUNT elements from element START)

Network: the tensor index of every repo (st_index.index, cached), then one range request per repo for the tensor
(or the slice); model.layers.12.self_attn.q_proj.weight is 6,553,600 values, 13.1 MB per repo.
"""

import re
import sys

import numpy as np

import st_index


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    argv = sys.argv[1:]
    start, count = 0, None
    if argv[0] == "--slice":
        start, count = (int(x) for x in argv[1].split(":"))
        argv = argv[2:]
    name, parent, children = argv[0], argv[1], argv[2:]

    p = st_index.read_bf16_bits(parent, name, start, count)
    p_sign, p_mag = p >> 15, p & 0x7FFF
    exponent = (p_mag >> 7) - 127  # unbiased exponent of the BF16 value (8 exponent bits, 7 mantissa bits)
    value = st_index.bf16_to_float32(p)
    print(f"tensor {name} | {p.size:,} values | parent RMS {np.sqrt(np.mean(value.astype(np.float64) ** 2)):.5f}")
    binades = [e for e in range(-16, 2) if (exponent == e).sum() > 0.002 * p.size]
    for repo in children:
        c = st_index.read_bf16_bits(repo, name, start, count)
        c_sign, c_mag = c >> 15, c & 0x7FFF
        same_sign = p_sign == c_sign
        steps = np.where(same_sign, c_mag - p_mag, -(c_mag + p_mag))  # signed along the magnitude axis
        changed = c != p
        print(
            f"\n{re.sub(r'^.*?67B-A2B-', '', repo)}: changed {100 * changed.mean():.2f}% | "
            f"sign flips {100 * (~same_sign).mean():.4f}%"
        )
        print("  binade   step size   share of values   changed %   mean |steps|   one step %   away from zero %")
        for e in binades:
            m = exponent == e
            ch = m & changed
            if ch.sum() == 0:
                print(f"  2^{e:<4d} {2.0 ** (e - 7):11.3e} {100 * m.mean():14.2f} {0:11.3f}")
                continue
            s = np.abs(steps[ch])
            print(
                f"  2^{e:<4d} {2.0 ** (e - 7):11.3e} {100 * m.mean():14.2f} {100 * ch.sum() / m.sum():11.3f} "
                f"{s.mean():14.3f} {100 * (s == 1).mean():12.2f} {100 * (steps[ch] > 0).mean():18.2f}"
            )


if __name__ == "__main__":
    main()
