## 4. Capability runs issued from the published tree by a driver (phase 2), fixed before any capability reply was sampled

`run_phase2.sh` stops when the harness tree differs from the one recorded in the filled pre-registration of
phase 2 (`09fbae2901d2da97e274226b9fef0be81393a3fa7dd47021fa059b6e2e4431a9`). That tree holds the eligibility
condition that deviation 3 corrects, and with it the scored arm would be N. Steps 22 to 24 therefore run from
the published tree (commit `12f2727ff362b0a19c26512940ad92f2b18bf81a`, tree
`f3f10e1aa86657f932b77de9fe2adbc8dd61caf25fa5fdd1679e93f60d3658a4`) through a driver, `run_phase2b.sh`, that
issues the commands of steps 22 to 24 of `run_phase2.sh` with that script's default settings and the scored arm
G, the candidate by the pre-registered text.

Before the first capability run the driver checks, and stops otherwise, that:

- the two trees differ in `swaptest/` only in the one eligibility line of `analyze_phase2.py`, so every module
  that generates or grades replies is byte-identical to the pre-registered tree;
- the task prompts and the merge spec have the SHA-256 recorded in `prereg_hashes.txt`, and the filled
  pre-registration has the SHA-256 recorded beside it;
- the runaway summary given to the capability analysis, `runaway_reanalysis.json` (the published code on the
  replies and teacher-forced rows of the runaway run), is byte-identical to the published file.

No other setting changes: 270 prompts per task, a cap of 4,096 new tokens, batch size 384, seed 9984,
10,000 bootstrap draws, level 0.95. Written on 2026-11-14 before step 22.
