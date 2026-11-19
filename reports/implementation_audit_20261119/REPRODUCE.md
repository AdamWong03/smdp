# Reproduce the implementation audit

This audit performs source reading and CPU numerical contract checks. It does
not train, select a model, run MD, relabel data, modify a checkpoint, install a
new environment, or benchmark external models. Historical experiment ranking
is excluded from the architecture-design rationale.

## Repository and assets

- Own source base: `ce5c6ee`, audit branch `research/implementation-audit-20261119`.
- Worktree: `/data1/wyd/research/foundation_mlip_adaptation/smdp-future-control`.
- Research root: `/data1/wyd/research/foundation_mlip_adaptation`.
- Existing interpreter: `/data1/wyd/envs/trajcast-c52f107/bin/python`.
- PyTorch: `2.5.1+cu121`; execution device: CPU; PyTorch threads: 2.
- DSRT frozen asset: `experiments/water6_reference_20261113/dsrt_radius6/best.pt`.
- Native frozen asset: `experiments/water6_reference_20261113/trajcast_native/best.pt`.
- Cache is read from the DSRT checkpoint config and is also recorded in the JSON.
- Pinned native source: `upstream/trajcast-c52f107`, commit
  `c52f107a02176671b21a420b2d4c67e0327b04f7`.

The native checkpoint belongs to stopped training. It is used to trace the
implemented computation, not presented as a completed or newly evaluated baseline.

## Run

From the worktree:

```bash
SMDP_RESEARCH_ROOT=/data1/wyd/research/foundation_mlip_adaptation \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/data1/wyd/envs/trajcast-c52f107/bin/python \
  scripts/audit_representation_contracts.py \
  --output reports/implementation_audit_20261119
```

This replaces only `representation_contracts.json` in the selected output
directory. To keep an earlier trace, pass a new `--output` directory. The script
uses the existing cache; it does not create a new dataset. Moving assets to
another server also requires resolving the cache path stored in the checkpoint.

The 16 indices are fixed by `linspace(0,4999,16,dtype=int)`. All are TRAIN
samples. The script records graph degree, velocity encoding, physical target
and residual scales, per-block activations, parameter counts, a normalized-loss
identity check, and a single batch-8 backward pass without an optimizer.

The FP32/FP64 comparison copies the same stored FP32 weights into FP64 and also
casts the original graph inputs into the corresponding precision. It therefore
includes input rounding and forward arithmetic; it does not compare training
in two precisions or long-time roundoff. Numerical residual scales are pooled
over equally sized graphs, not reported as a new validation/test result.

The native model mutates label fields when normalizing a fresh graph. The
script clones inputs and verifies the inverse scaling with `rtol=atol=1e-12`.
Observed maximum absolute roundtrip errors were 1.39e-17 Å and 3.47e-18 Å/fs.
The normalized-loss identity had maximum absolute discrepancy 1.46e-11.

## Source provenance

`SOURCE_MANIFEST.json` gives pinned commits, local locations, inspected scope
and downloaded paths. Source downloads are small text files only; no new
pretrained weights or training datasets were acquired. Upstream license files
are retained next to those local files. No upstream code is vendored into this
commit.

For a listed file, its immutable upstream URL is:

```text
https://raw.githubusercontent.com/{repository}/{commit}/{path}
```

The GitHub connector failed to connect during the audit; official GitHub API
and raw-file endpoints were used as the read-only fallback. The E2Former-V2
README names a `src/molfm/scripts/pretrain_template.sh` path absent from the
pinned tree. Training source and available configs were read directly; the
missing launcher is not represented as reviewed or reproduced.

Current FlashMD/metatrain source is explicitly separated from the v0.1.2 paper
reproduction interface. Likewise, MDtrajNet library defaults and E2Former
templates are not labeled as fully recovered paper experiment configurations.

## Deliverables

- `AUDIT.md`: own computation path and external implementation/training contracts.
- `FRONTIER_OPERATORS.md`: E2Former-V2/EquiformerV3 operators, config dispatch,
  numerical assumptions and execution-cost implications.
- `SOURCE_MANIFEST.json`: immutable provenance and disclosed verification limits.
- `representation_contracts.json`: raw CPU observations, including per-sample values.
- `scripts/audit_representation_contracts.py`: executable contract trace.

No result here establishes a causal explanation of long-term dynamics or the
performance of an unimplemented replacement network.
