"""Block-wise translation ACF integrals of the reference NVT trajectory."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from gle_kernel_study import translation_acf

def main():
    ref = Path('/data1/wyd/research/foundation_mlip_adaptation/data/smdp_reference_md_v1/water/physical_reference/nvt_2p5ns/production.h5')
    out = Path('experiments/reference_acf_blocks_20261126.json')
    block_ps = 100.0
    n_blocks = 25
    rows = []
    for i in range(n_blocks):
        start = i * block_ps
        acf, dt = translation_acf(ref, start, block_ps, 5.0, 5000.0)
        t = np.arange(len(acf)) * dt
        integral = float(np.trapezoid(acf, t))
        integral_2ps = float(np.trapezoid(acf[t <= 2000], t[t <= 2000]))
        rows.append(dict(block=i, start_ps=start, integral_fs=integral,
                         integral_2ps_fs=integral_2ps,
                         acf5=float(np.interp(5, t, acf)),
                         acf25=float(np.interp(25, t, acf)),
                         acf100=float(np.interp(100, t, acf)),
                         acf200=float(np.interp(200, t, acf)),
                         acf500=float(np.interp(500, t, acf))))
        print(i, start, integral, flush=True)
    vals = np.array([r['integral_fs'] for r in rows])
    payload = dict(reference=str(ref), block_ps=block_ps, n_blocks=n_blocks,
                   blocks=rows,
                   mean=float(vals.mean()), std=float(vals.std(ddof=1)),
                   min=float(vals.min()), max=float(vals.max()),
                   sem=float(vals.std(ddof=1)/np.sqrt(len(vals))))
    out.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps(payload, indent=2))

if __name__ == '__main__':
    main()
