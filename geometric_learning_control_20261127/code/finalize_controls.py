"""Consolidate the bounded geometric-learning controls into one evidence file."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import torch

from control_data import load_states
from evaluate_existing import stat
from positive_control_trajcast import (WATER_CFG, evaluate, make_pairs)


def odd_residual(arrays, states, horizon):
    n = states["q0"].shape[0]
    na = states["q0"].shape[1]
    m = states["masses"][None, :, None].double()
    v0 = states["v0"].double()
    p0 = states["p0"].double()
    qo = arrays["qo"].double().reshape(n, na, 3)
    qo_t = arrays["qo_t"].double().reshape(n, na, 3)
    po = (m * arrays["po"].double().reshape(n, na, 3))
    po_t = (m * arrays["po_t"].double().reshape(n, na, 3))
    return dict(
        Q_odd_residual=stat(qo - horizon * v0, qo_t - horizon * v0),
        P_odd_residual=stat(po - p0, po_t - p0),
    )


def stratify(arrays, states, pairs, horizon):
    n, na = states["q0"].shape[0], states["q0"].shape[1]
    m = states["masses"][None, :, None].double()
    qe = arrays["qe"].double().reshape(n, na, 3)
    qo = arrays["qo"].double().reshape(n, na, 3)
    pe = m * arrays["pe"].double().reshape(n, na, 3)
    po = m * arrays["po"].double().reshape(n, na, 3)
    qe_t = arrays["qe_t"].double().reshape(n, na, 3)
    qo_t = arrays["qo_t"].double().reshape(n, na, 3)
    pe_t = m * arrays["pe_t"].double().reshape(n, na, 3)
    po_t = m * arrays["po_t"].double().reshape(n, na, 3)
    z = states["atomic_numbers"]
    result = dict(per_species={}, error_covariance={}, neighbor_bins=[])
    for zz in sorted(set(int(v) for v in z.tolist())):
        mask = z == zz
        result["per_species"][str(zz)] = {
            name: stat(pred[:, mask], target[:, mask])
            for name, pred, target in (
                ("Q_even", qe, qe_t), ("Q_odd", qo, qo_t),
                ("P_even", pe, pe_t), ("P_odd", po, po_t))
        }
    errs = torch.stack([
        (qe - qe_t).reshape(-1), (qo - qo_t).reshape(-1),
        (pe - pe_t).reshape(-1), (po - po_t).reshape(-1)], 0).double()
    cov = torch.cov(errs)
    std = cov.diag().sqrt()
    result["error_covariance"] = dict(
        covariance=cov.tolist(),
        correlation=(cov / (std[:, None] * std[None, :]).clamp_min(1e-30)).tolist(),
        names=["Q_even", "Q_odd", "P_even", "P_odd"])
    counts = []
    for p in pairs:
        e = p["forward"].edge_index[0].cpu()
        counts.append(torch.bincount(e, minlength=na)[:na].double())
    counts = torch.stack(counts)
    qs = torch.quantile(counts.flatten(),
                        torch.tensor([1 / 3, 2 / 3], dtype=counts.dtype))
    for label, mask in (
            ("low", counts <= qs[0]),
            ("mid", (counts > qs[0]) & (counts <= qs[1])),
            ("high", counts > qs[1])):
        entry = dict(bin=label, atoms=int(mask.sum()))
        for name, pred, target in (
                ("Q_even", qe, qe_t), ("Q_odd", qo, qo_t),
                ("P_even", pe, pe_t), ("P_odd", po, po_t)):
            entry[name] = stat(pred[mask], target[mask])
        result["neighbor_bins"].append(entry)
    return result


def eval_trajcast(system, index, checkpoint, device, pretrained=False,
                  dtype=torch.float64, rms=None, cfg=None):
    torch.set_default_dtype(dtype)
    states = load_states(index, "test", 64, 20261127, dtype)
    pairs = make_pairs(states, float(states["cutoff"]),
                       float(states["horizon_fs"]), dtype)
    if pretrained:
        model_cfg = deepcopy(cfg)
        model_cfg["avg_num_neighbors"] = 89.7544
        model = None
        from trajcast.model.models import EfficientTrajCastModel
        model = EfficientTrajCastModel(
            config=model_cfg,
            predicted_fields=["displacements", "update_velocities"],
            rms_targets=rms, mean_targets=[0.0, 0.0]).to(device)
        sd = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=True)
    else:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model_cfg = ckpt["config"]
        rms = ckpt["rms"]
        from trajcast.model.models import EfficientTrajCastModel
        model = EfficientTrajCastModel(
            config=model_cfg,
            predicted_fields=["displacements", "update_velocities"],
            rms_targets=rms, mean_targets=[0.0, 0.0]).to(device)
        model.load_state_dict(ckpt["model"])
    model.eval()
    res = evaluate(model, pairs, rms, device, dtype, batch_pairs=1,
                   return_arrays=True)
    arrays = res.pop("arrays")
    res["odd_residual"] = odd_residual(
        arrays, states, float(states["horizon_fs"]))
    res.update(stratify(arrays, states, pairs,
                        float(states["horizon_fs"])))
    res.update(system=system, checkpoint=str(checkpoint),
               params=sum(p.numel() for p in model.parameters()),
               cutoff=float(states["cutoff"]),
               horizon_fs=float(states["horizon_fs"]), rms=rms,
               precision=64 if dtype == torch.float64 else 32)
    return res


def main():
    root = Path("experiments/parityflow_20261127")
    controls = root / "controls"
    torch.set_default_dtype(torch.float64)
    water_idx = root / "indices/water5"
    quartz_idx = root / "indices/quartz30"
    official = ("/data1/wyd/research/foundation_mlip_adaptation/upstream/"
                "trajcast-official-water-reference/state_dict_e3nn.pt")
    result = dict(
        status="completed",
        official_trajcast_pretrained_water=eval_trajcast(
            "water", water_idx, official, "cuda", pretrained=True,
            dtype=torch.float64, rms=[0.0575, 0.0130],
            cfg=WATER_CFG),
        official_trajcast_random_2k_water=eval_trajcast(
            "water", water_idx, controls / "trajcast_water_2k.pt", "cuda",
            dtype=torch.float64),
        official_trajcast_random_2k_quartz=eval_trajcast(
            "quartz", quartz_idx, controls / "trajcast_quartz_2k.pt", "cuda",
            dtype=torch.float64),
        official_trajcast_parity_optimizer_1k_water=eval_trajcast(
            "water", water_idx, controls / "trajcast_water_parity_1k.pt",
            "cuda", dtype=torch.float32),
        geometry_only_painn=dict(
            water=json.loads((controls / "painn_water_5k.json").read_text()),
            quartz=json.loads((controls / "painn_quartz_2k.json").read_text()),
        ),
        existing_propagators=json.loads(
            (root / "existing_propagators_results.json").read_text()),
        parityflow_odd_residual=json.loads(
            (root / "odd_residual_diagnostics.json").read_text()),
        provenance=dict(
            trajcast_commit="c52f107a02176671b21a420b2d4c67e0327b04f7",
            trajcast_official_checkpoint=official,
            trajcast_config="upstream/trajcast-official-water-reference/"
                            "water/config_e3nn.yaml (retrieved from the "
                            "official Hugging Face model repository)",
            r3_water="/data1/.../relational_transition/water5_20261103_r3_opt/best.pt",
            r3_quartz="/data1/.../relational_transition/quartz30_20261103_r3_opt/best.pt",
            phase_water="/data1/.../phase_space_20261120/water5/best.pt",
            phase_quartz="/data1/.../phase_space_20261120/quartz30/best.pt",
            parityflow_water=str(root / "paused_v1/water5/last.pt"),
            parityflow_quartz=str(root / "paused_v1/quartz30/last.pt"),
            note="Paired evaluation is matched on physical states/targets, not "
                 "on training data, potential, cutoff, normalization, or "
                 "architecture. R3-Opt/Phase-Space use future-conditioned "
                 "relations; TrajCast and PaiNN use central state/geometry only.",
        ),
    )
    out = root / "positive_control_results.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "official_pretrained": result["official_trajcast_pretrained_water"]["parity"],
        "trajcast_2k_water": result["official_trajcast_random_2k_water"]["parity"],
        "trajcast_2k_quartz": result["official_trajcast_random_2k_quartz"]["parity"],
        "painn_water": result["geometry_only_painn"]["water"]["final"],
        "painn_quartz": result["geometry_only_painn"]["quartz"]["final"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
