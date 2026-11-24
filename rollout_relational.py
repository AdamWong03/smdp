"""Autonomous 30 fs endpoint rollout for a trained relational model."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import h5py
import numpy as np
import torch

from periodic import PeriodicGraphBuilder
from physics.units import KB_EV_PER_K, KINETIC_TO_EV
from relational_inference import load_relational, step


def temperature(velocities, masses):
    return KINETIC_TO_EV * np.sum(masses[:, None] * velocities**2) / ((3*len(masses)-3)*KB_EV_PER_K)


class CSVR:
    def __init__(self, masses, dt_fs, tau_fs, seed):
        self.masses = masses
        self.dof = 3 * len(masses) - 3
        self.c = math.exp(-dt_fs / tau_fs)
        self.rng = np.random.default_rng(seed)

    def __call__(self, velocities):
        com = (velocities * self.masses[:, None]).sum(0) / self.masses.sum()
        internal = velocities - com
        kinetic = .5 * KINETIC_TO_EV * np.sum(self.masses[:, None] * internal**2)
        if not math.isfinite(kinetic) or kinetic <= 0:
            raise FloatingPointError("Invalid pre-thermostat kinetic energy")
        normal = self.rng.normal()
        chi = self.rng.chisquare(self.dof - 1)
        ratio = (1-self.c) * (.5*KB_EV_PER_K*300) / kinetic
        alpha2 = self.c + ratio*(normal*normal+chi) + 2*normal*math.sqrt(self.c*ratio)
        alpha = math.copysign(math.sqrt(max(alpha2, 0)), math.sqrt(self.c)+math.sqrt(ratio)*normal)
        return com + alpha*internal, alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ensemble", choices=["nve", "nvt"], required=True)
    parser.add_argument("--analysis-ensemble", choices=["nve", "nvt"], default=None,
                        help="Override the ensemble tag stored for analysis without changing dynamics")
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--thermalization-steps", type=int, default=0)
    parser.add_argument("--thermostat-metadata", type=Path)
    parser.add_argument("--seed", type=int, default=2026110301)
    parser.add_argument("--device", default="cuda")
    parser.add_argument('--execution-note', default='GPU exclusivity not recorded')
    parser.add_argument('--dense-query-fs', type=float, default=0.,
                        help='Optional SMDP-R interior output; zero preserves original endpoint path')
    parser.add_argument('--fd-readout', action='store_true',
                        help='apply the fluctuation-dissipation velocity readout')
    parser.add_argument('--position-consistent', action='store_true',
                        help='integrate positions with the GLE velocity instead of the backbone displacement')
    parser.add_argument('--driven-gle', action='store_true',
                        help='AR memory/noise plus backbone drift, with position integration')
    parser.add_argument('--langevin-drift', action='store_true',
                        help='stable Langevin integrator with the backbone drift')
    parser.add_argument('--langevin-gamma', type=float, default=0.1,
                        help='friction coefficient in 1/fs for --langevin-drift')
    parser.add_argument('--langevin-global-rescale', action='store_true',
                        help='global Gamma/CSVR on the total KE in the Langevin integrator')
    parser.add_argument('--fd-blend', type=float, default=1.0)
    parser.add_argument('--fd-blend-translation', type=float, default=None)
    parser.add_argument('--fd-blend-rotation', type=float, default=None)
    parser.add_argument('--fd-blend-internal', type=float, default=None)
    parser.add_argument('--fd-correct-displacement', action='store_true')
    parser.add_argument('--fd-displacement-weight', type=float, default=None,
                        help='Partial displacement correction weight (0=off, 1=full)')
    parser.add_argument('--fd-translation-displacement-weight', type=float, default=0.0,
                        help='Add only the translation-mode velocity correction to the displacement')
    parser.add_argument('--fd-com-displacement-scale', type=float, default=1.0,
                        help='Scale each molecule COM displacement by this factor (internal geometry unchanged)')
    parser.add_argument('--fd-rescale', action='store_true')
    parser.add_argument('--fd-stochastic-rescale', action='store_true')
    parser.add_argument('--fd-rescale-tau', type=float, default=0.0,
                        help='Weak-coupling time (fs) for stochastic rescale; 0 = fresh Gamma each step')
    parser.add_argument('--no-fd-rescale', action='store_true')
    parser.add_argument('--fd-global-rescale', action='store_true',
                        help='global Gamma/CSVR rescale of the total kinetic energy')
    parser.add_argument('--glr-noise-off', action='store_true')
    parser.add_argument('--glr-slow-tau', type=float, default=0.0,
                        help='translation slow-mode relaxation time (fs)')
    parser.add_argument('--glr-slow-weight', type=float, default=0.0,
                        help='translation slow-mode weight in the ACF')
    parser.add_argument('--glr-coefficients', type=Path)
    parser.add_argument('--species-glr-coefficients', type=Path)
    parser.add_argument('--phonon-glr-basis', type=Path,
                        help='phonon_basis.npz from the finite-difference Hessian')
    parser.add_argument('--phonon-glr-coefficients', type=Path,
                        help='per-mode AR coefficients (ar10_diagnostics.npz)')
    parser.add_argument('--phonon-memory-off', action='store_true',
                        help='phonon variance rescale only; no AR memory')
    parser.add_argument('--phonon-rescale-partition',
                        choices=['mode', 'bands', 'species', 'global', 'none'],
                        default='species',
                        help='granularity of the stochastic KE rescale')
    parser.add_argument('--phonon-static-scale', type=Path,
                        help='fixed per-mode amplitude calibration (mode_static_scale.npz)')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if args.output.exists() or args.output.with_suffix(".summary.json").exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model, saved = load_relational(args.checkpoint, device)
    if args.phonon_glr_basis:
        from phonon_glr import PhononBasis, PhononGLRReadout, PhononGLRTransition
        basis = PhononBasis(args.phonon_glr_basis, args.phonon_glr_coefficients)
        readout = None
        if not args.phonon_memory_off:
            if not basis.has_coefficients:
                raise ValueError('--phonon-glr-coefficients is required unless '
                                 '--phonon-memory-off is set')
            readout = PhononGLRReadout(
                basis.coefficients, basis.prediction_error,
                basis.target_variance, use_noise=not args.glr_noise_off)
        static_scale = None
        if args.phonon_static_scale:
            static_scale = np.load(args.phonon_static_scale)['scale']
        model = PhononGLRTransition(
            model, basis, readout=readout, blend=args.fd_blend,
            rescale=((args.fd_rescale or not args.phonon_memory_off)
                     and not args.no_fd_rescale),
            stochastic_rescale=args.fd_stochastic_rescale,
            rescale_tau=args.fd_rescale_tau,
            rescale_partition=args.phonon_rescale_partition,
            static_scale=static_scale).to(device)
    elif args.species_glr_coefficients:
        from species_glr import SpeciesGLRReadout, SpeciesGLRTransition
        data = json.loads(args.species_glr_coefficients.read_text())
        species = [int(s) for s in data['species']]
        readout = SpeciesGLRReadout(
            [data['modes'][str(s)]['coefficients'] for s in species],
            [data['modes'][str(s)]['prediction_error_ratio'] for s in species],
            species,
            [data['modes'][str(s)]['dof'] for s in species],
            use_noise=not args.glr_noise_off)
        model = SpeciesGLRTransition(model, readout, blend=args.fd_blend).to(device)
    elif args.fd_readout or args.glr_coefficients or args.langevin_drift:
        from fd_readout import FDTransition
        translation = (args.fd_blend_translation
                       if args.fd_blend_translation is not None else args.fd_blend)
        rotation = (args.fd_blend_rotation
                    if args.fd_blend_rotation is not None else args.fd_blend)
        internal = (args.fd_blend_internal
                    if args.fd_blend_internal is not None else args.fd_blend)
        blend = (translation, rotation, internal)
        readout = None
        if args.glr_coefficients:
            from glr_readout import GLRReadout
            data = json.loads(args.glr_coefficients.read_text())
            keys = ('translation', 'rotation', 'internal')
            readout = GLRReadout(
                [data['modes'][key]['coefficients'] for key in keys],
                [data['modes'][key]['prediction_error_ratio'] for key in keys],
                use_noise=not args.glr_noise_off,
                slow_tau=args.glr_slow_tau,
                slow_weight=args.glr_slow_weight,
                horizon_fs=float(model.cfg.horizon_fs))
        if args.langevin_drift:
            from langevin_drift import LangevinDriftTransition
            model = LangevinDriftTransition(
                model, gamma=args.langevin_gamma,
                global_rescale=args.langevin_global_rescale).to(device)
        elif args.driven_gle:
            if readout is None:
                raise ValueError('--driven-gle requires --glr-coefficients')
            from driven_gle import DrivenGLETransition
            model = DrivenGLETransition(model, readout=readout).to(device)
        elif args.position_consistent:
            if readout is None:
                raise ValueError('--position-consistent requires --glr-coefficients')
            from position_consistent import PositionConsistentTransition
            model = PositionConsistentTransition(
                model, readout=readout, blend=args.fd_blend).to(device)
        else:
            model = FDTransition(model, readout=readout, blend=blend,
                                 correct_displacement=args.fd_correct_displacement,
                                 displacement_weight=args.fd_displacement_weight,
                                 translation_displacement_weight=args.fd_translation_displacement_weight,
                                 com_displacement_scale=args.fd_com_displacement_scale,
                                 stochastic_rescale=args.fd_stochastic_rescale,
                                 rescale_tau=args.fd_rescale_tau,
                                 global_rescale=args.fd_global_rescale,
                                 rescale=((args.fd_rescale or bool(args.glr_coefficients))
                                          and not args.no_fd_rescale)).to(device)
    dt = model.cfg.horizon_fs
    if args.dense_query_fs:
        if not hasattr(model, 'query') or args.dense_query_fs!=1. or not float(dt).is_integer():
            raise ValueError('Dense output requires SMDP-R and a true 1 fs cadence')
    with h5py.File(args.initial) as source:
        if not 0 <= args.initial_frame < len(source["positions_unwrapped_A"]):
            raise IndexError("initial-frame outside source trajectory")
        q = source["positions_unwrapped_A"][args.initial_frame].astype(np.float32)
        v = source["velocities_A_per_fs"][args.initial_frame].astype(np.float32)
        cell = source["cell_A"][args.initial_frame]
        z, masses = source["atomic_numbers"][:], source["masses_u"][:]
        molecule_ids = source['molecule_ids'][:] if 'molecule_ids' in source else None
    thermostat = None
    thermostat_metadata = {}
    if args.ensemble == "nvt":
        thermostat_metadata = json.loads(args.thermostat_metadata.read_text())
        if thermostat_metadata["ensemble"] != "NVT-CSVR" or thermostat_metadata["temperature_K"] != 300:
            raise ValueError("Expected the frozen 300 K NVT-CSVR protocol")
        thermostat = CSVR(masses, dt, thermostat_metadata["thermostat_tau_fs"], args.seed)
    total_steps = args.thermalization_steps + args.steps
    neighbors = PeriodicGraphBuilder(model.cfg.cutoff_angstrom)
    near = PeriodicGraphBuilder(3.0)
    checksum = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    start = time.perf_counter()
    model_seconds = graph_seconds = 0.0
    failure = None
    completed = 0
    with h5py.File(args.output, "x") as output:
        stored_ensemble = args.analysis_ensemble or args.ensemble
        output.attrs.update(ensemble=stored_ensemble, timestep_fs=dt, requested_steps=args.steps,
                            thermalization_steps=args.thermalization_steps, completed_steps=0,
                            stable=False, failure_reason="", checkpoint=str(args.checkpoint),
                            checkpoint_sha256=checksum, seed=args.seed,
                            initial_trajectory=str(args.initial), initial_frame=args.initial_frame,
                            execution_note=args.execution_note,
                            thermostat_metadata=json.dumps(thermostat_metadata),
                            reference_forces_used_during_propagation=False)
        output["atomic_numbers"] = z; output["masses_u"] = masses; output["cell"] = cell
        if molecule_ids is not None:
            output['molecule_ids'] = molecule_ids
        species = np.unique(z)
        output['kinetic_species'] = species
        species_ke = lambda velocity: np.array([
            .5*KINETIC_TO_EV*np.sum(masses[z==element, None]*velocity[z==element]**2)
            for element in species])
        pre_species = output.create_dataset('pre_thermostat_species_KE_eV',
            (total_steps+1, len(species)), dtype='f8')
        post_species = output.create_dataset('post_thermostat_species_KE_eV',
            (total_steps+1, len(species)), dtype='f8')
        shape = (total_steps+1, len(z), 3)
        positions = output.create_dataset("positions", shape, dtype="f4", chunks=(128, len(z), 3), compression="lzf")
        velocities = output.create_dataset("velocities", shape, dtype="f4", chunks=(128, len(z), 3), compression="lzf")
        scalars = {name: output.create_dataset(name, (total_steps+1,), dtype="f8") for name in
                   ["time_fs", "temperature_K", "pre_thermostat_temperature_K",
                    "thermostat_alpha", "minimum_distance_A"]}
        positions[0], velocities[0] = q, v
        scalars["temperature_K"][0] = scalars["pre_thermostat_temperature_K"][0] = temperature(v, masses)
        scalars["thermostat_alpha"][0] = 1.0
        dense_q = dense_v = None
        previous_derivative = None
        if args.dense_query_fs:
            group = output.create_group('dense')
            group.attrs.update(output_interval_fs=1.,
                boundary_velocity='right continuous: post-CSVR macro velocity at segment start',
                interior_velocity='neural query before endpoint CSVR; q/v consistency not guaranteed')
            dense_shape = (total_steps*int(dt)+1, len(z), 3)
            dense_q = group.create_dataset('positions', dense_shape, dtype='f4',
                                           chunks=(128, len(z), 3), compression='lzf')
            dense_v = group.create_dataset('velocities', dense_shape, dtype='f4',
                                           chunks=(128, len(z), 3), compression='lzf')
            dense_q[0], dense_v[0] = q, v
            join = output.create_dataset('segment_join_derivative_RMS_A_fs', (total_steps+1,), dtype='f8')
            consistency = output.create_dataset('segment_kinematic_RMS_A_fs', (total_steps+1,), dtype='f8')
            join[0] = consistency[0] = 0.
        pre_species[0] = post_species[0] = species_ke(v)
        edges, shifts = near.one(q, cell)
        scalars["minimum_distance_A"][0] = np.linalg.norm(q[edges[0]]+shifts-q[edges[1]], axis=1).min()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            for index in range(1, total_steps+1):
                torch.cuda.synchronize(device); tic = time.perf_counter()
                if args.dense_query_fs:
                    from relational_inference import vesin_graph
                    graph = vesin_graph(q, v, cell, z, masses, model, neighbors, device)
                    with torch.inference_mode():
                        endpoint = model(graph, return_context=True)
                        queried = model.query(graph, endpoint, torch.arange(int(dt)+1, device=device))
                    qn, vn = endpoint['positions'], endpoint['velocities']
                    segment_q = queried['positions'].cpu().numpy()
                    segment_v = queried['velocities'].cpu().numpy()
                else:
                    qn, vn = step(model, q, v, cell, z, masses, device, builder=neighbors,
                                  molecule_ids=molecule_ids)
                torch.cuda.synchronize(device); model_seconds += time.perf_counter()-tic
                q, v = qn.cpu().numpy(), vn.cpu().numpy()
                pre = temperature(v, masses)
                edges, shifts = near.one(q, cell)
                minimum = float(np.linalg.norm(q[edges[0]]+shifts-q[edges[1]], axis=1).min())
                if not np.isfinite(q).all() or not np.isfinite(v).all():
                    raise FloatingPointError("Non-finite predicted phase space")
                if minimum < .65 or pre > 3000 or pre < 1:
                    raise RuntimeError(f"physical safety stop: min={minimum}, preT={pre}")
                alpha = 1.0
                pre_ke = species_ke(v)
                if thermostat is not None:
                    v, alpha = thermostat(v)
                post = temperature(v, masses)
                if dense_q is not None:
                    begin = (index-1)*int(dt)
                    # Do not save a rejected segment as completed dynamics.
                    dense_q[begin:begin+int(dt)] = segment_q[:-1]
                    dense_v[begin:begin+int(dt)] = segment_v[:-1]
                    dense_q[begin+int(dt)] = q
                    dense_v[begin+int(dt)] = v
                    difference = np.diff(segment_q, axis=0)
                    mismatch = difference-.5*(segment_v[1:]+segment_v[:-1])
                    consistency[index] = float(np.sqrt(np.mean(mismatch**2)))
                    join[index] = (float(np.sqrt(np.mean((difference[0]-previous_derivative)**2)))
                                   if previous_derivative is not None else 0.)
                    previous_derivative = difference[-1]
                positions[index], velocities[index] = q, v
                scalars["time_fs"][index] = index*dt
                scalars["pre_thermostat_temperature_K"][index] = pre
                scalars["temperature_K"][index] = post
                scalars["thermostat_alpha"][index] = alpha
                scalars["minimum_distance_A"][index] = minimum
                pre_species[index], post_species[index] = pre_ke, species_ke(v)
                completed = index; output.attrs["completed_steps"] = completed
                if index % 1000 == 0:
                    output.flush()
                    print(json.dumps(dict(step=index, total=total_steps, temperature_K=post,
                                          minimum_A=minimum, elapsed_s=time.perf_counter()-start)), flush=True)
        except Exception as error:
            failure = f"{type(error).__name__}: {error}"
        elapsed = time.perf_counter()-start
        output.attrs.update(stable=failure is None, failure_reason=failure or "")
        output.flush()
    result = dict(completed_steps=completed, production_steps=max(0, completed-args.thermalization_steps),
                  simulated_ps=completed*dt/1000, stable=failure is None, failure_reason=failure,
                  elapsed_seconds=elapsed, model_and_graph_seconds=model_seconds,
                  ns_day_end_to_end=completed*dt*1e-6/max(elapsed, 1e-9)*86400,
                  peak_cuda_bytes=torch.cuda.max_memory_allocated(device),
                  hardware=torch.cuda.get_device_name(device), parameters=sum(p.numel() for p in model.parameters()),
                  selected_epoch=saved["epoch"]+1, checkpoint_sha256=checksum)
    result['execution_note'] = args.execution_note
    args.output.with_suffix(".summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
