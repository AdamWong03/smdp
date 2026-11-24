# Methods Draft — GLR Architecture and Protocol

**Status:** technical draft for Phase 4; numbers to be finalized after the diffusion gate.
**Companion:** `paper/OUTLINE.md`, `paper/EVIDENCE_MAP.md`, `method_development/FD_READOUT_20261121.md`.

---

## 1. Force-free propagator

The backbone is a direct, force-free propagator trained on 5 fs pairs from NVE trajectories. It
maps a state \(x_t=(q_t,v_t)\) to a displacement and a velocity:

\[
(\Delta q_t, v_t^{\rm pred}) = f_\theta(q_t,v_t).
\]

The next position is \(q_{t+1}=q_t+\Delta q_t\). No forces, energies, or integrator are used at
inference. The backbone is frozen in all GLR experiments.

## 2. Mode decomposition

For each molecule \(m\) with atoms \(i\in m\), masses \(m_i\), positions \(r_i\), and velocities
\(v_i\), define the centre of mass and its velocity,

\[
R_m=\frac{\sum_{i\in m} m_i r_i}{M_m},\qquad
V_m=\frac{\sum_{i\in m} m_i v_i}{M_m},\qquad M_m=\sum_{i\in m}m_i .
\]

With \(r_{mi}=r_i-R_m\) (minimum image) and \(v_{mi}=v_i-V_m\), the angular velocity is

\[
\omega_m=I_m^{-1}\sum_{i\in m} m_i\, r_{mi}\times v_{mi},\qquad
I_m=\sum_{i\in m}m_i\left(|r_{mi}|^2\mathbf 1-r_{mi}r_{mi}^{\mathsf T}\right),
\]

using a pseudo-inverse for linear molecules. The three mass-weighted modes are

\[
v^{\rm trans}_i=V_m,\qquad
v^{\rm rot}_i=\omega_m\times r_{mi},\qquad
v^{\rm int}_i=v_{mi}-v^{\rm rot}_i .
\]

They are orthogonal under the mass-weighted inner product,
\(\sum_i m_i v^a_i\cdot v^b_i=0\) for \(a\ne b\), so the kinetic energy splits exactly:

\[
K=\sum_{a\in\{\rm trans,rot,int\}} K_a,\qquad
K_a=\frac12\sum_i m_i |v^a_i|^2 .
\]

The mode degrees of freedom are \(N_{\rm trans}=3N_{\rm mol}-3\),
\(N_{\rm rot}=3N_{\rm mol}\), \(N_{\rm int}=3N_{\rm atom}-6N_{\rm mol}\).

## 3. Mode-resolved AR(p) readout

For each mode \(a\), the reference NVT velocity ACF is

\[
\alpha_a(\ell)=\frac{\sum_i m_i\, \langle v^a_i(t)\cdot v^a_i(t+\ell\Delta t)\rangle}
{\sum_i m_i\, \langle |v^a_i(t)|^2\rangle}.
\]

We fit an AR(p) process by Yule–Walker:

\[
v^a(t)=\sum_{k=1}^{p} c^a_k\, v^a(t-k\Delta t)+\varepsilon^a(t),
\qquad
\mathbf R_a \mathbf c_a=\mathbf r_a,
\]

where \((\mathbf R_a)_{ij}=\alpha_a(|i-j|)\) and \((\mathbf r_a)_k=\alpha_a(k)\). The
prediction-error ratio is

\[
\rho_a=1-\sum_{k=1}^{p} c^a_k\,\alpha_a(k)
=\frac{\operatorname{Var}(\varepsilon^a)}{\operatorname{Var}(v^a)} .
\]

For a stationary AR(p) process, the noise variance that reproduces the target mode variance is
therefore

\[
\operatorname{Var}(\varepsilon^a)=\rho_a\,\frac{N_a k_B T}{2}.
\]

In the implementation, a fresh Gaussian field \(\xi\) is decomposed into the three modes and each
mode component is rescaled to the target noise kinetic energy:

\[
\varepsilon^a=\sqrt{\frac{\rho_a N_a k_B T/2}
{\frac12\sum_i m_i|\xi^a_i|^2}}\;\xi^a .
\]

This is the fluctuation–dissipation (FD) condition: the memory kernel (the AR coefficients) and
the noise amplitude are tied to the same reference ACF.

## 4. Readout, blend, and exact variance rescaling

At each step, the readout produces

\[
\tilde v_t=\sum_{a}\left(\sum_{k=1}^{p} c^a_k v^a_{t-k}+\varepsilon^a_t\right).
\]

It is blended with the backbone velocity,

\[
v_t^{\rm blend}=(1-\lambda)v_t^{\rm pred}+\lambda\tilde v_t,
\]

and then each mode is rescaled to its canonical kinetic energy:

\[
v_t^{\rm GLR}=\sum_a s_a\, (v_t^{\rm blend})^a,\qquad
s_a=\sqrt{\frac{N_a k_B T/2}
{\frac12\sum_i m_i |(v_t^{\rm blend})^a_i|^2}} .
\]

Finally, the total centre-of-mass velocity is projected out. The position update remains

\[
q_{t+1}=q_t+\Delta q_t,
\]

i.e. the backbone supplies the propagator and the readout supplies the velocity variable that
enters the next backbone call. The readout is not integrated as a second position update; this
avoids a double integrator and keeps the learned displacement in distribution.

**What is guaranteed.** The rescaling enforces the canonical **mean** kinetic partition
\(K_a=N_a k_B T/2\) at every step, and the AR(p) stationary ACF matches the reference ACF up to
lag \(p\). The position marginal is not guaranteed analytically; it is learned by the backbone and
checked empirically (RDF, VDOS, diffusion). The per-step rescale also suppresses the equilibrium
fluctuations of the mode kinetic energy (the reference std is ~0.24 eV; the readout std is
~0.002 eV), so the guarantee is on the first moment, not the full kinetic-energy distribution.
The marginal velocity distribution is nevertheless close to the reference: the component
kurtosis is 2.93/5.49/5.93 (translation/rotation/internal) for the GLR and 3.04/5.62/5.57 for the
reference. This distinction is stated explicitly in the paper.

## 5. Fitting and simulation protocol

- Reference: SPC/Fw water NVT, 300 K, CSVR \(\tau=50\) fs, 5 fs spacing; fit on a 100 ps block
  starting at 500 ps.
- AR order \(p=10\) for the main results; \(p=1,5,20\) for the order ablation. \(p=20\) is the best
  ACF match (internal 100/200 fs: 0.147/0.068 vs reference 0.147/0.072).
- Blend \(\lambda=0.1\); variance rescaling on; COM projection on.
- Rollouts: 5 fs, 100 ps screens and 2.5 ns production; no thermostat other than the readout
  (the `--ensemble nve` flag means no external CSVR).
- Metrics: mode kinetic partition; mode ACFs; O–O/O–H/H–H RDF; VDOS; MSD and diffusion with
  block averaging; all against the reference NVT trajectory.

## 6. Baselines and controls

- 250k frozen backbone without readout (structure-only control).
- FDR v1 (AR(1) readout) and GLR AR(p) orders.
- Published TrajCast checkpoint under the same 100 ps / 2.5 ns protocol.
- M0 NVT-pair fine-tune and CSRV moment-loss rollout as negative controls.
- Two initial conditions: val_04 NVE frame and an independent NVT trajectory.

## 7. Ablations

| Component | Configurations |
|---|---|
| Memory order | \(p=1,5,10,20\) |
| FD noise | on / off |
| Variance rescale | on / off |
| Internal blend | \(\lambda_{\rm int}=0.1,1.0\) |
| Seeds | ≥3 for the final configuration |
| Backbone | frozen vs end-to-end (scope, optional) |
| Backbone identity | 250k vs native ep145 (agnosticism, optional) |

## 8. Limitations to state

- The readout is fitted from a short reference NVT trajectory; it is training-free but not
  data-free.
- The analytic guarantee covers the mode kinetic partition and the velocity ACF, not the full
  joint position–velocity distribution.
- One water model and one backbone so far; long-range electrostatics and large-scale physics are
  outside the scope.
- The internal ACF at 500 fs remains too persistent for \(p=10\); \(p=20\) reduces this but does
  not eliminate it.
- Single seed in the current production runs; error bars are pending.
