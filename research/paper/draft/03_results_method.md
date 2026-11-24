# Draft — Results 2: The mode-resolved GLR (provisional)

**Status:** pre-draft for Phase 4; equations to be checked against `METHODS_DRAFT.md`.
**Figure:** Fig. 2.

---

## A readout that separates structure from thermodynamics

The diagnostic says that the missing object is a mode-resolved stationary measure. We therefore
leave the backbone unchanged and add a readout whose job is to enforce that measure. The
architecture is deliberately asymmetric. The frozen backbone supplies the displacement
\(\Delta q_t\); the readout supplies the velocity \(v_t\) that enters the next backbone call and
the kinetic diagnostics. The position update remains

\[
q_{t+1}=q_t+\Delta q_t ,
\]

so the learned displacement stays in distribution and the readout is not integrated as a second
position update. We tested the obvious alternative—adding the velocity correction to the
displacement—and it crashed at 24.8 ps (minimum interatomic distance 0.647 Å), confirming that the
decoupling is a stability requirement rather than a stylistic choice.

## Mode decomposition

For each molecule \(m\), we compute the centre-of-mass velocity \(V_m\) and the angular velocity
\(\omega_m\) from the instantaneous coordinates and velocities. The velocity of atom \(i\) in
molecule \(m\) is split into three orthogonal components:

\[
v_i = \underbrace{V_m}_{v_i^{\rm trans}}
+ \underbrace{\omega_m\times r_{mi}}_{v_i^{\rm rot}}
+ \underbrace{v_{mi}-\omega_m\times r_{mi}}_{v_i^{\rm int}} .
\]

The components are orthogonal under the mass-weighted inner product, so the kinetic energy splits
exactly into translation, rotation, and internal contributions. The corresponding degrees of
freedom are \(N_{\rm trans}=3N_{\rm mol}-3\), \(N_{\rm rot}=3N_{\rm mol}\), and
\(N_{\rm int}=3N_{\rm atom}-6N_{\rm mol}\). The decomposition is differentiable and uses a
pseudo-inverse for the inertia tensor, so linear molecules are handled without special cases.

## Memory kernel and fluctuation–dissipation noise

For each mode \(a\), we compute the reference velocity autocorrelation \(\alpha_a(\ell)\) from a
short NVT trajectory and fit an AR(\(p\)) process by Yule–Walker:

\[
v^a(t)=\sum_{k=1}^{p} c^a_k\,v^a(t-k\Delta t)+\varepsilon^a(t),
\qquad
\mathbf R_a\mathbf c_a=\mathbf r_a ,
\]

where \((\mathbf R_a)_{ij}=\alpha_a(|i-j|)\) and \((\mathbf r_a)_k=\alpha_a(k)\). The
prediction-error ratio

\[
\rho_a=1-\sum_{k=1}^{p}c^a_k\alpha_a(k)
\]

is the fraction of the mode variance not explained by the memory kernel. The
fluctuation–dissipation condition fixes the noise variance:

\[
\operatorname{Var}(\varepsilon^a)=\rho_a\,\frac{N_a k_B T}{2}.
\]

In practice, a fresh Gaussian field is decomposed into the three modes and each mode component is
rescaled to the target noise kinetic energy before it is added to the AR prediction. The memory
kernel and the noise amplitude are therefore tied to the same reference ACF; the readout is not a
single gain with an ad hoc noise level.

The order \(p\) is the only structural knob. A single gain (AR(1)) matches the 5 fs correlation
but not the 25–100 fs memory; it leaves the internal mode too persistent and produces the wrong
O–H RDF. AR(10) matches the short- and mid-lag ACF, and AR(20) also captures the 100–200 fs
internal tail. The paper reports AR(10) as the main configuration and uses the order sweep as an
ablation.

## Exact per-mode variance rescaling

The AR prediction and the FD noise are blended with the backbone velocity,

\[
v_t^{\rm blend}=(1-\lambda)v_t^{\rm pred}+\lambda\tilde v_t ,
\]

and each mode is then rescaled to its canonical kinetic energy:

\[
v_t^{\rm GLR}=\sum_a s_a\,(v_t^{\rm blend})^a,\qquad
s_a=\sqrt{\frac{N_a k_B T/2}
{\frac12\sum_i m_i\,|(v_t^{\rm blend})^a_i|^2}} .
\]

The total centre-of-mass velocity is projected out after the rescale. The blend \(\lambda=0.1\)
keeps the position head in distribution; the rescale, not the blend, is what enforces the
partition. Even a small admixture of the readout is enough because the rescale acts on the full
mode vector at every step.

## What is and is not guaranteed

The construction guarantees two things. First, the instantaneous kinetic partition is canonical:
\(K_a=N_a k_B T/2\) for every mode at every step. Second, the stationary velocity ACF of the
AR(\(p\)) process matches the fitted reference ACF up to lag \(p\). The position marginal is not
guaranteed analytically; it is learned by the backbone and validated empirically by the RDF,
VDOS, and diffusion. This is the honest scope of the method. The guarantee is on the mode kinetic
measure and the velocity memory, not on the full joint position–velocity distribution. The
empirical results show that this partial guarantee is sufficient to repair the failure mode
identified in the previous section.

The readout is fitted, not trained: it requires a short reference NVT trajectory (100 ps in our
water experiments), but no backbone gradients, no forces, and no new labels. The cost is
negligible compared with the backbone forward pass. The next section applies the readout to the
same frozen 250k backbone that failed the NVT screen without it.
