# How large should the pilot study be?

Research note for issue #13 (pilot study) and #29 / #21 (repeated estimator trials, predicted vs observed variance).
Researched 2026-10-29. Primary sources only; every claim is cited. Anything marked **[Inference]** is our reasoning, not a source's statement. Arithmetic we did ourselves is marked **[Computed]**.

---

## TL;DR (plain English)

- **The pilot is the "survey" that MXMCPy trusts blindly.** MXMCPy takes the pilot's covariance matrix (how the three render levels co-vary) as the truth. It uses it to decide how many samples to spend at each level, to set the control-variate weights, *and* to compute the variance it promises you. A noisy pilot hurts all three, and the predicted variance comes out **too optimistic** on average, not just noisy ([Stanley et al. 2026, Sec. 3](https://arxiv.org/abs/2605.23016); [Coons et al. 2025, Sec. 1](https://arxiv.org/abs/2508.18490)).
- **MXMCPy's own report measures this.** On a simple, well-behaved Gaussian test with 3 models, 10 pilot samples gave about 100% average error in the predicted estimator variance. 20 samples gave about 27–36%, 80 gave about 11–13%, and 320 gave about 4%. With the exact covariance, the error was about 1% (Table 3 checked against the PDF) ([Bomarito et al., NASA/TM-2020-220585, Sec. 4.3, Table 3](https://ntrs.nasa.gov/citations/20200003111)). Our QoI is much less well-behaved than that test.
- **Simpler estimators are more robust to pilot noise.** MFMC (our headline method) and weighted MLMC-style estimators hold up best with small pilots. Flexible ACV variants and MLBLUE overfit pilot noise and can under-predict their true variance by about an order of magnitude ([Stanley et al. 2026, Sec. 3, Figs. 3.1–3.3](https://arxiv.org/abs/2605.23016)).
- **Our QoI makes this worse.** A rare-event indicator is Bernoulli-like. With failure rates of 2–10%, a 20-sample pilot expects only 0.4–2 high-fidelity failures. The variance of such a variable needs roughly "kurtosis-many" samples to estimate ([Giles 2015, Sec. 3.3](https://people.maths.ox.ac.uk/gilesm/files/acta15.pdf)). For our rates the kurtosis is about 8–48 **[Computed]**. No source gives a pilot size for low-correlation, rare-event indicator QoIs. The commonly quoted 10–100 comes from smooth, highly correlated problems.
- **Recommendation for #13 [Inference]:** don't pick a fixed small number. Run the pilot in stages, starting with about 50–100 joint samples (all 3 levels on the same inputs). Store the raw navigation errors, not just indicators. Bootstrap the correlations and MXMCPy's predicted variance, and keep adding samples until the decisions they drive stop flipping across bootstrap replicates. Those decisions are which estimator wins, whether multifidelity beats MC, and the sample allocation. Expect to need enough samples for roughly 10–20 high-fidelity "near-failures", i.e. a few hundred at p ≈ 5%. Report the pilot cost and confidence intervals on the correlations.
- **Recommendation for #29 [Inference]:** separate "pilot noise" from "MXMCPy is wrong". Build an oracle covariance from a large joint sample, e.g. by running the cheap levels on the brute-force inputs. Then run two checks. (A) Fix the allocation, repeat trials, and check observed variance against the variance formula evaluated at the oracle covariance; this tests the math. (B) Give each trial its own fresh pilot and compare observed MSE against the brute-force reference; this tests the end-to-end workflow. Don't reuse pilot samples in the validation trials.

---

## 1. How noisy pilot covariances bias the allocation and the predicted variance

**What MXMCPy does with the pilot.** MXMCPy has no notion of pilot uncertainty. `OutputProcessor.compute_covariance_matrix` is a plain `np.cov(..., ddof=1)` over the pilot outputs. That matrix goes into `Optimizer(model_costs, covariance)` to get the allocation and the predicted variance. The same matrix goes into `Estimator(allocation, covariance)`, which computes the control-variate weights `alpha` from it by `np.linalg.solve` ([`mxmc/output_processor.py`](https://github.com/nasa/MXMCPy/blob/main/mxmc/output_processor.py); [`mxmc/estimators/acv_estimator.py`](https://github.com/nasa/MXMCPy/blob/main/mxmc/estimators/acv_estimator.py), checked at commit `367c7de`). The TM says the variance of the final estimator is the `min_variance` from the optimization step ([TM, Sec. 2, Step 5](https://ntrs.nasa.gov/citations/20200003111)). So the "predicted variance" is just the variance formula evaluated at the pilot covariance.

**Three distinct effects.** In Stanley et al.'s notation, ξ̂ is the design (allocation plus weights) chosen from the sample covariance Σ̂, and ξ* is the design chosen from the true covariance Σ ([Stanley et al. 2026, Secs. 2–3](https://arxiv.org/abs/2605.23016)).

1. **Suboptimal allocation and weights.** The true variance of the pilot-tuned estimator, v(ξ̂; Σ), is always at least the oracle optimum v(ξ*; Σ). Stanley et al. show that, for the bivariate case, optimizing with the sample correlation always gives a worse true variance than the oracle, for every ρ in (0,1) (Sec. 2.1). Peherstorfer et al. say the same qualitatively: with sample estimates plugged in, the allocation and coefficients are "in general not the optimal" ones ([PWG 2016, Sec. 3.6, after Alg. 2](https://doi.org/10.1137/15M1046472)).
2. **Optimistic predicted variance.** The projected variance v(ξ̂; Σ̂) tends to be *smaller* than the true v(ξ̂; Σ). This is an "overconfidence" risk. The more general estimators can underpredict by about an order of magnitude at small pilot sizes ([Stanley et al. 2026, Sec. 3, Fig. 3.3](https://arxiv.org/abs/2605.23016)). Coons et al. state that too few pilots can underestimate the true variance "by orders of magnitude" ([Coons et al. 2025, Sec. 1, Fig. 1](https://arxiv.org/abs/2508.18490)). **[Inference]** The mechanism is selection: the optimizer exploits correlations that look high by chance, and the same flattering numbers then feed the variance it reports.
3. **Model ordering and selection can flip.** MFMC requires the models to be ordered by correlation and to satisfy a cost/correlation condition ([PWG 2016, condition (3.12), Sec. 3.4, Fig. 1](https://doi.org/10.1137/15M1046472)). MXMCPy's MFMC optimizer sorts models by estimated correlation and raises `InconsistentModelError` if the condition fails ([`mxmc/optimizers/mfmc.py`](https://github.com/nasa/MXMCPy/blob/main/mxmc/optimizers/mfmc.py)). **[Inference]** With noisy correlations, a pilot can reorder the levels, drop a level, or raise this error, even when the true ladder is fine.

**Which piece of the covariance matters most.** For bi-fidelity MFMC, a Shapley sensitivity analysis attributes most of the suboptimality to the **correlation coefficient**. Its average Shapley value is 0.74, against 0.09 and 0.17 for the two standard deviations ([Stanley et al. 2026, Sec. 3.1, Fig. 3.4](https://arxiv.org/abs/2605.23016)). Peherstorfer et al. make the matching observation that the weights depend only on ratios σ₁/σᵢ, which are more stable than the variances themselves ([PWG 2016, Sec. 4.1, Table 2](https://doi.org/10.1137/15M1046472)).

**Which estimators are most sensitive.**
- Stanley et al. compared MFMC, WRDIFF (weighted MLMC-style), ACVIS, the generalized recursive ACV families (GMFMR, GRDMR, GISMR) and MLBLUE, with 4 models. MLBLUE and generalized ACV win with many pilot samples. With few samples they can have "far greater" true variance than MFMC/WRDIFF. In their tests MFMC came out best whenever there were fewer than 10 pilot samples and more than two models. Generalized estimators are "unequivocally more sensitive" in relative terms ([Stanley et al. 2026, Sec. 3, Figs. 3.1–3.2](https://arxiv.org/abs/2605.23016)). They give two causes. Flexible allocations overfit spurious pilot cross-correlations, while MFMC/WRDIFF depend on only M−1 correlations. Generalized ACV also inverts a dense covariance block that amplifies noise, where MFMC/WRDIFF invert diagonal ones. MLBLUE is called the most sensitive (same section).
- The ACV paper's own pilot experiment used 20 pilot samples out of a total cost of 100. It found every estimator "less efficient" with an estimated covariance, though ACV-MF still generally outperformed the alternatives ([Gorodetsky et al. 2020, arXiv v-preprint, Sec. 4.2.2, Fig. 4.7](https://arxiv.org/abs/1811.04988)).
- Dixon et al. (2023/24) found that the ACV estimator's variance spread is widest with few pilot samples ([Dixon, Warner, Bomarito & Gorodetsky, Sec. 4.2.2, Fig. 6](https://arxiv.org/abs/2310.00125)).
- **MLMC in MXMCPy** has fixed weights of −1 (`MLMCEstimator._calculate_alpha` returns `-1.0*np.ones(...)`). Pilot noise therefore only affects its allocation and predicted variance, which come from `Var(Y_l − Y_{l−1})` derived from the covariance, not its weights ([`mxmc/estimators/mlmc_estimator.py`](https://github.com/nasa/MXMCPy/blob/main/mxmc/estimators/mlmc_estimator.py), [`mxmc/optimizers/mlmc.py`](https://github.com/nasa/MXMCPy/blob/main/mxmc/optimizers/mlmc.py)). The catch is that unweighted recursive differences can lose to plain MC when correlations are modest, where the weighted version can still win ([Gorodetsky et al. 2020, Sec. 4.2.1, Fig. 4.6](https://arxiv.org/abs/1811.04988)).

**How robust is "robust"?** Peherstorfer et al. state that a small pilot "is typically sufficient" because small perturbations have small effects ([PWG 2016, Sec. 3.6; Sec. 3.4, Fig. 2](https://doi.org/10.1137/15M1046472)). Their supporting example (the plate, Table 2) has correlations of **0.995–0.99999996**. **[Inference]** That evidence does not carry over to a QoI whose correlations may be 0.5–0.9. Stanley et al. explicitly frame the "large-pilot / exact-covariance" assumption as common in the literature and impractical when the high-fidelity model is expensive ([Stanley et al. 2026, Sec. 1](https://arxiv.org/abs/2605.23016)).

---

## 2. How many pilot samples? What sources actually say

No source gives a universal number. All numbers below are tied to the problem they came from.

| Source | Pilot size(s) and finding | Problem context / caveat |
|---|---|---|
| [MXMCPy TM, Sec. 4.3, Table 3](https://ntrs.nasa.gov/citations/20200003111) | Mean relative error of the **predicted estimator variance**: 10 → 102.1–107.9%; 20 → 27.2–36.2%; 40 → 16.4–18.6%; 80 → 10.8–13.4%; 160 → 7.5–7.8%; 320 → 3.8–4.0%; exact Σ → 0.8–1.2% (MFMC, ACVIS, GRDMR). Checked against the PDF. Says the magnitude is "problem dependent". | 3 linear-Gaussian models, ρ = 0.9/0.8/0.9, costs 1/0.1/0.01, target cost 80; ten pilots per size, 10,000 estimator repeats each. Best-case statistics. (The printed Σ has a typo, "8.0" for 0.8, per the Cholesky factor shown.) |
| [MXMCPy examples](https://github.com/nasa/MXMCPy/tree/main/examples) (`ishigami`, `spring_mass`) | `num_pilot_samples = 10` | Demonstration code, not a recommendation. |
| [PWG 2016, Sec. 3.6, Sec. 4.1 Table 2, Sec. 4.2](https://doi.org/10.1137/15M1046472) | "A small number m is typically sufficient"; compared 10/100/1000 (plate) and used 100 (tubular reactor). | Correlations ≥ 0.995 in the plate example. |
| [Gorodetsky et al. 2020, Sec. 4.2.2](https://arxiv.org/abs/1811.04988) | 20 pilot samples within a total cost of 100; estimators lose efficiency but ACV-MF still generally best. | Tunable synthetic model. |
| [Dixon et al. 2023/24, Sec. 4.2.1, Fig. 5](https://arxiv.org/abs/2310.00125) | For mean estimation, a "sharp transition at 10 pilot samples" before the 5th-percentile variance reduction beats MC. For main-effect **variance** estimation, about 25 to beat ACV and about 250 to converge. | Synthetic 9-input system; the "worst-case" 5th percentile over 1000 pilot draws. Shows higher-order statistics need far more pilots. |
| [Stanley et al. 2026, Sec. 1, Sec. 3](https://arxiv.org/abs/2605.23016) | Practical pilots are "often just 5 to 20 pilot samples" (no source cited for this). MFMC best below about 10 pilots with >2 models. | Synthetic Gaussian covariances plus a NASA EDL trajectory dataset. Authors are the MXMCPy team. |
| [Coons et al. 2025, Sec. 5](https://arxiv.org/abs/2508.18490) | Bayesian adaptive stopping picked on average about 8–61 pilots (monomial benchmark, budgets equal to 50–200 pilots, prior-dependent). Darcy flow: loss plateaued at 10–14 pilots. | ACV; Gaussian-type priors on Σ. |
| [Giles 2015, Sec. 3.3](https://people.maths.ox.ac.uk/gilesm/files/acta15.pdf) | For an MLMC level variance, "as few as 10" may give a rough estimate, but O(κ) samples are needed, where κ is the kurtosis. The 0/1-output case is called out as extreme. | Applies directly to indicator QoIs (see Q5). |
| [Croci, Willcox & Wright 2023, Sec. 4](https://arxiv.org/abs/2301.07831) | Used 100 pilots, "larger than needed in practice in most cases", chosen to remove pilot error from comparisons. Pilot cost excluded from comparisons. | MLBLUE/MLMC/MFMC comparison; cites Ng & Willcox 2014 for the "more than needed" claim (not checked). |
| [Konrad et al. 2022, Sec. 4.2](https://arxiv.org/abs/2103.07539) | 50 pilot samples. | Correlations about 0.996–0.998. |
| [Warner, Bomarito, Geraci & Eldred 2026, Secs. 2.4, 4.1](https://arxiv.org/abs/2509.16007) | In a model-tuning setting, pilots of 10 to 100. Performance improved with N_pilot with diminishing returns; N_pilot = 10 often settled on suboptimal settings. Pilot size is "typically" a modest share of the budget. | Pilots used to tune model hyperparameters as well as allocate. |

**Theory for choosing the pilot size.**
- **Explore/exploit theory.** Dixon, Gorodetsky, Jakeman, Narayan & Xu derive an optimal number of pilot ("exploration") samples under a linear-model assumption: q*_S = B / (c_r + √(c_r γ(S)/k(S))). It grows **linearly with the total budget B** ([Dixon et al. 2026, arXiv:2505.09828, Eq. (24)–(25)](https://arxiv.org/abs/2505.09828)). Their algorithm starts at q = n + 2 pilots (n = number of low-fidelity models) and grows adaptively (Alg. 1). This builds on the bandit "AETC" framing of [Xu, Keshavarzzadeh, Kirby & Narayan 2022](https://arxiv.org/abs/2103.15342).
- **Bayesian stopping.** Coons et al. put a posterior on Σ and stop when the projected accuracy gain no longer pays for the extra pilot cost ([Coons et al. 2025, Alg. 1](https://arxiv.org/abs/2508.18490)).
- **Robust correlation adjustment.** Stanley et al.'s DDMM shrinks the sample correlation with a minimax rule over a confidence set. They report improvement over the raw sample correlation for small pilots, e.g. N = 5 on the EDL data, with an average variance-reduction-ratio gain of 8.6% ([Stanley et al. 2026, Secs. 4–5, Table 1](https://arxiv.org/abs/2605.23016)). The method assumes Gaussian outputs. Its empirical gains were smaller than theory on non-Gaussian QoIs (Sec. 5.2).
- **Pilot as a share of budget.** Dixon et al. (2023/24) show a U-shape. Too few pilots inflate variance, and too large a pilot fraction starves the estimator; a larger budget needs a smaller pilot fraction ([Sec. 4.2.2, Fig. 6](https://arxiv.org/abs/2310.00125)).
- **Rank floor.** A sample covariance from n samples has rank at most n, so take enough to avoid an artificially singular matrix ([Croci et al. 2023, Remark 2.4](https://arxiv.org/abs/2301.07831)). With 3 levels this is a trivial floor, but it matters for the dense solve in ACV.

---

## 3. Can or should pilot samples be reused in the final estimate?

**What sources say.**
- **Reuse introduces bias, usually small.** Konrad, Farcas, Peherstorfer et al. state that reusing pilot samples "can introduce a bias". They add that the MSE is typically dominated by the limited sample count rather than by that bias ([Konrad et al. 2022, Remark 1, Sec. 2](https://arxiv.org/abs/2103.07539)). In their plasma example (50 pilots, correlations ≥ 0.996), reuse had an "insignificant effect" on MSE for both mean and variance estimates (Sec. 4.2, Fig. 9, Table 11).
- **Reuse means you must measure MSE, not variance.** Dixon et al. append the pilot samples to the estimation sets ("MOACV-Reuse"). They note this creates bias, so MSE must be measured instead of just variance. Reuse beat the no-reuse estimator only at large pilot fractions. An adaptive variant that grows the pilot and then folds it in beat no-reuse at every pilot fraction, and the bias "seems negligible" in that example ([Dixon et al. 2023/24, Appendix C, Fig. 12](https://arxiv.org/abs/2310.00125)).
- **Theory: bias is asymptotically negligible under a linear model.** Dixon et al. (2026) show their LRMC estimator stops being exactly unbiased when exploration samples are reused. The extra MSE is asymptotically negligible under their linear-model assumption ([arXiv:2505.09828, Thm. 4.11](https://arxiv.org/abs/2505.09828)). In experiments they still chose **not** to reuse, finding negligible effects on bias and accuracy either way (Sec. 5).
- **Some workflows reuse by design.** "Iterated ACV" in Dakota starts from a small pilot and grows it as shared samples ([Warner et al. 2026, Secs. 2.4, 2.6](https://arxiv.org/abs/2509.16007)). Dai, Tchelepi & Tartakovsky report that reusing pilots lowered MSE in their vector-QoI experiments (abstract only checked) ([arXiv:2610.25016](https://arxiv.org/abs/2610.25016)).
- **Estimated weights alone can add bias.** The ACV paper reports a bias from estimating the CV weights of order 10⁻³ over 5000 realizations, called negligible ([Gorodetsky et al. 2020, Fig. 4.7 caption](https://arxiv.org/abs/1811.04988)).

**What MXMCPy does.** MXMCPy has **no pilot-reuse feature**. The documented workflow draws `num_total_samples` fresh inputs and partitions them with `allocate_samples_to_models` ([TM, Sec. 2, Step 3](https://ntrs.nasa.gov/citations/20200003111)). A user could reuse samples manually in two ways. One is passing outputs as the leading samples of the allocation. The other is re-estimating the covariance from the final outputs with `OutputProcessor.compute_covariance_matrix(model_outputs, sample_allocation)`, which the code supports for non-shared inputs ([`output_processor.py`](https://github.com/nasa/MXMCPy/blob/main/mxmc/output_processor.py)). **[Inference]** Both make the weights or the allocation depend on the same data as the estimate, which is exactly the source of the bias above.

**[Inference] For Fidelis specifically.** The spec sets the threshold r *from the pilot*, so the QoI itself depends on the pilot draw. Reusing pilot samples in an estimate of P(error > r) then couples three things to the same data: the threshold, the allocation and weights, and the estimate. The cited "negligible bias" evidence all comes from smooth, highly correlated outputs, not rare events. Since the whole point of #29 is an unbiased comparison against a brute-force reference, **don't reuse pilot samples in the validation trials**. If reuse is ever used for the single showable M3 estimate, say so and report MSE against the reference.

---

## 4. Comparing MXMCPy's predicted variance with observed variance

**What the sources do.** MXMCPy's verification computes the allocation from the **exact** covariance, repeats the sampling-and-estimation steps 10,000 times, and compares the sample variance of the estimates with the predicted variance. The relative error is about 1% ([TM, Sec. 4.2, Table 2](https://ntrs.nasa.gov/citations/20200003111)). It then repeats with pilot-estimated covariances, ten pilots per pilot size, to show the extra error from the pilot alone (Sec. 4.3, Table 3). Konrad et al. plot analytic MSE beside empirical MSE from 20 replicates ([Fig. 9](https://arxiv.org/abs/2103.07539)). Stanley et al. evaluate each pilot-tuned design under the oracle covariance and report the ratio of true to projected variance ([Sec. 3, Fig. 3.3](https://arxiv.org/abs/2605.23016)).

**The decomposition to use [Inference, built on Stanley et al.'s notation].** For each trial, keep four numbers apart:

| Quantity | Meaning | How to get it |
|---|---|---|
| v_pred = v(ξ̂; Σ̂) | What MXMCPy promises | `opt_result.variance` from the trial's pilot covariance |
| v_true\|pilot = v(ξ̂; Σ*) | True variance of *this* pilot-tuned design | Variance formula with the trial's allocation **and its pilot weights**, evaluated at the oracle Σ* |
| v_opt = v(ξ*; Σ*) | Best achievable | `Optimizer(costs, Σ*)` |
| observed | Spread of estimates across trials, or MSE vs the reference | From the repeated trials |

- v_pred < v_true|pilot means the pilot made MXMCPy **overconfident**, which is pilot noise.
- v_true|pilot > v_opt means the pilot made the design **suboptimal**, which is also pilot noise.
- observed ≠ v_true|pilot, beyond its sampling uncertainty, is a **genuine mismatch**. Possible causes are a bug, wrong sample sharing, non-independent seeds, cost drift, or a broken assumption.

**Code gotcha [Inference from source].** `Estimator(allocation, Σ*).approximate_variance` recomputes the weights from Σ*. For ACV/MFMC that gives the oracle-weight variance for the pilot allocation, *not* v_true|pilot with pilot weights. Getting v_true|pilot needs a few lines: take `alpha` from the pilot-built estimator, and the `get_k0_matrix()`/`get_k_matrix()` terms evaluated with Σ*, mirroring `ACVEstimator._get_approximate_variance` ([`acv_estimator.py`](https://github.com/nasa/MXMCPy/blob/main/mxmc/estimators/acv_estimator.py)). For MXMCPy's `mlmc` the weights are fixed at −1, so `Estimator(alloc, Σ*)` is already correct.

**Where Σ\* comes from [Inference].** It needs all three levels evaluated on the *same* inputs, at a sample size where its own uncertainty is small. The brute-force reference already pays for many high-fidelity runs. Evaluating the two cheap levels on those same inputs gives a large joint sample at small extra cost. Its own bootstrap CI should be reported. Running those cheap levels is a GPU spend that needs sign-off.

**Should each trial get its own pilot? [Inference]** Both designs answer different questions, so run both if budget allows:

- **(A) Formula check (shared allocation).** Fix one allocation (from Σ* or from one pilot), run R independent trials with fresh samples, and compare observed variance with v(ξ; Σ*) for that fixed design. This isolates "is MXMCPy's variance formula right for splat levels and our QoI?" It is the direct analogue of [TM Sec. 4.2](https://ntrs.nasa.gov/citations/20200003111). The formula needs only second moments and correct sample sharing, so a mismatch here is a red flag, not pilot noise.
- **(B) End-to-end check (fresh pilot per trial).** Each trial runs its own pilot, optimizes, and estimates, with pilot cost counted in the budget. Observed MSE against the brute-force reference reflects the real workflow, roughly the average of v_true|pilot plus bias². Comparing it with the average of v_pred measures overconfidence, like Stanley et al.'s Fig. 3.3. Without fresh pilots, R trials only tell you about one pilot's luck.

**Sampling uncertainty on "observed" [Computed, standard chi-square approximation for roughly normal estimates].** The relative standard error of a sample variance from R trials is about √(2/(R−1)). That is about 32% at R = 20, 20% at R = 50 and 14% at R = 100. A predicted-vs-observed gap smaller than this is not evidence of anything. Report a CI on the observed variance, not a point. Also subtract or bound the brute-force reference's own variance when computing MSE against it.

**Winner's curse [Inference].** If each trial picks "the estimator with the smallest predicted variance" from the same pilot, the choice favors whichever estimator's prediction is most flattering. Per Stanley et al., that is the flexible estimators, which are also the most overconfident. For #29, fix the estimator per experiment arm rather than auto-selecting per trial. The same applies to `auto_model_selection=True` in `Optimizer.optimize`.

---

## 5. Specific issues for indicator / smoothed-indicator QoIs

- **Bernoulli-like outputs have heavy kurtosis, so variance estimates are poor.** Giles: the sample variance's standard deviation is roughly √((κ−1)/N)·E[X²], so O(κ) samples are needed. For outputs that are always 0 or 1, the level difference is mostly 0, κ ≈ 1/(p+q) is large, and a small sample may show *all zeros* and an estimated variance of zero ([Giles 2015, Sec. 3.3](https://people.maths.ox.ac.uk/gilesm/files/acta15.pdf)). This is exactly the structure of differences between our fidelity levels' failure indicators, which feed MLMC's `Var(Y_l − Y_{l−1})`.
- **[Computed]** For a hard 0/1 indicator with failure rate p, the centered kurtosis is (1−3p+3p²)/(p(1−p)). That gives about 48 at p = 2%, 18 at p = 5% and 8 at p = 10%. Using Giles's approximation, the relative error of the estimated high-fidelity variance alone is:

  | N pilot | p = 2% | p = 5% | p = 10% |
  |---|---|---|---|
  | 20 | ~150% (expect 0.4 failures; P(no failures) ≈ 0.67) | ~92% (1.0; 0.36) | ~60% (2.0; 0.12) |
  | 50 | ~97% (1.0; 0.36) | ~58% (2.5; 0.08) | ~38% (5.0; 0.005) |
  | 100 | ~69% (2.0; 0.13) | ~41% (5.0; 0.006) | ~27% (10; ~0) |
  | 200 | ~48% (4.0; 0.02) | ~29% (10; ~0) | ~19% (20; ~0) |

  The sigmoid smoothing in the spec spreads the 0/1 jump over a band around r. That lowers kurtosis below these hard-indicator values, by an amount that depends on steepness. Treat the table as a pessimistic bound, not a prediction. We don't know Fidelis's actual numbers until the pilot runs.
- **Correlations of indicators are fragile.** The cross-level correlation of a rare indicator is driven by the handful of samples near or above r. One extra or missing failure can swing ρ̂ a lot **[Inference]**. Stanley et al.'s correction and its theory assume Gaussian outputs. They saw an "off trend" result for their least-correlated, non-Gaussian QoI and gains below theory on non-Gaussian data ([Stanley et al. 2026, Sec. 5.2](https://arxiv.org/abs/2605.23016)). **[Inference]** Normal-theory confidence intervals for ρ (Fisher z) are not trustworthy here. Use a bootstrap, and note that the bootstrap also fails when the pilot holds only one or two failures.
- **Low correlation shrinks the payoff and raises the stakes.** MFMC's variance reduction collapses as ρ drops, and with ρ about 0.4–0.6 some model combinations do worse than plain MC ([PWG 2016, Sec. 3.4, Fig. 1](https://doi.org/10.1137/15M1046472)). **[Inference]** If the true correlations sit near that boundary, pilot noise decides whether MXMCPy even *predicts* a multifidelity win. That is the spec's "smoothing must preserve enough correlation" risk, and it can only be judged with a CI on ρ, not a point estimate.
- **Higher-order statistics need more pilots.** The spec also estimates the variance of navigation error. Variance estimators need covariances of squared outputs (fourth-order information) ([Dixon et al. 2023/24, Sec. 3, Prop. 3.4](https://arxiv.org/abs/2310.00125)). Their experiments needed several times more pilots for variance-type statistics than for means (Sec. 4.2.1).
- **The threshold comes from the pilot.** **[Inference]** Picking r so that failure is 2–10% means estimating an upper quantile from the pilot. With N = 50 and a 5% target there are only about 2.5 samples above r. Both r and every covariance depending on it carry this noise.
- **Covariance-free alternatives exist for rare events.** Chakroborty, Dhulipala & Shields propose a bi-fidelity control-variate plus importance-sampling estimator whose tuning avoids estimating the covariance, aimed at rare failure probabilities ([arXiv:2405.03834, abstract](https://arxiv.org/abs/2405.03834)). Only the abstract was checked; noted as a possible fallback, not a recommendation.

---

## What this means for Fidelis

### For #13 (pilot study) — all **[Inference]** unless cited

1. **Run the pilot in stages, not at a fixed size.** Start with about 50–100 joint samples: the same scenario seeds at all three fidelity levels, with GPU type pinned per `docs/carc.md`. Extend in batches until the decisions are stable. This mirrors the adaptive schemes of Dixon et al. (Appendix C "Adapt"; [2026 Alg. 1](https://arxiv.org/abs/2505.09828)), [Coons et al.](https://arxiv.org/abs/2508.18490) and Dakota's online pilot ([Warner et al. 2026, Sec. 2.4](https://arxiv.org/abs/2509.16007)), but with a simple bootstrap rule instead of their machinery. The stopping rule: across bootstrap resamples of the pilot, (a) the sign of "MFMC beats MC" doesn't flip, (b) the MFMC level ordering doesn't change, and (c) the predicted variance-reduction ratio's CI is narrow enough to matter, with the threshold written down before the run. Each batch is real GPU time and needs sign-off under the compute guardrail.
2. **Size by failures, not by samples.** Aim for roughly 10–20 high-fidelity outcomes in the sigmoid's transition band or beyond. At p ≈ 5% that means a few hundred pilots; at 10%, about 100–200. This is our heuristic, motivated by Giles's O(κ) rule and the table in Q5. No source states it.
3. **Store raw navigation-error magnitudes for every level**, not just indicators. Then the covariance can be recomputed for any r and sigmoid steepness, as the spec's "results across a range of r" needs, without new renders.
4. **Report covariance uncertainty.**
   - Give bootstrap 95% CIs on each cross-level correlation, on σ_HF, and on the level differences (the spec's "do the levels differ" check).
   - Give the bootstrap distribution of MXMCPy's predicted variance and allocation for MFMC, MLMC and ACV, not just the point values.
   - Report the number of pilot failures at the chosen r.
   - Report the pilot's cost, so the equal-compute comparison can either charge it or state that it's excluded. Croci et al. exclude it; Warner et al. and Gorodetsky et al. charge it.
5. **Lead with MFMC; treat ACV variants as pilot-sensitive.** This matches the spec's choice of MFMC as headline. Per [Stanley et al. 2026](https://arxiv.org/abs/2605.23016), flexible ACV variants are the most likely to look great on paper with a small pilot and disappoint in practice. Use weighted MLMC (`wrdiff`) alongside plain `mlmc`, since unweighted differences can lose to MC at modest correlations ([Gorodetsky et al. 2020, Fig. 4.6](https://arxiv.org/abs/1811.04988)).
6. **Keep the pilot out of the final estimates** used for validation (see Q3).

### For #29 / #21 (predicted vs observed)

1. **Build the oracle Σ\*** from a large joint sample. The cheapest route is to run the two cheap levels on the brute-force reference inputs. Report its bootstrap CI.
2. **Run Check A (formula check):** one fixed allocation, R trials with fresh samples, observed variance with a CI against v(ξ; Σ*). A failure here is a real finding about MXMCPy or our pipeline, not pilot noise.
3. **Run Check B (end-to-end):** a fresh independent pilot per trial, pilot cost charged, MSE against the brute-force reference. Report the mean of v_pred, the mean of v_true|pilot (computed with pilot weights at Σ*; see the code gotcha in Q4), v_opt, and the observed MSE side by side. The gaps between them attribute the error to overconfidence, suboptimality, or a genuine mismatch.
4. **Put CIs on everything.** With R trials the observed variance is only known to about √(2/(R−1)). Set R from the GPU-hour budget, and state up front what size of gap R can resolve.
5. **Fix the estimator per arm.** Don't auto-select per trial by smallest predicted variance (winner's curse).
6. **If budget allows only one check, do B with fresh pilots.** It is the honest version of "does the optimizer's promise hold". Use A as a cheap diagnostic only if B shows a mismatch.

---

## Open questions / not verified

- **No source treats pilot sizing for rare-event indicator QoIs with MFMC/ACV.** The kurtosis argument (Giles) is the closest. The failure-count heuristic in "What this means" is ours and should be checked in the pilot itself, e.g. by bootstrapping the pilot at sub-sizes.
- **Stanley et al.'s "5 to 20" practical range** has no citation in their text.
- **Coons et al.'s and Dixon et al.'s pilot counts** come from synthetic or PDE problems with Gaussian-ish outputs. Their transfer to splat renders is unknown.
- **Konrad et al.'s "insignificant" reuse bias** was at correlations ≥ 0.996, and Dixon et al.'s at high correlations. Not verified for low-correlation or indicator QoIs.
- **Ng & Willcox 2014**, cited by Croci et al. for "100 pilots is more than needed", was not read.
- **Schaden & Ullmann 2020 (MLBLUE)** was not read directly. MLBLUE's pilot sensitivity here comes from Stanley et al.
- **Pham & Gorodetsky (ensemble ACV)** and **Xu et al. 2022 (AETC)**: only abstracts checked. Their lower bounds on samples needed for guaranteed variance reduction were not extracted.
- **Classical control-variate "loss factor"** for estimated coefficients (Lavenberg & Welch 1981; Nelson 1990) is relevant background but was not verified from the originals.
- **The ACV paper** was read in its arXiv version (1811.04988). Section and figure numbers may differ from the JCP version.
- **The MXMCPy code** was checked at commit `367c7de` (Jan 2024; `setup.py` version 1.0).
- **Dai, Tchelepi & Tartakovsky (2026)** reuse result: abstract only.
- **The kurtosis-based error table** uses Giles's large-N approximation, which is poor when the pilot holds very few failures. Read it as order-of-magnitude only.

---

## References

1. Peherstorfer, B., Willcox, K., Gunzburger, M. (2016). Optimal model management for multifidelity Monte Carlo estimation. *SIAM J. Sci. Comput.* 38(5), A3163–A3194. https://doi.org/10.1137/15M1046472 (authors' copy: https://kiwi.oden.utexas.edu/papers/multi-fidelity-monte-carlo-peherstorfer-willcox-gunzburger.pdf)
2. Peherstorfer, B., Willcox, K., Gunzburger, M. (2018). Survey of multifidelity methods in uncertainty propagation, inference, and optimization. *SIAM Review* 60(3), 550–591. https://doi.org/10.1137/16M1082469 (checked: its MFMC algorithm, Sec. 3.2.2, takes variances and correlations as given inputs and gives no pilot-size guidance)
3. Gorodetsky, A. A., Geraci, G., Eldred, M. S., Jakeman, J. D. (2020). A generalized approximate control variate framework for multifidelity uncertainty quantification. *J. Comput. Phys.* 408, 109257. https://doi.org/10.1016/j.jcp.2020.109257 ; preprint used: https://arxiv.org/abs/1811.04988
4. Bomarito, G. F., Warner, J. E., Leser, P. E., Leser, W. P., Morrill, L. (2020). Multi Model Monte Carlo with Python (MXMCPy). NASA/TM-2020-220585. https://ntrs.nasa.gov/citations/20200003111
5. MXMCPy source code, NASA. https://github.com/nasa/MXMCPy (commit 367c7de)
6. Giles, M. B. (2015). Multilevel Monte Carlo methods. *Acta Numerica* 24, 259–328. https://doi.org/10.1017/S096249291500001X ; author PDF: https://people.maths.ox.ac.uk/gilesm/files/acta15.pdf
7. Stanley, M., Coons, T., Bomarito, G., Leser, P., Pribe, J., Warner, J. (2026). Sample correlation adjustments for robust multi-fidelity Monte Carlo under limited pilot sampling. arXiv:2605.23016. https://arxiv.org/abs/2605.23016
8. Coons, T. E., Jivani, A., Huan, X. (2025). Bayesian covariance uncertainty for adaptive pilot-sampling termination in multi-fidelity uncertainty quantification. arXiv:2508.18490. https://arxiv.org/abs/2508.18490
9. Dixon, T. O., Warner, J. E., Bomarito, G. F., Gorodetsky, A. A. (2024). Covariance expressions for multifidelity sampling with multioutput, multistatistic estimators: Application to approximate control variates. *SIAM/ASA J. Uncertain. Quantif.* 12, 1005–1049. Preprint: https://arxiv.org/abs/2310.00125
10. Dixon, T., Gorodetsky, A., Jakeman, J., Narayan, A., Xu, Y. (2026). Optimally balancing exploration and exploitation to automate multi-fidelity statistical estimation. arXiv:2505.09828. https://arxiv.org/abs/2505.09828
11. Xu, Y., Keshavarzzadeh, V., Kirby, R. M., Narayan, A. (2022). A bandit-learning approach to multifidelity approximation. *SIAM J. Sci. Comput.* 44(1), A150–A175. https://doi.org/10.1137/21M1408312 ; https://arxiv.org/abs/2103.15342
12. Konrad, J., Farcas, I.-G., Peherstorfer, B., Di Siena, A., Jenko, F., Neckel, T., Bungartz, H.-J. (2022). Data-driven low-fidelity models for multi-fidelity Monte Carlo sampling in plasma micro-turbulence analysis. *J. Comput. Phys.* 451, 110898. https://doi.org/10.1016/j.jcp.2021.110898 ; https://arxiv.org/abs/2103.07539
13. Croci, M., Willcox, K. E., Wright, S. J. (2023). Multi-output multilevel best linear unbiased estimators via semidefinite programming. *Comput. Methods Appl. Mech. Engrg.* 413, 116130. https://arxiv.org/abs/2301.07831
14. Warner, J. E., Bomarito, G. F., Geraci, G., Eldred, M. S. (2026). Automated model tuning for multifidelity uncertainty propagation in trajectory simulation. arXiv:2509.16007. https://arxiv.org/abs/2509.16007
15. Pham, T., Gorodetsky, A. A. (2022). Ensemble approximate control variate estimators: Applications to multifidelity importance sampling. *SIAM/ASA J. Uncertain. Quantif.* 10(3), 1250–1292. https://doi.org/10.1137/21M1390426 ; https://arxiv.org/abs/2101.02786 (abstract only)
16. Schaden, D., Ullmann, E. (2020). On multilevel best linear unbiased estimators. *SIAM/ASA J. Uncertain. Quantif.* 8(2), 601–635. https://doi.org/10.1137/19M1263534 (not read directly)
17. Chakroborty, P., Dhulipala, S. L. N., Shields, M. D. (2024). Covariance-free bi-fidelity control variates importance sampling for rare event reliability analysis. arXiv:2405.03834. https://arxiv.org/abs/2405.03834 (abstract only)
18. Dai, Z., Tchelepi, H. A., Tartakovsky, D. M. (2026). Multifidelity Monte Carlo estimator for high-dimensional quantities of interest. arXiv:2610.25016. https://arxiv.org/abs/2610.25016 (abstract only)
