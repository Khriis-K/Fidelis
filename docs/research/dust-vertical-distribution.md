# Dust vertical distribution near the surface at Jezero

**Question:** how is Mars dust spread vertically in the lowest ~2.5 km above the ground, at Jezero (18.4°N, 77.5°E) during Mars 2020 EDL (18 Feb 2021, Ls ≈ 5°, early northern spring)? Issue #37 models the haze between LCAM and the ground as τ_path = τ × (1 − exp(−h/H)), with camera height h = 2.2–2.45 km, and *assumed* H = 11.1 km (dust well mixed with the gas). This note sources or replaces that H. Investigated 2026-11-14.

**Status:** research only. No code or config changed. Every number is quoted from a source, with its page or section, or marked **[Computed]**, **[Inference]** (our reasoning) or **[Assumption]**. Sibling of `lvs-error-sources.md`.

## Summary (plain English)

Nobody has measured the dust profile in the lowest few km over Jezero during EDL. Orbiters looking at the limb (MCS, CRISM) can't see that low, and the rover's sky cameras measure the whole column, not its shape. So we have to bracket it.

- **The 11.1 km assumption is the "dust well mixed with the air, all the way up" case.** If dust stays a fixed fraction of the air, the share of the column's dust below a height is just the share of the air's mass below it: 1 − exp(−h/H) with H the *gas* scale height. That is exactly the formula in #37, and it's the textbook default (Conrath 1975, ν = 0.007). It gives **18% of column τ below 2.2 km and 20% below 2.45 km** [Computed].
- **It's the low-haze end of what the sources support, not the middle.** The modern limb observations say that in low-dust seasons the dust is *capped lower*, within 10–20 km of the surface (Smith et al. 2013). The same column τ squeezed into a shallower layer puts more of it below the camera: **26–35% below 2.2–2.45 km** for a 10–20 km dust top [Computed]. In #37's formula that's an effective H of about **7.4 km (20 km top) to 5.7 km (10 km top)**, not 11.1 km.
- **The other direction exists too.** Detached dust layers aloft are common in northern spring and summer (Smith et al. 2013 §1, §4.2). Dust sitting at 15–25 km puts *less* of the column below the camera than well mixed. We found no profile we could turn into a number for this.
- **Recommendation:** keep #37's formula, but treat H as uncertain. Use **H = 11 km as the optimistic bound** (it now has a source: the gas scale height) and **H ≈ 7.4 km as the central haze case, with 5.7 km as the pessimistic bound**. Confidence: **low**. The direction (lower dust top → more haze below the camera) is solid physics; the actual top height at Jezero on landing day is not sourced.
- **Bigger open problem than H:** the 0.47 τ in `lvs-error-sources.md` has no stated wavelength in our notes. Column dust opacities from TES/MCS are infrared; the Mars 2020 atmosphere team scaled TES IR opacity by **2.25** to get visible (Mischna et al. 2022, Fig. 2 caption). If 0.47 is IR, the visible τ LCAM sees is about twice as big, which matters more than any choice of H here.

## 1. The formula, and why H is "where the dust is", not just "how fast the air thins"

Optical depth below height h is the extinction integrated from the ground to h. With dust mass mixing ratio q(z) and air density ∝ exp(−z/H):

  f(h) = τ(0→h) / τ(column) = ∫₀ʰ q(z) e^(−z/H) dz ⁄ ∫₀^∞ q(z) e^(−z/H) dz   **[Computed, definition]**

- **Well mixed** (q constant): f(h) = 1 − exp(−h/H). This is #37's formula with H = gas scale height. Smith et al. 2013 state the same thing for their retrieval: Δp/p_surf "is equivalent to the fraction of the column mass contained within a particular layer", and for a well-mixed aerosol the mixing ratio is constant for all levels (Smith et al. 2013 §3.2, ¶25, p.5).
- **Conrath profile.** Conrath's steady-state balance of settling and mixing gives q(z) = q₀ exp{ν[1 − exp(z/H)]}, eq. (22), with ν = t_D/t_s(0), the ratio of diffusion time to settling time at the surface. He defines the dust "top" z_t = −H ln ν and, from limb pictures showing dust up to at least 50 km in the 1971 storm, takes "Z_t ≈ 50km corresponding to ν = .007" (Conrath 1975 preprint p.10). He assumed H = 10 km throughout (p.9). As ν → 0, "the specific concentration of the dust becomes independent of height" (p.10).

**Why the dust top matters even near the surface [Computed].** For any reasonable ν, q barely changes in the lowest 2.45 km: q(2.45 km)/q₀ = 0.998 (z_t = 50 km), 0.963 (z_t = 20 km), 0.903 (z_t = 10 km), with H = 10 km. So near the camera the dust *is* well mixed in every Conrath case. What changes is the denominator: a lower top removes dust from aloft, so the same τ means more dust per km down low. That's why the dust-top height, not the near-surface shape, sets the haze below the camera.

## 2. Gas scale height at Jezero

- NASA's Mars fact sheet gives "Scale height: 11.0 km", average temperature ~214 K, mean molecular weight 43.49, surface gravity 3.73 m/s² (Williams, NSSDCA Mars Fact Sheet). That is a global mean, and likely where 11.1 km came from (the fact sheet says 11.0).
- H = RT/(μg) **[Computed]**: R = 8.314 J/(mol K), μ = 0.04349 kg/mol, g = 3.73 m/s² gives 10.0 km at 195 K, 11.0 km at 214 K, 12.0 km at 234 K.
- Jezero's afternoon air temperatures in the lowest 2.5 km during EDL: **not found** as numbers. Mischna et al. 2022 show them only in figures (Fig. 9 histogram; MCS profiles to "within 2 km of the surface", §3.6.2). The 195–234 K range above is an **[Assumption]** bracket, not a sourced range.
- Sensitivity is small next to the dust-top question: well-mixed f(2.2 km) is 0.198 at H = 10 km and 0.168 at 12 km [Computed].

## 3. What the sources say about the vertical distribution

**Orbital limb sounding (cannot see the lowest ~10 km).**
- CRISM limb: dust and ice profiles "often show a significant vertical structure for nearly all seasons and latitudes that is not consistent with the well-mixed or Conrath-ν assumptions" (Smith et al. 2013, abstract).
- Dust "is confined within 10–20 km of the surface when the column-integrated optical depth of dust is low (aphelion season and winter high latitudes)" and reaches about 40 km when dust is significant. "The equinoctal seasons tend to show the most structure" and are transitional (Smith et al. 2013 §4.2, ¶37, p.8). Ls 5° is just past equinox, at the start of the aphelion season, so it sits in the transitional case **[Inference]**.
- **Coverage limit:** for a clear case with column optical depth 0.4 there is sensitivity "down to approximately 10 km above the surface"; dustier cases stop at 25 km or higher (Smith et al. 2013 ¶26, p.5). In their worked example the column below the lowest level was filled by "extrapolating the lowest retrieved dust mixing ratio to the surface" (¶27, p.6). So limb profiles say nothing direct about the lowest 2.5 km.
- MCS found departures from well mixed "during the northern spring and summer seasons in the tropics" (Smith et al. 2013 ¶5, p.2, citing Heavens et al. 2011 and Kleinböhl et al.). The Heavens et al. 2011 detail (a mixing-ratio maximum at 15–25 km in the tropics) came to us only through search-engine summaries; we didn't read the paper or its abstract, so it's listed under "Not found".

**Models used for Mars 2020 EDL.**
- "Atmospheric dust loading is typically fixed to seasonal TES-based values with a seasonally and latitudinally dependent Conrath-nu vertical profile" (Mischna et al. 2022 §3.1.2). The OSU mesoscale model built its dust depth from MCS density-scaled opacity, converted to "a number of scale heights" (§3.1.1). The values at Jezero are only in Fig. 2.
- MCD v2.0 (1999) dust scenarios use a modified Conrath form with z_max, "the altitude of the top of the dust layer (where the dust mixing ratio is one thousandth of its value at p₀)", and z_max(Ls, φ) = (60 + 18 sin(Ls − 158°) − 22 sin²φ) km (Lewis, MCD v2.0 dust scenarios page, eqs. 2 and 5). At Ls 5°, φ = 18.4°: z_max ≈ 49.6 km **[Computed]**. This is a 1999 pre-MCS parameterisation, so it inherits the well-mixed bias that Smith et al. 2013 criticise **[Inference]**.

**Ground-based, lowest few km.**
- Phoenix lidar (68°N, sol 45, Ls 97°): "The dust was distributed evenly up to a height of 4 km because of the daytime vertical mixing" in the planetary boundary layer (Whiteway et al. 2009 p.68). The EGU abstract adds that the dust "decreased at greater heights" (Whiteway et al. 2010). Different latitude and season from Jezero EDL, but it's the only direct measurement of the lowest km: daytime convection mixes dust evenly through the boundary layer.
- Jezero's convective boundary layer during EDL was expected to be "very shallow over Jezero crater—between ∼2 and 4 km" (Mischna et al. 2022 §3.1.1), with a weak inversion near 5 km (§3.1.2). So **the camera at 2.2–2.45 km is near the top of the mixed layer** [Inference].
- At Jezero during a regional storm (sols 313–318, not EDL conditions), the boundary layer was "dust-depleted by a factor of ~2 compared to the air above", reaching 3–4 by sol 318 (Lemmon et al., MAMO-7 abstract). This shows the near-surface layer can be *cleaner* than well mixed, at least while a storm clears.

## 4. Fraction of column τ below the camera

All **[Computed]** from the formulas in §1 (numerical integration for Conrath; `f = 1 − exp(−h/H)` for well mixed). H_eff is the H that makes #37's formula give the same f.

| Case | Source for the shape | f below 2.2 km | f below 2.45 km | H_eff for #37 |
|---|---|---|---|---|
| Well mixed, H = 12 km (234 K) | Conrath ν → 0; T bracket [Assumption] | 0.168 | 0.185 | 12 km |
| Well mixed, H = 11.1 km (#37 as written) | — | 0.180 | 0.198 | 11.1 km |
| Well mixed, H = 11.0 km | NSSDCA fact sheet | 0.181 | 0.200 | 11.0 km |
| Well mixed, H = 10 km | Conrath 1975 p.9 | 0.198 | 0.217 | 10 km |
| Conrath ν = 0.007 (z_t ≈ 50 km), H = 10 km | Conrath 1975 p.10; ≈ MCD v2.0 z_max | 0.203 | 0.224 | ≈ 9.7 km |
| Conrath z_t = 40 km (ν = 0.018), H = 10 km | Dusty-season top, Smith et al. 2013 §4.2 | 0.211 | 0.232 | ≈ 9.3 km |
| **Conrath z_t = 20 km (ν = 0.135), H = 10 km** | Low-dust top, Smith et al. 2013 §4.2 | **0.256** | **0.281** | **≈ 7.4 km** |
| Conrath z_t = 10 km (ν = 0.368), H = 10 km | Low-dust top, lower end, same | 0.318 | 0.348 | ≈ 5.7 km |
| All dust well mixed inside a 4 km boundary layer, none above | Phoenix-style PBL; Jezero CBL 2–4 km (Mischna 2022) | 0.595 | 0.655 | — |

The last row is a **ceiling only, not a supported case**: Smith et al. 2013 always see dust up to at least 10 km. It shows how fast the haze grows if most of the dust is local and confined to the boundary layer.

## 5. Verdict on H = 11.1 km

- **Supported as the well-mixed case.** The gas scale height is ~11.0 km (NSSDCA), so #37's formula with 11.1 km is a correct implementation of "dust well mixed to great height", which is Conrath's ν = 0.007 storm profile and the old MCD default.
- **Probably too high for the season.** The best observational source we read says low-dust seasons cap dust at 10–20 km, which corresponds to H_eff ≈ 5.7–7.4 km and 1.4–1.8× more haze below the camera [Computed: 0.256/0.180 = 1.42, 0.318/0.180 = 1.77 at 2.2 km].
- **Which way is pessimistic:** a *smaller* H puts more of the column below the camera, so more haze. H = 11.1 km errs **optimistic** (least haze) among the supported cases. The only supported cases with *less* haze are detached high-altitude layers, which we couldn't quantify.
- **Prediction from the sources (CLAUDE.md literature null-case test):** for τ = 0.47 (wavelength unknown), the haze optical depth between camera and ground is τ_path ≈ 0.085–0.093 at 11 km and ≈ 0.12–0.13 at H_eff 7.4 km [Computed: 0.47 × f]. If 0.47 is IR, multiply by ~2.25 for visible (Mischna et al. 2022 Fig. 2 caption): about 0.27–0.30 at H_eff 7.4 km [Computed]. Whether any of these push LCAM contrast below LVS's ~4 DN limit is **not covered by these sources**; it depends on the renderer's scene contrast and needs a measurement.

## Not found

- Any measured dust profile in the lowest ~10 km over Jezero, or any tropical site, near Ls 5°.
- The dust "depth in scale heights" or Conrath-ν that the Mars 2020 EDL models used at Jezero (only in Mischna et al. 2022 Fig. 2, which we couldn't read as numbers).
- Afternoon air temperatures (K) in the lowest 2.5 km at Jezero at landing time, for a site-specific gas scale height.
- Heavens et al. 2011 (JGR 116, E04003): abstract and full text not reachable (publisher 403, Caltech thesis server down). Its 15–25 km tropical dust maximum is unverified by us.
- The wavelength and instrument of the τ ≈ 0.47 cited from Mohan et al. 2021 p.16 (NTRS PDF link 404'd in this session). Mischna et al. 2022 §3.6.2 says MCS produced "daily dust and water-ice column opacity estimates at the landing site" (values only in Fig. 19); MCS opacities are IR, so this is the likely origin, but that's **[Inference]**.
- Conrath 1975 Icarus journal version: read only via the 1974 NASA preprint (OCR text). The ν = 0.007 / 50 km statement is from that preprint.
- Montabone et al. column dust climatology and Kleinböhl et al. MCS retrieval papers: not read this session.

## Sources

1. Conrath, B. J. (1975). Thermal structure of the Martian atmosphere during the dissipation of the dust storm of 1971. *Icarus* 24, 36–46. https://doi.org/10.1016/0019-1035(75)90156-6 . Read: NASA Goddard preprint of the same title (1974), NTRS https://ntrs.nasa.gov/citations/19740018179 (PDF https://ntrs.nasa.gov/api/citations/19740018179/downloads/19740018179.pdf). Page numbers are the preprint's printed pages.
2. Smith, M. D., Wolff, M. J., Clancy, R. T., Kleinböhl, A., Murchie, S. L. (2013). Vertical distribution of dust and water ice aerosols from CRISM limb-geometry observations. *J. Geophys. Res. Planets* 118. https://doi.org/10.1002/jgre.20047 . Read: NTRS copy https://ntrs.nasa.gov/api/citations/20140005705/downloads/20140005705.pdf
3. Mischna, M. A., Villar, G., Kass, D. M., Dutta, S., Rafkin, S., Tyler, D., Barnes, J., Cantor, B., Lewis, S. R., Hinson, D., Pla-García, J., Kleinböhl, A., Karlgaard, C. (2022). Pre- and Post-entry, Descent and Landing Assessment of the Martian Atmosphere for the Mars 2020 Rover. *Planet. Sci. J.* 3, 147. https://doi.org/10.3847/PSJ/ac7148 (open access; cited by section, figures not read as numbers).
4. Whiteway, J. A., et al. (2009). Mars water-ice clouds and precipitation. *Science* 325, 68–70. https://doi.org/10.1126/science.1172344 . Read: copy at https://www.eoas.ubc.ca/~mjelline/212_website/212_live/resources/Required-Reading/Whiteway_2009_lidar_water_ice_clouds.pdf
5. Whiteway, J., Dickinson, C., Komguem, L., Illnicki, M., Junkermann, W., Connolly, P., Hacker, J. (2010). Lidar atmospheric measurements on Mars and Earth. *Geophys. Res. Abstracts* 12, EGU2010-6885. https://meetingorganizer.copernicus.org/EGU2010/EGU2010-6885.pdf
6. Lemmon, M. T., et al. Inside an active Martian storm in Jezero Crater. Conference abstract (MAMO-7, tagged; venue inferred). https://web.lmd.jussieu.fr/~aspiga/antichambre/ABS/antoinem/oral_Lemmon_Mark.txt
7. Lewis, S. R. (1999). Dust distribution scenarios in the MCD. *Mars Climate Database v2.0* documentation. https://www-mars.lmd.jussieu.fr/mars/esadoc/aopp/ddd/node8.html
8. Williams, D. R. Mars Fact Sheet. NASA NSSDCA. https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html (accessed 2026-11-14).
9. Heavens, N. G., et al. (2011). The vertical distribution of dust in the Martian atmosphere during northern spring and summer: Observations by the Mars Climate Sounder and analysis of zonal average vertical dust profiles. *J. Geophys. Res.* 116, E04003. https://doi.org/10.1029/2010JE003691 (not read; see Not found).
