# CTX terrain for a wider tile

**Question:** the HiRISE products only give a ~10 km tile around the landing site, but tilted LVS-altitude views (10–20° at ~2.4 km) need ground out to 6–10 km from the centre. Can a lower-resolution outer ring come from the USGS Mars 2020 TRN CTX products, and do they line up with the HiRISE DTM and orthoimage we already use? Investigated 2026-11-01.

**Status:** research only. Nothing built; no data downloaded beyond metadata and the 9.7 MB CTX DTM, which was read remotely for the two measurements below.

## Answer

- **Two USGS products fit:** a 20 m/pixel CTX DTM mosaic (32-bit float, 9.7 MB) and a 6 m/pixel CTX orthoimage mosaic (8-bit, 36.3 MB). Both use **the same projection as the HiRISE mosaics** (equirectangular, sphere R = 3,396,190 m, `lat_ts=0`, `lon_0=0`), and they share the HiRISE mosaics' grid origin.
- **They cover a 27.7 km square centred on the landing site** (map metres; 26.3 km of ground east–west), and that square is fully valid. The south edge sets the limit, 13.85 km from the landing site. A 25 km centred square fits with room to spare; a 30 km square does not fit at all.
- **The HiRISE products were built to match CTX, not the other way round.** The HiRISE DTM was controlled to the CTX DTM and then affine-shifted onto the CTX ortho. USGS gives a median HiRISE–CTX vertical offset of ~0.4 m and a 2.1 m median horizontal displacement. Our own check over the whole overlap measured a median vertical offset of +0.54 m, with CTX higher, and a 4.1 m standard deviation.
- **The CTX ortho is the actual onboard LVS map**, per USGS. The catch: rendering the outer ring from it and also matching against it would make the camera image and the map come from the same photo there. See "Implications" below.
- **The CTX DTM has only modest stated accuracy.** CTX DTMs agree with each other to 3.8 m vertically, after a 183.3 m shift to HRSC; the HRSC reference itself is ~13 m; the TRN requirement was <20 m. No absolute vertical accuracy is stated for the CTX DTM itself.

Recommendation: the CTX pair is a credible outer ring for terrain and texture, with no reprojection and with a known, small offset to HiRISE. Decide separately whether the matching map stays HiRISE-derived or becomes the real CTX LVS map. That choice matters more for the result than the terrain does.

## 1. The two TRN CTX products

Both were published by USGS Astrogeology on 2020-07-24 (Fergason, Galuszka, Hare, Mayer, Redding, Smith). Values below are from the FGDC metadata XML and the PDS3 labels that sit next to each GeoTIFF, from the Astropedia pages, and from the GeoTIFF headers read with rasterio.

| | CTX DTM mosaic | CTX orthoimage mosaic |
|---|---|---|
| Title | Context Camera Digital Terrain Model Mosaic for Mars 2020 Terrain Relative Navigation (Astropedia: "Mars 2020 Terrain Relative Navigation CTX DTM Mosaic") | Context Camera Orthorectifiedimage Mosaic for Mars 2020 Terrain Relative Navigation |
| DOI | 10.5066/P906QQT8 | 10.5066/P9GV1ND3 |
| File | `JEZ_ctx_B_soc_008_DTM_MOLAtopography_DeltaGeoid_20m_Eqc_latTs0_lon0.tif` | `JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0.tif` |
| URL | https://planetarymaps.usgs.gov/mosaic/mars2020_trn/CTX/ + file (302 redirect to `asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/CTX/`) | same directory |
| Size (HTTP Content-Length) | 9,662,387 bytes (Astropedia: 9.3 MB) | 36,269,402 bytes (Astropedia: 35 MB) |
| Pixel | 20 m | 6 m |
| Grid | 1596 × 1512 | 5322 × 5040 |
| Type | float32 (PDS3: `IEEE_REAL`, 32 bits); nodata −3.4028226550889045e+38 | uint8; nodata 0; overviews 2–63× |
| CRS (GeoTIFF) | `+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +R=3396190 +units=m` | identical |
| Upper-left corner | x 4,573,663.25, y 1,109,693.14 | identical |
| Extent (metadata bounding box) | 18.2110–18.7212 N, 77.1605–77.6990 E | 18.2110–18.7212 N, 77.1605–77.6992 E |
| Values (metadata) | −2710.41 to −1522.23 m | 3 to 255 |

The CRS string is the same one the HiRISE mosaics in `config/jezero.yaml` use, so no reprojection is needed. The 5.4% east–west scale fix from `descent-altitude-and-tile-size.md` applies unchanged.

### How much has data (measured)

Measured 2026-11-01. The DTM was read at full resolution; the orthoimage was read from its 4× overview (24 m, nearest neighbour). The tile centre is `config/jezero.yaml`'s 18.4447 N, 77.4508 E.

- **DTM: 100% valid** (no nodata cells in the whole 1596 × 1512 grid).
- **Ortho: 100% non-zero at 24 m.** Holes smaller than about 24 m are not ruled out.
- Distance from the landing site to each mosaic edge, in map metres: W 17,209, E 14,711, S 13,851, N 16,389.
- Centred squares: 10, 15, 20 and 25 km lie inside the extent and are fully valid. 30 km runs past the south and east edges.
- **So the largest centred square is 27.7 km of map**, which is 27.7 km north–south and 26.3 km east–west on the ground (× cos 18.44° = 0.9486).

Measured min/max of the DTM file: −2710.70 / −1515.54 m. That is slightly different from the metadata's −2710.41 / −1522.23 m. Not explained; possibly the metadata predates a final edit.

### How it was made (metadata process steps; LPSC abstract)

- Three CTX stereo pairs, the "minimum number of high quality stereo pairs" covering the TRN region: J03_046060/J03_045994, F05_037607/F05_037752, J21_052811/J22_053233.
- A small gap in the **northwest corner** was filled with CTX image J03_046060 orthorectified to the HRSC DTM, "since we did not generate a CTX DTM" there. So that corner's ortho rests on HRSC terrain. Its exact extent is not stated (not verified). The DTM has no nodata there, so the DTM's NW corner presumably comes from another source; also not verified.
- CTX images were preprocessed with JPL's distortion, jitter and timing corrections (script v7.1), which let USGS skip bundle adjustment, because "TRN is sensitive to non-linear distortions".
- DTMs were made in SOCET SET and aligned to each other with ASP `pc_align`, translation only. Then they were aligned as a group to the DLR HRSC Level 5 DTM for absolute position, exported at 20 m, and mosaicked with ASP `dem_mosaic`. The orthoimages come from the nadir-most member of each pair, at 6 m.

## 2. Registration to the HiRISE products, and heights

**Horizontal.** The HiRISE ortho was registered *to* the CTX ortho. After bundle adjustment, a single 2-D affine (translation, rotation, scale) corrected "a scale distortion between the CTX LVS orthomosaic and the HiRISE orthomosaic" (Fergason et al. 2020). The same transform was applied to the HiRISE DTM. Reported result:

- LPSC abstract: median horizontal displacement of HiRISE features from CTX is 2.1 m; "95.4% of features are displaced <6 meters".
- HiRISE DTM metadata: "94.1% of matched features have displacements less than 6 meters", which they call conservative. The two documents disagree by 1.3 points; the metadata is the later one.
- CTX-to-CTX: "The individual CTX images have an average displacement of 9.6 meters" (IMCORR).

**Vertical.** The HiRISE DTM was tied to the CTX DTM: tie points got their heights from the CTX DTM, and the HiRISE mosaic was aligned to "the CTX LVS elevation map" with `pc_align` (HiRISE DTM metadata).

- USGS: "median vertical offset from the CTX DTM mosaic ... of ~0.4 meters", with CTX "floating above" HiRISE.
- **Measured here:** HiRISE DTM (`data/jezero/raw/dtm.tif`) area-averaged onto the CTX 20 m grid over the full overlap, with a 2-cell erosion at HiRISE nodata edges (754,461 cells). CTX minus HiRISE: **median +0.54 m, mean +0.53 m, std 4.09 m, 5th–95th percentile −5.82 to +7.12 m**, median absolute deviation 2.35 m. This agrees with USGS's ~0.4 m. The spread mixes real noise with the gap between 20 m CTX and averaged 1 m HiRISE detail, so expect seams of a few metres at a CTX/HiRISE boundary unless they are blended.

**Vertical datum: the sources disagree on paper, but the two DTMs match each other.**

- The CTX metadata says elevations "refer to the Mars 2020 IAU Sphere with a radius of 3396190m".
- But the filename says `MOLAtopography_DeltaGeoid`, and USGS's own naming convention (stated in the 2024 MSR TRN HiRISE README, from the same group) defines that as heights relative to the Mars geoid, made with ASP `dem_geoid`, "analogous to the MOLA Gridded Topography product". A height relative to the IAU sphere would instead be called `DeltaRadius`.
- The value range (−2.7 to −1.5 km) also looks like MOLA topography of Jezero. That is an inference, not verified.
- For Fidelis it doesn't matter which is right: the HiRISE DTM carries the same `MOLAtopography_DeltaGeoid` label and was aligned to this CTX DTM, and the measured offset is ~0.5 m. **The two can share one mesh without a datum correction.** Over 25 km, the geoid's slope relative to the sphere is a separate question. Not checked.

**Version note.** The MSR TRN README says the 2024 MSR HiRISE products were controlled to "Mars 2020 TRN CTX DTM Mosaic B (version 9)". The public file is `soc_008`. Whether a v9 differs materially is not verified.

## 3. Is the CTX ortho the actual LVS map?

USGS says yes, without qualification:

- Ortho metadata and Astropedia: "This product is the Lander Visions System (LVS) map", the "truth dataset" TRN uses during EDL.
- Fergason et al. 2020: the LVS map is "generated from three Context Camera (CTX) orthorectified images" and will be onboard.
- The HiRISE DTM metadata calls the CTX DTM the "LVS elevation map", so the onboard LVS map also had an elevation layer at this 20 m resolution (USGS wording; how LVS used it is not covered here).

Not verified: whether JPL cropped, resampled or otherwise processed the USGS mosaic before uplink. Cheng, Ansar & Johnson 2021 would say, but the publisher returned 403 again. Its Table 1 6 m/pixel requirement is already quoted in `config/jezero.yaml` from an earlier read.

**Implications for Fidelis.**

- **Today the matching map and the renders share one source.** The map is the HiRISE ortho averaged to 6 m, and the renders are textured from the same HiRISE ortho. So the "map vs descent image" difference comes from resolution, rendering and shading, not from different photos (a different date, sun angle or sensor). `docs/EXPLAINER.md` already lists that the map "isn't the real onboard LVS map".
- **The real LVS map is now available at no extra cost** (36 MB, same grid). Using it as the matching map would make the matching problem *more* realistic in the inner tile: HiRISE-textured renders matched against a CTX map from different images. That is the flight situation. The likely effect is a harder match and a higher failure rate. That is an expectation, not a measurement.
- **If the outer ring is textured from the CTX ortho, don't also match against the CTX ortho there.** That makes the render and the map the same photo in the ring, which flatters the matcher exactly where views are most oblique. Either keep a HiRISE-derived map and accept that it only covers the inner 10 km, or use the CTX map and report the ring's matches separately.
- **The texture changes at the seam.** CTX images were taken at different times and sun angles from the HiRISE images, and both orthos have their own lighting baked in. A HiRISE/CTX boundary will show a brightness and shading discontinuity in renders unless it is blended. Whether that confuses the matcher is not measured.
- **Resolution is roughly adequate for a ring.** At 2.2–4.2 km, LCAM pixels cover 4.3–8.2 m at image centre and about twice that at the edges, where an outer ring would appear. A 6 m texture is near that; a 20 m DTM is coarser than the 4 m HiRISE mesh, so relief and shadows in the ring will be smoothed.

## 4. Stated accuracy of the CTX DTM

From the metadata ("Vertical positional accuracy") and Fergason et al. 2020:

- "The vertical registration between individual CTX DTMs is 3.8 meters after adjusting the mosaic by 183.3 meters vertically to align with HRSC."
- That is "within the theoretical maximum vertical precision of the HRSC DTM (~13 meters)" and within the TRN requirement of "vertical offsets of less than 20 meters".
- Astropedia lists horizontal and vertical accuracy as "Accurate to Control Net" (MOLA).
- **Not stated:** an absolute vertical accuracy for the CTX DTM, its effective (as opposed to posted) resolution, or a per-pixel precision.
- Kirk et al. 2020 evaluated HRSC and CTX DTMs at Jezero against HiRISE. The abstract gives "height errors are on the order of half a pixel" and horizontal resolution of "10–20 pixels". The abstract does not separate CTX from HRSC, and the PDF's text could not be extracted, so no CTX-specific number is recorded here (not verified). If "pixel" means the ~6 m CTX image pixel, effective DTM resolution would be 60–120 m. That reading is not verified, but if it holds, the 20 m posting overstates the real detail.

## 5. Other public DTMs over a 20–30 km square

- **Mars 2020 Science Investigation CTX DEM Mosaic** (JPL / F. Calef; USGS Astrogeology, 2021-10-14; PDS4 `urn:nasa:pds:mars2020_ctx_basemap_2021`).
  - Products and sizes: a 20 m DEM (`M20_JezeroCrater_CTXDEM_20m.tif`, float32, 4456 × 5067, 90,345,300 bytes) and a 5 m ortho (`M20_JezeroCrater_CTXortho_mosaic_5m.tif`, 8-bit, 17766 × 20234, 359,599,280 bytes). Both are in `mars2020_trn/CTX/ScienceInvestigationMaps_JPL/`.
  - Extent: 17.58–19.29 N, 76.997–78.583 E, the whole crater. Astropedia says it "includes the original Mars 2020 Jezero CTX TRN DEM" and adds ASP DEMs "localize[d] to the Mars MOLA dataset vertically".
  - **Different projection:** its PDS4 label gives `standard_parallel_1` = 18.4663°, not 0. The radius is 3,396,190 in the label but 3,396,160 on the Astropedia page, which conflicts. Using it would mean reprojecting.
  - No accuracy statement, and no stated co-registration to the TRN products (not verified). Only worth it if more than 27.7 km is ever needed.
- **Mars Sample Return TRN HiRISE DTM and Orthoimage Mosaics** (USGS, 2025-02-28, doi:10.5066/P13CPYYU).
  - Resolution: 1 m DTM and 0.25 m ortho, in the same `eqc`, `lat_ts=0` projection. They are controlled to the Mars 2020 TRN CTX DTM (v9).
  - Extent: 18.135–18.588 N, 77.057–77.380 E. That is **west and southwest of the landing site; its east edge (77.380 E) is ~4.2 km of map (4.0 km of ground) west of the site** (computed from the bounds, not measured on the data).
  - It could extend HiRISE detail westward, but not into a centred square. Download: 5.47 GB zip of mosaics; 8 of 10 planned stereo pairs completed.
- **HRSC.** DLR's HRSC Level 5 DTM was the absolute reference for the CTX DTM, with ~13 m vertical precision per USGS. Its public availability and pixel size were not verified.

## Sources

All read 2026-11-01.

1. USGS Astrogeology, *Mars 2020 Terrain Relative Navigation CTX DTM Mosaic*, doi:10.5066/P906QQT8. Astropedia: https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_ctx_dtm_mosaic . FGDC metadata, PDS3 label and ISIS label: https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/CTX/JEZ_ctx_B_soc_008_DTM_MOLAtopography_DeltaGeoid_20m_Eqc_latTs0_lon0.xml (`.lbl`, `_pds3.lbl`).
2. USGS Astrogeology, *Mars 2020 Terrain Relative Navigation Context Camera Orthorectified Image Mosaic*, doi:10.5066/P9GV1ND3. Astropedia: https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_context_camera_orthorectified_image_mosaic . FGDC metadata and PDS3 label: same S3 directory, `JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0.xml` / `_pds3.lbl` / `_dd.vrt`.
3. S3 directory listing with byte sizes: https://asc-pds-services.s3.us-west-2.amazonaws.com/?list-type=2&prefix=mosaic/mars2020_trn/ ; index page https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/CTX/index.html .
4. USGS, HiRISE DTM mosaic FGDC metadata (doi:10.5066/P9REJ9JN): https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/HiRISE/JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40.xml .
5. Fergason, R. L., et al. (2020). Mars 2020 Terrain Relative Navigation Flight Product Generation: Digital Terrain Model and Orthorectified Image Mosaics. LPSC 51, abstract 2020. https://www.hou.usra.edu/meetings/lpsc2020/pdf/2020.pdf .
6. USGS, *Mars Sample Return HiRISE Mosaics* README v003 (2024), defining the `DeltaGeoid` naming: https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Mars/MSR_TRN_HiRISE_soc_003_USGS_release_aug2024/README.md ; Astropedia https://astrogeology.usgs.gov/search/map/mars-sample-return-terrain-relative-navigation-hirise-dtm-mosaic (doi:10.5066/P13CPYYU).
7. USGS Astrogeology, *Mars 2020 Science Investigation CTX DEM Mosaic*: https://astrogeology.usgs.gov/search/map/mars_2020_science_investigation_ctx_dem_mosaic ; PDS4 labels in `mars2020_trn/CTX/ScienceInvestigationMaps_JPL/`.
8. Kirk, R. L., et al. (2020). Evaluating stereo DTM quality at Jezero Crater, Mars with HRSC, CTX, and HiRISE images. ISPRS Archives XLIII-B3-2020, 1129. https://isprs-archives.copernicus.org/articles/XLIII-B3-2020/1129/2020/ (abstract only; PDF text not extractable).
9. Cheng, Y., Ansar, A., Johnson, A. (2021). Making an Onboard Reference Map From MRO/CTX Imagery for Mars 2020 Lander Vision System. *Earth and Space Science* 8, doi:10.1029/2020EA001560. Not read: publisher returned 403 (HTML and PDF). Unpaywall lists only the publisher and DOAJ copies.
10. Malin, M. C., et al. (2007). Context Camera Investigation on board the Mars Reconnaissance Orbiter. *JGR* 112, E05S04, doi:10.1029/2006JE002808. Cited by sources 1–2 as the CTX reference; not re-read here.

The scratch scripts for the coverage and HiRISE–CTX offset measurements were not committed. The numbers above are the ones they printed.
