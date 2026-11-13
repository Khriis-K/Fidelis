# How the CTX map and the HiRISE texture were actually lit

**Question:** which orbital images make up the USGS CTX orthomosaic (the LVS map Fidelis matches against) and the USGS HiRISE orthomosaic (the texture Fidelis renders from) over the 10 km landing tile, and where was the sun in each? JPL's field tests only missed when the map-to-image sun difference passed 100° (`lvs-error-sources.md`, section 3). Investigated 2026-11-12.

**Status:** research only. No code or config changed. Every number is quoted from a source, or marked **[Computed]** (our arithmetic on sourced numbers) or **[Inference]** (our reasoning). Small metadata files were read; no rasters were saved. Two **[Computed]** checks read small pixel windows of the public GeoTIFFs over HTTP.

## Summary (plain English)

- **The map over our tile is one CTX image: `J03_045994_1986_XN_18N282W`**, taken 2016-05-19 at 15.2 h local time. Sun: **42.4° high, bearing 267.3°** (just south of due west). The whole mosaic is three CTX images; the other two don't appear on top anywhere we sampled in the tile.
- **The render texture over the tile is a patchwork of at least four HiRISE images.** The centre, including the landing site, is `ESP_045994_1985`. That is the **same MRO pass** as the CTX map image (orbit 45994, 14 seconds apart), so there the map and the texture carry *identical* baked lighting (0.1° apart).
- **The east strip of the tile is the outlier:** `ESP_048842_1985`, sun 36.6° high at bearing 219.2° (south-west, a winter afternoon). It sits 37° from the map's sun.
- **Largest angle between any Fidelis sun and the map's sun over the tile: 39.0°** (the 310°/25° sun). Against any of the three CTX images in the mosaic: 47.1°. Against the baked HiRISE lighting: 76.0° (east strip). **Everything is well inside JPL's 100° no-miss envelope.**
- **MRO flies a 3:00 pm sun-synchronous orbit**, which is why every image here falls between 14.2 h and 15.8 h local time.
- **Push back.** Fidelis' five suns can't reach the regime where LVS actually failed. The widest map-to-sun gap we can produce is 39–47°. And the render texture has shadows baked in from its own sun, so a "relit" image at 310° still carries shading from 219–267°. The simulated sun angle overstates how different the render really looks from the map.

## 1. Which CTX images make up the map

The USGS metadata names three stereo pairs: "J03_046060_1986_XN_18N282W.cal.pgm/J03_045994_1986_XN_18N282W.cal.pgm; F05_037607_2008_XN_20N282W.cal_sub.pgm/F05_037752_2008_XN_20N282W.cal_sub.pgm; and J21_052811_1983_XN_18N282W.pgm/J22_053233_1984_XN_18N282W.pgm" (source 1, Completeness Report). The mosaic itself is "generated from three Context Camera (CTX) orthorectified images" (source 1, Purpose). They are the "nadir-most member of each stereopair", mosaicked with ASP's `dem_mosaic` (source 3, p.1). The same directory publishes all six orthoimages; the three without `_offnadir` in the filename are `J03_045994`, `F05_037752` and `J21_052811` (source 2). Their emission angles confirm these are the nadir-most (table below).

**Over the tile.** The metadata doesn't give the mosaic order, and all three images' footprints cover the tile (source 2 bounding boxes; ODE footprints, source 4). **[Computed]** We compared 600 m patches of the mosaic with each of the six orthoimages at 25 points on a 2 km grid across the tile. `J03_045994` matched best at all 25 (correlation 0.89–0.995; the next best was always lower). It isn't 1.0 because USGS applied "manual editing, tone matching, and seam removal" (source 3, Fig. 1), and `dem_mosaic` blends overlaps by default (source 10). So the tile is `J03_045994`, with the caveat that this is measured, not documented.

A small gap in the mosaic's north-west corner was filled with `J03_046060` (source 1). It's outside the tile.

## 2. Sun geometry of the CTX images

From the PDS CTX cumulative index record for each image, as returned by ODE (source 4). Column meanings are from the index label (source 5).

| CTX image | Role | Start time (UTC) | Local true solar time (h) | Emission (°) | Incidence (°) | North az. (°) | Sub-solar az. (°) | Sun elevation **[Computed]** | Sun bearing **[Computed]** |
|---|---|---|---|---|---|---|---|---|---|
| J03_045994_1986_XN_18N282W | **in mosaic; on top over the tile** | 2016-05-19T08:25:22 | 15.23 | 3.78 | 47.59 | 277.19 | 184.48 | 42.41° | 267.29° |
| F05_037752_2008_XN_20N282W | in mosaic | 2014-08-16T04:23:57 | 15.65 | 8.88 | 57.34 | 277.34 | 173.69 | 32.66° | 256.35° |
| J21_052811_1983_XN_18N282W | in mosaic | 2017-11-01T13:37:03 | 14.93 | 11.46 | 41.29 | 277.36 | 204.89 | 48.71° | 287.53° |
| J03_046060_1986_XN_18N282W | stereo partner; NW gap fill | 2016-05-24T11:51:26 | 15.38 | 19.87 | 49.92 | 276.54 | 183.25 | 40.08° | 266.71° |
| F05_037607_2008_XN_20N282W | stereo partner (DTM only) | 2014-08-04T21:15:26 | 15.74 | 12.35 | 57.60 | 276.70 | 176.66 | 32.40° | 259.96° |
| J22_053233_1984_XN_18N282W | stereo partner (DTM only) | 2017-12-04T10:51:37 | 15.14 | 12.31 | 44.05 | 276.73 | 204.33 | 45.95° | 287.60° |

**How the bearing was converted.** The PDS index measures both azimuths in the *image*, not on the ground: "The angle in degrees clockwise from the reference axis of the image (a line from the center to the right edge of the image) to the direction to the north pole" and, likewise, "to the subsolar point" (source 5). So neither is a compass bearing on its own. Subtracting them gives the compass bearing of the sun, clockwise from north:

> sun bearing = SUB_SOLAR_AZIMUTH − NORTH_AZIMUTH (mod 360). Example: 184.48 − 277.19 = −92.71 → **267.29°**.

Sun elevation = 90 − incidence ("incidence angle at center of image", source 5). All six records have USAGE_NOTE = "N" (not flipped), so no flip is needed first.

**Cross-check [Computed].** The index also gives the sub-solar point's latitude and (west) longitude. Spherical trigonometry from the landing site to that point gives 267.47° / 42.45° for `J03_045994`, 287.45° / 48.63° for `J21_052811`, and 257.92° / 33.07° for `F05_037752`. The two methods agree within 0.2°, except the F05 images (1.6° in bearing), whose centres are about 2.4° north of the landing site.

## 3. The HiRISE images under the tile

The HiRISE mosaic has six stereo pairs, one split in two (source 6, Completeness Report). Five pairs' footprints touch the tile (source 7 bounding boxes); Jezero_CR is west of it. The accuracy statement speaks of "orthoimages derived from the nadir-most members of each stereopair" (source 6).

**What `_first` means.** The metadata says "The order of images in the mosaic was chosen to limit the number and length of image seams within the landing ellipse and to ensure the highest quality DTM data was on top" (source 6). It doesn't list the order. `dem_mosaic --first` means "Keep the first encountered DEM value (in the input order)" (source 10). **[Inference]** `_first` is that option: no blending, first image in the list wins. (The sibling DTM is `_blend40`.)

**Which image is on top [Computed].** We compared 64 m patches of the mosaic with the five nadir orthoimages at 17 points across the tile:

| Part of the tile | Image on top | Correlation with mosaic (runner-up) |
|---|---|---|
| Centre, incl. landing site; north and south centre | ESP_045994_1985 (Jezero_C) | 0.95–0.98 (≤ 0.89) at all 10 points it covers |
| East strip (x = +3.5 and +4.8 km) | ESP_048842_1985 (Jezero_E) | 0.95–0.99 (0.91) |
| West edge (x = −3.5, −4.8 km, mid-height) | PSP_002387_1985 (Jezero_DL) | 0.88–0.92 (0.85) — the narrowest margin |
| South-west corner | ESP_037396_1985 (Jezero_W) | 0.99 |
| Not seen on top at any sampled point | ESP_037818_1990 (Jezero_N) | — |

This is a 17-point sample, not a seam map.

**Sun geometry**, from each image's HiRISE RDR label (source 8). These products are map-projected north-up, so NORTH_AZIMUTH is 270.0 for all, and the same conversion gives sun bearing = SUB_SOLAR_AZIMUTH − 270 (convention: source 9, same wording as CTX's).

| HiRISE image | Start time (UTC) | Local time (h) | Incidence (°) | Sub-solar az. (°) | Sun elevation **[Computed]** | Sun bearing **[Computed]** |
|---|---|---|---|---|---|---|
| ESP_045994_1985 | 2016-05-19T08:25:36 | 15.24 | 47.583 | 177.367 | 42.42° | 267.37° |
| ESP_048842_1985 | 2016-12-27T07:00:49 | 14.22 | 53.357 | 129.222 | 36.64° | 219.22° |
| PSP_002387_1985 | 2007-01-29T10:49:15 | 15.62 | 55.637 | 169.842 | 34.36° | 259.84° |
| ESP_037396_1985 | 2014-07-19T10:41:26 | 15.71 | 55.311 | 175.496 | 34.69° | 265.50° |
| ESP_037818_1990 | 2014-08-21T07:48:54 | 15.77 | 58.897 | 167.430 | 31.10° | 257.43° |

`ESP_045994_1985` and CTX `J03_045994` are the same pass: same orbit number, the CTX record's rationale is "Ride-along with HiRISE", and the start times are 14 s apart (sources 4, 8).

## 4. MRO's orbit

The PDS mission catalog: the Primary Science Orbit "is sun-synchronous with an ascending node orientation that provides a Local Mean Solar Time (LMST) of 3:00 p.m. at the equator", and "true solar time varies by nearly 45 minutes" over a Mars year (source 11). That matches the 14.2–15.8 h spread above.

## 5. Cross-check against Cheng et al. 2021

Cheng's Tables 4 and 6 list 13 images for JPL "Mosaic B". Its stereo pairs are "the top three pairs": J03_046060/J03_045994, F05_037607/F05_037752 and D14_032794/D15_033216 (source 12, Table 4, Figs. 9 and 16). J21_052811 and J22_053233 are in the same table, lower down, among the auxiliary images.

**So the USGS mosaic is not built from exactly JPL's three pairs.** Two pairs are the same. The third differs: USGS's metadata names J21/J22, where JPL's Mosaic B used D14/D15. Cheng says USGS used "the same image data" (source 12, section 3); the USGS metadata is the more specific source for the USGS product. The image over our tile, J03_045994, is in both.

Cheng's Table 6 against the PDS-derived values (elevation / bearing):

| Image | Cheng Table 6 | PDS index (section 2) | Note |
|---|---|---|---|
| J03_045994 | 42.4 / 237.6 | 42.41 / 267.29 | elevation agrees; **bearing differs by 29.7°** |
| J03_046060 | 40.1 / 240.3 | 40.08 / 266.71 | elevation agrees; **bearing differs by 26.4°** |
| F05_037752 | 32.4 / 255.57 | 32.66 / 256.35 | Cheng's two F05 elevations look swapped |
| F05_037607 | 32.7 / 261.8 | 32.40 / 259.96 | |
| J21_052811 | 47.71 / 294.89 | 48.71 / 287.53 | Cheng's bearing = sub-solar azimuth + 90 exactly |
| J22_053233 | 45.95 / 294.33 | 45.95 / 287.60 | same: 204.33 + 90 = 294.33 |

**[Inference]** For J21/J22, Cheng's bearing equals the raw sub-solar azimuth plus 90°, which assumes north is exactly "up" in the raw image. It's really at 277°, hence the 7° gap. For the two J03 images we can't reproduce Cheng's bearing from the index at all. Three independent readings (CTX index azimuths, the HiRISE label of the same pass, and the sub-solar point) all give 267°. We use the PDS values. Cheng's Table 6 also repeats J03_046060's values on the F02_036618 row, so some rows look shifted in the published table.

**Requirement check [Computed].** With the landing sun at elevation 39.6–42.2° and bearing 255.8–257.2° (source 12, section 2.5), the three mosaic images sit at most 9.5° off in elevation (F05_037752) and 31.7° off in bearing (J21_052811). All three pass ±15° / ±35° on the PDS values.

## 6. Angles between suns [Computed]

Angle = arccos(sin e₁ sin e₂ + cos e₁ cos e₂ cos(a₁ − a₂)), using the elevation and bearing from sections 2 and 3.

**Fidelis' five suns (bearing, elevation) against each image's sun:**

| Image | (257, 40) | (240, 25) | (240, 55) | (310, 25) | (310, 55) | Largest |
|---|---|---|---|---|---|---|
| CTX J03_045994 (**map over the tile**) | 8.1° | 28.4° | 21.7° | 39.0° | 30.3° | **39.0°** |
| CTX F05_037752 (map, elsewhere) | 7.4° | 16.2° | 25.1° | 47.1° | 43.3° | 47.1° |
| CTX J21_052811 (map, elsewhere) | 23.3° | 43.8° | 29.4° | 29.5° | 15.1° | 43.8° |
| HiRISE ESP_045994 (tile centre) | 8.2° | 28.4° | 21.8° | 39.0° | 30.2° | 39.0° |
| HiRISE ESP_048842 (east strip) | 29.6° | 21.2° | 23.2° | 76.0° | 61.1° | **76.0°** |
| HiRISE PSP_002387 (west edge) | 6.1° | 19.6° | 24.8° | 44.1° | 40.0° | 44.1° |
| HiRISE ESP_037396 (south-west) | 8.6° | 24.1° | 26.9° | 39.5° | 36.6° | 39.5° |
| HiRISE ESP_037818 (not seen on top) | 8.9° | 16.5° | 26.9° | 46.4° | 43.8° | 46.4° |

**HiRISE texture sun against CTX map sun:**

| HiRISE image | vs J03_045994 (**tile**) | vs F05_037752 | vs J21_052811 |
|---|---|---|---|
| ESP_045994 (centre) | **0.1°** | 13.1° | 15.4° |
| ESP_048842 (east strip) | **37.0°** | 30.6° | 49.9° |
| PSP_002387 (west edge) | 9.9° | 3.4° | 25.0° |
| ESP_037396 (south-west) | 7.8° | 7.9° | 21.5° |
| ESP_037818 | 13.8° | 1.8° | 28.7° |

Among the three CTX mosaic images themselves: 13.0° (J03–F05), 15.4° (J03–J21), 28.3° (F05–J21).

**For Fidelis [Inference].**
- Over most of the tile the map and the texture were lit within 10° of each other, and identically in the centre. The CTX-vs-HiRISE match is therefore an easier appearance test than "two different images" suggests. Only the east strip (37°) has a real baked-lighting difference.
- The two lighting differences stack in a way the sun index hides. A render at sun (310, 25) over the east strip is 39° from the map by its simulated sun but its texture shadows were cast from 219°, which is 37° from the map on the other side.
- To probe the >100° regime where LVS missed, Fidelis would need suns outside the LVS design box (for example a morning sun). That would be a stress test beyond the flight envelope, not a realism fix.

## Not found

- **The CTX mosaic's image order or seam lines.** Not in the FGDC metadata, the PDS3 label, or the LPSC abstract. The "J03_045994 on top" result is our 25-point correlation check.
- **The HiRISE mosaic's image order or a seam map.** The metadata gives the rule but not the list. Our 17-point check found four images on top; exact seam positions inside the tile are unknown.
- **Confirmation that `_first` is `dem_mosaic --first`.** The file's ISIS history holds only a GDAL conversion step.
- **Why Cheng's Table 6 bearings for the J03 pair (237.6°, 240.3°) differ from the PDS index by 26–30°.** The paper says the angles are "directly obtained from the CTX header files" and doesn't give a convention.
- **Why USGS used the J21/J22 pair where JPL's Mosaic B used D14/D15.** Neither source says.
- **Dust opacity for J21/J22.** Cheng's Table 6 lists tau as "ND" for both (MCS dust opacity 0.5).
- **HiRISE's web pages (uahirise.org)** weren't read. The HiRISE numbers come from the team's PDS labels, which carry the same fields.
- The planetarymaps.usgs.gov directory listings returned HTTP 403 to an automated fetch. The same files were read from the USGS S3 bucket that host fronts.

## Sources

1. USGS Astrogeology, *Context Camera Orthorectified Image Mosaic for Mars 2020 Terrain Relative Navigation*, doi:10.5066/P9GV1ND3. FGDC metadata read: https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/CTX/JEZ_ctx_B_soc_008_orthoMosaic_6m_Eqc_latTs0_lon0.xml
2. USGS, per-image CTX orthoimage metadata, same directory: `rectJ03_045994_1986_XN_18N282W_v7pt1_6m_Eqc_latTs0_lon0.xml`, `rectF05_037752_2008_XN_20N282W_v7pt1_6m_Eqc_latTs0_lon0.xml`, `rectJ21_052811_1983_XN_18N282W_v7pt1_6m_Eqc_latTs0_lon0.xml`, and the three `..._offnadir.xml` files. Directory listing: https://asc-pds-services.s3.us-west-2.amazonaws.com/?list-type=2&prefix=mosaic/mars2020_trn/
3. Fergason, R. L., Hare, T. M., Mayer, D. P., Galuszka, D. M., Redding, B. L., Smith, E. D., Shinaman, J. R., Cheng, Y., Otero, R. E. (2020). Mars 2020 Terrain Relative Navigation Flight Product Generation: Digital Terrain Model and Orthorectified Image Mosaics. LPSC 51, abstract 2020. https://www.hou.usra.edu/meetings/lpsc2020/pdf/2020.pdf
4. Mars Orbital Data Explorer (ODE) REST API, PDS Geosciences Node. One query per CTX product; each result embeds the PDS cumulative index record. https://oderest.rsl.wustl.edu/live2/?query=product&results=m&output=JSON&target=mars&ihid=MRO&iid=CTX&pt=EDR&pdsid=J03_045994_1986_XN_18N282W (and the same with the other five product IDs). HiRISE: `iid=HIRISE&pt=RDRV11&pdsid=ESP_045994_1985_RED`.
5. PDS, MRO CTX EDR cumulative index label (column definitions): https://planetarydata.jpl.nasa.gov/img/data/mro/ctx/mrox_2578/index/cumindex.lbl
6. USGS Astrogeology, *HiRISE Orthorectified Image Mosaic for Mars 2020 Terrain Relative Navigation*, doi:10.5066/P9QJDP48. FGDC metadata read: https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/mars2020_trn/HiRISE/JEZ_hirise_soc_006_orthoMosaic_25cm_Eqc_latTs0_lon0_first.xml ; history file `..._first.History.IsisCube` in the same directory.
7. USGS, per-image HiRISE orthoimage metadata, same directory, e.g. `ESP_045994_1985_Edited_affine_25cm_Eqc_latTs0_lon0.xml` (14 files; 13 read, all but `ESP_037119_1985_SOUTH`, which is outside the tile).
8. HiRISE team PDS RDR labels: https://hirise-pds.lpl.arizona.edu/PDS/RDR/ESP/ORB_045900_045999/ESP_045994_1985/ESP_045994_1985_RED.LBL ; `.../ESP/ORB_048800_048899/ESP_048842_1985/ESP_048842_1985_RED.LBL` ; `.../PSP/ORB_002300_002399/PSP_002387_1985/PSP_002387_1985_RED.LBL` ; `.../ESP/ORB_037300_037399/ESP_037396_1985/ESP_037396_1985_RED.LBL` ; `.../ESP/ORB_037800_037899/ESP_037818_1990/ESP_037818_1990_RED.LBL`
9. HiRISE RDR cumulative index label (column definitions): https://hirise-pds.lpl.arizona.edu/PDS/INDEX/RDRCUMINDEX.LBL
10. NASA Ames Stereo Pipeline documentation, `dem_mosaic`: https://stereopipeline.readthedocs.io/en/latest/tools/dem_mosaic.html
11. PDS, MRO mission catalog (`mission.cat`), "Primary Science" section: https://planetarydata.jpl.nasa.gov/img/data/mro/ctx/mrox_2578/catalog/mission.cat
12. Cheng, Y., Ansar, A., Johnson, A. (2021). Making an Onboard Reference Map From MRO/CTX Imagery for Mars 2020 Lander Vision System. *Earth and Space Science* 8, e2020EA001560. https://doi.org/10.1029/2020EA001560 . Read from the copy in this folder: section 2.5, Tables 4 and 6, Figs. 9 and 16, section 3.
