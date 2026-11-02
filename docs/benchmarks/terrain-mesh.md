# Terrain mesh checks (issues #4 and #32)

The Jezero ground-truth mesh (`snakemake jezero_mesh`, code in `src/fidelis/terrain/mesh.py`), checked against the tile it was built from on 2026-11-01. Issue #4 built a 2 km tile at full resolution; issue #32 replaced it with a 10 km tile at 4 m mesh / 2 m texture, in true ground metres. The numbers below are for the 10 km mesh unless they say otherwise. Reproduce with:

    uv run python scripts/check_terrain_mesh.py data/jezero/mesh data/jezero/tile data/jezero/mesh/render_vs_ortho.png

## The mesh

- **The tile:** 10,020 × 10,020 map metres centred on the landing site (`size_m` 10,000, grown outward to a 12 m grid so its edges are whole mesh, texture and matching-map pixels). `fidelis-tile` area-averages the 1 m DTM in 4 × 4 blocks and the 0.25 m orthoimage in 8 × 8 blocks, reading the window in strips. Neither tile has any nodata pixels.
- One vertex per (4 m) DTM pixel, at the pixel centre: 2505 × 2505 = 6,275,025 vertices and 12,540,032 triangles. The binary PLY is 289 MB, and `texture.png` (the 5010 × 5010 orthoimage at 2 m) is 17 MB. Building the tile, mesh and matching map took 41 s on the laptop.
- Local coordinates: **ground metres** east and north of the tile centre (recorded as `origin` in `mesh.json`), with z the DTM elevation (−2652.1 to −2393.2 m). Raw map metres (~4.59e6) would leave float32 vertices only 0.5 m apart.
- The mesh stops half a pixel (2 m) inside the tile on each side, because its outer vertices are pixel centres. So it spans ±4750.7 m east–west and ±5008.0 m north–south.

Issue #4's 2 km mesh had one vertex per 1 m DTM pixel (4,016,016 vertices, 185 MB PLY, 33 MB texture at 0.25 m), in map metres.

## East–west scale (issue #32)

The USGS mosaics are equirectangular with the standard parallel at the equator (`lat_ts=0`), so one map metre east–west is cos(latitude) of a ground metre, while north–south is exact. Issue #4's mesh treated map metres as ground metres, which made it 5.4% too wide east–west: east–west slopes 5.4% too gentle, and shadows cast east or west 5.4% too long.

- `fidelis-tile` records `east_west_ground_scale` = cos(latitude at the tile centre) in `tile.json`: **0.94863**. GDAL's inverse projection at the landing site gives 1000 map metres east–west = 948.6 ground metres (`docs/research/descent-altitude-and-tile-size.md`).
- `fidelis-mesh` multiplies local x by it and records it as `east_scale` in `mesh.json`. Map metres → mesh: x = (map x − origin x) × `east_scale`, y = map y − origin y.
- One factor for the whole tile: across 10 km, cos(latitude) varies by less than 0.05%.
- Unit test: neighbouring DTM pixels, 1 map metre apart both ways, come out `east_scale` metres apart east–west and 1 m apart north–south. The elevation and texture-registration unit tests run on a scaled mesh, converting back to map metres through the recorded scale.

## Elevation (acceptance: within a stated tolerance)

**Stated tolerance: 1 mm at every vertex** (`elevation_tolerance_m` in `mesh.json`). Vertices *are* DTM samples stored as float32, so only rounding can move them.

- Unit tests (`tests/terrain/test_mesh.py`): every vertex of a rough synthetic DTM, read back through Mitsuba's PLY loader, sits on a pixel centre at that pixel's height within 1 mm.
- Real tile: 10,000 random vertices, read straight from the PLY bytes, against `dtm.tif` (the 4 m tile) sampled at their map coordinates (x divided by `east_scale`). **Max error 0.0 m** on both the 2 km and the 10 km mesh.

**Caveat:** this checks the mesh against the 4 m tile, not the 1 m DTM. Each 4 m height is the mean of 16 DTM pixels, so relief finer than about 4 m is averaged away; between vertices the mesh interpolates linearly. At LVS altitudes a descent-camera pixel covers 4.3–8.2 m of ground, so the camera couldn't resolve that relief either, but low-sun shadows from small, steep features should come out softer (not measured).

## Texture registration

- **Unit test, through the renderer's own lookup.** The synthetic orthoimage is a ramp (value = column + 5 × row), so the value Mitsuba samples at a vertex's UV says exactly where in the image it landed. Every vertex samples the ramp value at its own map position, within 0.05 grey levels (0.05 ortho pixels).
- **Real tile, top-down render, in ground metres.** A Mitsuba orthographic render from straight above (uniform sky, 9 km × 9 km of ground at 4 m/px, 2250 × 2250 px, 16 samples per pixel, `cuda_ad_rgb` on the laptop's RTX 5070) against the orthoimage area-averaged onto the same ground grid (each 4 m ground pixel is 4.217 map metres east–west). **Correlation 0.978, phase-correlation shift +0.01, −0.01 px.** Controls: mirrored north–south the render correlates at 0.173, mirrored east–west at 0.137, and against the orthoimage laid out in uncorrected map metres at **0.529**. So the check detects a wrong texture and a wrong east–west scale. The remaining mismatch is expected: crater floors see less sky, so they render darker, and 16 samples leave noise.
- Issue #4's 2 km mesh, rendered at 2 m/px over 2 km: correlation 0.972, shift 0.00 px, mirrored −0.109 (north–south) and 0.071 (east–west).

**Caveat:** phase correlation resolves a fraction of a 4 m pixel, not centimetres. The sub-pixel guarantee comes from the unit test on synthetic data.

## Opens in a standard viewer

Viewer checks were on issue #4's 2 km mesh; the 10 km mesh uses the same PLY layout and wasn't reopened.

- **MeshLab** opened the PLY on the laptop: 4,016,016 vertices and 8,024,018 faces, loaded in 6.7 s. The first build named its UVs `u`/`v`, and MeshLab silently dropped them, so it showed the mesh untextured. The PLY now uses `texture_u`/`texture_v`, the names MeshLab itself writes, plus a `comment TextureFile texture.png` line that tells it which image to load. A test pins those names. After the rename, MeshLab 2025.07 showed the whole mesh textured at 500 FPS. Craters sit in their rims, and the layout matches the orthoimage, rotated but not mirrored. **Laptop gotcha:** Windows first ran MeshLab on the integrated GPU (`nvidia-smi` showed no process on the RTX 5070). There it fell back to immediate-mode rendering at 1.4 FPS and never finished drawing the mesh. Setting MeshLab to *High performance* in Settings → System → Display → Graphics fixed it.
- **three.js** r160's stock `PLYLoader` (in a browser, served from localhost) loaded all 4,016,016 vertices with their UVs, both before and after the rename. With `texture.png` applied, crater rims and dune fields in the texture sit on the matching relief.
- Mitsuba 3.9 reads `texture_u`/`texture_v` too, and the registration tests run through it.

## UV convention: a gotcha for the renderer

The PLY stores UVs with v = 0 at the image's **bottom** row, the convention MeshLab, Blender and three.js use. Mitsuba 3.9's PLY loader keeps UVs as written, but its bitmap textures put v = 0 at the **top** row. So `mitsuba_shape()` flips v on lookup (`to_uv`). Load the mesh into Mitsuba through that function, not as a bare `ply` shape, or the texture comes out mirrored north–south.

`mitsuba_shape()` also loads the texture raw (no sRGB decoding), on the assumption that the orthoimage's 8-bit values scale reflectance linearly. Issue #5 checked the USGS product description: the values are tone-matched DN, not calibrated reflectance (`docs/benchmarks/mesh-render.md`).
