# Terrain mesh checks (issue #4)

The Jezero ground-truth mesh (`snakemake jezero_mesh`, code in `src/fidelis/terrain/mesh.py`), checked against the tile it was built from on 2026-11-01. Reproduce with:

    uv run python scripts/check_terrain_mesh.py data/jezero/mesh data/jezero/tile data/jezero/mesh/render_vs_ortho.png

## The mesh

- One vertex per DTM pixel, at the pixel centre: 2004 × 2004 = 4,016,016 vertices and 8,024,018 triangles. The binary PLY is 185 MB, and `texture.png` (the 8016 × 8016 orthoimage) is 33 MB.
- Local coordinates: metres east and north of the tile centre (recorded as `origin` in `mesh.json`), with z the DTM elevation (−2607.6 to −2544.7 m). Raw map metres (~4.59e6) would leave float32 vertices only 0.5 m apart.
- The mesh stops half a pixel (0.5 m) inside the tile on each side, because its outer vertices are pixel centres.

## Elevation (acceptance: within a stated tolerance)

**Stated tolerance: 1 mm at every vertex** (`elevation_tolerance_m` in `mesh.json`). Vertices *are* DTM samples stored as float32, so only rounding can move them.

- Unit tests (`tests/terrain/test_mesh.py`): every vertex of a rough synthetic DTM, read back through Mitsuba's PLY loader, sits on a pixel centre at that pixel's height within 1 mm.
- Real tile: 10,000 random vertices, read straight from the PLY bytes, against `dtm.tif` sampled at their map coordinates. **Max error 0.0 m.**

**Caveat:** between vertices the mesh interpolates linearly across each triangle. The DTM has no "true" value there either, since 1 m is its resolution. Sub-metre relief that the DTM doesn't resolve isn't in the mesh.

## Texture registration

- **Unit test, through the renderer's own lookup.** The synthetic orthoimage is a ramp (value = column + 5 × row), so the value Mitsuba samples at a vertex's UV says exactly where in the image it landed. Every vertex samples the ramp value at its own map position, within 0.05 grey levels (0.05 ortho pixels).
- **Real tile, top-down render.** A Mitsuba orthographic render from straight above (uniform sky, 1000 × 1000 px at 2 m/px, 16 samples per pixel, `cuda_ad_rgb` on the laptop's RTX 5070) against the orthoimage averaged to 2 m. **Correlation 0.972, phase-correlation shift 0.00 px** in both axes. Mirrored north–south, the same render correlates at −0.109, and mirrored east–west at 0.071, so the check does detect a wrong texture. The remaining mismatch is expected: crater floors see less sky, so they render darker, and 16 samples leave noise.

**Caveat:** phase correlation resolves a fraction of a 2 m pixel, not centimetres. The sub-pixel guarantee comes from the unit test on synthetic data.

## Opens in a standard viewer

- **MeshLab** opened the PLY on the laptop: 4,016,016 vertices and 8,024,018 faces, loaded in 6.7 s. The first build named its UVs `u`/`v`, and MeshLab silently dropped them, so it showed the mesh untextured. The PLY now uses `texture_u`/`texture_v`, the names MeshLab itself writes, plus a `comment TextureFile texture.png` line that tells it which image to load. A test pins those names. *Rechecking the texture in MeshLab after the rename: pending.*
- **three.js** r160's stock `PLYLoader` (in a browser, served from localhost) loaded all 4,016,016 vertices with their UVs, both before and after the rename. With `texture.png` applied, crater rims and dune fields in the texture sit on the matching relief.
- Mitsuba 3.9 reads `texture_u`/`texture_v` too, and the registration tests run through it.

## UV convention: a gotcha for the renderer

The PLY stores UVs with v = 0 at the image's **bottom** row, the convention MeshLab, Blender and three.js use. Mitsuba 3.9's PLY loader keeps UVs as written, but its bitmap textures put v = 0 at the **top** row. So `mitsuba_shape()` flips v on lookup (`to_uv`). Load the mesh into Mitsuba through that function, not as a bare `ply` shape, or the texture comes out mirrored north–south.

`mitsuba_shape()` also loads the texture raw (no sRGB decoding), on the assumption that the orthoimage's 8-bit values scale reflectance linearly. Issue #5 (mesh rendering) should confirm that against the USGS product description before tuning lighting.
