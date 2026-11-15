# Changelog

## 1.7.0 — 2026-11-15

Fixes from a review of the whole code base. Every input that succeeded with 1.6.1 gives byte-identical GLB, comparison and proof (portrait grid and cyan luminance case compared).

- **Yaw fallback.** When the requested `yaw` finds no face on the mesh, yaw + 180, + 90 and − 90 are tried and the first that works is used; the proof records `yaw_fallback` with every failed attempt, and the camera block names the yaw used. The job mesh of 1.6.1 rotated by 180° is found at yaw −180. If every turn fails and the mesh is clearly longest along X or Z, the error says the node expects an upright Y-up mesh.
- **Same face in both views.** The native head re-render must show the face that placed its camera: nose within 0.5 eye spans and eye span within ×0.6–1.67 of the locating detection, both in model units. On the portrait grid, the cyan case and the 1.6.1 job mesh the measured deviation was at most 0.02 eye spans and 2 %.
- **One face reported twice** in a single view (noses within 0.35 eye spans) counts as one face. A 200 px face low in the render was reported twice by MediaPipe and refused as "multiple faces".
- **Small faces when locating.** The 12 px eye-span and anchors-inside-the-image checks now apply only where landmarks drive registration (photo, head re-render), not to the detection that merely places the head camera.
- **Worker arguments.** Numeric values are passed as `--name=value`; `--yaw -1e-05` was read by argparse as an option and the worker refused to start. `worker.py` exposes `argument_parser()`.
- **Clean error messages.** A refused input (`FaceTextureError`) leaves the worker with exit code 3 and one `FaceTextureError: …` line; the node shows just that message. Other failures keep the traceback without MediaPipe log lines and Python warnings.
- **Work directories.** A failed or cancelled run removes its `glb_face_retexture_*` directory; a successful run keeps only the returned GLB, which `FILE_3D` references by path.
- **Model check.** The worker refuses a model file whose SHA-256 differs from the pinned download.
- **GLB Face Regions** refuses a GLB that already carries `_FACE_REGION`/`_FACE_DEPTH` or `extras.glb_face_regions` instead of failing in the export check.
- Smaller fixes: an unreadable or oversized (> 67,108,864 px) reference photo is a `FaceTextureError`; outputs get the normal umask permissions instead of 0600; the error for several candidate faces states their number; a triangle whose bounding box exceeds the raster budget is rasterised in row bands (same rows in the same order) instead of allocating it whole.
- 14 new tests: bridge (negative exponent values through the real parser, cleanup on success/failure/cancel, input-error message, log filtering, model hash) and core (double report, small locating face, yaw fallback and hint, view consistency, labelled GLB, bad photo, file mode, raster bands).

## 1.6.1 — 2026-11-15

- Find the head before trusting a face in the full source render. On a full-body character (head about 70 px wide in the 768 px render, stylised black eyes) the head was missed and the chest, with the nipples as eyes, was the only "face", so the photo was painted onto the chest (job d2d91062). The head is now read from the silhouette of the full render: going down from the top, head, narrower neck, then a shoulder row at least 1.6 head widths wide. A face whose nose lies below that row is discarded, and a native render of head and neck is searched instead; the same happens when the full render finds no face at all. There the job's head was found with the nose 52 px above the shoulder row. A face that is still below the shoulders is refused.
- A silhouette without that shape (head-only mesh, bust without neck, side view) keeps the 1.6.0 path, as does every face found above the shoulders: the portrait grid and the cyan luminance case give byte-identical GLB, comparison and proof. When the head window was used, `source_full_observation` holds its attempts plus `head_window` (shoulder row, window, reason, nose row and the rejected full-render observation).
- `GLB Face Regions` shares the head location and gets the same fix.
- Add tests with a synthetic figure (head, neck, torso) and a size-limited fake detector: chest face replaced by the head, head window when the full render finds nothing, unchanged path for a face above the shoulders, refusal without a face on the head, shoulder row and head-only silhouette.

## 1.6.0 — 2026-11-12

- New node **GLB Face Regions (Locate Only)** (`GLBFaceRegions`; worker `--locate`, `--preview-png`, `--depth-window`). It labels every vertex from MediaPipe landmarks of the head render: right/left eye with lid rim, right/left brow, lips and nose (`_FACE_REGION` 1–6, 0 = none). Hidden vertices up to `depth_window` (default 0.3 eye spans) behind the visible surface are included, such as eye sockets, the inside of the mouth and lash flaps. `_FACE_DEPTH` gives their distance behind the surface in eye spans. No photo is needed and no vertex moves: geometry, UVs, vertex order, images and materials stay byte-identical, and the original binary is an exact prefix.
- `extras.glb_face_regions` in the output GLB carries the legend, the eye span in model units and twelve landmarks in scene coordinates, so a consumer needs nothing but the GLB. The proof adds per-region counts; the preview tints the regions.
- On UniForge3D characters (Kai, Kira, Kira's raw mesh with lash flaps) 1,600–3,100 vertices are labelled in about 2 s. A large share lies more than 0.02 eye spans behind the visible skin: these meshes carry inner layers under the face.
- The retexture node is unchanged (byte-identical output); its head location is shared with the new node.

## 1.5.0 — 2026-11-12

- Add `seam_blend` (default on): a two-band seam blend at the projection edge. The photo keeps its detail, but within a band along the edge its low frequencies (σ 0.1 eye spans) are pulled toward the mesh skin just outside the edge. The pull is a per-channel linear gain, at most ×2/×½, measured on the outer feather and carried inward. It is full across the feather and fades to none at feather + 0.25 eye spans. Eyes, brows, lips and the inside of the face are untouched, and nothing changes where the outside is hair or background. Only texels the projection paints anyway are affected, so nothing reaches other UV islands or empty texture. Light and tone matching only evened out the average colour; on generated meshes the edge still showed a coloured band.
- New fourth node output `projection_mask` (IMAGE) and worker option `--projection-mask-png`: the face atlas's per-texel blend amount as greyscale, 255 = projected, 0 = original.
- The proof gains a `seam_blend` block. `render_delta_e` is the ΔE across the feather in the head render. `delta_e` is the ±3 texel ΔE along the texture mask within one UV island, also without the areas around eyes, brows and lips. On two UniForge3D characters `render_delta_e` p50/p95 fell from 9.0/13.3 to 7.0/11.4 and from 16.3/34.5 to 9.7/20.6.
- `seam_blend` off gives the 1.4.4 GLB and comparison byte for byte; the proof only gains the measurement block. Output is deterministic for the same input.
- Add synthetic tests for the seam field (edge match, untouched interior and features, hairline, determinism), the mask, the gain and the ΔE measurement, plus a real vision run on a warm-shifted mesh.

## 1.4.4 — 2026-11-09

- Ignore a second "face" on body texture. On a full-body render (head about 60 px wide in the 768 px render) no face was found in the whole image, and the upper search crop reported the head plus a second face on the chest, so the job failed with `Source mesh render: multiple faces are unsupported`. When a view of the source mesh (full render or head re-render) now reports more than one face, each candidate is re-detected in its own face crop and kept only if a face appears there at the same nose position (within 0.35 outer-eye spans). The chest vanished in its own crop, the head was found again within 0.02 eye spans. Two confirmed faces in a render are still refused. The reference photo is unchanged: it refuses every second face, confirmed or not, so a hard-to-detect second person cannot slip through.
- The proof lists the extra detections as `<kind>_confirmation` attempts.
- Otherwise unchanged: a view with at most one face takes the same path as in 1.4.3, so every input that succeeded before gives byte-identical GLB, comparison and proof.
- Add unit tests with a fake detector: an unconfirmed body detection, a confirmation crop that only shows a neighbouring face, two separate and two adjacent confirmed faces in a render, and a photo with a second detection.

## 1.4.3 — 2026-11-05

- Accept small bearded faces in the full source render. A full-body character with a full beard (outer-eye span about 41 px in the 768 px render) failed with `Source mesh render: crop refinement changed the face implausibly`: the bbox refinement agreed with the first detection within 3 px on eyes, nose and mouth, but the beard-hidden chin (landmark 152) moved 0.43 eye spans, just above the 0.4 limit. That detection only places the camera for the native head re-render, so it now keeps the first detection instead of failing. The photo and the head re-render, whose landmarks drive registration, keep the strict check.
- Every detection in the proof records `refinement_rejected`; `refinement_used` is false when the refinement was rejected.
- Otherwise unchanged: GLB, comparison and proof (apart from the new field) are byte-identical to 1.4.2 on the portrait grid (defaults and all controls active) and the cyan luminance case.
- Add unit tests for the strict and the locate-only refinement paths with a fake detector.

## 1.4.2 — 2026-10-31

- Security: the worker venv now installs Pillow 13.0.0 instead of 12.3.0. 12.3.0 carries about 40 advisories, among them heap overflows and use-after-free in the JPEG, TIFF, GIF and BMP decoders and a bundled libopenjp2 with a heap-buffer-overflow write; the worker decodes the photo and every embedded GLB texture with Pillow. Re-run `install.py` (or reinstall the node) so an existing `.venv` picks it up.
- No change to detection or output.

## 1.4.1 — 2026-10-22

- Find small faces with saturated non-skin colours. A full-body cyan character (head about 60 px wide in the 768 px source render, glowing eyes) failed with `Source mesh render: no unique usable face detected` at every `render_resolution`, while a tight head crop or a grey copy was detected. When the colour search finds no face, `detect_face` now repeats the same search on a luminance (BT.601) copy and refines there; landmarks are geometry, so registration is unchanged. The proof records those attempts with a `luminance_` prefix.
- Inputs whose face is found in colour produce byte-identical GLB, comparison and proof to 1.4.0.
- Add a real vision regression: the fixture portrait as 160 px cyan face on a 768 px canvas fails in colour and is retextured through the luminance pass.

## 1.4.0 — 2026-10-20

- Support embedded WebP textures through `EXT_texture_webp`, including UniForge3D exports and PNG/JPEG fallbacks, with shared texture-reference resolution.
- Validate every embedded image signature, format, dimensions and single-frame decode; limit textures to 8192 pixels per side and 67,108,864 pixels total.
- Preserve original WebP images, binary prefix, extension declarations and metallic/roughness maps while new face and optional recoloured body textures reference PNG atlases without copied WebP extensions.
- Add lossless WebP pixel, export preservation, fallback, JPEG and malformed-input regression coverage.

## 1.3.0 — 2026-10-16

- Add optional `skin_recolor` (0–1, default 0) to the node, worker CLI and workflow to correct generator-shifted skin across the original atlas toward the light-balanced photo skin tone in linear light.
- Model full-front mesh skin with robust colour statistics, softly weight skin-coloured texels (tolerant toward redder and brighter, so the red shift and baked highlights are included), exclude partial alpha, and respect `baseColorFactor`. Only the colour moves; each texel keeps its luminance. On a strongly red generated mesh the plain per-channel gain had clipped scalp highlights into white blotches. Blend the face into the recoloured atlas and tone-match toward the recoloured face skin.
- Append a separate recoloured body atlas/material while preserving original images, binary prefix and geometry; verify protected body pixels against the expected recolour and record measurements in `skin_recolor`.
- Retain byte-identical v1.2.0 outputs with recolour off; extend synthetic, export, validation, CLI, bridge and real vision regression coverage.

## 1.2.0 — 2026-10-16

- `light_balance` now removes the photo's broad shading instead of only balancing left against right: a skin-only low pass of the luminance (sigma a quarter eye span) is divided out and replaced by the median skin luminance, refined in three passes. Besides side light this removes the darker cheek sides where the face turns away from the camera, which 1.1 left as a shadow along the projection edge. Features stay and nothing is mirrored. Measured on a generated face view: broad-shading p10/p90 .57 -> .89, left/right .73 -> .96. The proof adds `shading_p10_p90_before/after`.

## 1.1.1 — 2026-10-16

- Tone match now targets the mesh's own face skin (semantic face mask of the source render) instead of a ring around the face. On generated meshes that ring was mostly the neck under the chin, with baked shadow, and the hairline, with hair roots, so 1.1.0 pulled the face darker and browner than the skin around it. Measured on a generated humanoid: ring median sRGB 170/125/97 against face skin 203/153/136.

## 1.1.0 — 2026-10-16

- Add optional `light_balance`, `tone_match` and `brightness` controls to the node and CPU worker CLI; defaults retain byte-identical output and proof.
- Balance broad side light along a landmark-fitted rolled midline without mirroring facial features, then match linear skin colour to a semantic ring around the rendered source face (with recorded face fallback), and apply final brightness.
- Record active adjustment measurements in `photo_adjust` and show the adjusted photo in the comparison.
- Extend synthetic, bridge, CLI and real portrait regression coverage; update the editable workflow with optional widgets after required controls.

## 1.0.2 — 2026-10-12

- Fix face color under a `baseColorFactor` below 1: the photo is now inverted in linear light as glTF specifies, so viewers show the photo color instead of an sRGB-space approximation (for example 146 instead of 100). Unrepresentable bright colors are counted in `factor_clipped_texture_pixels`.
- Fix gutter bleed: atlas padding no longer overwrites masked-out or occluded texels inside face triangles; it extends only outside the face UV footprint and continues the island's own edge colors.
- Make the registration residual gate independent: the three exactly fitted anchors are no longer averaged into the 30 % eye-span check.
- Vectorize triangle rasterization for rendering and atlas painting with identical z-buffer tie rules (about 18× faster; 203k triangles at 768 px in 2.5 s instead of 44 s).
- Cancelling a prompt in ComfyUI now stops the CPU worker; a stuck worker is stopped after 1800 s (`GLB_FACE_RETEXTURE_TIMEOUT`). Worker logs go to files and are decoded as UTF-8, so large or non-ASCII output can neither block nor hide an error.
- Add `install.py --recreate` to rebuild an unusable private `.venv`; the installer and node now point to it instead of looping on the plain installer.
- Pass package-index, certificate, proxy and cache variables (`PIP_INDEX_URL`, `PIP_CERT`, `UV_INDEX_URL`, `UV_SYSTEM_CERTS`, ...) to pip/uv while still dropping install-target overrides.
- Reject malformed glTF references (missing attributes, negative or out-of-range indices, undecodable images) as clear `FaceTextureError`s and refuse images smaller than 32 px.
- Proof schema 2: render checks use the atlases decoded from the written GLB, `output_triangle_count` is read from the emitted index accessors, and the reference is hashed both as file (`reference_file_sha256`) and as decoded RGB (`reference_rgb_sha256`).
- Publish GLB, comparison and proof only together, so a late failure leaves no partial result.
- Extend the suite from 14 to 31 tests, including the ComfyUI bridge and a real CPU MediaPipe end-to-end run on a public-domain portrait fixture.
- Add a release workflow: pushing a `v*` tag that matches `pyproject.toml` creates the GitHub release from this changelog section.

## 1.0.1 — 2026-10-12

- Fix installation on Linux ARM64: select MediaPipe 0.10.18 only for Linux aarch64 workers, using its official CPython 3.12 wheel; retain 0.10.21 on other platforms.
- Verify all 14 worker tests on Linux ARM64 and x86-64, plus a real native ComfyUI workflow and AI-Hub upload/result round trip on NVIDIA GB10. Original geometry and protected body texture remain unchanged.

## 1.0.0 — 2026-10-12

- Extract the face-only GLB texture projection node from AI-Hub into the independent Apache-2.0 package ComfyUI-GLB-Face-Retexture, retaining original attribution.
- Expose `GLBFaceRetexture`, displayed as GLB Face Retexture (Preserve Geometry), with strength, feather, render_resolution and yaw controls and native FILE_3D / IMAGE / STRING outputs.
- Preserve the original core algorithm and geometry; partition original triangles into body/face primitives for the projected face texture rather than claiming byte-identical GLB output.
- Provide a portable Python installer for an isolated Python 3.12 CPU worker and hash-checked official model downloads. Keep ComfyUI project dependencies empty.
- Include an editable seven-node workflow with empty user-upload selectors and no bundled source assets.
- Verify private-worker installation and tests in Ubuntu and Windows CI, including CPU vision execution. Verify the editable workflow end-to-end in Linux ComfyUI 0.37.0; Windows ComfyUI GUI execution remains unverified.
- Comfy Registry publication requires the owner's actual publisher identity and API key; GitHub installation is available independently.
