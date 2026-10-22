# Changelog

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
