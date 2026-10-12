# Changelog

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
