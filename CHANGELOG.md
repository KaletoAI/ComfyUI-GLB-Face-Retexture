# Changelog

## 1.0.0 — 2026-10-12

- Extract the face-only GLB texture projection node from AI-Hub into the independent Apache-2.0 package ComfyUI-GLB-Face-Retexture, retaining original attribution.
- Expose `GLBFaceRetexture`, displayed as GLB Face Retexture (Preserve Geometry), with strength, feather, render_resolution and yaw controls and native FILE_3D / IMAGE / STRING outputs.
- Preserve the original core algorithm and geometry; partition original triangles into body/face primitives for the projected face texture rather than claiming byte-identical GLB output.
- Provide a portable Python installer for an isolated Python 3.12 CPU worker and hash-checked official model downloads. Keep ComfyUI project dependencies empty.
- Include an editable seven-node workflow with empty user-upload selectors and no bundled source assets.
- Add private-worker unit-test CI for Ubuntu and Windows. Windows ComfyUI execution remains unverified; Comfy Registry publication requires the owner's actual publisher identity and API key.
