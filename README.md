# ComfyUI-GLB-Face-Retexture

Project a matching RGB face photo onto the visible face texture of an existing static GLB. **GLB Face Retexture (Preserve Geometry)** (`GLBFaceRetexture`) uses face landmarks and semantic masks to register and blend the photo, without reconstructing or deforming the mesh.

Geometry is preserved, but the output is **not a byte-identical GLB**: original triangles are partitioned into body and face primitives and a dedicated face texture/material is added. Original vertex positions, UVs, normals, tangents and original image bytes remain unchanged.

## Requirements

- A modern ComfyUI with native `Load3DAdvanced`, `Preview3DAdvanced`, `SaveGLB` and `comfy_api.latest.Types.File3D`. The supplied workflow records Comfy core **0.37.0**, the version in the verified source setup. Earlier versions are not claimed compatible.
- The ComfyUI Python interpreter must support Python 3.10 or newer and already provide ComfyUI's torch, NumPy and Pillow dependencies.
- A separate **Python 3.12** installation, or [uv](https://docs.astral.sh/uv/getting-started/installation/) on PATH to obtain it. Network access is needed for worker dependencies and official model downloads.
- Your own supported GLB and matching RGB photo. No source meshes, photos or sample assets are bundled.

The CPU worker runs in a private `.venv` with its own MediaPipe dependencies. It does not install worker dependencies into ComfyUI, use ComfyUI's GPU environment, or silently fall back to the ComfyUI interpreter. Windows installation commands are provided, but Windows execution has not yet been verified.

## Install

### ComfyUI Manager

Use Manager's **Install via Git URL** with:

```text
https://github.com/KaletoAI/ComfyUI-GLB-Face-Retexture
```

Then run the installer below from the installed custom-node directory. Installing the repository alone does not prepare the private worker.

### Manual installation

From your ComfyUI directory:

```sh
git clone https://github.com/KaletoAI/ComfyUI-GLB-Face-Retexture custom_nodes/ComfyUI-GLB-Face-Retexture
cd custom_nodes/ComfyUI-GLB-Face-Retexture
```

### Prepare the CPU worker

Use the **actual ComfyUI interpreter** to launch the stdlib installer. For a typical Linux ComfyUI venv, starting in the custom-node directory:

```sh
../../venv/bin/python install.py
```

If you normally launch ComfyUI with an activated environment instead:

```sh
python install.py
```

For Windows portable, starting in `ComfyUI_windows_portable\ComfyUI\custom_nodes\ComfyUI-GLB-Face-Retexture`, run in Command Prompt:

```bat
..\..\..\python_embeded\python.exe install.py
```

The three parent steps are intentional: the embedded interpreter is beside `ComfyUI`, not inside it. In PowerShell prepend `&` to that interpreter command.

The installer finds an existing Python 3.12 or uses uv. To select either explicitly, append one of these options to the interpreter command above:

```text
install.py --python /path/to/python3.12
install.py --uv /path/to/uv
```

`PYTHON312` and `UV_BIN` can also select these executables. Use an actual Python 3.12 executable for `--python`; ComfyUI itself may run another supported version. If neither prerequisite is available, the installer reports a clear error rather than changing ComfyUI's environment.

The installer creates `.venv`, installs `requirements-worker.txt` there, and downloads hash-checked models into `models/`. Restart ComfyUI after installation, then hard-refresh the browser so the node and native widgets load.

## Use the editable workflow

Open [`workflows/glb-face-retexture.json`](workflows/glb-face-retexture.json) in ComfyUI. Its seven visible nodes are fully editable; upload selectors are deliberately empty:

1. **Load3DAdvanced:** upload/select your own static textured source GLB.
2. **LoadImage:** upload/select its matching RGB face photo, exactly one image.
3. **GLB Face Retexture:** adjust the four controls below and queue the workflow.
4. **Preview3DAdvanced:** inspect the corrected GLB in the native viewer.
5. **PreviewImage:** inspect the reference/before/after comparison.
6. **SaveGLB:** save the corrected model using the default `3d/GLBFaceRetexture` prefix or your own prefix.
7. **Note:** usage reminders, not an execution dependency.

| Control | Default / node range | Purpose |
| --- | --- | --- |
| `strength` | 1.0 / 0.01–1.0 | Photo blending strength within the selected face region. |
| `feather` | 0.18 / 0–0.5 | Mask-edge feathering relative to landmark eye span. |
| `render_resolution` | 768 / 256–2048, step 64 | CPU registration/render resolution; higher settings require more work. |
| `yaw` | 0 / −180–180 degrees | Source viewing yaw for face registration; not mesh deformation. |

Outputs are native **`FILE_3D`** (`corrected_glb`), **`IMAGE`** (`comparison`) and **`STRING`** (`proof_metrics`). The metrics are also shown as node text and describe geometry/texture preservation observations. The corrected GLB can connect directly to native preview and save nodes.

## Supported inputs and limits

- A single mesh with one indexed triangle primitive and an embedded, uncompressed opaque texture.
- Static assets only: no rigs, skinning, morph targets or external assets. Orient the model **Y-up, +Z front**.
- One matching RGB reference image, not a batch. A detectable face must be visible in both the source registration view and photo.
- This is planar photo projection, not 3D face reconstruction. It cannot improve silhouette, add missing geometry, or repair hidden eye anatomy. Perspective and expression differences can remain visible.
- Registration, masking and texture projection run locally on CPU. There is no cloud generation service and no automatic model regeneration.

## Models and licenses

The repository code is licensed under [Apache License 2.0](LICENSE). See [NOTICE](NOTICE) for the retained AI-Hub attribution and derivation note.

The installer downloads (does not bundle) official Google MediaPipe model assets:

- [Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker), `face_landmarker.task`, from the [official model source](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task).
- [Image Segmenter](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter), `selfie_multiclass.tflite`, from the [official model source](https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/1/selfie_multiclass_256x256.tflite).

MediaPipe software and model assets are separate upstream works. Consult the [MediaPipe upstream license](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE) and linked model documentation for their applicable terms; this repository's license does not replace those terms. Other worker dependencies retain their own licenses. Only use source meshes/photos you have permission to process and redistribute.

## Development and registry publication

Tests run in the private worker environment, not the ComfyUI environment:

```sh
.venv/bin/python -m unittest discover -s tests -t .
```

On Windows:

```bat
.venv\Scripts\python.exe -m unittest discover -s tests -t .
```

CI installs that environment with Python 3.12 on Ubuntu and Windows and runs the tests there. CI coverage does not by itself establish a verified Windows ComfyUI session.

The intended package ID is **`glb-face-retexture`**, version **1.0.0**. It is **not yet listed in Comfy Registry**; use the GitHub URL installation above. Workflow `cnr_id` metadata records the intended ID, not evidence of registration.

Before publishing, the owner must obtain the actual Comfy Registry publisher identity, add its real `PublisherId` under `[tool.comfy]` in `pyproject.toml`, obtain the corresponding registry API key, and publish using the [official registry instructions](https://docs.comfy.org/registry/publishing). No publisher identity or credentials are fabricated here, and there is no automatic publishing workflow.
