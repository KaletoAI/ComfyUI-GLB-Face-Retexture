"""ComfyUI bridge for face-only texture projection on an existing static GLB."""

import json
import os
from pathlib import Path
import subprocess
import tempfile

import comfy.model_management
import folder_paths
import numpy as np
from PIL import Image
import torch
from comfy_api.latest import Types

from .worker_runtime import MODELS, installer_command, run_worker, worker_environment, worker_python


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_TIMEOUT_SECONDS = 1800


def worker_timeout() -> float:
    """Seconds before a stuck worker is killed; GLB_FACE_RETEXTURE_TIMEOUT overrides it."""
    value = os.environ.get("GLB_FACE_RETEXTURE_TIMEOUT", "").strip()
    if not value:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        seconds = float(value)
    except ValueError:
        seconds = 0
    if not 0 < seconds < float("inf"):
        raise ValueError(f"GLB_FACE_RETEXTURE_TIMEOUT must be a positive number of seconds, not {value!r}.")
    return seconds


class GLBFaceRetexture:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_glb": ("FILE_3D", {"tooltip": "Original static textured Trellis GLB. Geometry is preserved, not regenerated."}),
                "reference_image": ("IMAGE", {"tooltip": "One matching RGB face photo. Face detection and registration run anew."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "feather": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 0.5, "step": 0.01}),
                "render_resolution": ("INT", {"default": 768, "min": 256, "max": 2048, "step": 64}),
                "yaw": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0, "tooltip": "Source viewing yaw in degrees for face registration."}),
            },
            "optional": {
                "light_balance": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Removes the photo's broad shading from the face (side light, darker face edges where the cheeks turn away) so the texture carries skin colour, not light. Features are kept and nothing is mirrored. 0 = off."}),
                "tone_match": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Pulls the photo face's overall skin colour and brightness toward the mesh's own face skin, so the projection blends into the surrounding texture without a seam. 0 = off."}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.05, "tooltip": "Final brightness scale of the photo face before projection. 1 = unchanged."}),
                "skin_recolor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Recolours the mesh's own skin (face, neck, arms, hands — every skin-coloured texel of the original texture) toward the photo's skin tone, for meshes whose generator shifted the skin colour. Clothes and hair are left alone. 0 = off."}),
            }
        }

    RETURN_TYPES = ("FILE_3D", "IMAGE", "STRING")
    RETURN_NAMES = ("corrected_glb", "comparison", "proof_metrics")
    FUNCTION = "retexture"
    CATEGORY = "3d/texture"
    DESCRIPTION = "Project a matching photo onto only the visible face texture; preserve the source GLB geometry and original assets. CPU worker runs in a separate process."

    def retexture(self, source_glb, reference_image, strength, feather, render_resolution, yaw, light_balance=0.0, tone_match=0.0, brightness=1.0, skin_recolor=0.0):
        repair = installer_command(PACKAGE_DIR)
        if not isinstance(source_glb, Types.File3D) or source_glb.format != "glb":
            raise ValueError("GLBFaceRetexture requires a static textured GLB from the native Load 3D node.")
        if reference_image.ndim != 4 or reference_image.shape[0] != 1 or reference_image.shape[-1] != 3:
            raise ValueError("GLBFaceRetexture requires exactly one RGB reference image, not an image batch.")

        python = str(worker_python(PACKAGE_DIR))
        model_dir = PACKAGE_DIR / "models"
        for filename, _, _ in MODELS:
            if not (model_dir / filename).is_file():
                raise RuntimeError(f"GLBFaceRetexture is missing {model_dir / filename}. Run: {repair}")

        folder_paths_temp = Path(folder_paths.get_temp_directory())
        folder_paths_temp.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix="glb_face_retexture_", dir=folder_paths_temp))
        source = source_glb.get_source()
        if isinstance(source, str):
            source_path = Path(source).resolve()
        else:
            source_path = work_dir / "source.glb"
            source_glb.save_to(str(source_path))
        if not source_path.is_file():
            raise ValueError(f"GLBFaceRetexture source GLB does not exist: {source_path}")

        reference_path = work_dir / "reference.png"
        pixels = reference_image[0].detach().cpu().numpy()
        Image.fromarray(np.rint(np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8)).save(reference_path)
        output_path = work_dir / "GLBFaceRetexture.glb"
        comparison_path = work_dir / "comparison.png"
        proof_path = work_dir / "proof.json"
        command = [
            python, "-E", "-s", "-X", "utf8", str(PACKAGE_DIR / "worker.py"),
            "--source-glb", str(source_path), "--reference-image", str(reference_path),
            "--output-glb", str(output_path), "--comparison-png", str(comparison_path),
            "--proof-json", str(proof_path), "--model-dir", str(model_dir),
            "--strength", str(strength), "--feather", str(feather),
            "--render-resolution", str(render_resolution), "--yaw", str(yaw),
            "--light-balance", str(light_balance), "--tone-match", str(tone_match), "--brightness", str(brightness),
            "--skin-recolor", str(skin_recolor),
        ]
        timeout = worker_timeout()
        try:
            # Cancelling the prompt in ComfyUI raises here and kills the worker.
            returncode, stdout, stderr = run_worker(
                command, cwd=PACKAGE_DIR, env=worker_environment(), log_dir=work_dir, timeout=timeout,
                check=comfy.model_management.throw_exception_if_processing_interrupted,
            )
        except OSError as exc:
            raise RuntimeError(f"GLBFaceRetexture could not start the CPU worker ({python}): {exc}. Run: {repair}") from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"GLBFaceRetexture CPU worker exceeded {timeout:g} s and was stopped. "
                               "Lower render_resolution or raise GLB_FACE_RETEXTURE_TIMEOUT.") from exc
        if returncode:
            details = stderr.strip() or stdout.strip() or "Worker produced no error details."
            raise RuntimeError(f"GLBFaceRetexture CPU worker failed (exit {returncode}):\n{details}")
        for output in (output_path, comparison_path, proof_path):
            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError(f"GLBFaceRetexture worker did not produce its required output: {output}")
        proof = json.loads(proof_path.read_text(encoding="utf-8"))
        with Image.open(comparison_path) as comparison_image:
            comparison = np.asarray(comparison_image.convert("RGB"), dtype=np.float32) / 255.0
        comparison_tensor = torch.from_numpy(comparison).unsqueeze(0)
        proof_text = json.dumps(proof, ensure_ascii=False, indent=2)
        return {"ui": {"text": [proof_text]}, "result": (Types.File3D(str(output_path)), comparison_tensor, proof_text)}


NODE_CLASS_MAPPINGS = {"GLBFaceRetexture": GLBFaceRetexture}
NODE_DISPLAY_NAME_MAPPINGS = {"GLBFaceRetexture": "GLB Face Retexture (Preserve Geometry)"}
