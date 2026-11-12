"""CPU-only command line entrypoint; never imported into ComfyUI's process."""

import argparse
from pathlib import Path
import sys
import traceback

from worker_runtime import installer_command, private_venv_directory, validate_worker_identity, worker_python


def main():
    parser = argparse.ArgumentParser(description="Face-only texture projection with unchanged source geometry.")
    parser.add_argument("--locate", action="store_true",
                        help="Only label the face regions of every vertex (no photo); writes --preview-png.")
    parser.add_argument("--source-glb", required=True, type=Path)
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--output-glb", required=True, type=Path)
    parser.add_argument("--comparison-png", type=Path)
    parser.add_argument("--preview-png", type=Path)
    parser.add_argument("--proof-json", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--depth-window", type=float, default=0.3)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--feather", type=float, default=0.18)
    parser.add_argument("--render-resolution", type=int, default=768)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--light-balance", type=float, default=0.0)
    parser.add_argument("--tone-match", type=float, default=0.0)
    parser.add_argument("--brightness", type=float, default=1.0)
    parser.add_argument("--skin-recolor", type=float, default=0.0)
    parser.add_argument("--seam-blend", choices=("true", "false"), default="true")
    parser.add_argument("--projection-mask-png", type=Path)
    arguments = parser.parse_args()
    if arguments.locate and arguments.preview_png is None:
        parser.error("--locate requires --preview-png")
    if not arguments.locate and (arguments.reference_image is None or arguments.comparison_png is None):
        parser.error("--reference-image and --comparison-png are required without --locate")
    package_dir = Path(__file__).resolve().parent
    repair = installer_command(package_dir)
    try:
        worker_python(package_dir)
        validate_worker_identity(
            {"version": list(sys.version_info[:2]), "prefix": sys.prefix, "base_prefix": sys.base_prefix},
            private_venv_directory(package_dir),
        )
    except (OSError, RuntimeError) as exc:
        print(f"GLBFaceRetexture refuses this worker environment: {exc}.\nRun: {repair}", file=sys.stderr)
        return 2
    try:
        # This import belongs here, not in the Comfy node's module. MediaPipe and
        # its CPU runtime do not share state with ComfyUI's torch interpreter.
        import face_core
        if arguments.locate:
            face_core.run_locate(source_glb=arguments.source_glb, output_glb=arguments.output_glb,
                                 preview_png=arguments.preview_png, proof_json=arguments.proof_json,
                                 model_dir=arguments.model_dir, render_resolution=arguments.render_resolution,
                                 yaw=arguments.yaw, depth_window=arguments.depth_window)
            return 0
        face_core.run_retexture(
            source_glb=arguments.source_glb,
            reference_image=arguments.reference_image,
            output_glb=arguments.output_glb,
            comparison_png=arguments.comparison_png,
            proof_json=arguments.proof_json,
            model_dir=arguments.model_dir,
            strength=arguments.strength,
            feather=arguments.feather,
            render_resolution=arguments.render_resolution,
            yaw=arguments.yaw,
            light_balance=arguments.light_balance, tone_match=arguments.tone_match, brightness=arguments.brightness,
            skin_recolor=arguments.skin_recolor,
            seam_blend=arguments.seam_blend == "true",
            projection_mask_png=arguments.projection_mask_png,
        )
    except ImportError as exc:
        print(f"GLBFaceRetexture CPU dependency unavailable in {sys.executable}: {exc}.\nRun: {repair}", file=sys.stderr)
        return 2
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
