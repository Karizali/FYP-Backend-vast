"""
Vast.ai GPU Worker — 3D Gaussian Splatting
==========================================

This script runs on a rented Vast.ai GPU instance as a long-running process.
Instead of waiting for Runpod to call a handler, it polls the Redis Bull queue
directly — picks up jobs, processes them, and reports results back via HTTP.

Pipeline:
  1. Poll Redis for queued jobs
  2. Download input files from Cloudinary URLs
  3. (Optional) Enhance images with Real-ESRGAN
  4. Run COLMAP for camera pose estimation
  5. Train 3D Gaussian splatting model
  6. Convert .ply → .glb
  7. Upload GLB + thumbnail to Cloudinary
  8. PATCH job status back to the API via HTTP

Setup on Vast.ai instance:
  1. SSH into the instance
  2. git clone your repo
  3. pip install -r requirements.vast.txt
  4. cp .env.vast .env && fill in values
  5. python vast_worker.py
"""

import os
import sys
import time
import json
import shutil
import signal
import requests
import subprocess
import cloudinary
import cloudinary.uploader
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL        = os.environ["API_BASE_URL"]          # your Railway API URL
WORKER_SECRET       = os.environ["WORKER_SECRET"]         # shared secret for auth
CLOUDINARY_CLOUD    = os.environ["CLOUDINARY_CLOUD_NAME"]
CLOUDINARY_KEY      = os.environ["CLOUDINARY_API_KEY"]
CLOUDINARY_SECRET   = os.environ["CLOUDINARY_API_SECRET"]

POLL_INTERVAL       = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
WORK_DIR            = Path(os.getenv("WORK_DIR", "/workspace"))
GAUSSIAN_REPO       = Path("/gaussian-splatting")
ESRGAN_SCRIPT       = Path("/Real-ESRGAN/inference_realesrgan.py")

# ─── Cloudinary ───────────────────────────────────────────────────────────────
cloudinary.config(
    cloud_name = CLOUDINARY_CLOUD,
    api_key    = CLOUDINARY_KEY,
    api_secret = CLOUDINARY_SECRET,
    secure     = True,
)

# ─── Graceful shutdown ────────────────────────────────────────────────────────
running = True
def handle_signal(sig, frame):
    global running
    print("\n[Worker] Shutdown signal received — finishing current job then stopping...")
    running = False

signal.signal(signal.SIGINT,  handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# ─── API helpers ──────────────────────────────────────────────────────────────
HEADERS = {
    "Content-Type":  "application/json",
    "X-Worker-Secret": WORKER_SECRET,
}

def api_patch_status(job_id, status, progress_pct=None, output=None, error=None):
    """Report job progress/completion back to the Node.js API."""
    payload = { "status": status }
    if progress_pct is not None: payload["progressPct"]  = progress_pct
    if output is not None:       payload["output"]       = output
    if error is not None:        payload["error"]        = error

    try:
        res = requests.patch(
            f"{API_BASE_URL}/api/jobs/{job_id}/worker-update",
            json    = payload,
            headers = HEADERS,
            timeout = 15,
        )
        res.raise_for_status()
        print(f"[{job_id}] Status updated → {status} ({progress_pct}%)")
    except Exception as e:
        print(f"[{job_id}] WARNING: Could not update status: {e}", file=sys.stderr)

def api_poll_next_job():
    """Ask the API for the next queued job. Returns job dict or None."""
    try:
        res = requests.post(
            f"{API_BASE_URL}/api/jobs/worker-dequeue",
            headers = HEADERS,
            timeout = 10,
        )
        if res.status_code == 204:
            return None   # queue empty
        res.raise_for_status()
        return res.json().get("job")
    except Exception as e:
        print(f"[Worker] Poll error: {e}", file=sys.stderr)
        return None

# ─── Main processing pipeline ─────────────────────────────────────────────────
def process_job(job):
    job_id      = job["jobId"]
    input_files = job["inputFiles"]
    input_type  = job["inputType"]
    settings    = job["settings"]
    enhance     = settings.get("enhanceImages", True)
    iterations  = quality_to_iterations(settings.get("quality", "balanced"))

    work = WORK_DIR / job_id
    work.mkdir(parents=True, exist_ok=True)

    try:
        print(f"\n[{job_id}] ═══ Starting pipeline ═══")
        print(f"[{job_id}] type={input_type} enhance={enhance} iter={iterations}")

        # ── Stage 1: Download inputs ───────────────────────────────────────────
        api_patch_status(job_id, "preprocessing", 10)
        raw_dir = work / "raw"
        raw_dir.mkdir(exist_ok=True)

        if input_type == "video":
            video_path = download_file(input_files[0]["url"], raw_dir / "input.mp4")
            images_dir = extract_frames(video_path, work / "frames", job_id)
        else:
            images_dir = raw_dir
            for i, f in enumerate(input_files):
                ext = Path(f["originalName"]).suffix or ".jpg"
                download_file(f["url"], raw_dir / f"{i:04d}{ext}")
            print(f"[{job_id}] Downloaded {len(input_files)} images")

        # ── Stage 2: Image enhancement ────────────────────────────────────────
        api_patch_status(job_id, "preprocessing", 20)
        if enhance and ESRGAN_SCRIPT.exists():
            print(f"[{job_id}] Enhancing with Real-ESRGAN...")
            enhanced_dir = work / "enhanced"
            enhanced_dir.mkdir(exist_ok=True)
            run_cmd([
                "python", str(ESRGAN_SCRIPT),
                "-i", str(images_dir),
                "-o", str(enhanced_dir),
                "--model_name", "RealESRGAN_x4plus",
                "--outscale", "2",
                "--fp32",
            ], job_id)
            images_dir = enhanced_dir
        else:
            print(f"[{job_id}] Skipping enhancement")

        # ── Stage 3: COLMAP ───────────────────────────────────────────────────
        api_patch_status(job_id, "training", 30)
        print(f"[{job_id}] Running COLMAP...")
        colmap_dir = work / "colmap"
        colmap_dir.mkdir(exist_ok=True)
        run_colmap(images_dir, colmap_dir, job_id)

        # ── Stage 4: Gaussian splatting training ──────────────────────────────
        api_patch_status(job_id, "training", 40)
        print(f"[{job_id}] Training ({iterations} iterations)...")
        output_dir = work / "output"
        run_cmd([
            "python", str(GAUSSIAN_REPO / "train.py"),
            "-s", str(colmap_dir),
            "-m", str(output_dir),
            "--iterations", str(iterations),
            "--densification_interval", "100",
            "--quiet",
        ], job_id)
        api_patch_status(job_id, "training", 80)

        # ── Stage 5: Convert .ply → .glb ─────────────────────────────────────
        api_patch_status(job_id, "converting", 85)
        print(f"[{job_id}] Converting to GLB...")
        ply_path = find_final_ply(output_dir, job_id)
        glb_path = work / "scene.glb"
        convert_ply_to_glb(ply_path, glb_path, job_id)

        # ── Stage 6: Generate thumbnail ───────────────────────────────────────
        thumbnail_path = generate_thumbnail(glb_path, work / "thumbnail.jpg", job_id)

        # ── Stage 7: Upload to Cloudinary ─────────────────────────────────────
        api_patch_status(job_id, "converting", 90)
        print(f"[{job_id}] Uploading to Cloudinary...")

        glb_result = cloudinary.uploader.upload(
            str(glb_path),
            resource_type = "raw",
            folder        = f"gaussian-outputs/{job_id}",
            public_id     = "scene",
            tags          = [f"job_{job_id}", "output"],
        )

        thumb_result = None
        if thumbnail_path and thumbnail_path.exists():
            thumb_result = cloudinary.uploader.upload(
                str(thumbnail_path),
                resource_type = "image",
                folder        = f"gaussian-outputs/{job_id}",
                public_id     = "thumbnail",
                tags          = [f"job_{job_id}", "output"],
            )

        # ── Stage 8: Report completion ────────────────────────────────────────
        output = {
            "glbCloudinaryId":       glb_result["public_id"],
            "glbSecureUrl":          glb_result["secure_url"],
            "thumbnailCloudinaryId": thumb_result["public_id"]  if thumb_result else None,
            "thumbnailSecureUrl":    thumb_result["secure_url"] if thumb_result else None,
            "fileSizeBytes":         glb_path.stat().st_size,
        }

        api_patch_status(job_id, "done", 100, output=output)
        print(f"[{job_id}] ✓ Done! GLB: {glb_result['secure_url']}")

    except Exception as e:
        error_msg = str(e)
        print(f"[{job_id}] ✗ FAILED: {error_msg}", file=sys.stderr)
        api_patch_status(job_id, "failed", error={
            "message": humanize_error(error_msg),
            "code":    parse_error_code(error_msg),
            "stage":   "processing",
        })

    finally:
        shutil.rmtree(work, ignore_errors=True)
        print(f"[{job_id}] Workspace cleaned up")


# ─── Main loop ────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Vast.ai GPU Worker — Gaussian Splatting")
    print(f"  API: {API_BASE_URL}")
    print(f"  Poll interval: {POLL_INTERVAL}s")
    print("=" * 50)

    while running:
        job = api_poll_next_job()

        if job:
            process_job(job)
        else:
            print(f"[Worker] Queue empty — waiting {POLL_INTERVAL}s...", end="\r")
            time.sleep(POLL_INTERVAL)

    print("[Worker] Stopped cleanly.")


# ─── Pipeline helpers ─────────────────────────────────────────────────────────

def download_file(url: str, dest: Path) -> Path:
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest


def extract_frames(video_path: Path, output_dir: Path, job_id: str, fps: int = 2) -> Path:
    output_dir.mkdir(exist_ok=True)
    run_cmd([
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(output_dir / "%04d.jpg"),
        "-y",
    ], job_id)
    count = len(list(output_dir.glob("*.jpg")))
    print(f"[{job_id}] Extracted {count} frames at {fps} fps")
    if count < 10:
        raise ValueError(f"Too few frames ({count}). Video may be too short.")
    return output_dir


def run_colmap(images_dir: Path, colmap_dir: Path, job_id: str):
    db = colmap_dir / "database.db"
    sparse = colmap_dir / "sparse"
    sparse.mkdir(exist_ok=True)

    run_cmd([
        "colmap", "feature_extractor",
        "--database_path", str(db),
        "--image_path",    str(images_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1",
    ], job_id)

    run_cmd([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db),
        "--SiftMatching.use_gpu", "1",
    ], job_id)

    run_cmd([
        "colmap", "mapper",
        "--database_path", str(db),
        "--image_path",    str(images_dir),
        "--output_path",   str(sparse),
    ], job_id)

    if not any(sparse.iterdir()):
        raise RuntimeError(
            "COLMAP failed to reconstruct the scene. "
            "Try images with more overlap and better lighting."
        )


def find_final_ply(output_dir: Path, job_id: str) -> Path:
    candidates = sorted(output_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
    if not candidates:
        raise FileNotFoundError(f"No .ply output found in {output_dir}")
    return candidates[-1]


def convert_ply_to_glb(ply_path: Path, glb_path: Path, job_id: str):
    """Convert Gaussian splatting .ply to .glb using Blender."""
    script = ply_path.parent / "convert.py"
    script.write_text(f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_mesh.ply(filepath='{ply_path}')
bpy.ops.export_scene.gltf(filepath='{glb_path}', export_format='GLB')
print("GLB export complete")
""")
    run_cmd(["blender", "--background", "--python", str(script)], job_id)
    if not glb_path.exists():
        raise FileNotFoundError("GLB conversion failed — output not found")


def generate_thumbnail(glb_path: Path, output_path: Path, job_id: str):
    try:
        script = glb_path.parent / "thumbnail.py"
        script.write_text(f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath='{glb_path}')
bpy.context.scene.render.filepath = '{output_path}'
bpy.context.scene.render.image_settings.file_format = 'JPEG'
bpy.context.scene.render.resolution_x = 512
bpy.context.scene.render.resolution_y = 512
bpy.ops.render.render(write_still=True)
""")
        run_cmd(["blender", "--background", "--python", str(script)], job_id)
        return output_path if output_path.exists() else None
    except Exception as e:
        print(f"[{job_id}] Thumbnail failed (non-fatal): {e}")
        return None


def run_cmd(cmd: list, job_id: str = ""):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}):\n"
            f"{result.stdout[-2000:]}"
        )
    return result.stdout


def quality_to_iterations(quality: str) -> int:
    return {"fast": 7_000, "balanced": 30_000, "high": 100_000}.get(quality, 30_000)


def parse_error_code(message: str) -> str:
    msg = message.lower()
    if "colmap" in msg or "sfm" in msg:         return "COLMAP_FAILED"
    if "out of memory" in msg or "oom" in msg:   return "GPU_OOM"
    if "too few" in msg:                          return "TOO_FEW_IMAGES"
    if "cuda" in msg:                             return "CUDA_ERROR"
    return "WORKER_ERROR"


def humanize_error(message: str) -> str:
    code = parse_error_code(message)
    return {
        "COLMAP_FAILED":   "Could not reconstruct 3D geometry. Try more overlapping images.",
        "GPU_OOM":         "GPU ran out of memory. Try 'fast' quality or fewer images.",
        "TOO_FEW_IMAGES":  "Not enough usable images. Upload at least 20 from different angles.",
        "CUDA_ERROR":      "A GPU error occurred. Please try again.",
        "WORKER_ERROR":    "Processing failed. Please try again.",
    }.get(code, "Processing failed. Please try again.")


if __name__ == "__main__":
    main()
