"""
Vast.ai GPU Worker — 3D Gaussian Splatting
==========================================
"""

import os
import sys
import time
import shutil
import signal
import requests
import subprocess
import cloudinary
import cloudinary.uploader
from pathlib import Path
from dotenv import load_dotenv

# ─── Headless display fix ─────────────────────────────────────────────────────
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL     = os.environ["API_BASE_URL"]
WORKER_SECRET    = os.environ["WORKER_SECRET"]
CLOUDINARY_CLOUD = os.environ["CLOUDINARY_CLOUD_NAME"]
CLOUDINARY_KEY   = os.environ["CLOUDINARY_API_KEY"]
CLOUDINARY_SECRET = os.environ["CLOUDINARY_API_SECRET"]

POLL_INTERVAL  = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
WORK_DIR       = Path(os.getenv("WORK_DIR", "/workspace"))
GAUSSIAN_REPO  = Path("/gaussian-splatting")
ESRGAN_SCRIPT  = Path("/Real-ESRGAN/inference_realesrgan.py")

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
    "Content-Type":    "application/json",
    "X-Worker-Secret": WORKER_SECRET,
}

def api_patch_status(job_id, status, progress_pct=None, output=None, error=None):
    payload = {"status": status}
    if progress_pct is not None: payload["progressPct"] = progress_pct
    if output is not None:       payload["output"]      = output
    if error is not None:        payload["error"]       = error
    try:
        res = requests.patch(
            f"{API_BASE_URL}/api/jobs/{job_id}/worker-update",
            json=payload, headers=HEADERS, timeout=15,
        )
        res.raise_for_status()
        print(f"[{job_id}] Status updated → {status} ({progress_pct}%)")
    except Exception as e:
        print(f"[{job_id}] WARNING: Could not update status: {e}", file=sys.stderr)

def api_poll_next_job():
    try:
        res = requests.post(
            f"{API_BASE_URL}/api/jobs/worker-dequeue",
            headers=HEADERS, timeout=10,
        )
        if res.status_code == 204:
            return None
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
                "python3", str(ESRGAN_SCRIPT),   # python3 not python
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
        colmap_out = run_colmap(images_dir, colmap_dir, job_id)

        # ── Stage 4: Gaussian splatting training ──────────────────────────────
        api_patch_status(job_id, "training", 40)
        print(f"[{job_id}] Training ({iterations} iterations)...")
        output_dir = work / "output"
        run_cmd([
            "python3", str(GAUSSIAN_REPO / "train.py"),   # python3 not python
            "-s", str(colmap_out),
            "-m", str(output_dir),
            "--iterations", str(iterations),
            "--densification_interval", "100",
            "--quiet",
        ], job_id)
        api_patch_status(job_id, "training", 80)

        # ── Stage 5: Upload .ply to Cloudinary ───────────────────────────────
        api_patch_status(job_id, "converting", 85)
        print(f"[{job_id}] Finding .ply output...")
        ply_path = find_final_ply(output_dir, job_id)

        api_patch_status(job_id, "converting", 90)
        print(f"[{job_id}] Compressing .ply...")
        ply_path = compress_ply(ply_path, work / "scene_compressed.ply", job_id)

        print(f"[{job_id}] Uploading .ply to Cloudinary ({ply_path.stat().st_size // 1024 // 1024}MB)...")
        ply_result = cloudinary.uploader.upload(
            str(ply_path),
            resource_type = "raw",
            folder        = f"gaussian-outputs/{job_id}",
            public_id     = "scene",
            tags          = [f"job_{job_id}", "output"],
        )

        output = {
            "glbCloudinaryId":       ply_result["public_id"],    # reusing field for ply
            "glbSecureUrl":          ply_result["secure_url"],   # reusing field for ply
            "thumbnailCloudinaryId": None,
            "thumbnailSecureUrl":    None,
            "fileSizeBytes":         ply_path.stat().st_size,
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

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
    db       = colmap_dir / "database.db"
    sparse   = colmap_dir / "sparse"
    sparse.mkdir(exist_ok=True)
    dense    = colmap_dir / "sparse_undistorted"

    # Force PINHOLE camera model — Gaussian splatting only accepts
    # PINHOLE or SIMPLE_PINHOLE (not OPENCV, RADIAL, etc.)
    run_cmd([
        "colmap", "feature_extractor",
        "--database_path",             str(db),
        "--image_path",                str(images_dir),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model",  "PINHOLE",
        "--SiftExtraction.use_gpu",    "0",
    ], job_id)

    run_cmd([
        "colmap", "exhaustive_matcher",
        "--database_path",        str(db),
        "--SiftMatching.use_gpu", "0",
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

    # Undistort images — converts to PINHOLE if any distortion model slipped in.
    # image_undistorter output structure:
    #   dense/
    #   ├── images/        ← undistorted images
    #   └── sparse/        ← cameras.bin, images.bin, points3D.bin
    dense.mkdir(exist_ok=True)
    model_dir = next(sparse.iterdir())   # e.g. sparse/0
    run_cmd([
        "colmap", "image_undistorter",
        "--image_path",   str(images_dir),
        "--input_path",   str(model_dir),
        "--output_path",  str(dense),
        "--output_type",  "COLMAP",
    ], job_id)

    # train.py expects: <source>/sparse/0/cameras.bin
    # image_undistorter outputs: <dense>/sparse/cameras.bin (no subdirectory)
    # So we move dense/sparse/ → dense/sparse/0/ to match expected structure
    undist_sparse = dense / "sparse"
    target_0 = undist_sparse / "0"
    if undist_sparse.exists() and not target_0.exists():
        target_0.mkdir()
        for f in list(undist_sparse.iterdir()):
            if f.name != "0":
                f.rename(target_0 / f.name)

    return dense


def find_final_ply(output_dir: Path, job_id: str) -> Path:
    candidates = sorted(output_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
    if not candidates:
        raise FileNotFoundError(f"No .ply output found in {output_dir}")
    return candidates[-1]


def compress_ply(input_path: Path, output_path: Path, job_id: str) -> Path:
    """
    Reduce .ply file size by:
    1. Keeping only the most opaque Gaussians (prune low-opacity ones)
    2. Quantizing float32 → float16 for color/scale/rotation properties
    This typically reduces file size by 50-70% with minimal visual quality loss.
    """
    try:
        import numpy as np
        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(str(input_path))
        vertex  = plydata["vertex"]
        data    = {prop.name: vertex[prop.name] for prop in vertex.properties}

        original_count = len(data[list(data.keys())[0]])

        # Prune low-opacity Gaussians — opacity is stored as logit
        # logit(0.05) ≈ -2.94, so keep anything above that threshold
        if "opacity" in data:
            opacity_mask  = data["opacity"] > -2.94   # > ~5% opacity
            kept          = int(opacity_mask.sum())
            print(f"[{job_id}] Pruning: {original_count} → {kept} Gaussians ({100*kept//original_count}% kept)")
            data = {k: v[opacity_mask] for k, v in data.items()}

        # Quantize non-position properties to float16 to halve their size
        position_props = {"x", "y", "z"}
        new_props = []
        arrays    = []
        for name, arr in data.items():
            if name not in position_props and arr.dtype == np.float32:
                arr = arr.astype(np.float16)
            new_props.append((name, arr.dtype.str))
            arrays.append(arr)

        # Rebuild structured array
        dtype   = [(name, arr.dtype) for name, arr in zip(data.keys(), arrays)]
        count   = len(arrays[0])
        new_arr = np.zeros(count, dtype=dtype)
        for name, arr in zip(data.keys(), arrays):
            new_arr[name] = arr

        el  = PlyElement.describe(new_arr, "vertex")
        out = PlyData([el], text=False)
        out.write(str(output_path))

        orig_mb = input_path.stat().st_size  / 1024 / 1024
        comp_mb = output_path.stat().st_size / 1024 / 1024
        print(f"[{job_id}] Compressed: {orig_mb:.1f}MB → {comp_mb:.1f}MB")
        return output_path

    except Exception as e:
        print(f"[{job_id}] Compression failed (using original): {e}")
        return input_path   # fall back to original if compression fails


def run_cmd(cmd: list, job_id: str = ""):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
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
    if "colmap" in msg or "sfm" in msg:        return "COLMAP_FAILED"
    if "out of memory" in msg or "oom" in msg: return "GPU_OOM"
    if "too few" in msg:                        return "TOO_FEW_IMAGES"
    if "cuda" in msg:                           return "CUDA_ERROR"
    return "WORKER_ERROR"


def humanize_error(message: str) -> str:
    return {
        "COLMAP_FAILED":  "Could not reconstruct 3D geometry. Try more overlapping images.",
        "GPU_OOM":        "GPU ran out of memory. Try 'fast' quality or fewer images.",
        "TOO_FEW_IMAGES": "Not enough usable images. Upload at least 20 from different angles.",
        "CUDA_ERROR":     "A GPU error occurred. Please try again.",
        "WORKER_ERROR":   "Processing failed. Please try again.",
    }.get(parse_error_code(message), "Processing failed. Please try again.")


if __name__ == "__main__":
    main()