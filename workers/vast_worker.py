"""
Vast.ai GPU Worker — 3D Gaussian Splatting
===========================================
Pipeline:
  1. Download video / images
  2. Extract frames (ffmpeg) — original, unmodified
  3. COLMAP  — feature extraction, matching, mapping, undistortion
  4. Gaussian Splatting training
  5. Compress .ply for mobile
  6. Upload to Backblaze B2

No image enhancement — ESRGAN removed entirely.
COLMAP and training both use the same raw frames.
"""

import os
import sys
import time
import math
import shutil
import signal
import logging
import requests
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from b2sdk.v1 import B2Api, InMemoryAccountInfo

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
load_dotenv()

# ─── Logging — writes to both stdout AND /workspace/worker.log ────────────────
LOG_FILE = Path("/workspace/worker.log")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),          # visible in Vast.ai console
        logging.FileHandler(str(LOG_FILE), mode="a"), # persists on disk
    ]
)
log = logging.getLogger()

# Redirect print() to logger so all existing print() calls also go to file
class _PrintToLog:
    def write(self, msg):
        if msg.strip():
            log.info(msg.rstrip())
    def flush(self):
        pass

sys.stdout = _PrintToLog()
sys.stderr = _PrintToLog()

# ─── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL  = os.environ["API_BASE_URL"]
WORKER_SECRET = os.environ["WORKER_SECRET"]

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
WORK_DIR      = Path(os.getenv("WORK_DIR", "/workspace"))
GAUSSIAN_REPO = Path("/gaussian-splatting")
VOCAB_TREE    = Path("/colmap/vocab_tree_flickr100K_words256K.bin")

B2_KEY_ID      = os.environ["B2_KEY_ID"]
B2_APP_KEY     = os.environ["B2_APP_KEY"]
B2_BUCKET_NAME = os.environ["B2_BUCKET_NAME"]

# ─── Quality profiles ─────────────────────────────────────────────────────────
#            fps  max_frames  iterations  grad_thresh  densify_until  min_opacity  mobile_target
QUALITY_PROFILES = {
    "fast":     ( 3,  100,        20_000,     0.0002,      10_000,        0.005,       500_000),
    "balanced": ( 4,  150,        50_000,     0.0001,      28_000,        0.004,       800_000),
    "high":     ( 5,  200,        80_000,     0.0001,      30_000,        0.003,     1_000_000),
}

# ─── Graceful shutdown ────────────────────────────────────────────────────────
running = True
def handle_signal(sig, frame):
    global running
    print("\n[Worker] Shutdown signal — finishing current job then stopping...")
    running = False

signal.signal(signal.SIGINT,  handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# ─── B2 ───────────────────────────────────────────────────────────────────────
def get_b2_bucket():
    info = InMemoryAccountInfo()
    api  = B2Api(info)
    api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
    return api.get_bucket_by_name(B2_BUCKET_NAME)

def upload_to_b2(file_path: Path, job_id: str) -> dict:
    bucket    = get_b2_bucket()
    file_name = f"outputs/{job_id}_scene.ply"
    print(f"[{job_id}] Uploading {file_path.stat().st_size/1024/1024:.1f}MB to B2...")
    file_info = bucket.upload_local_file(
        local_file=str(file_path),
        file_name=file_name,
        content_type="application/octet-stream",
    )
    return {"fileId": file_info.id_, "downloadUrl": bucket.get_download_url(file_name)}

def upload_thumbnail_to_b2(file_path: Path, job_id: str) -> dict:
    try:
        bucket    = get_b2_bucket()
        file_name = f"outputs/{job_id}/thumbnail.jpg"
        print(f"[{job_id}] Uploading thumbnail ({file_path.stat().st_size/1024:.0f}KB)...")
        file_info = bucket.upload_local_file(
            local_file=str(file_path),
            file_name=file_name,
            content_type="image/jpeg",
        )
        return {"fileId": file_info.id_, "downloadUrl": bucket.get_download_url(file_name)}
    except Exception as e:
        print(f"[{job_id}] Thumbnail upload failed (non-fatal): {e}", file=sys.stderr)
        return None

# ─── API helpers ──────────────────────────────────────────────────────────────
HEADERS = {"Content-Type": "application/json", "X-Worker-Secret": WORKER_SECRET}

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
        print(f"[{job_id}] Status → {status} ({progress_pct}%)")
    except Exception as e:
        print(f"[{job_id}] WARNING: status update failed: {e}", file=sys.stderr)

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

# ─── Pipeline ─────────────────────────────────────────────────────────────────
def process_job(job):
    job_id     = job["jobId"]
    input_files= job["inputFiles"]
    input_type = job["inputType"]
    settings   = job["settings"]
    quality    = settings.get("quality", "balanced")

    fps, max_frames, iterations, grad_thresh, densify_until, min_opacity, mobile_target = \
        QUALITY_PROFILES.get(quality, QUALITY_PROFILES["balanced"])

    work = WORK_DIR / job_id
    work.mkdir(parents=True, exist_ok=True)

    # CUDA memory fragmentation fix.
    # expandable_segments is NOT used — it causes an internal assert failure
    # in older PyTorch builds (confirmed crash at CUDACachingAllocator.cpp:2549).
    # max_split_size_mb + garbage_collection_threshold are safe on all versions.
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        "max_split_size_mb:64,"
        "garbage_collection_threshold:0.7"
    )

    try:
        print(f"\n[{job_id}] ═══ Starting pipeline ═══")
        print(f"[{job_id}] type={input_type}  quality={quality}  iter={iterations}")

        # ── Stage 1: Download ──────────────────────────────────────────────────
        api_patch_status(job_id, "preprocessing", 10)
        raw_dir = work / "raw"
        raw_dir.mkdir(exist_ok=True)

        if input_type == "video":
            video_path = download_file(input_files[0]["url"], raw_dir / "input.mp4")
            images_dir = extract_frames(video_path, work / "frames", job_id,
                                        fps=fps, max_frames=max_frames)
        else:
            images_dir = raw_dir
            for i, f in enumerate(input_files):
                ext = Path(f["originalName"]).suffix or ".jpg"
                download_file(f["url"], raw_dir / f"{i:04d}{ext}")
            print(f"[{job_id}] Downloaded {len(input_files)} images")

        # ── Stage 2: COLMAP ────────────────────────────────────────────────────
        api_patch_status(job_id, "training", 30)
        print(f"[{job_id}] Running COLMAP...")
        colmap_dir = work / "colmap"
        colmap_dir.mkdir(exist_ok=True)
        colmap_out = run_colmap(images_dir, colmap_dir, job_id,
                                is_video=(input_type == "video"))

        # ── Stage 3: Gaussian Splatting training ───────────────────────────────
        api_patch_status(job_id, "training", 40)
        print(f"[{job_id}] Training ({iterations} iters, grad_thresh={grad_thresh})...")
        output_dir = work / "output"

        ckpt_iters = list(range(5_000, iterations, 5_000))
        run_cmd([
            "python3", str(GAUSSIAN_REPO / "train.py"),
            "-s", str(colmap_out),
            "-m", str(output_dir),
            "--iterations",              str(iterations),
            "--save_iterations",         str(iterations),
            "--test_iterations",         "-1",
            "--densification_interval",  "100",
            "--densify_until_iter",      str(densify_until),
            "--densify_grad_threshold",  str(grad_thresh),
            "--opacity_reset_interval",  "3000",
            "--sh_degree",               "3",
            "--checkpoint_iterations",   *[str(i) for i in ckpt_iters],
            "--quiet",
        ], job_id)
        api_patch_status(job_id, "training", 80)

        # ── Stage 4: Compress .ply for mobile ─────────────────────────────────
        api_patch_status(job_id, "converting", 85)
        ply_path        = find_final_ply(output_dir, job_id)
        compressed_path = ply_path.parent / "point_cloud_mobile.ply"
        ply_path        = compress_ply(ply_path, compressed_path, job_id,
                                       min_opacity=min_opacity,
                                       mobile_target=mobile_target)

        # ── Stage 5: Upload ────────────────────────────────────────────────────
        api_patch_status(job_id, "converting", 90)
        b2_result = upload_to_b2(ply_path, job_id)

        # ── Stage 6: Render and upload thumbnail ────────────────────────────
        api_patch_status(job_id, "converting", 95)
        thumbnail_info = None
        thumbnail_path = work / "thumbnail.jpg"
        if render_thumbnail(output_dir, thumbnail_path, job_id):
            thumbnail_info = upload_thumbnail_to_b2(thumbnail_path, job_id)

        # ── Complete ───────────────────────────────────────────────────────
        output_data = {
            "glbB2Id":        b2_result["fileId"],
            "glbDownloadUrl": b2_result["downloadUrl"],
            "fileSizeBytes":  ply_path.stat().st_size,
        }
        if thumbnail_info:
            output_data["thumbnailB2Id"]       = thumbnail_info["fileId"]
            output_data["thumbnailDownloadUrl"] = thumbnail_info["downloadUrl"]

        api_patch_status(job_id, "done", 100, output=output_data)
        print(f"[{job_id}] ✓ Done → {b2_result['downloadUrl']}")


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
        print(f"[{job_id}] Workspace cleaned")


# ─── Main loop ────────────────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Vast.ai GPU Worker — Gaussian Splatting")
    print(f"  API: {API_BASE_URL}   Poll: {POLL_INTERVAL}s")
    print("=" * 50)
    run_startup_diagnostics()
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


def extract_frames(video_path: Path, output_dir: Path, job_id: str,
                   fps: int = 3, max_frames: int = 100) -> Path:
    output_dir.mkdir(exist_ok=True)
    run_cmd([
        "ffmpeg", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "1",        # highest JPEG quality — no lossy compression artifacts
        str(output_dir / "%04d.jpg"),
        "-y",
    ], job_id)

    frames = sorted(output_dir.glob("*.jpg"))
    count  = len(frames)
    print(f"[{job_id}] Extracted {count} frames at {fps} fps")

    if count > max_frames:
        keep = set(round(i * (count-1) / (max_frames-1)) for i in range(max_frames))
        for i, f in enumerate(frames):
            if i not in keep:
                f.unlink()
        print(f"[{job_id}] Subsampled to {max_frames} frames")

    remaining = len(list(output_dir.glob("*.jpg")))
    if remaining < 20:
        raise ValueError(f"Too few frames ({remaining}). Video may be too short.")
    return output_dir


def run_colmap(images_dir: Path, colmap_dir: Path, job_id: str,
               is_video: bool = False) -> Path:
    db     = colmap_dir / "database.db"
    sparse = colmap_dir / "sparse"
    sparse.mkdir(exist_ok=True)
    dense  = colmap_dir / "sparse_undistorted"

    sift_gpu = "1" if _cuda_available() else "0"

    # ── Feature extraction ────────────────────────────────────────────────────
    run_cmd([
        "colmap", "feature_extractor",
        "--database_path",                   str(db),
        "--image_path",                      str(images_dir),
        "--ImageReader.single_camera",       "1",
        "--ImageReader.camera_model",        "OPENCV",
        "--SiftExtraction.use_gpu",          sift_gpu,
        "--SiftExtraction.max_num_features", "16384",
        "--SiftExtraction.peak_threshold",   "0.003",
        "--SiftExtraction.edge_threshold",   "10",
        "--SiftExtraction.num_octaves",      "4",
    ], job_id)

    # ── Feature matching ──────────────────────────────────────────────────────
    match_gpu  = "1" if _cuda_available() else "0"
    has_vocab  = VOCAB_TREE.exists()
    image_count = len(list(images_dir.glob("*.*")))

    if is_video:
        cmd = [
            "colmap", "sequential_matcher",
            "--database_path",                            str(db),
            "--SiftMatching.use_gpu",                     match_gpu,
            "--SiftMatching.max_ratio",                   "0.85",
            "--SiftMatching.max_num_matches",             "32768",
            "--SequentialMatching.overlap",               "30",   # was 20 — wider window survives fast panning
            "--SequentialMatching.quadratic_overlap",     "1",    # also match at 2x,4x,8x gaps — bridges corners
            "--SequentialMatching.loop_detection",        "1" if has_vocab else "0",
            "--SequentialMatching.loop_detection_period", "5",    # was 10 — detect loops more frequently
            "--SequentialMatching.loop_detection_num_images", "80",  # was 50 — wider loop search
        ]
        if has_vocab:
            cmd += ["--SequentialMatching.vocab_tree_path", str(VOCAB_TREE)]
        else:
            print(f"[{job_id}] WARNING: vocab tree missing — loop detection disabled")
        run_cmd(cmd, job_id)
    elif image_count <= 150:
        run_cmd([
            "colmap", "exhaustive_matcher",
            "--database_path",        str(db),
            "--SiftMatching.use_gpu", match_gpu,
            "--SiftMatching.max_ratio",       "0.85",
            "--SiftMatching.max_num_matches", "32768",
        ], job_id)
    else:
        if has_vocab:
            run_cmd([
                "colmap", "vocab_tree_matcher",
                "--database_path",                         str(db),
                "--SiftMatching.use_gpu",                  match_gpu,
                "--VocabTreeMatching.vocab_tree_path",     str(VOCAB_TREE),
            ], job_id)
        else:
            run_cmd([
                "colmap", "exhaustive_matcher",
                "--database_path",        str(db),
                "--SiftMatching.use_gpu", match_gpu,
            ], job_id)

    # ── Mapper ────────────────────────────────────────────────────────────────
    run_cmd([
        "colmap", "mapper",
        "--database_path",                       str(db),
        "--image_path",                          str(images_dir),
        "--output_path",                         str(sparse),
        "--Mapper.ba_global_max_num_iterations", "50",
        "--Mapper.tri_min_angle",                "2.0",
    ], job_id)

    if not any(sparse.iterdir()):
        raise RuntimeError(
            "COLMAP reconstruction failed — no sparse model produced. "
            "Walk slowly with lots of overlap and good lighting."
        )

    _log_colmap_stats(sparse, job_id)

    # Merge fragmented sub-models if COLMAP produced more than one
    sub_models = sorted(sparse.iterdir())
    if len(sub_models) > 1:
        print(f"[{job_id}] WARNING: {len(sub_models)} sub-models — attempting merge...")
        _merge_colmap_models(sparse, sub_models, job_id)

    # ── Undistortion ──────────────────────────────────────────────────────────
    # Always use sparse/0 — _merge_colmap_models renames the result to 0
    dense.mkdir(exist_ok=True)
    model_dir = sparse / "0"
    if not model_dir.exists():
        model_dir = next(iter(sorted(sparse.iterdir())))
    run_cmd([
        "colmap", "image_undistorter",
        "--image_path",  str(images_dir),
        "--input_path",  str(model_dir),
        "--output_path", str(dense),
        "--output_type", "COLMAP",
    ], job_id)

    # Fix sparse subdir structure: train.py expects sparse/0/cameras.bin
    undist_sparse = dense / "sparse"
    target_0      = undist_sparse / "0"
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
    ply = candidates[-1]
    print(f"[{job_id}] Found .ply: {ply} ({ply.stat().st_size/1024/1024:.1f}MB)")
    return ply


def compress_ply(input_path: Path, output_path: Path, job_id: str,
                 min_opacity: float = 0.004,
                 mobile_target: int = 800_000) -> Path:
    """
    Mobile optimisation:
      1. Opacity prune   — remove invisible floaters
      2. Size prune      — remove oversized background blobs
      3. Importance sort — keep highest quality Gaussians up to mobile_target
      4. float16 quantise — halve non-position property size
    """
    try:
        import numpy as np
        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(str(input_path))
        vertex  = plydata["vertex"]
        data    = {p.name: vertex[p.name] for p in vertex.properties}
        n0      = len(data["x"])
        print(f"[{job_id}] Mobile optimise: {n0:,} Gaussians input")

        # Step 1: opacity prune
        if "opacity" in data:
            thresh = math.log(min_opacity / (1.0 - min_opacity))
            mask   = data["opacity"] > thresh
            data   = {k: v[mask] for k, v in data.items()}
            print(f"[{job_id}]   After opacity prune: {len(data['x']):,}")

        # Step 2: size prune — remove top 5% largest Gaussians
        scale_keys = sorted(k for k in data if k.startswith("scale_"))
        if scale_keys:
            scales    = np.stack([np.exp(data[k]) for k in scale_keys], axis=1)
            max_scale = scales.max(axis=1)
            thresh_sz = np.percentile(max_scale, 95)
            mask      = max_scale < thresh_sz
            data      = {k: v[mask] for k, v in data.items()}
            print(f"[{job_id}]   After size prune:    {len(data['x']):,}")

        # Step 3: importance sort + cap
        n_now = len(data["x"])
        if n_now > mobile_target:
            if "opacity" in data and scale_keys:
                opacity_sig = 1.0 / (1.0 + np.exp(-data["opacity"]))
                scales      = np.stack([np.exp(data[k]) for k in scale_keys], axis=1)
                importance  = opacity_sig / (scales.max(axis=1) + 1e-6)
            elif "opacity" in data:
                importance  = 1.0 / (1.0 + np.exp(-data["opacity"]))
            else:
                importance  = np.ones(n_now)

            top_idx = np.argpartition(importance, -mobile_target)[-mobile_target:]
            top_idx = np.sort(top_idx)
            data    = {k: v[top_idx] for k, v in data.items()}
            print(f"[{job_id}]   After mobile cap:    {len(data['x']):,}")

        # Step 4: float16 quantise (keep xyz as float32)
        pos    = {"x", "y", "z"}
        arrays = [
            (n, a if n in pos else
             (a.astype(np.float16) if a.dtype == np.float32 else a))
            for n, a in data.items()
        ]
        count   = len(arrays[0][1])
        dtype   = [(n, a.dtype) for n, a in arrays]
        new_arr = np.zeros(count, dtype=dtype)
        for n, a in arrays:
            new_arr[n] = a

        PlyData([PlyElement.describe(new_arr, "vertex")], text=False).write(str(output_path))
        orig_mb = input_path.stat().st_size  / 1024 / 1024
        comp_mb = output_path.stat().st_size / 1024 / 1024
        print(f"[{job_id}] ✓ {orig_mb:.0f}MB → {comp_mb:.0f}MB | {count:,} Gaussians")
        return output_path

    except Exception as e:
        print(f"[{job_id}] Compression failed — using original: {e}")
        return input_path


def render_thumbnail(model_dir: Path, output_path: Path, job_id: str) -> Path:
    """
    Render a preview thumbnail from the trained Gaussian Splatting model.
    Uses the render.py script from the gaussian-splatting repo.
    """
    try:
        render_py = GAUSSIAN_REPO / "render.py"
        if not render_py.exists():
            print(f"[{job_id}] render.py not found — skipping thumbnail")
            return None

        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Run render.py with specific parameters
        # This renders a single viewpoint as JPEG for preview
        run_cmd([
            "python3", str(render_py),
            "-m", str(model_dir),
            "-o", str(output_path.parent),
            "--quiet",
        ], job_id)

        # If render.py created a default output, use it
        if output_path.exists():
            print(f"[{job_id}] ✓ Thumbnail rendered ({output_path.stat().st_size/1024:.0f}KB)")
            return output_path

        # Otherwise look for any PNG/JPEG in the output directory
        for ext in ["*.png", "*.jpg", "*.jpeg"]:
            candidates = sorted(output_path.parent.glob(ext))
            if candidates:
                img_path = candidates[0]
                # Convert PNG to JPEG if needed
                if img_path.suffix.lower() == ".png":
                    import subprocess as sp
                    sp.run([
                        "convert", str(img_path), 
                        "-quality", "90",
                        str(output_path)
                    ], check=True)
                    img_path.unlink()
                    print(f"[{job_id}] ✓ Thumbnail rendered ({output_path.stat().st_size/1024:.0f}KB)")
                    return output_path
                else:
                    img_path.rename(output_path)
                    print(f"[{job_id}] ✓ Thumbnail rendered ({output_path.stat().st_size/1024:.0f}KB)")
                    return output_path

        print(f"[{job_id}] WARNING: render.py did not produce output")
        return None

    except Exception as e:
        print(f"[{job_id}] Thumbnail rendering failed (non-fatal): {e}", file=sys.stderr)
        return None


def run_cmd(cmd: list, job_id: str = "") -> str:
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode}):\n{result.stdout[-3000:]}"
        )
    return result.stdout


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _log_colmap_stats(sparse_dir: Path, job_id: str):
    try:
        import struct
        total_images = 0
        total_points = 0
        for model_dir in sorted(sparse_dir.iterdir()):
            images_bin = model_dir / "images.bin"
            points_bin = model_dir / "points3D.bin"
            if images_bin.exists():
                with open(images_bin, "rb") as f:
                    total_images += struct.unpack("<Q", f.read(8))[0]
            if points_bin.exists():
                with open(points_bin, "rb") as f:
                    total_points += struct.unpack("<Q", f.read(8))[0]
        print(f"[{job_id}] COLMAP: {total_images} images registered, "
              f"{total_points:,} 3D points")
        if total_images < 50:
            print(f"[{job_id}] ⚠ WARNING: only {total_images} images registered — "
                  f"reconstruction will be partial. Re-shoot more slowly.")
    except Exception as e:
        print(f"[{job_id}] Could not read COLMAP stats: {e}")


def _merge_colmap_models(sparse_dir: Path, sub_models: list, job_id: str):
    try:
        merged = sparse_dir / "merged"
        merged.mkdir(exist_ok=True)
        run_cmd([
            "colmap", "model_merger",
            "--input_path1", str(sub_models[0]),
            "--input_path2", str(sub_models[1]),
            "--output_path", str(merged),
        ], job_id)
        target = sparse_dir / "0"
        if target.exists():
            shutil.rmtree(target)
        merged.rename(target)
        print(f"[{job_id}] Models merged successfully")
    except Exception as e:
        print(f"[{job_id}] Merge failed ({e}) — using largest sub-model")
        largest = max(sub_models, key=lambda p: sum(1 for _ in p.glob("*.bin")))
        target  = sparse_dir / "0"
        if largest != target:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(largest, target)


def _ensure_virtual_display():
    if os.environ.get("DISPLAY"):
        return
    try:
        import subprocess as sp
        sp.Popen(["Xvfb", ":99", "-screen", "0", "1024x768x24"],
                 stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        os.environ["DISPLAY"] = ":99"
        time.sleep(1)
        print("[COLMAP] Started Xvfb on :99")
    except FileNotFoundError:
        pass


def parse_error_code(message: str) -> str:
    msg = message.lower()
    if "colmap" in msg or "sfm" in msg:        return "COLMAP_FAILED"
    if "out of memory" in msg or "oom" in msg: return "GPU_OOM"
    if "too few" in msg:                       return "TOO_FEW_IMAGES"
    if "cuda" in msg:                          return "CUDA_ERROR"
    return "WORKER_ERROR"

def humanize_error(message: str) -> str:
    return {
        "COLMAP_FAILED":  "Could not reconstruct 3D geometry. Walk slowly with lots of overlap.",
        "GPU_OOM":        "GPU ran out of memory. Try 'balanced' quality.",
        "TOO_FEW_IMAGES": "Not enough usable frames. Upload a longer video.",
        "CUDA_ERROR":     "A GPU error occurred. Please try again.",
        "WORKER_ERROR":   "Processing failed. Please try again.",
    }.get(parse_error_code(message), "Processing failed. Please try again.")


def run_startup_diagnostics():
    print("\n" + "="*50)
    print("  STARTUP DIAGNOSTICS")
    print("="*50)

    for tool in ["colmap", "ffmpeg", "python3"]:
        try:
            subprocess.run([tool, "--help"], capture_output=True, timeout=10)
            print(f"[DIAG] ✓ {tool} found")
        except FileNotFoundError:
            print(f"[DIAG] ✗ {tool} NOT FOUND — jobs will fail")
        except Exception:
            print(f"[DIAG] ✓ {tool} found")

    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            vram  = props.total_memory / 1024**3
            print(f"[DIAG] ✓ CUDA — {props.name} ({vram:.0f}GB VRAM)")
            if vram < 20:
                print(f"[DIAG] ⚠ <20GB VRAM — high quality may OOM")
        else:
            print("[DIAG] ✗ CUDA unavailable — training will fail")
    except Exception as e:
        print(f"[DIAG] ✗ torch: {e}")

    train_py = GAUSSIAN_REPO / "train.py"
    if train_py.exists():
        print(f"[DIAG] ✓ gaussian-splatting train.py found")
    else:
        print(f"[DIAG] ✗ train.py NOT FOUND at {train_py}")

    if VOCAB_TREE.exists():
        print(f"[DIAG] ✓ Vocab tree ({VOCAB_TREE.stat().st_size//1024//1024}MB)")
    else:
        print(f"[DIAG] ⚠ Vocab tree missing — loop detection disabled")

    print("="*50 + "\n")


if __name__ == "__main__":
    main()