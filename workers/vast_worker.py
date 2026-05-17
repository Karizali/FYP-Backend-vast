"""
Vast.ai GPU Worker — 3D Gaussian Splatting
==========================================
Room-scale reconstruction fixes:
  - COLMAP: sequential+loop matcher instead of exhaustive, proper camera model
  - Training: indoor-tuned hyperparams, scene extent, white background off
  - Compression: smarter opacity threshold
"""

import os
import sys
import time
import shutil
import signal
import requests
import subprocess
import json
import base64
import traceback
from pathlib import Path
from dotenv import load_dotenv
from b2sdk.v1 import B2Api, InMemoryAccountInfo

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
API_BASE_URL  = os.environ["API_BASE_URL"]
WORKER_SECRET = os.environ["WORKER_SECRET"]

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL_SECONDS", "10"))
WORK_DIR      = Path(os.getenv("WORK_DIR", "/workspace"))
GAUSSIAN_REPO = Path("/gaussian-splatting")
ESRGAN_SCRIPT = Path("/Real-ESRGAN/inference_realesrgan.py")

B2_KEY_ID      = os.environ["B2_KEY_ID"]
B2_APP_KEY     = os.environ["B2_APP_KEY"]
B2_BUCKET_NAME = os.environ["B2_BUCKET_NAME"]

# ─── Per-quality tuning (room-scale optimised) ────────────────────────────────
#
# Key changes vs original:
#   • fps bumped up — denser frame sampling covers walls/ceiling better
#   • max_frames raised — more views = fewer holes in walls
#   • iterations increased — rooms have far more Gaussians needed than objects
#   • grad_thresh lowered — lets the trainer densify fine wall/floor detail
#   • densify_until raised — keep adding Gaussians well into training
#   • min_opacity added — prune floaters without killing real geometry
#
QUALITY_PROFILES = {
    #            fps  max_frames  iterations  grad_thresh  densify_until  min_opacity
    "fast":     ( 3,  80,         20_000,     0.0002,      12_000,        0.005),
    "balanced": ( 4,  120,        60_000,     0.0001,      35_000,        0.004),
    "high":     ( 5,  180,        100_000,    0.00005,     60_000,        0.003),
}

# ─── Backblaze B2 ─────────────────────────────────────────────────────────────
def get_b2_bucket():
    info = InMemoryAccountInfo()
    api  = B2Api(info)
    api.authorize_account("production", B2_KEY_ID, B2_APP_KEY)
    return api.get_bucket_by_name(B2_BUCKET_NAME)

def upload_to_b2(file_path: Path, job_id: str) -> dict:
    bucket    = get_b2_bucket()
    file_name = f"outputs/{job_id}_scene.ply"
    print(f"[{job_id}] Uploading {file_path.stat().st_size / 1024 / 1024:.1f}MB to B2...")
    file_info = bucket.upload_local_file(
        local_file=str(file_path),
        file_name=file_name,
        content_type="application/octet-stream",
    )
    return {
        "fileId":      file_info.id_,
        "downloadUrl": bucket.get_download_url(file_name),
    }

# ─── Graceful shutdown ────────────────────────────────────────────────────────
running = True
def handle_signal(sig, frame):
    global running
    print("\n[Worker] Shutdown signal — finishing current job then stopping...")
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

# ─── Main pipeline ────────────────────────────────────────────────────────────
def process_job(job):
    job_id      = job["jobId"]
    input_files = job["inputFiles"]
    input_type  = job["inputType"]
    settings    = job["settings"]
    enhance     = settings.get("enhanceImages", True)
    quality     = settings.get("quality", "balanced")

    fps, max_frames, iterations, grad_thresh, densify_until, min_opacity = QUALITY_PROFILES[quality]

    work = WORK_DIR / job_id
    work.mkdir(parents=True, exist_ok=True)

    # Prevent CUDA OOM on large scenes — 512 MB chunks is safe for 24 GB VRAM
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

    try:
        print(f"\n[{job_id}] ═══ Starting pipeline ═══")
        print(f"[{job_id}] type={input_type} quality={quality} enhance={enhance} iter={iterations}")

        # ── Stage 1: Download inputs ───────────────────────────────────────────
        api_patch_status(job_id, "preprocessing", 10)
        raw_dir = work / "raw"
        raw_dir.mkdir(exist_ok=True)

        if input_type == "video":
            video_path = download_file(input_files[0]["url"], raw_dir / "input.mp4")
            images_dir = extract_frames(video_path, work / "frames", job_id, fps=fps, max_frames=max_frames)
        else:
            images_dir = raw_dir
            for i, f in enumerate(input_files):
                ext = Path(f["originalName"]).suffix or ".jpg"
                download_file(f["url"], raw_dir / f"{i:04d}{ext}")
            print(f"[{job_id}] Downloaded {len(input_files)} images")

        # ── Stage 2: Image enhancement ────────────────────────────────────────
        api_patch_status(job_id, "preprocessing", 20)
        if enhance and ESRGAN_SCRIPT.exists():
            if not _cuda_available():
                print(f"[{job_id}] CUDA unavailable (driver mismatch?) — skipping ESRGAN to avoid CPU OOM")
            else:
                print(f"[{job_id}] Enhancing with Real-ESRGAN (batch mode)...")
                enhanced_dir = work / "enhanced"
                enhanced_dir.mkdir(exist_ok=True)
                esrgan_ok = _run_esrgan_batched(
                    images_dir, enhanced_dir, job_id, batch_size=10
                )
                if esrgan_ok:
                    images_dir = enhanced_dir
        else:
            print(f"[{job_id}] Skipping enhancement")

        # ── Stage 3: COLMAP ───────────────────────────────────────────────────
        api_patch_status(job_id, "training", 30)
        print(f"[{job_id}] Running COLMAP (room-scale settings)...")
        colmap_dir = work / "colmap"
        colmap_dir.mkdir(exist_ok=True)
        colmap_out = run_colmap(images_dir, colmap_dir, job_id, is_video=(input_type == "video"))

        # ── Stage 4: Gaussian splatting training ──────────────────────────────
        api_patch_status(job_id, "training", 40)
        print(f"[{job_id}] Training ({iterations} iters, grad_thresh={grad_thresh})...")
        output_dir = work / "output"

        run_cmd([
            "python3", str(GAUSSIAN_REPO / "train.py"),
            "-s", str(colmap_out),
            "-m", str(output_dir),
            # ── Iteration schedule ──────────────────────────────────────────
            "--iterations",              str(iterations),
            "--save_iterations",         str(iterations),
            "--test_iterations",         "-1",           # skip test renders — saves VRAM
            # ── Densification (critical for rooms) ─────────────────────────
            "--densification_interval",  "100",
            "--densify_until_iter",      str(densify_until),
            "--densify_grad_threshold",  str(grad_thresh),
            # ── Opacity / pruning ───────────────────────────────────────────
            "--opacity_reset_interval",  "3000",
            # --min_opacity is not a train.py flag in this repo version.
            # Floater pruning is handled post-training in compress_ply().
            # ── Scene scale (IMPORTANT for rooms) ──────────────────────────
            # Rooms are large; default cameras_extent can be too small causing
            # Gaussians to be clipped. Let train.py auto-compute from COLMAP.
            # Do NOT pass --scene_extent — rely on automatic calculation.
            # ── Background ─────────────────────────────────────────────────
            # Rooms have no "outside" — white_background causes bright halos
            # on walls. Use black (default) so wall/floor edges blend cleanly.
            # (Do NOT pass --white_background)
            # ── SH degree ───────────────────────────────────────────────────
            # sh_degree=3 captures view-dependent effects on glossy floors,
            # windows, and mirrors common in indoor scenes.
            "--sh_degree",               "3",
            "--quiet",
        ], job_id)
        api_patch_status(job_id, "training", 80)

        # ── Stage 5: Compress .ply ────────────────────────────────────────────
        api_patch_status(job_id, "converting", 85)
        ply_path = find_final_ply(output_dir, job_id)
        compressed_path = ply_path.parent / "point_cloud_compressed.ply"
        ply_path = compress_ply(ply_path, compressed_path, job_id, min_opacity=min_opacity)

        # ── Stage 6: Upload ───────────────────────────────────────────────────
        api_patch_status(job_id, "converting", 90)
        b2_result = upload_to_b2(ply_path, job_id)

        output = {
            "glbB2Id":        b2_result["fileId"],
            "glbDownloadUrl": b2_result["downloadUrl"],
            "fileSizeBytes":  ply_path.stat().st_size,
        }
        api_patch_status(job_id, "done", 100, output=output)
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
def run_startup_diagnostics():
    """
    Runs once at container start. Prints results to stdout so they appear
    in Vast.ai logs. Identifies ESRGAN/CUDA issues before any job runs.
    """
    import traceback
    print("\n" + "="*50)
    print("  STARTUP DIAGNOSTICS")
    print("="*50)

    # 1. basicsr patch
    try:
        import pathlib, inspect
        import basicsr.data.degradations as deg
        src = pathlib.Path(inspect.getfile(deg)).read_text()
        if "functional_tensor" in src:
            print("[DIAG] ✗ basicsr patch MISSING — functional_tensor still present")
            print("       ESRGAN will fail. Rebuild image with sed patch in Dockerfile.")
        else:
            print("[DIAG] ✓ basicsr patch OK")
    except Exception as e:
        print(f"[DIAG] ✗ basicsr import failed: {e}")

    # 2. CUDA
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[DIAG] ✓ CUDA OK — {torch.cuda.get_device_name(0)}")
        else:
            print("[DIAG] ✗ CUDA unavailable — ESRGAN will be skipped")
    except Exception as e:
        print(f"[DIAG] ✗ torch import failed: {e}")

    # 3. ESRGAN dry run on a synthetic image
    try:
        import subprocess, tempfile
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            tin  = Path(tmp) / "in";  tin.mkdir()
            tout = Path(tmp) / "out"; tout.mkdir()
            Image.new("RGB", (64, 64), color=(100, 100, 100)).save(tin / "test.jpg")
            r = subprocess.run(
                ["python3", str(ESRGAN_SCRIPT),
                 "-i", str(tin), "-o", str(tout),
                 "--model_name", "RealESRGAN_x4plus",
                 "--outscale", "2", "--fp32",
                 "--tile", "256", "--tile_pad", "16", "--pre_pad", "0"],
                capture_output=True, text=True, timeout=120
            )
            out_files = list(tout.iterdir())
            if r.returncode != 0:
                print(f"[DIAG] ✗ ESRGAN crashed (exit {r.returncode})")
                print(f"       stderr: {(r.stderr or r.stdout)[-500:]}")
            elif not out_files:
                print(f"[DIAG] ✗ ESRGAN ran but wrote 0 files")
                print(f"       stdout: {r.stdout[-500:]}")
            else:
                print(f"[DIAG] ✓ ESRGAN OK — wrote {out_files[0].name}")
    except Exception as e:
        print(f"[DIAG] ✗ ESRGAN test failed: {e}\n{traceback.format_exc()}")

    # 4. COLMAP
    try:
        r = subprocess.run(["colmap", "help"], capture_output=True, text=True)
        first_line = (r.stdout or r.stderr).splitlines()[0] if (r.stdout or r.stderr) else "no output"
        print(f"[DIAG] ✓ COLMAP OK — {first_line}")
    except Exception as e:
        print(f"[DIAG] ✗ COLMAP not found: {e}")

    # 5. Vocab tree
    vt = Path("/colmap/vocab_tree_flickr100K_words256K.bin")
    if vt.exists():
        print(f"[DIAG] ✓ Vocab tree OK ({vt.stat().st_size // 1024 // 1024}MB)")
    else:
        print(f"[DIAG] ✗ Vocab tree MISSING at {vt} — loop detection will be disabled")

    print("="*50 + "\n")


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

def _ensure_virtual_display():
    """
    Start a minimal Xvfb virtual display if available.
    COLMAP's GPU SIFT needs an OpenGL context even on headless servers.
    This is a best-effort helper — if Xvfb isn't installed it's a no-op
    because we already force use_gpu=0 for feature extraction.
    Sets DISPLAY=:99 so any subsequent OpenGL call can find it.
    """
    if os.environ.get("DISPLAY"):
        return  # already set
    try:
        subprocess.Popen(
            ["Xvfb", ":99", "-screen", "0", "1024x768x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        os.environ["DISPLAY"] = ":99"
        time.sleep(1)  # give Xvfb a moment to start
        print("[COLMAP] Started Xvfb virtual display on :99")
    except FileNotFoundError:
        pass  # Xvfb not installed — fine, we use CPU SIFT anyway


def _cuda_available() -> bool:
    """
    Check CUDA availability without importing torch at module level.
    Returns False if driver/HW mismatch (error 804) or no GPU present.
    """
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _run_esrgan_batched(images_dir: Path, enhanced_dir: Path,
                        job_id: str, batch_size: int = 10) -> bool:
    """
    Run ESRGAN in small batches to avoid OOM.
    Staging dirs live inside enhanced_dir to guarantee correct path resolution.
    Returns True if at least some frames were enhanced, False on total failure.
    """
    import shutil as _shutil

    # Only process actual image files — skip hidden files or directories
    all_frames = sorted(
        f for f in images_dir.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    total     = len(all_frames)
    succeeded = 0

    if total == 0:
        print(f"[{job_id}] WARNING: No image files found in {images_dir}")
        return False

    # Use a fixed tmp dir inside the work directory (same filesystem as images_dir)
    # so renames are atomic and paths are always predictable.
    tmp_root = enhanced_dir.parent / "_esrgan_tmp"
    tmp_root.mkdir(exist_ok=True)

    try:
        for batch_idx, batch_start in enumerate(range(0, total, batch_size)):
            batch     = all_frames[batch_start:batch_start + batch_size]
            batch_in  = tmp_root / f"in_{batch_idx}"
            batch_out = tmp_root / f"out_{batch_idx}"
            batch_in.mkdir(exist_ok=True)
            batch_out.mkdir(exist_ok=True)

            for f in batch:
                _shutil.copy2(f, batch_in / f.name)

            # Sanity check — make sure files actually landed
            staged = list(batch_in.iterdir())
            if not staged:
                print(f"[{job_id}] WARNING: Staging copy failed for batch {batch_idx}")
                for f in batch:
                    _shutil.copy2(f, enhanced_dir / f.name)
                continue

            try:
                output = run_cmd([
                    "python3", str(ESRGAN_SCRIPT),
                    "-i", str(batch_in),
                    "-o", str(batch_out),
                    "--model_name", "RealESRGAN_x4plus",
                    "--outscale", "2",
                    "--fp32",
                    "--tile", "256",        # process in 256×256 tiles — prevents OOM on 4K frames
                    "--tile_pad", "16",     # overlap between tiles to avoid seam artifacts
                    "--pre_pad", "0",
                ], job_id)
                # Log last 20 lines of ESRGAN output for diagnosis
                tail = "\n".join(output.strip().splitlines()[-20:])
                print(f"[{job_id}] ESRGAN stdout tail:\n{tail}")

                out_files = list(batch_out.iterdir())
                if not out_files:
                    raise RuntimeError(
                        f"ESRGAN ran without error but wrote 0 files "
                        f"(input had {len(staged)} files in {batch_in})\n"
                        f"ESRGAN output was:\n{tail}"
                    )
                for out_f in out_files:
                    out_f.rename(enhanced_dir / out_f.name)
                succeeded += len(out_files)
                pct = int(100 * min(batch_start + len(batch), total) / total)
                print(f"[{job_id}] ESRGAN batch {batch_idx+1}: "
                      f"{succeeded}/{total} frames ({pct}%)")

            except RuntimeError as e:
                print(f"[{job_id}] WARNING: ESRGAN batch {batch_idx+1} failed — "
                      f"using originals. Error: {e}")
                for f in batch:
                    _shutil.copy2(f, enhanced_dir / f.name)
            finally:
                _shutil.rmtree(batch_in,  ignore_errors=True)
                _shutil.rmtree(batch_out, ignore_errors=True)

    finally:
        _shutil.rmtree(tmp_root, ignore_errors=True)

    enhanced_count = len(list(enhanced_dir.iterdir()))
    if enhanced_count == 0:
        print(f"[{job_id}] WARNING: ESRGAN produced no output — using originals")
        return False

    print(f"[{job_id}] Enhanced {enhanced_count}/{total} frames total")
    return True


def download_file(url: str, dest: Path) -> Path:
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest


def extract_frames(video_path: Path, output_dir: Path, job_id: str,
                   fps: int = 3, max_frames: int = 120) -> Path:
    output_dir.mkdir(exist_ok=True)

    # Use scene_change detection + fixed fps together for better coverage.
    # select filter: emit frame if it is a scene-change OR every N seconds.
    # This ensures uniform coverage even in slow-panning room walkthroughs.
    select_filter = (
        f"select='isnan(prev_selected_t)+gte(t-prev_selected_t,1/{fps})',"
        f"setpts=N/FRAME_RATE/TB"
    )
    run_cmd([
        "ffmpeg", "-i", str(video_path),
        "-vf", select_filter,
        "-vsync", "vfr",
        "-q:v", "1",          # highest JPEG quality — COLMAP needs clean textures
        str(output_dir / "%04d.jpg"),
        "-y",
    ], job_id)

    frames = sorted(output_dir.glob("*.jpg"))
    count  = len(frames)
    print(f"[{job_id}] Extracted {count} frames at ~{fps} fps")

    # Uniform subsample if still over cap
    if count > max_frames:
        keep = set(round(i * (count - 1) / (max_frames - 1)) for i in range(max_frames))
        for i, f in enumerate(frames):
            if i not in keep:
                f.unlink()
        print(f"[{job_id}] Subsampled to {max_frames} frames")

    remaining = len(list(output_dir.glob("*.jpg")))
    if remaining < 20:
        raise ValueError(f"Too few frames ({remaining}). Video may be too short or featureless.")
    return output_dir


def run_colmap(images_dir: Path, colmap_dir: Path, job_id: str, is_video: bool = False):
    db     = colmap_dir / "database.db"
    sparse = colmap_dir / "sparse"
    sparse.mkdir(exist_ok=True)
    dense  = colmap_dir / "sparse_undistorted"

    # COLMAP built with EGL — GPU SIFT works headless via EGL context.
    # Fall back to CPU if CUDA unavailable (driver mismatch on some instances).
    sift_use_gpu = "1" if _cuda_available() else "0"
    _ensure_virtual_display()
    run_cmd([
        "colmap", "feature_extractor",
        "--database_path",                    str(db),
        "--image_path",                       str(images_dir),
        "--ImageReader.single_camera",        "1",
        "--ImageReader.camera_model",         "OPENCV",
        "--SiftExtraction.use_gpu",           sift_use_gpu,
        "--SiftExtraction.max_num_features",  "16384",
        "--SiftExtraction.peak_threshold",    "0.003",
        "--SiftExtraction.edge_threshold",    "10",
        "--SiftExtraction.num_octaves",       "4",
    ], job_id)

    # ── Feature matching ──────────────────────────────────────────────────────
    # For video: sequential matcher with large overlap window + loop detection.
    # For photos: exhaustive up to 120 images, vocab-tree beyond that.
    # The original always used exhaustive — fine for small sets, but misses
    # cross-room connections for video sequences.

    image_count = len(list(images_dir.glob("*")))

    # COLMAP is built with EGL enabled (see Dockerfile), so SiftGPU can create
    # a GPU OpenGL context with no display server. GPU matching is safe to use.
    match_use_gpu = "1" if _cuda_available() else "0"
    print(f"[{job_id}] Feature matching — GPU={match_use_gpu}")

    vocab_tree = Path("/colmap/vocab_tree_flickr100K_words256K.bin")
    has_vocab  = vocab_tree.exists()

    if is_video:
        cmd = [
            "colmap", "sequential_matcher",
            "--database_path",                          str(db),
            "--SiftMatching.use_gpu",                   match_use_gpu,
            "--SiftMatching.max_ratio",                 "0.85",
            "--SiftMatching.max_num_matches",           "32768",
            "--SequentialMatching.overlap",             "20",
            "--SequentialMatching.loop_detection",      "1" if has_vocab else "0",
            "--SequentialMatching.loop_detection_period","10",
            "--SequentialMatching.loop_detection_num_images", "50",
        ]
        if has_vocab:
            cmd += ["--SequentialMatching.vocab_tree_path", str(vocab_tree)]
        else:
            print(f"[{job_id}] WARNING: vocab tree missing — loop detection disabled")
        run_cmd(cmd, job_id)
    elif image_count <= 150:
        run_cmd([
            "colmap", "exhaustive_matcher",
            "--database_path",        str(db),
            "--SiftMatching.use_gpu", match_use_gpu,
            "--SiftMatching.max_ratio",       "0.85",
            "--SiftMatching.max_num_matches", "32768",
        ], job_id)
    else:
        vocab_tree_path = Path("/colmap/vocab_tree_flickr100K_words256K.bin")
        if not vocab_tree_path.exists():
            run_cmd([
                "colmap", "exhaustive_matcher",
                "--database_path",        str(db),
                "--SiftMatching.use_gpu", match_use_gpu,
            ], job_id)
        else:
            run_cmd([
                "colmap", "vocab_tree_matcher",
                "--database_path",          str(db),
                "--SiftMatching.use_gpu",   match_use_gpu,
                "--VocabTreeMatching.vocab_tree_path", str(vocab_tree_path),
            ], job_id)

    # ── Mapper ────────────────────────────────────────────────────────────────
    run_cmd([
        "colmap", "mapper",
        "--database_path",                     str(db),
        "--image_path",                        str(images_dir),
        "--output_path",                       str(sparse),
        # Allow more re-triangulation attempts for rooms where many points
        # are initially missed on textureless walls/ceilings
        "--Mapper.ba_global_max_num_iterations", "50",
        "--Mapper.tri_min_angle",              "2.0",  # lower than default — helps flat surfaces
    ], job_id)

    if not any(sparse.iterdir()):
        raise RuntimeError(
            "COLMAP reconstruction failed — no sparse model produced. "
            "Ensure video covers all walls with slow, overlapping sweeps."
        )

    # If COLMAP produced multiple sub-models (fragmented reconstruction),
    # merge them into model 0 so training uses the complete scene.
    sub_models = sorted(sparse.iterdir())
    if len(sub_models) > 1:
        print(f"[{job_id}] WARNING: COLMAP produced {len(sub_models)} sub-models — merging...")
        _merge_colmap_models(sparse, sub_models, db, images_dir, job_id)

    # ── Image undistortion ────────────────────────────────────────────────────
    dense.mkdir(exist_ok=True)
    model_dir = next(sparse.iterdir())
    run_cmd([
        "colmap", "image_undistorter",
        "--image_path",  str(images_dir),
        "--input_path",  str(model_dir),
        "--output_path", str(dense),
        "--output_type", "COLMAP",
    ], job_id)

    # Fix sparse subdir structure expected by train.py (sparse/0/cameras.bin)
    undist_sparse = dense / "sparse"
    target_0      = undist_sparse / "0"
    if undist_sparse.exists() and not target_0.exists():
        target_0.mkdir()
        for f in list(undist_sparse.iterdir()):
            if f.name != "0":
                f.rename(target_0 / f.name)

    return dense


def _merge_colmap_models(sparse_dir: Path, sub_models: list, db: Path,
                         images_dir: Path, job_id: str):
    """
    Attempt to merge fragmented COLMAP sub-models using model_merger.
    Falls back to keeping the largest sub-model if merging fails.
    """
    try:
        merged_dir = sparse_dir / "merged"
        merged_dir.mkdir(exist_ok=True)
        run_cmd([
            "colmap", "model_merger",
            "--input_path1",  str(sub_models[0]),
            "--input_path2",  str(sub_models[1]),
            "--output_path",  str(merged_dir),
        ], job_id)
        # Re-register the merged model as model 0
        target = sparse_dir / "0"
        if target.exists():
            shutil.rmtree(target)
        merged_dir.rename(target)
        print(f"[{job_id}] Models merged successfully")
    except Exception as e:
        print(f"[{job_id}] Model merge failed ({e}) — using largest sub-model")
        # Pick sub-model with most images as the best reconstruction
        largest = max(sub_models, key=lambda p: sum(1 for _ in p.glob("*.bin")))
        target = sparse_dir / "0"
        if largest != target:
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(largest, target)


def find_final_ply(output_dir: Path, job_id: str) -> Path:
    candidates = sorted(output_dir.glob("point_cloud/iteration_*/point_cloud.ply"))
    if not candidates:
        raise FileNotFoundError(f"No .ply output found in {output_dir}")
    return candidates[-1]


def compress_ply(input_path: Path, output_path: Path, job_id: str, min_opacity: float = 0.004) -> Path:
    """
    Prune floaters + quantize float32 → float16.
    min_opacity comes from the quality profile so we don't over-prune
    subtle geometry like curtains or thin furniture legs.
    """
    try:
        import numpy as np
        from plyfile import PlyData, PlyElement

        plydata = PlyData.read(str(input_path))
        vertex  = plydata["vertex"]
        data    = {prop.name: vertex[prop.name] for prop in vertex.properties}
        original_count = len(data[list(data.keys())[0]])

        # logit(min_opacity) threshold — prune near-invisible Gaussians
        import math
        logit_thresh = math.log(min_opacity / (1 - min_opacity))

        if "opacity" in data:
            mask = data["opacity"] > logit_thresh
            kept = int(mask.sum())
            print(f"[{job_id}] Pruning: {original_count} → {kept} Gaussians "
                  f"({100*kept//original_count}% kept, threshold={min_opacity})")
            data = {k: v[mask] for k, v in data.items()}

        position_props = {"x", "y", "z"}
        arrays = []
        for name, arr in data.items():
            if name not in position_props and arr.dtype == np.float32:
                arr = arr.astype(np.float16)
            arrays.append((name, arr))

        dtype   = [(name, arr.dtype) for name, arr in arrays]
        count   = len(arrays[0][1])
        new_arr = np.zeros(count, dtype=dtype)
        for name, arr in arrays:
            new_arr[name] = arr

        PlyData([PlyElement.describe(new_arr, "vertex")], text=False).write(str(output_path))

        orig_mb = input_path.stat().st_size  / 1024 / 1024
        comp_mb = output_path.stat().st_size / 1024 / 1024
        print(f"[{job_id}] Compressed: {orig_mb:.1f}MB → {comp_mb:.1f}MB")
        return output_path

    except Exception as e:
        print(f"[{job_id}] Compression failed — using original: {e}")
        return input_path


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


def parse_error_code(message: str) -> str:
    msg = message.lower()
    if "colmap" in msg or "sfm" in msg:        return "COLMAP_FAILED"
    if "out of memory" in msg or "oom" in msg: return "GPU_OOM"
    if "too few" in msg:                        return "TOO_FEW_IMAGES"
    if "cuda" in msg:                           return "CUDA_ERROR"
    return "WORKER_ERROR"

def humanize_error(message: str) -> str:
    return {
        "COLMAP_FAILED":  "Could not reconstruct 3D geometry. Walk slowly around the room with lots of overlap.",
        "GPU_OOM":        "GPU ran out of memory. Try 'fast' quality or a shorter video.",
        "TOO_FEW_IMAGES": "Not enough usable frames. Upload a longer video or more photos.",
        "CUDA_ERROR":     "A GPU error occurred. Please try again.",
        "WORKER_ERROR":   "Processing failed. Please try again.",
    }.get(parse_error_code(message), "Processing failed. Please try again.")


if __name__ == "__main__":
    main()