#fix.py

#!/usr/bin/env python3
# Fix already-generated LIBERO HDF5 files so they match what LiberoHandler reads.
#
# Problem: the handler does cv2.imdecode(frame) per image, which expects ENCODED
# (JPEG) bytes. Our preprocessing stored RAW uint8 pixel arrays [T,H,W,3], so
# cv2.imdecode fails with the CV_8U assertion.
#
# Fix: rewrite each file with the image datasets stored as variable-length bytes
# (one JPEG blob per frame), so the handler's cv2.imdecode succeeds.
#
# Channel convention (critical): at EVAL the libero client sends the model raw
# obs["agentview_image"] (RGB) and never round-trips through cv2.imdecode, so the
# model expects RGB. At TRAIN the handler does cv2.imdecode(...) then
# Image.fromarray(...) with NO color conversion. cv2.imencode is byte-faithful,
# so we encode the RGB array DIRECTLY (no RGB->BGR); cv2.imdecode then returns
# RGB, matching eval.
#
# Why multiprocessing (not threads): the work is CPU-bound JPEG encoding plus
# HDF5 I/O. The GIL serializes Python-level work and HDF5 is not safe for
# concurrent writes, so processes (independent files, no shared handles) are the
# correct parallelism here.
#
# Each file is rewritten to a temp file then atomically os.replace()'d, so a
# crash mid-run never corrupts an original, and the output file doesn't carry
# dead space from in-place deletes.
#
# Run in EITHER env (needs h5py, numpy, opencv):
#   python fix_libero_h5_images.py \
#       --root /data2/daniel/libero/libero_h5/libero_10 --workers 8
#
# Idempotent: files already in encoded (1-D vlen) form are skipped.

import os
import argparse
import glob
import multiprocessing as mp

import numpy as np
import h5py
import cv2

IMAGE_KEYS = ["agentview_rgb", "eye_in_hand_rgb"]
VLEN = h5py.special_dtype(vlen=np.uint8)


def _encode_frames(arr, quality):
    """[T,H,W,3] uint8 RGB -> object array of 1-D uint8 JPEG buffers."""
    T = arr.shape[0]
    out = np.empty(T, dtype=object)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    for i in range(T):
        ok, buf = cv2.imencode(".jpg", arr[i], params)
        if not ok:
            raise RuntimeError(f"JPEG encode failed at frame {i}")
        out[i] = buf.reshape(-1)
    return out


def fix_one(job):
    """Worker: rewrite a single file. Returns (path, status)."""
    path, quality = job
    try:
        with h5py.File(path, "r") as f:
            present_imgs = [k for k in IMAGE_KEYS if k in f]
            if present_imgs and all(f[k].ndim == 1 for k in present_imgs):
                return (path, "skipped")

            payload = {}
            for k in f.keys():
                if k in IMAGE_KEYS:
                    payload[k] = ("img", f[k][()])
                else:
                    payload[k] = ("raw", f[k][()])

        tmp = path + ".tmp"
        with h5py.File(tmp, "w") as g:
            for k, (kind, val) in payload.items():
                if kind == "img":
                    enc = _encode_frames(val, quality)
                    d = g.create_dataset(k, shape=(enc.shape[0],), dtype=VLEN,
                                         compression="gzip")
                    for i in range(enc.shape[0]):
                        d[i] = enc[i]
                else:
                    if getattr(val, "ndim", 0) == 0:
                        g.create_dataset(k, data=val)
                    else:
                        g.create_dataset(k, data=val, compression="gzip")

        os.replace(tmp, path)
        return (path, "ok")
    except Exception as e:
        try:
            if os.path.exists(path + ".tmp"):
                os.remove(path + ".tmp")
        except OSError:
            pass
        return (path, f"ERROR: {type(e).__name__}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="Folder of .hdf5 files (e.g. .../libero_h5/libero_10).")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (1-100).")
    ap.add_argument("--glob", default="*.hdf5")
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() // 2))
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.root, args.glob)))
    if not files:
        raise SystemExit(f"No files matched {os.path.join(args.root, args.glob)}")

    jobs = [(p, args.quality) for p in files]
    print(f"Re-encoding {len(files)} files across {args.workers} workers (q={args.quality})...")

    ok = skipped = errors = 0
    err_msgs = []
    with mp.Pool(processes=args.workers) as pool:
        for i, (path, status) in enumerate(pool.imap_unordered(fix_one, jobs), 1):
            if status == "ok":
                ok += 1
            elif status == "skipped":
                skipped += 1
            else:
                errors += 1
                err_msgs.append(f"{os.path.basename(path)}: {status}")
            if i % 25 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  (ok={ok} skipped={skipped} err={errors})")

    print(f"\nDone. ok={ok} skipped={skipped} errors={errors}")
    if err_msgs:
        print("Failures:")
        for m in err_msgs[:20]:
            print("  ", m)


if __name__ == "__main__":
    main()
