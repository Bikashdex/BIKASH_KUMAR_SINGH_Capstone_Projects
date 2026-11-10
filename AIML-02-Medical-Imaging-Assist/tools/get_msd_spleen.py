"""Download and unpack Medical Segmentation Decathlon Task09_Spleen, then prove what landed.

Run it once per machine:

    python tools/get_msd_spleen.py                 # into data/raw/msd/
    python tools/get_msd_spleen.py --dest D:/msd   # somewhere with more space

Why a script instead of `DecathlonDataset(..., download=True)` inside a notebook: the notebook then cannot
tell you *why* it has no data. This prints the URL, the expected md5 MONAI ships for that tar, the number of
bytes actually received, and the manifest of `imagesTr`/`labelsTr`, so a truncated mirror or a moved URL is a
message rather than a mystery. The md5 is the same value `monai.apps.datasets.DecathlonDataset` uses, copied
here so the check survives a MONAI upgrade changing its table.

The 1.6 GB tar and the ~2.5 GB of extracted volumes are git-ignored and DVC-tracked; the manifest this script
writes to `results/` is what a reviewer can read without the data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
import time
import urllib.error
import urllib.request

URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar"
MD5 = "410d4a301da4e5b2f6f86ec3ddba524e"  # MONAI's own table for this task
TASK = "Task09_Spleen"
EXPECTED_VOLUMES = 41


def md5_of(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download(dest: str, force: bool = False) -> str:
    """Stream the tar to disk with a byte counter, resuming a partial file when the server allows it."""
    if os.path.exists(dest) and not force:
        size = os.path.getsize(dest)
        print(f"{dest} already exists ({size:,} bytes) - verifying before re-downloading")
        if size and md5_of(dest) == MD5:
            print("md5 matches, skipping the download")
            return dest
        print("md5 does not match, re-downloading")
        os.remove(dest)

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = dest + ".part"
    done = os.path.getsize(part) if os.path.exists(part) and not force else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    print(f"GET {URL}")
    print(f"starting at byte {done:,}" if done else "starting from scratch")
    req = urllib.request.Request(URL, headers=headers)
    t0 = time.time()
    last = t0
    with urllib.request.urlopen(req, timeout=120) as resp, open(part, "ab" if done else "wb") as out:
        total = done + int(resp.headers.get("Content-Length") or 0)
        print("expected size:", f"{total:,} bytes")
        moved = 0
        while True:
            block = resp.read(1 << 22)
            if not block:
                break
            out.write(block)
            moved += len(block)
            now = time.time()
            if now - last > 20:
                rate = (moved) / (now - t0) / 1e6
                print(f"  {done + moved:,} / {total:,} bytes ({100 * (done + moved) / max(total, 1):.1f} %)"
                      f"  {rate:.1f} MB/s", flush=True)
                last = now
    os.replace(part, dest)
    print(f"downloaded {os.path.getsize(dest):,} bytes in {time.time() - t0:.0f} s")
    return dest


def extract(tar_path: str, dest_dir: str) -> str:
    root = os.path.join(dest_dir, TASK)
    if os.path.isdir(os.path.join(root, "imagesTr")):
        print(f"{root} already extracted")
        return root
    print(f"extracting {os.path.basename(tar_path)} -> {dest_dir}")
    with tarfile.open(tar_path, "r:") as tf:  # "r:" = uncompressed tar; a .gz here would fail loudly
        tf.extractall(dest_dir, filter="data")
    return root


def manifest(task_root: str) -> dict:
    """Counts and sizes of the real volumes.

    The MSD tarballs were packed on macOS, so `imagesTr` contains an AppleDouble `._spleen_N.nii.gz` beside
    every `spleen_N.nii.gz` - 82 entries for 41 spleens. Counting them would make this script report a
    dataset that no loader can open, so the junk is filtered here exactly as `src.data.ct_volume` filters it.
    """
    out: dict = {"task": TASK, "url": URL, "tar_md5_expected": MD5}
    for sub in ("imagesTr", "labelsTr"):
        d = os.path.join(task_root, sub)
        if not os.path.isdir(d):
            out[sub] = {"error": f"{d} does not exist"}
            continue
        all_names = sorted(n for n in os.listdir(d) if n.endswith(".nii.gz"))
        names = [n for n in all_names if not n.startswith("._")]
        junk = len(all_names) - len(names)
        out[sub] = {"n": len(names),
                    "appledouble_entries_ignored": junk,
                    "total_bytes": sum(os.path.getsize(os.path.join(d, n)) for n in names),
                    "first": names[:3], "last": names[-3:]}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=os.path.join("data", "raw", "msd"),
                    help="directory for the tar and the extracted Task09_Spleen/ (default data/raw/msd)")
    ap.add_argument("--force", action="store_true", help="re-download even if a matching file exists")
    ap.add_argument("--skip-download", action="store_true", help="only verify and extract what is on disk")
    args = ap.parse_args(argv)

    tar_path = os.path.join(args.dest, TASK + ".tar")
    if not args.skip_download:
        download(tar_path, force=args.force)

    if not os.path.exists(tar_path):
        print(f"no {tar_path} and --skip-download was given", file=sys.stderr)
        return 2
    got = md5_of(tar_path)
    print(f"md5 {got}  {'MATCHES' if got == MD5 else 'DOES NOT MATCH'} expected {MD5}")
    if got != MD5:
        print("refusing to extract a tar that is not the one MONAI documents", file=sys.stderr)
        return 3

    root = extract(tar_path, args.dest)
    man = manifest(root)
    print(json.dumps(man, indent=1))
    n_img = man["imagesTr"].get("n", 0)
    n_lab = man["labelsTr"].get("n", 0)
    if n_img != EXPECTED_VOLUMES or n_lab != EXPECTED_VOLUMES:
        print(f"expected {EXPECTED_VOLUMES} images and {EXPECTED_VOLUMES} labels, "
              f"got {n_img} and {n_lab}", file=sys.stderr)
        return 4

    results_dir = os.path.join("results")
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "msd_spleen_manifest.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1)
    print(f"wrote {path}")
    print("\nNext: dvc add data/raw/msd/Task09_Spleen && dvc push, then run notebooks/05_segment_spleen.ipynb")
    return 0


if __name__ == "__main__":
    sys.exit(main())
