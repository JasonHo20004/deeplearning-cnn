#!/usr/bin/env python3
"""
Download animal images and organize into train/test splits.
Uses icrawler (Bing) to collect ~300-1000 images per class.

Usage: python scripts/download_animal_dataset.py
"""

import shutil
import random
import subprocess
from pathlib import Path

# 5 animal classes
CLASSES = ["dog", "cat", "bird", "horse", "elephant"]

# ~500 images per class (within 300-1000), 80% train / 20% test
IMAGES_PER_CLASS = 500
TRAIN_RATIO = 0.8
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def ensure_icrawler():
    """Ensure icrawler is installed."""
    try:
        from icrawler.builtin import BingImageCrawler
        return BingImageCrawler
    except ImportError:
        print("Installing icrawler...")
        subprocess.check_call(["pip", "install", "icrawler"])
        from icrawler.builtin import BingImageCrawler
        return BingImageCrawler


def create_folder_structure(dataset_dir: Path) -> None:
    """Create dataset/train/{class}/ and dataset/test/{class}/."""
    for split in ["train", "test"]:
        for cls in CLASSES:
            (dataset_dir / split / cls).mkdir(parents=True, exist_ok=True)


def main():
    base = Path(__file__).resolve().parent.parent
    dataset_dir = base / "dataset"
    download_base = base / "dataset" / "_temp_downloads"
    dataset_dir.mkdir(exist_ok=True)
    create_folder_structure(dataset_dir)

    BingImageCrawler = ensure_icrawler()

    print("=== Downloading animal images (this may take several minutes) ===\n")

    for cls in CLASSES:
        download_dir = download_base / cls
        download_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading '{cls}' (up to {IMAGES_PER_CLASS} images)...")
        try:
            crawler = BingImageCrawler(
                downloader_threads=4,
                storage={"root_dir": str(download_dir)},
            )
            crawler.crawl(
                keyword=f"{cls} animal photo",
                filters=None,
                max_num=IMAGES_PER_CLASS,
            )
        except Exception as e:
            print(f"  Warning: {e}")

        # Collect downloaded images
        files = []
        for f in download_dir.rglob("*"):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                files.append(f)

        if files:
            random.shuffle(files)
            files = files[:IMAGES_PER_CLASS]
            n_train = int(len(files) * TRAIN_RATIO)
            for i, f in enumerate(files):
                split = "train" if i < n_train else "test"
                dst = dataset_dir / split / cls / f.name
                try:
                    shutil.copy2(f, dst)
                except Exception as e:
                    print(f"  Copy error: {e}")
            print(f"  -> {cls}: {n_train} train, {len(files) - n_train} test\n")
        else:
            print(f"  -> No images found for {cls}\n")

    # Cleanup temp downloads
    if download_base.exists():
        shutil.rmtree(download_base, ignore_errors=True)
        print("Temporary downloads removed.")

    # Summary
    print("\n=== Dataset summary ===")
    total_train = total_test = 0
    for split in ["train", "test"]:
        for cls in CLASSES:
            count = len(list((dataset_dir / split / cls).glob("*.*")))
            if count > 0:
                print(f"  {split}/{cls}: {count} images")
                if split == "train":
                    total_train += count
                else:
                    total_test += count
    print(f"\nTotal: {total_train} train, {total_test} test")


if __name__ == "__main__":
    random.seed(42)
    main()
