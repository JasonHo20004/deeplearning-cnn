#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download more animal images and prepare the dataset for training.
Target: 1500 train + 200 validate per class for 5 classes.

Uses icrawler (Bing) with diverse search queries per class.
Automatically resizes to 224x224 and 32x32.

Usage:
    python scripts/expand_dataset.py
"""

import os
import sys
import random
import shutil
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / 'datasets'

TRAIN_224_DIR = DATASETS_DIR / 'train' / '224x224'
TRAIN_32_DIR = DATASETS_DIR / 'train' / '32x32'
VAL_DIR = DATASETS_DIR / 'validate'
TEMP_DIR = DATASETS_DIR / '_temp_crawl'

# Target counts
TARGET_TRAIN = 1500   # images per class for training
TARGET_VAL = 200      # images per class for validation

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

# Diverse search queries per class for better coverage
SEARCH_QUERIES = {
    'cat': [
        'cat animal photo',
        'domestic cat close up',
        'kitten photo',
        'cat face portrait',
        'cat sitting outdoor',
        'cat different breeds',
    ],
    'chicken': [
        'chicken animal photo',
        'hen rooster photo',
        'chicken farm bird',
        'chicken poultry close up',
        'baby chick photo',
        'chicken outdoor farm',
    ],
    'cow': [
        'cow animal photo',
        'cattle farm photo',
        'cow close up face',
        'cow grazing field',
        'dairy cow photo',
        'cow different breeds',
    ],
    'dog': [
        'dog animal photo',
        'pet dog close up',
        'puppy photo',
        'dog face portrait',
        'dog outdoor playing',
        'dog different breeds',
    ],
    'monkey': [
        'monkey animal photo',
        'primate monkey close up',
        'monkey face portrait',
        'monkey wildlife photo',
        'baby monkey photo',
        'monkey sitting tree',
    ],
}

CLASSES = list(SEARCH_QUERIES.keys())


# ============================================================
# Utilities
# ============================================================
def count_images(folder):
    """Count valid image files in a folder."""
    if not folder.exists():
        return 0
    return len([f for f in folder.iterdir()
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS])


def validate_image(filepath):
    """Check if an image file is valid and can be opened."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        # Re-open to check if readable
        with Image.open(filepath) as img:
            img.load()
            # Minimum size check
            if img.size[0] < 32 or img.size[1] < 32:
                return False
        return True
    except Exception:
        return False


def resize_image(src_path, dst_path, size):
    """Resize an image to the specified size and save."""
    try:
        with Image.open(src_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_resized = img.resize((size, size), Image.LANCZOS)
            img_resized.save(dst_path, 'JPEG', quality=95)
        return True
    except Exception:
        return False


def ensure_icrawler():
    """Ensure icrawler is installed and return the crawler class."""
    try:
        from icrawler.builtin import BingImageCrawler
        return BingImageCrawler
    except ImportError:
        print("Installing icrawler...")
        import subprocess
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'icrawler'])
        from icrawler.builtin import BingImageCrawler
        return BingImageCrawler


# ============================================================
# Download
# ============================================================
def download_images(cls, queries, target_count, download_dir, BingImageCrawler):
    """Download images for a class using multiple search queries."""
    download_dir.mkdir(parents=True, exist_ok=True)
    existing = count_images(download_dir)

    if existing >= target_count:
        print(f"  [{cls}] Already have {existing}/{target_count} images. Skipping download.")
        return existing

    needed = target_count - existing
    per_query = max(needed // len(queries) + 50, 100)  # extra buffer for failed downloads

    print(f"  [{cls}] Have {existing}, need {needed} more. Downloading ~{per_query} per query...")

    for i, query in enumerate(queries):
        current = count_images(download_dir)
        if current >= target_count:
            break

        query_dir = download_dir / f'_q{i}'
        query_dir.mkdir(exist_ok=True)

        print(f"    Query {i+1}/{len(queries)}: '{query}' (max {per_query})...")
        try:
            crawler = BingImageCrawler(
                downloader_threads=4,
                storage={'root_dir': str(query_dir)},
            )
            crawler.crawl(
                keyword=query,
                filters=None,
                max_num=per_query,
            )
        except Exception as e:
            print(f"    Warning: {e}")

        # Move valid images to main folder
        moved = 0
        for f in query_dir.rglob('*'):
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                if validate_image(f):
                    # Use unique name to avoid collisions
                    dst = download_dir / f'{cls}_crawl_{existing + moved + 1:04d}{f.suffix}'
                    if not dst.exists():
                        shutil.move(str(f), str(dst))
                        moved += 1

        # Cleanup query dir
        shutil.rmtree(query_dir, ignore_errors=True)
        print(f"    -> Got {moved} valid images from this query")

    final_count = count_images(download_dir)
    print(f"  [{cls}] Total: {final_count} images")
    return final_count


# ============================================================
# Split and Resize
# ============================================================
def prepare_dataset():
    """Split downloaded images into train/val and resize to 224x224 and 32x32."""
    print("\n" + "=" * 60)
    print("  Preparing train/val splits and resizing...")
    print("=" * 60)

    random.seed(42)

    for cls in CLASSES:
        temp_cls = TEMP_DIR / cls
        if not temp_cls.exists():
            print(f"  [{cls}] No temp download folder found. Skipping.")
            continue

        # Collect all downloaded images
        all_images = [f for f in temp_cls.iterdir()
                      if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS]

        if not all_images:
            print(f"  [{cls}] No images found in temp folder. Skipping.")
            continue

        random.shuffle(all_images)

        # Count existing images
        existing_train_224 = count_images(TRAIN_224_DIR / cls)
        existing_train_32 = count_images(TRAIN_32_DIR / cls)
        existing_val = count_images(VAL_DIR / cls)

        need_train = max(0, TARGET_TRAIN - existing_train_224)
        need_val = max(0, TARGET_VAL - existing_val)
        total_needed = need_train + need_val

        print(f"\n  [{cls}] Downloaded: {len(all_images)}, "
              f"Existing train: {existing_train_224}, val: {existing_val}")
        print(f"  [{cls}] Need: {need_train} train + {need_val} val = {total_needed}")

        # Ensure directories exist
        (TRAIN_224_DIR / cls).mkdir(parents=True, exist_ok=True)
        (TRAIN_32_DIR / cls).mkdir(parents=True, exist_ok=True)
        (VAL_DIR / cls).mkdir(parents=True, exist_ok=True)

        # Split: first need_val go to val, rest go to train
        added_val = 0
        added_train = 0

        for img_path in all_images:
            if added_val + added_train >= total_needed:
                break

            if added_val < need_val:
                # Add to validation (original resolution, just copy)
                dst = VAL_DIR / cls / f'{cls}_new_{existing_val + added_val + 1:04d}.jpg'
                if resize_image(img_path, dst, 224):  # save at 224 for val
                    added_val += 1
            elif added_train < need_train:
                # Resize and save to both 224x224 and 32x32
                name = f'{cls}_new_{existing_train_224 + added_train + 1:04d}.jpg'
                dst_224 = TRAIN_224_DIR / cls / name
                dst_32 = TRAIN_32_DIR / cls / name

                ok_224 = resize_image(img_path, dst_224, 224)
                ok_32 = resize_image(img_path, dst_32, 32)
                if ok_224 and ok_32:
                    added_train += 1

        print(f"  [{cls}] Added: {added_train} train + {added_val} val")

    # Final counts
    print("\n" + "=" * 60)
    print("  FINAL DATASET COUNTS")
    print("=" * 60)
    for cls in CLASSES:
        t224 = count_images(TRAIN_224_DIR / cls)
        t32 = count_images(TRAIN_32_DIR / cls)
        v = count_images(VAL_DIR / cls)
        print(f"  {cls:>10s}: train_224={t224}, train_32={t32}, val={v}")


# ============================================================
# Cleanup
# ============================================================
def cleanup():
    """Remove temporary download directory."""
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        print("\nTemporary downloads cleaned up.")


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  DATASET EXPANSION — Target 1500 train + 200 val / class")
    print("=" * 60)

    # Check current status
    print("\nCurrent dataset:")
    for cls in CLASSES:
        t224 = count_images(TRAIN_224_DIR / cls)
        t32 = count_images(TRAIN_32_DIR / cls)
        v = count_images(VAL_DIR / cls)
        print(f"  {cls:>10s}: train_224={t224}, train_32={t32}, val={v}")

    # Ensure icrawler
    BingImageCrawler = ensure_icrawler()

    # Download phase
    print("\n" + "=" * 60)
    print("  DOWNLOADING IMAGES (this may take 15-30 minutes)...")
    print("=" * 60)

    total_target = TARGET_TRAIN + TARGET_VAL  # per class
    for cls in CLASSES:
        existing_total = count_images(TRAIN_224_DIR / cls) + count_images(VAL_DIR / cls)
        needed = max(0, total_target - existing_total)
        if needed == 0:
            print(f"\n  [{cls}] Already have enough images. Skipping.")
            continue

        print(f"\n  [{cls}] Need to download ~{needed + 200} images (buffer for invalid)...")
        download_images(
            cls=cls,
            queries=SEARCH_QUERIES[cls],
            target_count=needed + 200,  # buffer for invalid images
            download_dir=TEMP_DIR / cls,
            BingImageCrawler=BingImageCrawler,
        )

    # Prepare (split + resize)
    prepare_dataset()

    # Cleanup
    cleanup()

    print("\n" + "=" * 60)
    print("  DONE! You can now re-train M1:")
    print("    python train_M1_224.py")
    print("    python train_M1_32.py")
    print("=" * 60)


if __name__ == '__main__':
    random.seed(42)
    try:
        main()
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
