"""
Fix camera-key mismatches in metadata.yaml files and remove orphan img/ dirs.

Each metadata.yaml should have five camera-keyed dicts with identical key sets:
  intrinsic_matrices, extrinsic_matrices, distortion_matrices,
  camera_heights, camera_widths

When a video file is missing from a session, the calibration dicts (from the
JSON file) can end up with more cameras than the width/height dicts (derived
from the actual video files), causing a KeyError at training time. This script
takes the intersection and rewrites affected files.

Additionally, if a video failed to decode during preprocessing the producer may
have left a partial img/<cam>/ directory on disk even though that camera is not
in metadata.yaml, causing an IndexError at training time. This script removes
such orphan directories.

Usage:
    python -m posetail_preprocessing.scripts.fix_metadata_camera_keys <root> [<root2> ...]
    python -m posetail_preprocessing.scripts.fix_metadata_camera_keys <root> --dry-run
"""
import argparse
import shutil
import sys
from pathlib import Path

import yaml

CAMERA_DICTS = [
    'intrinsic_matrices',
    'extrinsic_matrices',
    'distortion_matrices',
    'camera_heights',
    'camera_widths',
]


def fix_file(path: Path, dry_run: bool) -> bool:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return False

    if not all(k in data for k in CAMERA_DICTS):
        return False

    key_sets = [set(data[k].keys()) for k in CAMERA_DICTS]
    final_cams = key_sets[0].intersection(*key_sets[1:])

    yaml_changed = not all(s == final_cams for s in key_sets)

    if yaml_changed:
        dropped = key_sets[0].union(*key_sets[1:]) - final_cams
        print(f'{"would fix" if dry_run else "fixing"} yaml: {path}  (dropping: {sorted(dropped)})')
        if not dry_run:
            for dict_key in CAMERA_DICTS:
                data[dict_key] = {k: v for k, v in data[dict_key].items() if k in final_cams}
            data['num_cameras'] = len(final_cams)
            with open(path, 'w') as f:
                yaml.dump(data, f)

    # remove orphan img/<cam>/ dirs not in final_cams
    img_dir = path.parent / 'img'
    orphans_removed = False
    if img_dir.is_dir():
        for cam_dir in sorted(img_dir.iterdir()):
            if cam_dir.is_dir() and cam_dir.name not in final_cams:
                print(f'{"would remove" if dry_run else "removing"} orphan: {cam_dir}')
                if not dry_run:
                    shutil.rmtree(cam_dir)
                orphans_removed = True

    return yaml_changed or orphans_removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', metavar='ROOT',
                        help='Root directories to search recursively')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would change without writing')
    args = parser.parse_args()

    fixed = 0
    current = 0

    for root in args.roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f'WARNING: {root} does not exist, skipping', file=sys.stderr)
            continue

        for yaml_path in sorted(root_path.rglob('metadata.yaml')):
            changed = fix_file(yaml_path, dry_run=args.dry_run)
            if changed:
                fixed += 1
            else:
                current += 1

    action = 'would fix' if args.dry_run else 'fixed'
    print(f'\n{action}: {fixed}, already consistent: {current}')


if __name__ == '__main__':
    main()
