"""
One-shot migration: rename legacy metadata.yaml keys to their canonical forms.

  cam_heights  -> camera_heights
  cam_widths   -> camera_widths
  n_frames     -> num_frames (only in the top-level calib dict, not CSV rows)

Usage:
    python -m posetail_preprocessing.scripts.migrate_metadata_keys <root> [<root2> ...]
    python -m posetail_preprocessing.scripts.migrate_metadata_keys <root> --dry-run
"""
import argparse
import sys
from pathlib import Path

import yaml

RENAMES = {
    'cam_heights': 'camera_heights',
    'cam_widths': 'camera_widths',
    'n_frames': 'num_frames',
}


def migrate_file(path: Path, dry_run: bool) -> bool:
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return False

    changed_keys = [k for k in RENAMES if k in data]
    if not changed_keys:
        return False

    if not dry_run:
        for old_key in changed_keys:
            data[RENAMES[old_key]] = data.pop(old_key)
        with open(path, 'w') as f:
            yaml.dump(data, f)

    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', metavar='ROOT',
                        help='Root directories to search recursively')
    parser.add_argument('--dry-run', action='store_true',
                        help='Report what would change without writing')
    args = parser.parse_args()

    migrated = 0
    current = 0

    for root in args.roots:
        root_path = Path(root)
        if not root_path.exists():
            print(f'WARNING: {root} does not exist, skipping', file=sys.stderr)
            continue

        for yaml_path in sorted(root_path.rglob('metadata.yaml')):
            changed = migrate_file(yaml_path, dry_run=args.dry_run)
            if changed:
                migrated += 1
                if args.dry_run:
                    print(f'would migrate: {yaml_path}')
            else:
                current += 1

    action = 'would migrate' if args.dry_run else 'migrated'
    print(f'\n{action}: {migrated}, already current: {current}')


if __name__ == '__main__':
    main()
