from __future__ import annotations

import argparse
from pathlib import Path

from plot_breakdown_impl import (
    FIG_MODULES,
    FIG_MODULES_PNG,
    FIG_MODULES_SVG,
    MODULES_TABLE_ROOT,
    load_csv_rows,
    load_top_manifest_from_table_root,
    plot_modules,
    prepare_breakdown_tables,
)


def resolve_output_root(manifest_path: str | None) -> Path:
    manifest, resolved_manifest_path = load_top_manifest_from_table_root(MODULES_TABLE_ROOT, manifest_path)
    if resolved_manifest_path is not None:
        return resolved_manifest_path.parent
    if manifest.get("suite_stamp") not in {None, "", "no-data"}:
        return MODULES_TABLE_ROOT / str(manifest["suite_stamp"])
    return MODULES_TABLE_ROOT


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, default=None)
    args = parser.parse_args(argv)

    manifest, _ = load_top_manifest_from_table_root(MODULES_TABLE_ROOT, args.manifest)
    output_root = resolve_output_root(args.manifest)
    _, module_rows = prepare_breakdown_tables(manifest, output_root)
    if not module_rows:
        module_rows = load_csv_rows(output_root / "BREAKDOWN_MODULES.csv")
    plot_modules(module_rows)
    print(f"Saved CSV: {output_root / 'BREAKDOWN_MODULES.csv'}")
    print(f"Saved PDF: {FIG_MODULES}")
    print(f"Saved PNG: {FIG_MODULES_PNG}")
    print(f"Saved SVG: {FIG_MODULES_SVG}")


if __name__ == "__main__":
    main()
