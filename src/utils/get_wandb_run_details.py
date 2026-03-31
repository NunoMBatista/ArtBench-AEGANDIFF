import argparse
import datetime
import fnmatch
import json
import os
from collections.abc import Mapping

import dotenv
import wandb


def _json_default(obj):
    """Best-effort conversion of common non-JSON-serializable objects."""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()

    # Mapping-like objects (e.g., wandb SummarySubDict)
    if isinstance(obj, Mapping):
        return dict(obj)

    if isinstance(obj, set):
        return list(obj)

    # Numpy scalars/arrays (optional dependency)
    try:
        import numpy as np  # type: ignore

        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    return str(obj)


def _resolve_run_path(run_arg: str, entity: str | None, project: str | None) -> str:
    # Allow full path: entity/project/run_id
    if run_arg.count("/") == 2:
        return run_arg

    if not entity or not project:
        raise ValueError(
            "WANDB_ENTITY/WANDB_PROJECT not set and run arg isn't a full path (entity/project/run_id)."
        )

    return f"{entity}/{project}/{run_arg}"


def fetch_run_details(
    run_path: str,
    output_dir: str,
    *,
    history_samples: int | None,
    history_keys: list[str] | None,
    download_files: bool,
    include_patterns: list[str] | None,
    exclude_patterns: list[str] | None,
    max_files: int | None,
    overwrite: bool,
) -> None:
    api = wandb.Api()
    run = api.run(run_path)

    os.makedirs(output_dir, exist_ok=True)

    run_json = {
        "path": run.path,  # [entity, project, run_id]
        "id": run.id,
        "name": run.name,
        "display_name": getattr(run, "display_name", None),
        "state": run.state,
        "url": getattr(run, "url", None),
        "entity": getattr(run, "entity", None),
        "project": getattr(run, "project", None),
        "group": getattr(run, "group", None),
        "job_type": getattr(run, "job_type", None),
        "tags": list(run.tags) if run.tags else [],
        "notes": getattr(run, "notes", None),
        "created_at": run.created_at,
        "heartbeat_at": getattr(run, "heartbeat_at", None),
        "runtime": getattr(run, "runtime", None),
        "config": run.config,
        "summary": dict(run.summary),
    }

    with open(os.path.join(output_dir, "run.json"), "w", encoding="utf-8") as f:
        json.dump(run_json, f, indent=2, default=_json_default)

    # History / curves
    # - If history_keys is provided, fetch those.
    # - Otherwise fetch all logged keys (W&B decides columns based on what's logged).
    history_kwargs: dict[str, object] = {}
    if history_samples is not None:
        history_kwargs["samples"] = history_samples
    if history_keys:
        history_kwargs["keys"] = history_keys

    history_df = run.history(**history_kwargs)  # type: ignore[arg-type]

    history_csv_path = os.path.join(output_dir, "history.csv")
    history_df.to_csv(history_csv_path, index=False)

    # Also write JSONL for easier streaming / grep
    history_jsonl_path = os.path.join(output_dir, "history.jsonl")
    with open(history_jsonl_path, "w", encoding="utf-8") as f:
        for row in history_df.to_dict(orient="records"):
            f.write(json.dumps(row, default=_json_default) + "\n")

    if download_files:
        files_dir = os.path.join(output_dir, "files")
        os.makedirs(files_dir, exist_ok=True)

        includes = include_patterns or ["*"]
        excludes = exclude_patterns or []

        downloaded: list[dict] = []
        skipped: list[dict] = []
        total_seen = 0
        total_downloaded = 0

        for file_obj in run.files():
            total_seen += 1
            name = getattr(file_obj, "name", None)
            if not name:
                skipped.append({"reason": "missing_name"})
                continue

            should_include = any(fnmatch.fnmatch(name, pat) for pat in includes)
            should_exclude = any(fnmatch.fnmatch(name, pat) for pat in excludes)

            if not should_include or should_exclude:
                skipped.append(
                    {
                        "name": name,
                        "reason": "filtered",
                        "included": should_include,
                        "excluded": should_exclude,
                    }
                )
                continue

            if max_files is not None and total_downloaded >= max_files:
                skipped.append({"name": name, "reason": "max_files_reached"})
                continue

            local_path = os.path.join(files_dir, name)
            if not overwrite and os.path.exists(local_path):
                skipped.append({"name": name, "reason": "already_exists"})
                continue

            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            file_obj.download(root=files_dir, replace=overwrite)
            downloaded.append({"name": name, "path": local_path})
            total_downloaded += 1

        manifest = {
            "run": run_path,
            "output_dir": output_dir,
            "files_dir": files_dir,
            "include_patterns": includes,
            "exclude_patterns": excludes,
            "max_files": max_files,
            "overwrite": overwrite,
            "total_seen": total_seen,
            "total_downloaded": total_downloaded,
            "downloaded": downloaded,
            "skipped": skipped,
        }

        with open(os.path.join(output_dir, "download_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=_json_default)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch a single Weights & Biases run details (config/summary/metadata) and export metric history "
            "(loss curves) to outputs. Optionally downloads run files/media."
        )
    )
    parser.add_argument(
        "run",
        help=(
            "Run id (e.g. m9hr4wn2) or full path (entity/project/run_id). "
            "If you pass just the id, WANDB_ENTITY and WANDB_PROJECT must be set."
        ),
    )
    parser.add_argument(
        "--entity",
        default=None,
        help="Override WANDB_ENTITY.",
    )
    parser.add_argument(
        "--project",
        default=None,
        help="Override WANDB_PROJECT.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/wandb_runs",
        help="Root output directory (default: outputs/wandb_runs).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help=(
            "Max number of history rows to fetch (wandb API 'samples' parameter). "
            "Omit to let wandb decide a reasonable default."
        ),
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help=(
            "Optional list of history keys (metrics) to fetch. If omitted, fetch all keys returned by W&B."
        ),
    )

    parser.add_argument(
        "--download-media",
        action="store_true",
        help="Download media files (default patterns: media/*).",
    )
    parser.add_argument(
        "--download-all-files",
        action="store_true",
        help="Download all run files (may be large).",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help=(
            "Include glob patterns for run file download (fnmatch style). "
            "Examples: media/*, media/images/*, *.png."
        ),
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Exclude glob patterns for run file download.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to download (safety limit).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing downloaded files.",
    )

    args = parser.parse_args()

    dotenv.load_dotenv()
    entity = args.entity or os.getenv("WANDB_ENTITY")
    project = args.project or os.getenv("WANDB_PROJECT")

    run_path = _resolve_run_path(args.run, entity, project)
    run_id = run_path.split("/")[-1]
    output_dir = os.path.join(args.output_root, run_id)

    print(f"Fetching run: {run_path}")
    print(f"Writing to: {output_dir}")

    download_files = bool(args.download_media or args.download_all_files or args.include)
    if args.download_all_files:
        include_patterns = args.include or ["*"]
    elif args.download_media:
        include_patterns = args.include or ["media/*"]
    else:
        include_patterns = args.include

    fetch_run_details(
        run_path,
        output_dir,
        history_samples=args.samples,
        history_keys=args.keys,
        download_files=download_files,
        include_patterns=include_patterns,
        exclude_patterns=args.exclude,
        max_files=args.max_files,
        overwrite=args.overwrite,
    )

    print("Done.")


if __name__ == "__main__":
    main()
