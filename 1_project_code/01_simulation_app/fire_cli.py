from __future__ import annotations

import argparse
import json
from pathlib import Path

from fire_core import inspect_legacy_project
from fire_core import load_cases
from fire_core import load_export_config
from fire_core import run_ca_baseline
from fire_core import run_case_export
from fire_core import run_hybrid_experiment
from fire_core import serialize_ca_run
from fire_core import serialize_hybrid_run
from fire_reports import generate_hybrid_report


def _common_project_root_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", default="../project", help="Path to the original notebook-based project folder.")


def _add_date_window_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--start-date", help="Burned-series date used as the initial fire map.")
    parser.add_argument("--end-date", help="Last burned-series date to evaluate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible runs.")


def _add_viirs_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--use-viirs-init", action="store_true", help="Initialize the fire state from accumulated VIIRS detections instead of the burned raster.")
    parser.add_argument("--viirs-history-days", type=int, default=None, help="Optional number of previous VIIRS days to merge. Defaults to all previous non-empty days.")


def _add_targeting_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--target-strategy",
        choices=["point_growth_frontier", "clustered_frontier", "clustered_growth_frontier", "clustered_growth_frontier_priority", "clustered_growth_frontier_diverse"],
        default="clustered_growth_frontier",
        help="Drone target-selection strategy for the MAS layer.",
    )
    parser.add_argument(
        "--use-spread-bias",
        action="store_true",
        help="Use drone observations to build a local next-day spread bias for the CA component.",
    )


def _add_hybrid_runtime_arguments(parser: argparse.ArgumentParser, *, include_output_dir: bool = False) -> None:
    parser.add_argument("--use-barriers", action="store_true", help="Enable barrier masking in the CA component of the hybrid model.")
    parser.add_argument("--drone-count", type=int, default=10, help="Number of drones in the MAS layer.")
    parser.add_argument("--vision", type=float, default=8.0, help="Drone sensing radius in pixels.")
    parser.add_argument("--move", type=float, default=35.0, help="Drone movement speed in pixels per step.")
    parser.add_argument("--steps-per-day", type=int, default=24, help="Number of intra-day drone update steps.")
    if include_output_dir:
        parser.add_argument("--output-dir", default="outputs/reports", help="Directory where CSV tables and figures will be written.")
        parser.add_argument("--num-days", type=int, default=6, help="How many daily drone trajectory figures to save.")


def _hybrid_run_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "project_root": args.project_root,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "seed": args.seed,
        "use_barriers": args.use_barriers,
        "drone_count": args.drone_count,
        "vision": args.vision,
        "move": args.move,
        "steps_per_day": args.steps_per_day,
        "target_strategy": args.target_strategy,
        "use_spread_bias": args.use_spread_bias,
        "use_viirs_init": args.use_viirs_init,
        "viirs_history_days": args.viirs_history_days,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for the reworked CA+MAS wildfire project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect the legacy project data layout and validation warnings.")
    _common_project_root_argument(inspect_parser)

    ca_parser = subparsers.add_parser("ca", help="Run the CA baseline and emit JSON metrics.")
    _common_project_root_argument(ca_parser)
    _add_date_window_arguments(ca_parser)
    _add_viirs_arguments(ca_parser)
    ca_parser.add_argument("--use-barriers", action="store_true", help="Enable barrier masking during spread simulation.")
    ca_parser.add_argument("--save-json", help="Optional path to save the CA result JSON.")

    hybrid_parser = subparsers.add_parser("hybrid", help="Run the CA+MAS hybrid baseline and emit JSON metrics.")
    _common_project_root_argument(hybrid_parser)
    _add_date_window_arguments(hybrid_parser)
    _add_viirs_arguments(hybrid_parser)
    _add_targeting_arguments(hybrid_parser)
    _add_hybrid_runtime_arguments(hybrid_parser)
    hybrid_parser.add_argument("--save-json", help="Optional path to save the hybrid result JSON.")

    report_parser = subparsers.add_parser("report", help="Run the hybrid model and write legacy-style tables and figures.")
    _common_project_root_argument(report_parser)
    _add_date_window_arguments(report_parser)
    _add_viirs_arguments(report_parser)
    _add_targeting_arguments(report_parser)
    _add_hybrid_runtime_arguments(report_parser, include_output_dir=True)
    report_parser.add_argument("--save-json", help="Optional path to save the hybrid result JSON.")

    export_parser = subparsers.add_parser("export-gee", help="Export raw wildfire case data from Google Earth Engine.")
    export_parser.add_argument("--defaults", default="01_simulation_app/simulation_config.example.yaml", help="Path to the defaults YAML file.")
    export_parser.add_argument("--cases", required=True, help="Path to the cases config file.")
    export_parser.add_argument("--case", dest="case_name", default=None, help="Optional single case name to export.")
    export_parser.add_argument("--dry-run", action="store_true", help="Resolve config and output paths without calling Earth Engine.")

    return parser


def _write_json_if_requested(payload: dict[str, object], output_path: str | None) -> None:
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect":
        payload = inspect_legacy_project(args.project_root)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "ca":
        run_result = run_ca_baseline(
            project_root=args.project_root,
            start_date=args.start_date,
            end_date=args.end_date,
            seed=args.seed,
            use_barriers=args.use_barriers,
            use_viirs_init=args.use_viirs_init,
            viirs_history_days=args.viirs_history_days,
        )
        payload = serialize_ca_run(run_result)
        _write_json_if_requested(payload, args.save_json)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "hybrid":
        run_result = run_hybrid_experiment(**_hybrid_run_kwargs(args))
        payload = serialize_hybrid_run(run_result)
        _write_json_if_requested(payload, args.save_json)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "report":
        baseline_run = run_ca_baseline(
            project_root=args.project_root,
            start_date=args.start_date,
            end_date=args.end_date,
            seed=args.seed,
            use_barriers=args.use_barriers,
            use_viirs_init=args.use_viirs_init,
            viirs_history_days=args.viirs_history_days,
        )
        run_result = run_hybrid_experiment(**_hybrid_run_kwargs(args))
        payload = serialize_hybrid_run(run_result)
        _write_json_if_requested(payload, args.save_json)
        report_payload = generate_hybrid_report(run_result, output_dir=args.output_dir, num_days_to_show=args.num_days, ca_run=baseline_run)
        print(json.dumps({"result": payload, "report": report_payload}, indent=2))
        return

    if args.command == "export-gee":
        export_config = load_export_config(args.defaults)
        cases = load_cases(args.cases)
        if args.case_name:
            cases = {args.case_name: cases[args.case_name]}

        results = {}
        for case_name, case_def in cases.items():
            results[case_name] = run_case_export(case_def, export_config, dry_run=args.dry_run)
        print(json.dumps(results, indent=2))
        return

    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()