from __future__ import annotations

import argparse
import json


def build_parser(stage_name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=stage_name)
    parser.add_argument("--config", help="Path to the training config.")
    parser.add_argument("--manifest", help="Optional dataset manifest override.")
    return parser


def placeholder_main(stage_name: str, objective: str) -> None:
    parser = build_parser(stage_name)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "stage": stage_name,
                "objective": objective,
                "config": args.config,
                "manifest": args.manifest,
                "status": "placeholder",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
