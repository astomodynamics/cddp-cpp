#!/usr/bin/env python3
"""Generate animated Python portfolio demos for the pycddp bindings."""

from __future__ import annotations

import argparse
from pathlib import Path

import python_portfolio_lib as portfolio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--demo",
        choices=["all", *sorted(portfolio.DEMO_BUILDERS)],
        default="all",
        help="Which demo to generate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/python_portfolio"),
        help="Directory where GIFs will be written.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Animation frame rate.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=110,
        help="Output DPI for the generated GIFs.",
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=2,
        help="Use every Nth solver state as an animation frame.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    demo_names = (
        sorted(portfolio.DEMO_BUILDERS)
        if args.demo == "all"
        else [args.demo]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name in demo_names:
        result = portfolio.build_demo(name)
        output_path = args.output_dir / f"{result.slug}.gif"
        portfolio.save_animation(
            result,
            output_path,
            fps=args.fps,
            dpi=args.dpi,
            frame_step=args.frame_step,
        )
        print(
            f"{result.title}: {output_path} "
            f"(solver={result.solver_name}, final_error={result.final_error:.4f})"
        )


if __name__ == "__main__":
    main()
