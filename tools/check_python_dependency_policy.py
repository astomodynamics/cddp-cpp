#!/usr/bin/env python3
"""Enforce a conservative dependency policy for Python packaging files."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
LOCK_PATH = REPO_ROOT / "uv.lock"
ALLOWLIST_PATH = REPO_ROOT / "security" / "python-direct-deps-allowlist.txt"
PYPI_SIMPLE = "https://pypi.org/simple"
NAME_RE = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def extract_requirement_name(spec: str) -> str:
    match = NAME_RE.match(spec)
    if not match:
        raise ValueError(f"Could not parse requirement name from {spec!r}")
    return normalize_name(match.group(1))


def is_direct_reference(spec: str) -> bool:
    lowered = spec.lower()
    return (
        " @" in spec
        or lowered.startswith(("git+", "http://", "https://", "file:"))
        or "git+" in lowered
        or "://" in spec
    )


def load_toml(path: Path) -> dict:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def load_allowlist(path: Path) -> set[str]:
    allowed: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        allowed.add(normalize_name(line))
    return allowed


def iter_dependency_specs(pyproject: dict) -> Iterable[tuple[str, str]]:
    build_system = pyproject.get("build-system", {})
    for spec in build_system.get("requires", []):
        yield ("build-system.requires", spec)

    project = pyproject.get("project", {})
    for spec in project.get("dependencies", []):
        yield ("project.dependencies", spec)

    for group, specs in project.get("optional-dependencies", {}).items():
        for spec in specs:
            yield (f"project.optional-dependencies.{group}", spec)

    for group, specs in pyproject.get("dependency-groups", {}).items():
        for spec in specs:
            yield (f"dependency-groups.{group}", spec)


def collect_direct_dependency_names(pyproject: dict) -> set[str]:
    names: set[str] = set()
    for _, spec in iter_dependency_specs(pyproject):
        names.add(extract_requirement_name(spec))
    return names


def load_base_pyproject(base_ref: str) -> dict | None:
    result = subprocess.run(
        ["git", "show", f"{base_ref}:pyproject.toml"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return tomllib.loads(result.stdout)


def load_base_lock(base_ref: str) -> dict | None:
    result = subprocess.run(
        ["git", "show", f"{base_ref}:uv.lock"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return tomllib.loads(result.stdout)


def validate_dependency_specs(pyproject: dict) -> list[str]:
    errors: list[str] = []
    for source, spec in iter_dependency_specs(pyproject):
        try:
            extract_requirement_name(spec)
        except ValueError as exc:
            errors.append(f"{source}: {exc}")
            continue

        if is_direct_reference(spec):
            errors.append(
                f"{source}: direct URL/git/file references are not allowed: {spec!r}"
            )
    return errors


def validate_lockfile(lock_data: dict) -> list[str]:
    errors: list[str] = []
    for package in lock_data.get("package", []):
        name = package.get("name", "<unknown>")
        source = package.get("source")
        if source is not None:
            if source.get("editable") == ".":
                continue

            registry = source.get("registry")
            if registry != PYPI_SIMPLE:
                errors.append(
                    f"uv.lock package {name!r} uses non-PyPI source {source!r}"
                )

        sdist = package.get("sdist")
        if sdist and "hash" not in sdist:
            errors.append(f"uv.lock package {name!r} is missing an sdist hash")

        for wheel in package.get("wheels", []):
            if "hash" not in wheel:
                errors.append(f"uv.lock package {name!r} has a wheel without a hash")
    return errors


def collect_lock_package_names(lock_data: dict | None) -> set[str]:
    if not lock_data:
        return set()
    return {
        normalize_name(package["name"])
        for package in lock_data.get("package", [])
        if "name" in package
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-ref",
        default="origin/master",
        help="Git ref used as the review baseline.",
    )
    args = parser.parse_args()

    pyproject = load_toml(PYPROJECT_PATH)
    lock_data = load_toml(LOCK_PATH)
    allowlist = load_allowlist(ALLOWLIST_PATH)
    base_pyproject = load_base_pyproject(args.base_ref)
    base_lock = load_base_lock(args.base_ref)

    errors = []
    errors.extend(validate_dependency_specs(pyproject))
    errors.extend(validate_lockfile(lock_data))

    current_direct = collect_direct_dependency_names(pyproject)
    base_direct = collect_direct_dependency_names(base_pyproject) if base_pyproject else set()
    added_direct = sorted(current_direct - base_direct)
    unapproved_direct = [name for name in added_direct if name not in allowlist]
    if unapproved_direct:
        errors.append(
            "New top-level Python dependencies must be explicitly allowlisted in "
            f"{ALLOWLIST_PATH.relative_to(REPO_ROOT)}: "
            + ", ".join(unapproved_direct)
        )

    current_locked = collect_lock_package_names(lock_data)
    base_locked = collect_lock_package_names(base_lock)
    added_locked = sorted(current_locked - base_locked)

    if errors:
        print("Dependency policy check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        if added_direct:
            print(
                "Direct dependencies added in this change: "
                + ", ".join(added_direct),
                file=sys.stderr,
            )
        if added_locked:
            print(
                "New locked packages introduced in this change: "
                + ", ".join(added_locked),
                file=sys.stderr,
            )
        return 1

    print("Dependency policy check passed.")
    if added_direct:
        print("Direct dependencies added for review: " + ", ".join(added_direct))
    if added_locked:
        print("New locked packages introduced for review: " + ", ".join(added_locked))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
