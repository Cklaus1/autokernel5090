#!/usr/bin/env python3
"""
CLI tool for ModelSpec: show, launch, compare, validate, and manage serving specs.

Usage:
    python3 tools/spec_cli.py show <spec.yaml>
    python3 tools/spec_cli.py launch <spec.yaml> [--output serve.sh]
    python3 tools/spec_cli.py docker <spec.yaml>
    python3 tools/spec_cli.py compare <spec_a.yaml> <spec_b.yaml>
    python3 tools/spec_cli.py validate <spec.yaml>
    python3 tools/spec_cli.py list-presets
    python3 tools/spec_cli.py generate-presets [--dir specs/]
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.model_spec import ModelSpec, PRESETS


def cmd_show(args: argparse.Namespace) -> None:
    """Display a spec in human-readable form."""
    spec = ModelSpec.load(args.spec)
    print(f"=== {spec.model_name} ===")
    print(f"Summary: {spec.summary()}")
    print()

    d = spec.to_dict()
    for section, values in d.items():
        if section in ("description", "notes"):
            continue
        print(f"[{section}]")
        if isinstance(values, dict):
            for k, v in values.items():
                print(f"  {k}: {v}")
        print()

    if spec.description:
        print(f"Description: {spec.description}")
    if spec.notes:
        print("Notes:")
        for note in spec.notes:
            print(f"  - {note}")


def cmd_launch(args: argparse.Namespace) -> None:
    """Generate a launch script from a spec."""
    spec = ModelSpec.load(args.spec)
    output = args.output or f"serve_{spec.model_name.lower().replace('-', '_')}.sh"
    spec.to_launch_script(output)
    print(f"Launch script written to: {output}")
    print(f"Run with: bash {output}")


def cmd_docker(args: argparse.Namespace) -> None:
    """Print the docker run command for a spec."""
    spec = ModelSpec.load(args.spec)
    print(spec.to_docker_command())


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare two specs."""
    a = ModelSpec.load(args.spec_a)
    b = ModelSpec.load(args.spec_b)
    print(ModelSpec.compare(a, b))


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate a spec for internal consistency."""
    spec = ModelSpec.load(args.spec)
    try:
        warnings = spec.validate()
        if warnings:
            print(f"Spec is valid with {len(warnings)} warning(s):")
            for w in warnings:
                print(f"  {w}")
        else:
            print("Spec is valid. No issues found.")
    except ValueError as e:
        print(f"VALIDATION FAILED:\n{e}", file=sys.stderr)
        sys.exit(1)


def cmd_list_presets(args: argparse.Namespace) -> None:
    """List available preset configurations."""
    print("Available presets:")
    print()
    for name, factory in PRESETS.items():
        spec = factory()
        print(f"  {name}")
        print(f"    {spec.summary()}")
        if spec.description:
            print(f"    {spec.description}")
        print()


def cmd_generate_presets(args: argparse.Namespace) -> None:
    """Generate YAML files for all presets."""
    out_dir = Path(args.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, factory in PRESETS.items():
        spec = factory()
        path = out_dir / f"{name}.yaml"
        spec.save(path)
        print(f"  {path}")

    print(f"\nGenerated {len(PRESETS)} spec files in {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ModelSpec CLI: manage declarative serving configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python3 tools/spec_cli.py show specs/gemma4_nvfp4_rtx5090_throughput.yaml
              python3 tools/spec_cli.py docker specs/gemma4_nvfp4_rtx5090_throughput.yaml
              python3 tools/spec_cli.py compare specs/throughput.yaml specs/latency.yaml
              python3 tools/spec_cli.py validate specs/gemma4_nvfp4_rtx5090_throughput.yaml
              python3 tools/spec_cli.py generate-presets --dir specs/
        """),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # show
    p = sub.add_parser("show", help="Display a spec in human-readable form")
    p.add_argument("spec", help="Path to spec YAML/JSON file")

    # launch
    p = sub.add_parser("launch", help="Generate a bash launch script")
    p.add_argument("spec", help="Path to spec YAML/JSON file")
    p.add_argument("--output", "-o", help="Output script path (default: auto-named)")

    # docker
    p = sub.add_parser("docker", help="Print docker run command")
    p.add_argument("spec", help="Path to spec YAML/JSON file")

    # compare
    p = sub.add_parser("compare", help="Compare two specs")
    p.add_argument("spec_a", help="First spec file")
    p.add_argument("spec_b", help="Second spec file")

    # validate
    p = sub.add_parser("validate", help="Validate a spec")
    p.add_argument("spec", help="Path to spec YAML/JSON file")

    # list-presets
    sub.add_parser("list-presets", help="List available preset configurations")

    # generate-presets
    p = sub.add_parser("generate-presets", help="Generate YAML files for all presets")
    p.add_argument("--dir", default="specs", help="Output directory (default: specs/)")

    args = parser.parse_args()

    commands = {
        "show": cmd_show,
        "launch": cmd_launch,
        "docker": cmd_docker,
        "compare": cmd_compare,
        "validate": cmd_validate,
        "list-presets": cmd_list_presets,
        "generate-presets": cmd_generate_presets,
    }
    commands[args.command](args)


import textwrap

if __name__ == "__main__":
    main()
