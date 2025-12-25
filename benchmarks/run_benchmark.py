import argparse
import importlib
from pathlib import Path

def discover_benchmarks():
    bench_dir = Path(__file__).parent
    files = sorted(
        p.stem
        for p in bench_dir.glob("*_set_*.py")
        if p.is_file()
    )
    return files


def run_benchmark(module_name: str, output_dir: Path):
    print(f"\n Running benchmark: {module_name}")
    mod = importlib.import_module(f"benchmarks.{module_name}")

    if not hasattr(mod, "run"):
        raise RuntimeError(
            f"{module_name}.py must define: run(output_dir)"
        )

    mod.run(str(output_dir))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        default="",
        help="Optional substring filter (e.g. 'class', 'regression', 'set_1')",
    )
    args = parser.parse_args()

    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = discover_benchmarks()

    if args.pattern:
        benchmarks = [b for b in benchmarks if args.pattern in b]

    if not benchmarks:
        raise RuntimeError("No benchmarks matched the given pattern.")

    print("Discovered benchmarks:")
    for b in benchmarks:
        print(f"  - {b}")

    for b in benchmarks:
        run_benchmark(b, output_dir)


if __name__ == "__main__":
    main()
