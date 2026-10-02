"""The two commands every benchmark ships: ``self_check.py`` and ``run.py``.

Both are identical across the three benchmarks once the data, metrics and
harness modules are supplied, so they live here and each benchmark's scripts
are a few lines of configuration.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from .loader import AdapterError, load_adapter, reject_private_path
from .report import banner, table, write_report


@dataclass
class BenchCli:
    """What the shared runners need to know about one benchmark."""

    benchmark: str  # slug used in the report
    title: str  # human-readable heading
    quality_label: str  # e.g. "route quality over time (70)"
    robustness_label: str
    engineering_label: str
    data: Any  # module: public_instances, SELF_CHECK_NAMES, budget_for, ...
    metrics: Any  # module: load_anchors, score_instance, score_suite, ...
    harness: Any  # module: run_one, check_determinism, set_verbose_errors
    reference_path: str
    baseline_factory: Callable[[], Any]  # for provisional practice scoring
    validity_hint: str  # printed when a submission produces nothing valid
    cost_name: str = "cost"


# ---------------------------------------------------------------------------
# self_check
# ---------------------------------------------------------------------------


def self_check_main(cli: BenchCli, argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"fast sanity run for {cli.title}: two small public instances"
    )
    parser.add_argument("--adapter", required=True, help="e.g. adapters.myteam:Solver")
    parser.add_argument("--budget-scale", type=float, default=1.0)
    parser.add_argument("--verbose-errors", action="store_true")
    args = parser.parse_args(argv)

    cli.harness.set_verbose_errors(args.verbose_errors)

    try:
        adapter = load_adapter(args.adapter)
    except AdapterError as exc:
        print(f"adapter problem: {exc}", file=sys.stderr)
        return 2

    try:
        anchors = cli.metrics.load_anchors(cli.reference_path)
    except cli.metrics.MissingAnchors as exc:
        print(f"cannot score: {exc}", file=sys.stderr)
        return 2

    instances = cli.data.public_instances(cli.data.SELF_CHECK_NAMES)
    print(banner(f"self-check: {args.adapter}"))

    rows = []
    records = []
    for instance in instances:
        budget = cli.data.budget_for(instance) * args.budget_scale
        record = cli.harness.run_one(instance, adapter, budget_s=budget)
        records.append(record)
        entry = cli.metrics.score_instance(record, anchors[instance.name])
        rows.append([
            instance.name,
            instance.size,
            round(budget, 1),
            "yes" if record.produced_valid else "NO",
            entry["best_cost"],
            entry["baseline_cost"],
            entry["reference_cost"],
            entry["score"],
        ])
        if record.error:
            print(f"  ! {instance.name}: {record.error}")
        if record.sink.get("rejected"):
            print(f"  ! {instance.name}: {record.sink['rejected']} rejected "
                  f"candidate(s): {record.sink['rejection_reasons']}")

    print()
    print(table(
        ["instance", "size", "budget", "valid", cli.cost_name,
         "baseline", "reference", "score"],
        rows,
    ))

    result = cli.metrics.score_suite(records, anchors, deterministic=None)
    print()
    print(f"mean instance score : {result['aggregate']['quality_mean']:.3f}")
    print(f"first valid output  : "
          f"{[r.sink.get('first_valid_at_s') for r in records]}")

    if not all(r.produced_valid for r in records):
        print("\nFAIL: at least one instance produced nothing valid.")
        print(cli.validity_hint)
        return 1

    print("\nOK: valid on both instances. Run run.py for the full public suite.")
    return 0


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def run_main(cli: BenchCli, argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=f"full public evaluation for {cli.title}"
    )
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", default="report.json")
    parser.add_argument("--seeds", nargs="*", type=int, default=None,
                        help="extra practice instances, scored provisionally")
    parser.add_argument("--only", nargs="*", default=None,
                        help="restrict the public suite to these instance names")
    parser.add_argument("--budget-scale", type=float, default=1.0,
                        help="multiply every per-instance budget (for quick passes)")
    parser.add_argument("--memory-mb", type=int, default=2048)
    parser.add_argument("--no-determinism-check", action="store_true")
    parser.add_argument("--verbose-errors", action="store_true")
    args = parser.parse_args(argv)

    reject_private_path(args.out)
    cli.harness.set_verbose_errors(args.verbose_errors)

    try:
        adapter = load_adapter(args.adapter)
        anchors = dict(cli.metrics.load_anchors(cli.reference_path))
    except (AdapterError, cli.metrics.MissingAnchors) as exc:
        print(f"{exc}", file=sys.stderr)
        return 2

    instances = cli.data.public_instances(args.only)
    print(banner(f"{cli.title} -- {args.adapter}"))
    print(f"{len(instances)} public instances, budget scale {args.budget_scale}")

    records: List[Any] = []
    for instance in instances:
        budget = cli.data.budget_for(instance) * args.budget_scale
        print(f"  running {instance.name:26s} size={instance.size:5d} "
              f"budget={budget:5.1f}s ... ", end="", flush=True)
        record = cli.harness.run_one(
            instance, adapter, budget_s=budget, memory_mb=args.memory_mb
        )
        records.append(record)
        if record.produced_valid:
            entry = cli.metrics.score_instance(record, anchors[instance.name])
            print(f"{cli.cost_name}={entry['best_cost']}  ({record.wall_s:.1f}s)")
        else:
            print(f"NOTHING VALID  ({record.error or 'rejected'})")

    deterministic = None
    if not args.no_determinism_check:
        probe = instances[0]
        print(f"\n  determinism replay on {probe.name} ... ", end="", flush=True)
        deterministic, first, second = cli.harness.check_determinism(
            probe, adapter, budget_s=cli.data.budget_for(probe) * args.budget_scale
        )
        # "stable" means the score reproduces (checkpoint + final cost), not
        # that the raw candidate stream is bit-for-bit identical -- a solver
        # that legitimately bounds its own work by wall-clock remaining time
        # can differ there and still be perfectly stable. See
        # benchkit.runner.scored_output_stability.
        print("stable" if deterministic else f"UNSTABLE (streams: {first} vs {second})")

    result = cli.metrics.score_suite(records, anchors, deterministic=deterministic)

    print(banner("public suite"))
    print(cli.metrics.format_summary(result))

    practice = _practice(cli, args, adapter, anchors) if args.seeds else None

    points = result["points"]
    print(banner("score (public estimate)"))
    print(table(
        ["axis", "points"],
        [
            [cli.quality_label, points["quality"]],
            [cli.robustness_label, points["robustness"]],
            [cli.engineering_label, points["engineering"]],
            ["TOTAL", points["total"]],
        ],
    ))
    if points["penalties"]:
        print(f"\npenalties applied: {', '.join(points['penalties'])}")
    print(f"\nengineering breakdown: {result['aggregate']['engineering_parts']}")
    print(
        "\nThis is a public estimate only. The private suite weights shifted\n"
        "families far more heavily, so a score that leans on the easy public\n"
        "families will not survive it."
    )

    write_report(
        args.out,
        cli.benchmark,
        {
            "adapter": args.adapter,
            "budget_scale": args.budget_scale,
            "generator_schema": cli.data.SCHEMA_VERSION,
            "result": result,
            "practice": practice,
        },
    )
    print(f"\nwrote {args.out}")
    return 0


def _practice(cli: BenchCli, args, adapter, anchors) -> dict:
    """Score self-generated instances against a live baseline run.

    These have no published anchors, so the baseline is run here and the
    reference point is projected from the public suite's ratio. Provisional by
    construction -- useful for iterating, not for predicting a placing.
    """
    ratios = [
        float(a["reference_cost_units"]) / float(a["baseline_cost_units"])
        for a in anchors.values()
        if a["baseline_cost_units"]
    ]
    projected = statistics.median(ratios)

    print(banner("practice seeds (provisional)"))
    print(f"reference projected as {projected:.3f} x the live baseline cost")

    rows = []
    entries = []
    for seed in args.seeds:
        instance = cli.data.practice_instance(seed)
        budget = cli.data.budget_for(instance) * args.budget_scale
        baseline = cli.harness.run_one(
            instance, cli.baseline_factory, budget_s=budget
        )
        record = cli.harness.run_one(
            instance, adapter, budget_s=budget, memory_mb=args.memory_mb
        )
        if baseline.final_cost is None:
            rows.append([instance.name, instance.size, "--", "--", "--"])
            continue
        entry = cli.metrics.score_instance(
            record,
            {
                "baseline_cost_units": baseline.final_cost,
                "reference_cost_units": baseline.final_cost * projected,
            },
        )
        entry["provisional"] = True
        entries.append(entry)
        rows.append([
            instance.name, instance.size,
            entry["best_cost"], entry["baseline_cost"], entry["score"],
        ])

    print(table(["instance", "size", cli.cost_name, "live baseline", "score"], rows))
    return {"projected_reference_ratio": projected, "instances": entries}
