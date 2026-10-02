# Delivery Route Planner

Plan one delivery route for each vehicle so that every customer is served
exactly once and every operational constraint is satisfied.

All times and distances inside the instance are integer seconds. Customer `0`
is the depot; delivery customers are numbered `1` through `instance.n`.

## What happens during evaluation

For each instance, the evaluator:

1. creates your `Router` class;
2. calls `solve(instance, submit_candidate)` once;
3. validates every candidate passed to `submit_candidate`;
4. treats the value returned by `solve` as one final candidate;
5. records the best valid cost available at the scoring checkpoints; and
6. recomputes all feasibility checks and costs itself.

Your code never supplies an objective value. Only the routes you return are
used.

## Instance data

The `instance` object provides:

| Field | Meaning |
| --- | --- |
| `n` | Number of customers, excluding the depot |
| `xs`, `ys` | Coordinates indexed by node; index `0` is the depot |
| `demand[c]` | Load delivered to customer `c` |
| `service[c]` | Service duration at customer `c` |
| `tw_open[c]`, `tw_close[c]` | Inclusive delivery window for arrival at customer `c` |
| `capacities[v]` | Capacity of vehicle `v` |
| `starts[v]` | Time at which vehicle `v` leaves the depot |
| `num_vehicles` | Number of available vehicles |
| `max_duration` | Maximum route duration, measured from the vehicle's start |
| `horizon` | Latest permitted return time at the depot |
| `profile` | Descriptive instance characteristics used for reporting |

Travel time between two nodes is the integer Euclidean distance:

```text
isqrt((x1 - x2)^2 + (y1 - y2)^2)
```

## Submission contract

Create `adapters/myteam.py` containing:

```python
class Router:
    def solve(self, instance, submit_candidate):
        return final_plan
```

A plan is a sequence of routes in vehicle order. Route `v` is driven by vehicle
`v`. Do not include the depot in a route.

```python
[
    [3, 7, 2],       # vehicle 0
    [],              # vehicle 1 is unused
    [5, 1, 9, 4],    # vehicle 2
]
```

You may also wrap the routes as `{"routes": routes}`. Trailing unused vehicles
may be omitted, but a plan cannot contain more routes than there are vehicles.

`submit_candidate(plan)` returns a receipt:

```python
{
    "accepted": bool,
    "reason": str | None,
    "cost": int | None,
    "best": int | None,
    "elapsed_s": float,
    "remaining_s": float,
}
```

You may submit multiple candidates. A rejected candidate is ignored for route
quality but is recorded by the evaluator. The returned value from `solve` is
validated in the same way as a submitted candidate.

## Validity rules

A plan is valid only when all of the following hold:

- every customer ID is an integer from `1` through `n`;
- every customer appears exactly once across all routes;
- route `v` carries no more than `capacities[v]`;
- vehicle `v` begins at the depot at `starts[v]`;
- travel, waiting, and service are applied in route order;
- arriving before `tw_open[c]` causes the vehicle to wait;
- arriving after `tw_close[c]` invalidates the plan;
- after its last customer, every vehicle returns to the depot;
- no route exceeds `max_duration`; and
- no vehicle returns after `horizon`.

Arrival exactly at a window's closing time is allowed. Waiting time counts
toward route duration.

## Objective

For every valid plan, the evaluator totals travel and waiting across all
vehicles:

```text
cost_units = 100 * total_travel_seconds
           +  15 * total_waiting_seconds
```

The displayed value is:

```text
minutes = cost_units / 6000
```

This is equivalent to travel minutes plus `0.15` times waiting minutes. Lower
cost is better.

## Time and scoring

The public per-instance budget is:

```text
4.0 + number_of_customers / 40.0 seconds
```

Route quality is measured at `1%`, `10%`, `50%`, and `100%` of that budget,
with respective weights `30%`, `30%`, `20%`, and `20%`. At each checkpoint,
the best valid candidate received by that time is used.

Each checkpoint cost is normalized between two published anchors in
`public_reference.json`:

- the published baseline corresponds to normalized score `0.25`;
- the organizer reference corresponds to normalized score `1.00`;
- missing or invalid output scores `0`; and
- values are clipped to the range `[0, 1]`.

The suite combines mean route quality and performance on shifted instance
families using a `70/30` point split. Reproducibility, valid-candidate rate,
instance integrity, and staying within the time budget form an engineering
multiplier; they cannot add points, but failures can reduce the total.

The public runner is a practice estimate. Official grading may include fresh
unpublished instances or seeds.

## Submission integrity

AI coding tools are allowed, but your adapter must be a genuine routing
algorithm that works from the supplied instance. Submit exactly one readable
Python adapter file. The organizers will review, hash, and rerun that exact
file.

Do not mutate the instance or any benchmark/evaluator object. Do not access
private attributes of `submit_candidate`, evaluator closures, caller frames,
globals, or other runtime internals. Do not call or reproduce generator
internals to recover a construction witness, and do not use instance names,
seeds, profiles, or digests to select hardcoded or precomputed routes.
Monkeypatching, reflection, dynamic imports, `eval`/`exec`, subprocesses,
filesystem or network access, environment inspection, native-code loading,
encoded payloads, and changes to clocks, limits, reports, or scoring state are
prohibited.

Calling `submit_candidate(plan)` and reading its receipt is allowed. Normal
routing code and the documented helpers in `adapter.py` are allowed. Unreadable
or unexplained generated code may be rejected, and a rules violation may
disqualify a submission regardless of its reported score.

## Running locally

Run these commands from `bench-delivery/`:

```bash
python self_check.py --adapter adapters.myteam:Router
python run.py --adapter adapters.myteam:Router --out report.json
```

`self_check.py` tests two smaller public instances. `run.py` evaluates the full
public suite and writes a detailed JSON report.

## Relevant files

```text
adapter.py             submission interface and public data helpers
data.py                deterministic public instance generation
validator.py           authoritative feasibility and objective calculation
harness.py             adapter execution and candidate handling
public_reference.json  public scoring anchors
adapters/               examples and your submitted adapter location
```
