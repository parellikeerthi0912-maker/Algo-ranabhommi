# Warehouse Robot Planner

Move every robot from its assigned start cell to its assigned goal cell on a
blocked grid, without allowing robots to collide.

Time is discrete. At time `0`, every robot occupies its start cell. Each later
entry in its path describes where it is after one additional time step.

## What happens during evaluation

For each instance, the evaluator:

1. creates your `Planner` class;
2. calls `solve(instance, submit_candidate)` once;
3. validates every complete multi-robot plan you submit;
4. treats the value returned by `solve` as one final candidate;
5. checks paths together on a common time axis; and
6. records the best valid cost available at each scoring checkpoint.

Your code returns paths only. The evaluator independently checks every move,
collision, arrival time, and objective term.

## Grid and instance data

A cell is represented by one integer:

```text
cell = y * width + x
x = cell % width
y = cell // width
```

The `instance` object provides:

| Field | Meaning |
| --- | --- |
| `width`, `height` | Grid dimensions |
| `passable[cell]` | Whether a robot may occupy the cell |
| `blocked` | Sorted blocked-cell IDs |
| `n_robots` | Number of robots |
| `starts[r]` | Start cell of robot `r` |
| `goals[r]` | Goal cell of robot `r` |
| `horizon` | Maximum permitted completion time |
| `lb_makespan` | Published makespan lower bound used by the objective |
| `lb_sum_of_costs` | Published sum-of-costs lower bound |
| `corridor_cells` | Cells included in the congestion calculation |
| `congestion_scale` | Published congestion divisor |
| `profile` | Descriptive instance characteristics used for reporting |

`instance.neighbours(cell)` returns the passable north, south, east, and west
neighbors of a cell. It does not include waiting in place.

## Submission contract

Create `adapters/myteam.py` containing:

```python
class Planner:
    def solve(self, instance, submit_candidate):
        return final_plan
```

A plan contains exactly one path per robot, in robot-index order:

```python
[
    [start_0, cell, cell, ..., goal_0],
    [start_1, cell, cell, ..., goal_1],
]
```

You may also submit `{"paths": paths}`. Every path must be non-empty and contain
only integer cell IDs.

`submit_candidate(paths)` returns a receipt:

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

You may submit multiple complete plans. The return value from `solve` is
validated as one final candidate.

## Movement and collision rules

For robot `r`:

- its first path cell must equal `starts[r]`;
- its final path cell must equal `goals[r]`;
- every occupied cell must be inside the grid and passable;
- one step may wait in place or move exactly one cell north, south, east, or
  west; and
- the path may contain no more than `horizon + 1` cells.

All robot paths are evaluated together. If paths have different lengths, each
shorter path is extended by repeating its final goal cell. Consequently, a
robot continues occupying its goal after it finishes.

At every time step:

- two robots may not occupy the same cell; and
- two robots may not exchange positions across the same edge.

The second case is an edge-swap conflict: if one robot moves `a -> b` while
another moves `b -> a` in the same step, the plan is invalid.

## Objective

A robot's arrival time is the first time in the final uninterrupted suffix in
which it remains at its goal. Earlier visits do not count if the robot later
leaves the goal.

```text
makespan     = maximum robot arrival time
sum_of_costs = sum of all robot arrival times
```

For congestion, the evaluator records which distinct robots visit each
`corridor_cell` up to their arrival times. If `k` robots visit the same corridor
cell, that cell contributes `k * (k - 1) // 2`. Contributions are summed across
all corridor cells.

The authoritative integer cost is:

```text
cost_units = 750000 * makespan     // lb_makespan
           + 150000 * sum_of_costs // lb_sum_of_costs
           + 100000 * congestion   // congestion_scale
```

The displayed cost is `cost_units / 1,000,000`. Lower cost is better.

## Time and scoring

The public per-instance budget is:

```text
4.0 + number_of_robots / 12.0 seconds
```

Plan quality is measured at `1%`, `10%`, `50%`, and `100%` of that budget,
with respective weights `30%`, `30%`, `20%`, and `20%`. At each checkpoint,
the best valid candidate received by that time is used.

Each checkpoint cost is normalized between two anchors in
`public_reference.json`: the published baseline corresponds to `0.25`, the
organizer reference corresponds to `1.00`, missing output scores `0`, and the
result is clipped to `[0, 1]`.

The suite combines mean plan quality and performance on shifted instance
families using a `70/30` point split. Reproducibility, valid-candidate rate,
instance integrity, and staying within budget form an engineering multiplier;
they cannot add points, but failures can reduce the total.

The public suite contains nine instances. The public runner is a practice
estimate; official grading may include fresh unpublished instances or seeds.

## Submission integrity

AI coding tools are allowed, but your adapter must be a genuine multi-robot
planning algorithm that works from the supplied instance. Submit exactly one
readable Python adapter file. The organizers will review, hash, and rerun that
exact file.

Do not mutate the instance or any benchmark/evaluator object. Do not access
private attributes of `submit_candidate`, evaluator closures, caller frames,
globals, or other runtime internals. Do not call or reproduce generator
internals to recover a construction witness, and do not use instance names,
seeds, profiles, or digests to select hardcoded or precomputed paths.
Monkeypatching, reflection, dynamic imports, `eval`/`exec`, subprocesses,
filesystem or network access, environment inspection, native-code loading,
encoded payloads, and changes to clocks, limits, reports, or scoring state are
prohibited.

Calling `submit_candidate(paths)` and reading its receipt is allowed. Normal
path-planning code and the documented helpers in `adapter.py` are allowed.
Unreadable or unexplained generated code may be rejected, and a rules violation
may disqualify a submission regardless of its reported score.

## Running locally

Run these commands from `bench-warehouse/`:

```bash
python self_check.py --adapter adapters.myteam:Planner
python run.py --adapter adapters.myteam:Planner --out report.json
```

`self_check.py` tests two smaller public instances. `run.py` evaluates the full
public suite and writes a detailed JSON report.

## Relevant files

```text
adapter.py             submission interface and public path utilities
data.py                deterministic public instance generation
validator.py           authoritative movement, collision, and objective checks
harness.py             adapter execution and candidate handling
public_reference.json  public scoring anchors
adapters/               examples and your submitted adapter location
```
