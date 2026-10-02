# Factory Scheduler

Choose the processing order of manufacturing operations on shared machines.
Every job is an ordered chain of operations, and every operation requires one
specific machine for a fixed duration.

All times are non-negative integers. Operations have global integer IDs from
`0` through `instance.total_ops - 1`.

## What happens during evaluation

For each instance, the evaluator:

1. creates your `Scheduler` class;
2. calls `solve(instance, submit_candidate)` once;
3. converts each candidate into one operation order per machine;
4. constructs the earliest schedule consistent with those orders;
5. rejects incomplete, misplaced, or cyclic orders;
6. treats the value returned by `solve` as one final candidate; and
7. records the best valid cost available at each scoring checkpoint.

Your code chooses operation order. The evaluator owns the resulting start
times, completion times, feasibility verdict, and cost.

## Instance data

The `instance` object provides:

| Field | Meaning |
| --- | --- |
| `n_jobs`, `n_machines` | Numbers of jobs and machines |
| `total_ops` | Total number of operations |
| `op_job[op]` | Job containing operation `op` |
| `op_machine[op]` | Machine required by operation `op` |
| `op_duration[op]` | Processing duration of operation `op` |
| `op_group[op]` | Setup group of operation `op` |
| `job_ops[j]` | Operations of job `j`, in required precedence order |
| `machine_ops[m]` | Operations requiring machine `m` |
| `release[j]` | Earliest time at which job `j` may begin |
| `due[j]` | Due time of job `j` |
| `weight[j]` | Tardiness weight of job `j` |
| `setup_machines` | Machines on which group changes require setup time |
| `setup_matrix[a][b]` | Setup time when group `a` is followed by group `b` |
| `lb_makespan` | Published makespan lower bound used by the objective |
| `twt_scale` | Published tardiness scale used by the objective |
| `profile` | Descriptive instance characteristics used for reporting |

An operation cannot begin before its job is released, before the preceding
operation of its job completes, or before the previous operation on its
machine completes. When applicable, setup time is added between consecutive
machine operations.

## Submission contract

Create `adapters/myteam.py` containing:

```python
class Scheduler:
    def solve(self, instance, submit_candidate):
        return final_schedule
```

The primary representation is one operation sequence per machine:

```python
{
    "sequences": [
        [operation_id, operation_id, ...],  # machine 0
        [operation_id, operation_id, ...],  # machine 1
    ]
}
```

A plain list of machine sequences is also accepted. Every operation must appear
exactly once and must be placed in the sequence of its required machine.

Alternatively, submit one proposed start value per operation:

```python
{"starts": [t0, t1, t2, ...]}
```

A mapping from operation ID to proposed start value is also accepted:

```python
{"starts": {0: t0, 1: t1, 2: t2, ...}}
```

The proposed values are used only to sort operations on each machine. They are
then discarded. The evaluator recomputes the earliest possible schedule for
the implied machine orders, so inserted idle time has no effect.

`submit_candidate(schedule)` returns a receipt:

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

You may submit multiple candidates. The return value from `solve` is validated
as one final candidate.

## Canonical scheduling and validity

The evaluator builds a directed graph containing:

- every job-precedence edge; and
- every consecutive pair in each submitted machine sequence.

If this graph contains a cycle, the schedule is rejected as a deadlock. If it
is acyclic, operations are processed in topological order and each operation is
started as early as its release, job predecessor, machine predecessor, and
required setup permit.

A candidate is valid only when:

- every operation ID is an integer in the published range;
- every operation appears exactly once;
- every operation is assigned to its required machine; and
- the combined job and machine orders are acyclic.

## Objective

For the canonical schedule:

```text
makespan = latest job completion time

total_weighted_tardiness =
    sum(weight[j] * max(0, completion[j] - due[j]))
```

The authoritative integer cost is:

```text
cost_units = 650000 * makespan // lb_makespan
           + 350000 * total_weighted_tardiness // twt_scale
```

The displayed cost is `cost_units / 1,000,000`. Lower cost is better.

## Time and scoring

The public per-instance budget is:

```text
4.0 + total_operations / 150.0 seconds
```

Schedule quality is measured at `1%`, `10%`, `50%`, and `100%` of that budget,
with respective weights `30%`, `30%`, `20%`, and `20%`. At each checkpoint,
the best valid candidate received by that time is used.

Each checkpoint cost is normalized between two anchors in
`public_reference.json`: the published baseline corresponds to `0.25`, the
organizer reference corresponds to `1.00`, missing output scores `0`, and the
result is clipped to `[0, 1]`.

The suite combines mean schedule quality and performance on shifted instance
families using a `65/35` point split. Reproducibility, valid-candidate rate,
instance integrity, and staying within budget form an engineering multiplier;
they cannot add points, but failures can reduce the total.

The public runner is a practice estimate. Official grading may include fresh
unpublished instances or seeds.

## Submission integrity

AI coding tools are allowed, but your adapter must be a genuine scheduling
algorithm that works from the supplied instance. Submit exactly one readable
Python adapter file. The organizers will review, hash, and rerun that exact
file.

Do not mutate the instance or any benchmark/evaluator object. Do not access
private attributes of `submit_candidate`, evaluator closures, caller frames,
globals, or other runtime internals. Do not call or reproduce generator
internals to recover a construction witness, and do not use instance names,
seeds, profiles, or digests to select hardcoded or precomputed schedules.
Monkeypatching, reflection, dynamic imports, `eval`/`exec`, subprocesses,
filesystem or network access, environment inspection, native-code loading,
encoded payloads, and changes to clocks, limits, reports, or scoring state are
prohibited.

Calling `submit_candidate(schedule)` and reading its receipt is allowed. Normal
scheduling code and the documented helpers in `adapter.py` are allowed.
Unreadable or unexplained generated code may be rejected, and a rules violation
may disqualify a submission regardless of its reported score.

## Running locally

Run these commands from `bench-factory/`:

```bash
python self_check.py --adapter adapters.myteam:Scheduler
python run.py --adapter adapters.myteam:Scheduler --out report.json
```

`self_check.py` tests two smaller public instances. `run.py` evaluates the full
public suite and writes a detailed JSON report.

## Relevant files

```text
adapter.py             submission interface and public representation helpers
data.py                deterministic public instance generation
validator.py           canonical scheduling, feasibility, and objective
harness.py             adapter execution and candidate handling
public_reference.json  public scoring anchors
adapters/               examples and your submitted adapter location
```
