# Algorithm Challenges

Four self-contained optimization challenges. Each challenge has its own instance generator, validator, benchmark runner, and adapter interface.

| Folder | Challenge |
| --- | --- |
| `delivery-route-planner/` | Build delivery routes that satisfy vehicle, capacity, time-window, and return constraints. |
| `factory-scheduler/` | Sequence manufacturing operations while respecting job precedence and shared machines. |
| `warehouse-robot-planner/` | Route robots through a warehouse without vertex or edge conflicts. |
| `adaptive-recall-engine/` | Choose per-dimension trust weights that help a frozen recall engine recover damaged stored patterns. |

Each challenge README defines the objective, submission contract, and local commands. Do not edit benchmark code, generators, validators, or runners. Submit only the named adapter file for the challenge you choose.

## Competition and submission integrity

AI coding tools are allowed for development. The submitted work must still be a
genuine, general-purpose solution to the selected optimization problem.

The score printed by a public runner is a practice result only. Official results
are produced by the organizers from the exact submitted adapter file, using
organizer-controlled runs that may include fresh unpublished instances or seeds.
Participants must not report or alter a locally produced score as an official
result.

Submit exactly one readable Python adapter file. The organizers will review it
before execution, hash it, and rerun that exact file. Deliberately obscured code
or unexplained generated payloads may be rejected. Embedded learned parameters
are allowed when they implement a general model and are disclosed; tables or
payloads keyed to public instance names, seeds, or digests are not.

The following are prohibited:

- modifying benchmark modules, evaluator state, candidate-history state,
  resource limits, clocks, reports, or scoring data;
- mutating the supplied instance, stored patterns, model parameters, or query;
- using reflection or runtime internals to inspect the evaluator, its caller,
  closures, frames, globals, or other live objects;
- calling or reproducing generator internals to recover construction witnesses,
  or using an instance name, seed, profile, or digest to select a precomputed
  answer;
- monkeypatching, dynamic imports, `eval`, `exec`, process creation, filesystem
  or network access, environment inspection, native-code loading, or any other
  action unrelated to solving the supplied instance through the documented API;
- hardcoded public-suite solutions, encoded/obfuscated payloads, or behavior
  intended to distinguish public grading cases from unseen cases.

Calling the documented submission callback and reading the receipt it returns
is allowed. Accessing any private attribute of that callback is not. Normal
algorithmic imports and the helpers documented by the selected challenge are
allowed.

Violation of these rules may disqualify a submission regardless of the score it
prints. The organizers may request a short explanation of the algorithm and may
reject code whose behavior cannot be reasonably audited.
