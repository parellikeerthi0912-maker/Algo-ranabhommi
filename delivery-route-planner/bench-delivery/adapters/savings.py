"""The published serious baseline. Match it exactly and you score 0.50.

Four stages:

1. **Parallel greedy append.** Customers sorted by closing time, each appended
   to the cheapest vehicle whose end state still works.  Every check is O(1)
   because each vehicle carries a running clock, so this finishes in
   milliseconds even at n=1000.  Its job is to get *something* valid on the
   board before the 1% checkpoint.
2. **Cheapest-insertion construction**, used when append fails -- which it does
   on tight windows, where a customer's window may have closed on every route
   that already exists.  Insertion can place it mid-route; appending cannot.
3. **Clarke-Wright savings** over the k-nearest graph, merging route tails into
   route heads while time windows hold.
4. **Iterated local search**: relocate, Or-opt on segments of 2 and 3, and
   intra-route 2-opt to convergence, then a random ruin of a few customers and
   another descent, keeping the result only if it improved.

Where the remaining points are.  The perturbation here is small and uniform,
the acceptance criterion is strictly-better-only, insertion is greedy rather
than regret-based, and every move is evaluated by re-simulating whole routes
instead of maintaining incremental time-window slack.  The organizer reference
fixes all four, and scores 1.00.
"""

from __future__ import annotations

import sys
import time
from math import isqrt
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapter import Router, assign_vehicles, neighbour_lists, route_cost, simulate_route  # noqa: E402
from benchkit.rng import Rng  # noqa: E402
from data import Instance  # noqa: E402

SUBMIT_INTERVAL = 0.05  # seconds between submissions while improving
RESERVE = 0.15  # stop this far from the deadline so the final submit lands


def _null_sink(_plan) -> dict:
    """Stands in for ``submit_candidate`` when construction runs standalone."""
    return {"accepted": False, "reason": "no sink", "cost": None, "best": None,
            "elapsed_s": 0.0, "remaining_s": None}


class SavingsRouter(Router):
    def prepare(self, instance: Instance, deadline: float) -> None:
        """Set up state without running anything.

        Split out so the organizer reference can reuse this construction as
        its starting point. That reuse is deliberate: it guarantees the
        reference is never worse than the baseline it anchors against, which
        would otherwise invert the scoring curve on an unlucky instance.
        """
        self.inst = instance
        self.n = instance.n
        self.started = time.perf_counter()
        self.deadline = deadline
        self.last_submit = 0.0
        self.best_cost: Optional[int] = None
        self.best_plan: Optional[List[List[int]]] = None
        # Seeded from the instance, never from the clock, so two runs of this
        # submission produce identical candidate streams.
        self.rng = Rng(instance.digest() ^ 0xC1A2_4E00)
        self.neighbours = neighbour_lists(instance, k=min(24, max(2, instance.n - 1)))
        self.submit = _null_sink

    def initial_plan(self) -> Optional[List[List[int]]]:
        """Construction only -- append, insertion, sweep, then Clarke-Wright."""
        plan = self._greedy_append()
        if plan is None:
            plan = self._insertion_construct()
        if plan is None:
            plan = self._sweep_construct()
        if plan is None:
            return None
        best, best_cost = plan, self._plan_cost(plan)
        merged = self._clarke_wright()
        if merged is not None:
            cost = self._plan_cost(merged)
            if cost is not None and (best_cost is None or cost < best_cost):
                best, best_cost = merged, cost
        return best

    def solve(self, instance: Instance, submit_candidate) -> object:
        self.prepare(instance, time.perf_counter() + 1e9)  # replaced by first receipt
        self.submit = submit_candidate

        plan = self._greedy_append()
        if plan is None:
            plan = self._insertion_construct()
        if plan is None:
            plan = self._sweep_construct()
        if plan is None:
            return []
        self._offer(plan)

        merged = self._clarke_wright()
        if merged is not None:
            self._offer(merged)

        self._iterated_local_search()
        return self.best_plan if self.best_plan is not None else plan

    # -- bookkeeping --------------------------------------------------------

    def _elapsed(self) -> float:
        return time.perf_counter() - self.started

    def _remaining(self) -> float:
        return self.deadline - time.perf_counter()

    def _offer(self, plan: Sequence[Sequence[int]], force: bool = True) -> bool:
        now = self._elapsed()
        if not force and now - self.last_submit < SUBMIT_INTERVAL:
            return False
        receipt = self.submit([list(route) for route in plan])
        self.last_submit = now
        # The receipt is the authoritative clock; one read tells us the real
        # deadline regardless of how the harness was configured.
        if receipt.get("remaining_s") is not None:
            self.deadline = time.perf_counter() + receipt["remaining_s"]
        if receipt["accepted"]:
            cost = receipt["cost"]
            if self.best_cost is None or cost < self.best_cost:
                self.best_cost = cost
                self.best_plan = [list(route) for route in plan]
            return True
        return False

    def _plan_cost(self, plan: Sequence[Sequence[int]]) -> Optional[int]:
        total = 0
        for vehicle, route in enumerate(plan):
            if not route:
                continue
            value = route_cost(self.inst, route, vehicle)
            if value is None:
                return None
            total += value
        return total

    # -- stage 1: parallel greedy append ------------------------------------

    def _greedy_append(self) -> Optional[List[List[int]]]:
        inst = self.inst
        orders = (
            sorted(range(1, self.n + 1), key=lambda c: (inst.tw_close[c], inst.tw_open[c], c)),
            sorted(range(1, self.n + 1), key=lambda c: (inst.tw_open[c], inst.tw_close[c], c)),
            sorted(range(1, self.n + 1), key=lambda c: (-inst.demand[c], inst.tw_close[c], c)),
        )
        best: Optional[List[List[int]]] = None
        best_cost: Optional[int] = None
        for order in orders:
            plan = self._append_pass(order)
            if plan is None:
                continue
            cost = self._plan_cost(plan)
            if cost is not None and (best_cost is None or cost < best_cost):
                best, best_cost = plan, cost
        return best

    def _append_pass(self, order: Sequence[int]) -> Optional[List[List[int]]]:
        inst = self.inst
        xs, ys = inst.xs, inst.ys
        vehicles = inst.num_vehicles

        routes: List[List[int]] = [[] for _ in range(vehicles)]
        clock = list(inst.starts)
        last = [0] * vehicles
        load = [0] * vehicles

        for customer in order:
            demand = inst.demand[customer]
            open_at, close_at = inst.tw_open[customer], inst.tw_close[customer]
            service = inst.service[customer]
            cx, cy = xs[customer], ys[customer]
            home = isqrt((cx - xs[0]) ** 2 + (cy - ys[0]) ** 2)

            chosen = -1
            chosen_cost = None
            chosen_clock = 0
            for v in range(vehicles):
                if load[v] + demand > inst.capacities[v]:
                    continue
                prev = last[v]
                step = isqrt((xs[prev] - cx) ** 2 + (ys[prev] - cy) ** 2)
                arrival = clock[v] + step
                if arrival > close_at:
                    continue
                departure = (open_at if arrival < open_at else arrival) + service
                if departure + home - inst.starts[v] > inst.max_duration:
                    continue
                if departure + home > inst.horizon:
                    continue
                # Marginal detour, not raw distance: prefers extending a route
                # over opening one, which is what keeps the fleet count down.
                back = isqrt((xs[prev] - xs[0]) ** 2 + (ys[prev] - ys[0]) ** 2)
                marginal = step + home - back
                if chosen_cost is None or marginal < chosen_cost:
                    chosen, chosen_cost, chosen_clock = v, marginal, departure

            if chosen < 0:
                return None
            routes[chosen].append(customer)
            clock[chosen] = chosen_clock
            last[chosen] = customer
            load[chosen] += demand

        return routes

    # -- stage 2: cheapest-insertion construction ---------------------------

    def _candidate_positions(
        self, plan: Sequence[List[int]], customer: int, loads: Sequence[int]
    ) -> List[Tuple[int, int, int]]:
        """(delta, vehicle, position), cheapest first, screened in O(1).

        Walks each route once, carrying the running clock, and rejects a
        position the moment the arrival time passes the customer's window.

        This screen is optimistic: it checks that *this* customer fits, not
        that the customers after it still fit once the insertion pushes them
        later, and it ignores the route-duration cap entirely.  Callers must
        verify with ``simulate_route`` and fall through to the next candidate,
        because on tight instances most high-ranked candidates fail that
        check.  Tightening this screen into an exact test -- by carrying
        backward time-window slack per route -- is one of the larger wins
        available over this baseline.
        """
        inst = self.inst
        demand = inst.demand[customer]
        close_at = inst.tw_close[customer]
        out: List[Tuple[int, int, int]] = []

        for vehicle, route in enumerate(plan):
            if loads[vehicle] + demand > inst.capacities[vehicle]:
                continue
            clock = inst.starts[vehicle]
            previous = 0
            for position in range(len(route) + 1):
                step = inst.travel(previous, customer)
                arrival = clock + step
                if arrival <= close_at:
                    following = route[position] if position < len(route) else 0
                    delta = (
                        step
                        + inst.travel(customer, following)
                        - inst.travel(previous, following)
                    )
                    out.append((delta, vehicle, position))
                elif previous != 0:
                    break  # later positions only arrive later
                if position < len(route):
                    node = route[position]
                    arrive = clock + inst.travel(previous, node)
                    if arrive > inst.tw_close[node]:
                        break
                    clock = max(arrive, inst.tw_open[node]) + inst.service[node]
                    previous = node
        out.sort()
        return out

    def _insertion_construct(self) -> Optional[List[List[int]]]:
        inst = self.inst
        plan: List[List[int]] = [[] for _ in range(inst.num_vehicles)]
        loads = [0] * inst.num_vehicles
        order = sorted(
            range(1, self.n + 1), key=lambda c: (inst.tw_close[c], inst.tw_open[c], c)
        )
        for customer in order:
            if self._remaining() < RESERVE:
                return None
            placed = False
            # Every candidate, cheapest first -- not a truncated shortlist.
            # An empty vehicle always ranks last on detour cost, so capping
            # the list would discard the one position guaranteed to work.
            for _, vehicle, position in self._candidate_positions(plan, customer, loads):
                route = plan[vehicle]
                trial = route[:position] + [customer] + route[position:]
                if simulate_route(inst, trial, vehicle) is not None:
                    plan[vehicle] = trial
                    loads[vehicle] += inst.demand[customer]
                    placed = True
                    break
            if not placed:
                return None
        return plan

    # -- stage 2b: sweep construction ---------------------------------------

    def _sweep_construct(self) -> Optional[List[List[int]]]:
        """Angular sweep into capacity-bounded wedges, each served in time order.

        The fallback of last resort, and the most reliable of the three.
        Greedy insertion strands customers because it commits vehicles early
        and discovers too late that a heavy or late customer has nowhere to
        go.  A sweep decides the *partition* first and the ordering second, so
        it degrades by splitting a wedge rather than by failing outright.

        Geographic wedges also happen to be what vehicle routing solutions
        look like, which is why this is a classic construction rather than a
        trick: adjacent customers ride together.
        """
        inst = self.inst
        depot_x, depot_y = inst.xs[0], inst.ys[0]

        def angle(customer: int) -> int:
            dx, dy = inst.xs[customer] - depot_x, inst.ys[customer] - depot_y
            total = abs(dx) + abs(dy)
            if total == 0:
                return 0
            if dx >= 0 and dy >= 0:
                return 10000 * dy // total
            if dx < 0 and dy >= 0:
                return 10000 + 10000 * (-dx) // total
            if dx < 0 and dy < 0:
                return 20000 + 10000 * (-dy) // total
            return 30000 + 10000 * dx // total

        order = sorted(range(1, self.n + 1), key=lambda c: (angle(c), c))
        # Vehicles largest-capacity and earliest-start first, so the wedges
        # that need the most room get it.
        vehicles = sorted(
            range(inst.num_vehicles),
            key=lambda v: (inst.starts[v], -inst.capacities[v], v),
        )

        plan: List[List[int]] = [[] for _ in range(inst.num_vehicles)]
        pending = list(order)
        for vehicle in vehicles:
            if not pending:
                break
            if self._remaining() < RESERVE:
                return None
            capacity = inst.capacities[vehicle]
            wedge: List[int] = []
            load = 0
            while pending:
                # Look past an awkward customer instead of closing the wedge
                # on it. One heavy drop or one late window should cost a
                # skipped customer, not half a vehicle's worth of capacity.
                taken = -1
                for offset in range(min(12, len(pending))):
                    customer = pending[offset]
                    if load + inst.demand[customer] > capacity:
                        continue
                    # Insert at the cheapest position in the wedge rather than
                    # re-sorting it by opening time. Sorting by time makes the
                    # route zigzag across the wedge, and the travel it wastes
                    # is what exhausts the duration cap -- wedges end up half
                    # the size the fleet was built for.
                    positions = []
                    for position in range(len(wedge) + 1):
                        previous = wedge[position - 1] if position else 0
                        following = wedge[position] if position < len(wedge) else 0
                        positions.append(
                            (
                                inst.travel(previous, customer)
                                + inst.travel(customer, following)
                                - inst.travel(previous, following),
                                position,
                            )
                        )
                    positions.sort()
                    for _, position in positions[:4]:
                        candidate = wedge[:position] + [customer] + wedge[position:]
                        if simulate_route(inst, candidate, vehicle) is not None:
                            wedge = candidate
                            load += inst.demand[customer]
                            taken = offset
                            break
                    if taken >= 0:
                        break
                if taken < 0:
                    break
                pending.pop(taken)
            plan[vehicle] = wedge

        return None if pending else plan

    # -- stage 3: Clarke-Wright --------------------------------------------

    def _clarke_wright(self) -> Optional[List[List[int]]]:
        inst = self.inst
        proxy = max(
            range(inst.num_vehicles),
            key=lambda v: (inst.starts[v] == 0, inst.capacities[v]),
        )
        cap_bound = inst.capacities[proxy]

        routes: dict = {}
        route_of: dict = {}
        for customer in range(1, self.n + 1):
            if simulate_route(inst, [customer], proxy) is None:
                return None  # cannot be served alone; CW has nothing to say
            routes[customer] = [customer]
            route_of[customer] = customer

        savings = []
        for i in range(1, self.n + 1):
            d0i = inst.travel(0, i)
            for j in self.neighbours[i]:
                if j <= 0 or j == i:
                    continue
                value = d0i + inst.travel(0, j) - inst.travel(i, j)
                if value > 0:
                    savings.append((-value, i, j))
        savings.sort()

        loads = {c: inst.demand[c] for c in range(1, self.n + 1)}
        for _, i, j in savings:
            if self._remaining() < RESERVE:
                break
            a, b = route_of[i], route_of[j]
            if a == b:
                continue
            if routes[a][-1] != i or routes[b][0] != j:
                continue  # CW only joins a tail to a head
            if loads[a] + loads[b] > cap_bound:
                continue
            merged = routes[a] + routes[b]
            if simulate_route(inst, merged, proxy) is None:
                continue
            routes[a] = merged
            loads[a] += loads[b]
            for customer in routes[b]:
                route_of[customer] = a
            del routes[b], loads[b]

        active = [route for route in routes.values() if route]
        if len(active) > inst.num_vehicles:
            return None  # the append/insertion plan already fits; keep it
        return assign_vehicles(inst, active)

    # -- stage 4: iterated local search -------------------------------------

    def _iterated_local_search(self) -> None:
        if self.best_plan is None:
            return
        plan = [list(route) for route in self.best_plan]
        costs = [route_cost(self.inst, route, v) or 0 for v, route in enumerate(plan)]
        where = {c: v for v, route in enumerate(plan) for c in route}

        self._descend(plan, costs, where)
        self._offer(plan)
        incumbent = sum(costs)

        # Tight windows make most random ruins unrepairable. That is a reason
        # to try a different ruin, not to stop searching -- but if nothing is
        # repairable at all, spinning to the deadline helps nobody.
        consecutive_failures = 0
        while self._remaining() > RESERVE * 2 and consecutive_failures < 200:
            trial = [list(route) for route in plan]
            trial_costs = list(costs)
            trial_where = dict(where)
            if not self._perturb(trial, trial_costs, trial_where):
                consecutive_failures += 1
                continue
            consecutive_failures = 0
            self._descend(trial, trial_costs, trial_where)
            total = sum(trial_costs)
            if total < incumbent:
                plan, costs, where, incumbent = trial, trial_costs, trial_where, total
                self._offer(plan, force=False)

        self._offer(plan)

    def _descend(self, plan, costs, where) -> None:
        improved = True
        while improved and self._remaining() > RESERVE:
            improved = False
            for length in (1, 2, 3):
                if self._remaining() <= RESERVE:
                    break
                if self._relocate_pass(plan, costs, where, length):
                    improved = True
            if self._remaining() > RESERVE and self._two_opt_pass(plan, costs):
                improved = True

    def _perturb(self, plan, costs, where) -> bool:
        """Rip out a handful of customers and greedily put them back.

        Deliberately mild: a few random removals with greedy reinsertion. The
        reference destroys 10-40% at a time with relatedness-aware operators,
        which is what lets it cross the barriers this cannot.
        """
        inst = self.inst
        occupied = [v for v, route in enumerate(plan) if route]
        if not occupied:
            return False

        count = min(self.rng.randint(2, 5), sum(len(plan[v]) for v in occupied))
        torn: List[int] = []
        for _ in range(count):
            occupied = [v for v, route in enumerate(plan) if route]
            if not occupied:
                break
            vehicle = self.rng.choice(occupied)
            route = plan[vehicle]
            customer = route[self.rng.below(len(route))]
            route.remove(customer)
            torn.append(customer)
            where.pop(customer, None)
            costs[vehicle] = route_cost(inst, route, vehicle) or 0 if route else 0

        loads = [sum(inst.demand[c] for c in route) for route in plan]
        for customer in torn:
            placed = False
            for _, vehicle, position in self._candidate_positions(plan, customer, loads):
                route = plan[vehicle]
                trial = route[:position] + [customer] + route[position:]
                value = route_cost(inst, trial, vehicle)
                if value is not None:
                    plan[vehicle] = trial
                    costs[vehicle] = value
                    loads[vehicle] += inst.demand[customer]
                    where[customer] = vehicle
                    placed = True
                    break
            if not placed:
                return False  # caller discards this trial
        return True

    def _relocate_pass(self, plan, costs, where, length: int) -> bool:
        inst = self.inst
        changed = False
        for source in range(len(plan)):
            position = 0
            while position + length <= len(plan[source]):
                if self._remaining() <= RESERVE:
                    return changed
                route = plan[source]
                segment = route[position : position + length]
                remainder = route[:position] + route[position + length :]
                new_source_cost = 0
                if remainder:
                    value = route_cost(inst, remainder, source)
                    if value is None:
                        position += 1
                        continue
                    new_source_cost = value

                best = None
                for neighbour in self.neighbours[segment[0]]:
                    target = where.get(neighbour)
                    if target is None:
                        continue
                    target_route = remainder if target == source else plan[target]
                    try:
                        index = target_route.index(neighbour)
                    except ValueError:
                        continue
                    for insert_at in (index, index + 1):
                        candidate = target_route[:insert_at] + segment + target_route[insert_at:]
                        value = route_cost(inst, candidate, target)
                        if value is None:
                            continue
                        if target == source:
                            delta = value - costs[source]
                        else:
                            delta = (new_source_cost + value) - (costs[source] + costs[target])
                        if delta < 0 and (best is None or delta < best[0]):
                            best = (delta, target, candidate, value)

                if best is None:
                    position += 1
                    continue

                _, target, candidate, value = best
                if target == source:
                    plan[source] = candidate
                    costs[source] = value
                else:
                    plan[source] = remainder
                    costs[source] = new_source_cost
                    plan[target] = candidate
                    costs[target] = value
                    for customer in segment:
                        where[customer] = target
                changed = True
                self._offer(plan, force=False)
        return changed

    def _two_opt_pass(self, plan, costs) -> bool:
        """Intra-route segment reversal. Often infeasible under tight windows,
        but cheap to test and occasionally unlocks a large gain."""
        inst = self.inst
        changed = False
        for vehicle in range(len(plan)):
            route = plan[vehicle]
            if len(route) < 4:
                continue
            for i in range(len(route) - 2):
                if self._remaining() <= RESERVE:
                    return changed
                for j in range(i + 2, min(len(route), i + 12)):
                    candidate = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
                    value = route_cost(inst, candidate, vehicle)
                    if value is not None and value < costs[vehicle]:
                        plan[vehicle] = route = candidate
                        costs[vehicle] = value
                        changed = True
                        self._offer(plan, force=False)
        return changed


Router_ = SavingsRouter
