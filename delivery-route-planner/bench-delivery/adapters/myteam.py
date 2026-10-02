from __future__ import annotations

import math
import time
from typing import List

from adapters.savings import SavingsRouter
from adapter import route_cost, simulate_route


SUBMIT_INTERVAL = 0.05
RESERVE = 0.15


class Router(SavingsRouter):
    """
    Stronger Delivery Route Planner adapter.

    Strategy:
      1. Keep the published SavingsRouter for small instances.
      2. Create a valid solution quickly.
      3. Use multiple geographic sweep constructions.
      4. Use Clarke-Wright savings.
      5. Improve routes with relocate / Or-opt / 2-opt.
      6. Use the published ILS on medium-sized instances.

    Only documented instance information and public helpers are used.
    """

    # ================================================================
    # MULTI-START GEOGRAPHIC SWEEP
    # ================================================================

    def _rotated_sweep(self, shift: int):
        inst = self.inst

        depot_x = inst.xs[0]
        depot_y = inst.ys[0]

        def angle(customer: int) -> float:
            a = math.atan2(
                inst.ys[customer] - depot_y,
                inst.xs[customer] - depot_x,
            )

            if a < 0.0:
                a += 2.0 * math.pi

            return a

        order = sorted(
            range(1, self.n + 1),
            key=lambda c: (angle(c), c),
        )

        if order:
            shift %= len(order)
            order = order[shift:] + order[:shift]

        # Prefer vehicles with earlier starts and larger capacity.
        vehicles = sorted(
            range(inst.num_vehicles),
            key=lambda v: (
                inst.starts[v],
                -inst.capacities[v],
                v,
            ),
        )

        plan: List[List[int]] = [
            [] for _ in range(inst.num_vehicles)
        ]

        pending = list(order)

        for vehicle in vehicles:

            if not pending:
                break

            if self._remaining() < RESERVE:
                return None

            route: List[int] = []
            load = 0

            while pending:

                chosen_index = -1

                # Look at several nearby customers rather than blindly
                # taking only the next customer.
                lookahead = min(12, len(pending))

                for offset in range(lookahead):

                    customer = pending[offset]

                    # Capacity check.
                    if (
                        load + inst.demand[customer]
                        > inst.capacities[vehicle]
                    ):
                        continue

                    insertion_positions = []

                    for position in range(len(route) + 1):

                        previous = (
                            route[position - 1]
                            if position > 0
                            else 0
                        )

                        following = (
                            route[position]
                            if position < len(route)
                            else 0
                        )

                        delta = (
                            inst.travel(previous, customer)
                            + inst.travel(customer, following)
                            - inst.travel(previous, following)
                        )

                        insertion_positions.append(
                            (delta, position)
                        )

                    insertion_positions.sort(
                        key=lambda x: x[0]
                    )

                    # Fully simulate only the best few positions.
                    for _, position in insertion_positions[:4]:

                        candidate = (
                            route[:position]
                            + [customer]
                            + route[position:]
                        )

                        result = simulate_route(
                            inst,
                            candidate,
                            vehicle,
                        )

                        if result is not None:

                            route = candidate

                            load += inst.demand[customer]

                            chosen_index = offset

                            break

                    if chosen_index >= 0:
                        break

                # No feasible customer for this vehicle.
                if chosen_index < 0:
                    break

                pending.pop(chosen_index)

            plan[vehicle] = route

        # Every customer must be assigned.
        if pending:
            return None

        return plan

    # ================================================================
    # MULTIPLE SWEEP STARTS
    # ================================================================

    def _best_sweeps(self):

        if self.n <= 120:

            fractions = (
                0.0,
                0.25,
                0.50,
                0.75,
            )

        elif self.n <= 350:

            fractions = (
                0.0,
                0.08,
                0.25,
                0.50,
                0.75,
            )

        else:

            fractions = (
                0.0,
                0.125,
                0.25,
                0.50,
                0.75,
            )

        best = None
        best_cost = None

        for fraction in fractions:

            if self._remaining() < RESERVE:
                break

            shift = int(self.n * fraction)

            candidate = self._rotated_sweep(shift)

            if candidate is None:
                continue

            cost = self._plan_cost(candidate)

            if cost is None:
                continue

            if (
                best_cost is None
                or cost < best_cost
            ):

                best = candidate
                best_cost = cost

        return best

    # ================================================================
    # MAIN SOLVER
    # ================================================================

    def solve(self, instance, submit_candidate):

        # ------------------------------------------------------------
        # SMALL INSTANCES
        # ------------------------------------------------------------
        #
        # The supplied SavingsRouter is already very strong on these.
        #
        if instance.n <= 120:

            return super().solve(
                instance,
                submit_candidate,
            )

        # ------------------------------------------------------------
        # PREPARE
        # ------------------------------------------------------------

        self.prepare(
            instance,
            time.perf_counter() + 1e9,
        )

        self.submit = submit_candidate

        # ------------------------------------------------------------
        # STEP 1:
        # FAST VALID CONSTRUCTION
        # ------------------------------------------------------------

        plan = self._greedy_append()

        if plan is None:

            plan = self._insertion_construct()

        if plan is None:

            plan = self._sweep_construct()

        if plan is not None:

            self._offer(plan)

        # ------------------------------------------------------------
        # STEP 2:
        # FIRST GEOGRAPHIC SWEEP
        # ------------------------------------------------------------

        first = self._rotated_sweep(0)

        if first is not None:

            self._offer(first)

        # ------------------------------------------------------------
        # STEP 3:
        # MULTI-START SWEEP
        # ------------------------------------------------------------

        sweep = self._best_sweeps()

        if sweep is not None:

            self._offer(sweep)

        # ------------------------------------------------------------
        # STEP 4:
        # CLARKE-WRIGHT SAVINGS
        # ------------------------------------------------------------

        if self._remaining() > RESERVE:

            merged = self._clarke_wright()

            if merged is not None:

                self._offer(merged)

        # ------------------------------------------------------------
        # STEP 5:
        # LOCAL SEARCH
        # ------------------------------------------------------------

        if (
            self.best_plan is not None
            and self._remaining() > RESERVE
        ):

            plan = [
                list(route)
                for route in self.best_plan
            ]

            costs = [
                route_cost(
                    self.inst,
                    route,
                    vehicle,
                ) or 0
                for vehicle, route
                in enumerate(plan)
            ]

            where = {
                customer: vehicle
                for vehicle, route
                in enumerate(plan)
                for customer in route
            }

            # Uses the public SavingsRouter local-search operators:
            #
            # relocate
            # Or-opt
            # 2-opt
            #
            self._descend(
                plan,
                costs,
                where,
            )

            self._offer(plan)

        # ------------------------------------------------------------
        # STEP 6:
        # ITERATED LOCAL SEARCH
        # ------------------------------------------------------------
        #
        # ILS is expensive on very large instances because many route
        # feasibility checks are performed.
        #
        # Therefore use it mainly on medium-sized instances.
        #

        if (
            instance.n < 500
            and self._remaining() > RESERVE
        ):

            self._iterated_local_search()

        # ------------------------------------------------------------
        # FINAL RESULT
        # ------------------------------------------------------------

        if self.best_plan is not None:

            return self.best_plan

        return []


Router_ = Router