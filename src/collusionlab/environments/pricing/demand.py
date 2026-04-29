"""Demand models for the pricing environment.

All downstream pricing code depends only on the `DemandModel` interface; the
specific model is selected via config and instantiated through `get_demand_model`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod


class DemandModel(ABC):
    """Maps a list of prices to per-firm quantities, plus equilibrium reference prices."""

    @abstractmethod
    def quantities(self, prices: list[float]) -> list[float]: ...

    @abstractmethod
    def nash_price(self) -> float:
        """Symmetric Nash-equilibrium price (continuous; round to grid where needed)."""

    @abstractmethod
    def monopoly_price(self) -> float:
        """Symmetric joint-profit-maximizing price (continuous)."""

    @property
    @abstractmethod
    def marginal_cost(self) -> float: ...


# ---------------------------------------------------------------------------
# Calvano logit demand
# ---------------------------------------------------------------------------


class CalvanoDemand(DemandModel):
    """Calvano et al. (2020) symmetric logit demand.

        q_i = exp((a_i - p_i) / mu) / (sum_j exp((a_j - p_j) / mu) + exp(a_0 / mu))

    Symmetric across firms: every firm shares the same quality index `a` and cost `c`.
    The default parameters are produced by `scripts/calibrate_calvano.py` so that Nash
    and monopoly equilibria land at desired integer positions on the price grid; they
    live in `configs/base.yaml`.
    """

    def __init__(
        self,
        n_agents: int,
        a: float,
        mu: float,
        a_0: float,
        c: float,
    ) -> None:
        if n_agents < 2:
            raise ValueError("CalvanoDemand requires n_agents >= 2")
        if mu <= 0:
            raise ValueError("mu must be positive")
        self.n_agents = n_agents
        self.a = float(a)
        self.mu = float(mu)
        self.a_0 = float(a_0)
        self.c = float(c)
        # Cache equilibria; they only depend on parameters.
        self._nash = self._solve_symmetric_nash()
        self._monopoly = self._solve_symmetric_monopoly()

    @property
    def marginal_cost(self) -> float:
        return self.c

    def quantities(self, prices: list[float]) -> list[float]:
        if len(prices) != self.n_agents:
            raise ValueError(
                f"expected {self.n_agents} prices, got {len(prices)}"
            )
        # Numerically stable softmax with the outside option included.
        exponents = [(self.a - p) / self.mu for p in prices] + [self.a_0 / self.mu]
        m = max(exponents)
        exps = [math.exp(e - m) for e in exponents]
        denom = sum(exps)
        return [exps[i] / denom for i in range(self.n_agents)]

    # --- equilibria -------------------------------------------------------

    def _own_profit_at_symmetric(self, p: float) -> float:
        prices = [p] * self.n_agents
        q = self.quantities(prices)[0]
        return (p - self.c) * q

    def _joint_profit_at_symmetric(self, p: float) -> float:
        return self.n_agents * self._own_profit_at_symmetric(p)

    def _solve_symmetric_nash(self) -> float:
        """Find p* such that p* is a best response when all rivals also play p*.

        Iterates: at each step, fix rivals at the current candidate and find the
        own-price best response by maximizing profit_i(p_i, rivals=p*). At a
        symmetric Nash, the best response equals the candidate (fixed point).
        """
        p = self.c + self.mu  # initial guess just above cost
        for _ in range(200):
            br = self._best_response(p)
            if abs(br - p) < 1e-7:
                p = br
                break
            p = 0.5 * (p + br)  # damped update for stability
        gap = abs(self._best_response(p) - p)
        if gap > 1e-4:
            raise RuntimeError(
                f"Calvano symmetric Nash solver failed to converge: fixed-point gap={gap:.6g}"
            )
        return p

    def _best_response(self, rival_p: float) -> float:
        """Maximize agent 0's profit holding all rivals at rival_p."""
        def neg_pi(p_own: float) -> float:
            prices = [rival_p] * self.n_agents
            prices[0] = p_own
            q0 = self.quantities(prices)[0]
            return -(p_own - self.c) * q0

        return _golden_section_min(neg_pi, lo=self.c, hi=self._upper_search_bound())

    def _solve_symmetric_monopoly(self) -> float:
        """Find p* maximizing joint profit at the symmetric profile (p, ..., p)."""
        def neg_joint(p: float) -> float:
            return -self._joint_profit_at_symmetric(p)

        return _golden_section_min(neg_joint, lo=self.c, hi=self._upper_search_bound())

    def _upper_search_bound(self) -> float:
        """Conservative upper bound on equilibrium prices.

        At p = c + a + |a_0| + mu, even the most generous interpretation of the
        utility numerator (a - p) is well into the negative regime and quantities
        are vanishing. Equilibria for symmetric Calvano logit lie comfortably
        below this.
        """
        return self.c + abs(self.a) + abs(self.a_0) + 5.0 * self.mu

    def nash_price(self) -> float:
        return self._nash

    def monopoly_price(self) -> float:
        return self._monopoly


def _golden_section_min(
    f, lo: float, hi: float, tol: float = 1e-8, max_iter: int = 200
) -> float:
    """Golden-section search for the minimizer of a unimodal `f` on [lo, hi].

    Profit functions for symmetric Calvano logit are unimodal in own price (or
    in the symmetric price), so this is robust without needing a sign change.
    """
    phi = (math.sqrt(5.0) - 1.0) / 2.0  # ≈ 0.618
    a, b = lo, hi
    c = b - phi * (b - a)
    d = a + phi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = f(d)
    return 0.5 * (a + b)


# ---------------------------------------------------------------------------
# Bertrand winner-take-all
# ---------------------------------------------------------------------------


class BertrandDemand(DemandModel):
    """Winner-take-all Bertrand: cheapest firm gets all of fixed demand Q.

    Ties split evenly. Marginal cost `c` is symmetric. `nash_price()` returns the
    continuous Bertrand benchmark (= marginal cost), not a grid-constrained price;
    `PricingGame` applies grid constraints when constructing default actions and
    reward-elevation baselines. Monopoly is unbounded above without a
    reservation price, so we expose one as a parameter and treat it as the
    monopoly price (a single firm at the reservation price extracts all surplus).
    """

    def __init__(
        self,
        n_agents: int,
        Q: float = 1.0,
        c: float = 1.0,
        reservation_price: float = 10.0,
    ) -> None:
        if n_agents < 2:
            raise ValueError("BertrandDemand requires n_agents >= 2")
        if reservation_price <= c:
            raise ValueError("reservation_price must exceed marginal cost")
        self.n_agents = n_agents
        self.Q = float(Q)
        self.c = float(c)
        self.reservation_price = float(reservation_price)

    @property
    def marginal_cost(self) -> float:
        return self.c

    def quantities(self, prices: list[float]) -> list[float]:
        if len(prices) != self.n_agents:
            raise ValueError(
                f"expected {self.n_agents} prices, got {len(prices)}"
            )
        min_p = min(prices)
        if min_p > self.reservation_price:
            return [0.0] * self.n_agents
        winners = [i for i, p in enumerate(prices) if p == min_p]
        share = self.Q / len(winners)
        return [share if i in winners else 0.0 for i in range(self.n_agents)]

    def nash_price(self) -> float:
        return self.c

    def monopoly_price(self) -> float:
        return self.reservation_price


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_demand_model(name: str, n_agents: int, params: dict) -> DemandModel:
    """Instantiate a demand model by name. `params` is the env-config sub-dict."""
    if name == "calvano":
        return CalvanoDemand(
            n_agents=n_agents,
            a=params["a"],
            mu=params["mu"],
            a_0=params["a_0"],
            c=params["c"],
        )
    if name == "bertrand":
        return BertrandDemand(
            n_agents=n_agents,
            Q=params.get("Q", 1.0),
            c=params["c"],
            reservation_price=params.get("reservation_price", 10.0),
        )
    raise ValueError(f"unknown demand model {name!r}")
