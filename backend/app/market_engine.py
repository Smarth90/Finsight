"""FinSight Synthetic Market Simulation Engine.

This module implements a multi-factor synthetic stock market simulation engine
for financial portfolio analysis and risk intelligence.

Mathematical Foundation:
------------------------
1. Price Evolution:
   Prices evolve via discrete geometric compounding over trading days t in {1, ..., T}:
       P_{i, t} = P_{i, t-1} * exp(r_{i, t}) = P_{i, 0} * exp(sum_{k=1}^t r_{i, k})
   Because exp(x) > 0 for all x in Real, prices are guaranteed strictly positive.

2. Daily Log-Return Decomposition:
   r_{i, t} = mu_daily + sigma_i * xi_{i, t}
   where:
       - mu_daily = annual_log_drift / 252 is the expected continuously compounded daily return.
       - sigma_i is the company's daily volatility (used directly without sqrt(252) scaling).
       - xi_{i, t} is the composite standardized innovation with Var(xi_{i, t}) = 1.0.

3. Three-Factor Composite Innovation:
   xi_{i, t} = beta_m * F_{m, t} + beta_s * F_{s, t} + beta_eps * eps_{i, t}^*
   with approved variance allocation:
       - Market Factor:    weight_m = 0.35  =>  beta_m   = sqrt(0.35) ~= 0.591608
       - Sector Factor:    weight_s = 0.35  =>  beta_s   = sqrt(0.35) ~= 0.591608
       - Company Factor:   weight_e = 0.30  =>  beta_eps = sqrt(0.30) ~= 0.547723
       Orthogonal variance: beta_m^2 + beta_s^2 + beta_eps^2 = 1.0.

4. Stationary AR(1) Factor Dynamics:
   Market and sector factors follow stationary AR(1) processes:
       F_0 ~ Normal(0, 1)
       F_t = phi * F_{t-1} + sqrt(1 - phi^2) * eta_t,  where eta_t ~ Normal(0, 1)
   Sampling F_0 from the stationary distribution guarantees Var(F_t) = 1.0 and
   E[F_t] = 0 for all t >= 0 with zero warmup/burn-in bias.

5. Standardized Student's t Innovations (df = 5):
   Idiosyncratic company shocks incorporate fat tails (excess kurtosis = 6.0):
       T_{i, t} ~ Student_t(df=5),  Var(T) = 5 / (5 - 2) = 5/3
       eps_{i, t}^* = T_{i, t} * sqrt(3/5)
   Standardization enforces E[eps^*] = 0 and Var(eps^*) = 1.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class MarketConfig:
    """Configuration parameters for synthetic market simulation.

    Attributes:
        trading_days: Number of trading days to simulate (T >= 1).
        random_seed: Seed for NumPy's default_rng generator for reproducibility.
        annual_log_drift: Expected annual continuously compounded growth rate (default: 0.08 = 8%).
        weight_market: Variance fraction explained by broad market moves (default: 0.35).
        weight_sector: Variance fraction explained by sector-level moves (default: 0.35).
        weight_company: Variance fraction explained by idiosyncratic company moves (default: 0.30).
        phi_market: AR(1) persistence coefficient for the market factor (-1 < phi < 1).
        phi_sector: AR(1) persistence coefficient for sector factors (-1 < phi < 1).
    """

    trading_days: int = 252
    random_seed: Optional[int] = None
    annual_log_drift: float = 0.08
    weight_market: float = 0.35
    weight_sector: float = 0.35
    weight_company: float = 0.30
    phi_market: float = 0.10
    phi_sector: float = 0.08

    # Derived parameters computed in __post_init__
    daily_drift: float = field(init=False)
    beta_market: float = field(init=False)
    beta_sector: float = field(init=False)
    beta_company: float = field(init=False)

    def __post_init__(self) -> None:
        """Validate configuration values and compute derived factor betas."""
        if not isinstance(self.trading_days, int) or self.trading_days < 1:
            raise ValueError(f"trading_days must be an integer >= 1, got {self.trading_days}")

        for name, val in [
            ("annual_log_drift", self.annual_log_drift),
            ("weight_market", self.weight_market),
            ("weight_sector", self.weight_sector),
            ("weight_company", self.weight_company),
            ("phi_market", self.phi_market),
            ("phi_sector", self.phi_sector),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{name} must be a finite real number, got {val}")

        total_weight = self.weight_market + self.weight_sector + self.weight_company
        if not math.isclose(total_weight, 1.0, rel_tol=1e-5):
            raise ValueError(
                f"Factor variance weights must sum to 1.0, got sum={total_weight:.6f} "
                f"({self.weight_market} + {self.weight_sector} + {self.weight_company})"
            )

        for name, w in [
            ("weight_market", self.weight_market),
            ("weight_sector", self.weight_sector),
            ("weight_company", self.weight_company),
        ]:
            if w < 0.0:
                raise ValueError(f"{name} must be non-negative, got {w}")

        for name, phi in [
            ("phi_market", self.phi_market),
            ("phi_sector", self.phi_sector),
        ]:
            if not (-1.0 < phi < 1.0):
                raise ValueError(
                    f"{name} must satisfy -1 < phi < 1 for stationary AR(1), got {phi}"
                )

        # Precompute daily drift and factor loadings (square root of variance weights)
        self.daily_drift = self.annual_log_drift / 252.0
        self.beta_market = math.sqrt(self.weight_market)
        self.beta_sector = math.sqrt(self.weight_sector)
        self.beta_company = math.sqrt(self.weight_company)


@dataclass
class CompanyProfile:
    """Company profile data required for market simulation.

    Attributes:
        ticker: Unique identifier symbol (e.g., 'AAPL', 'TECH01').
        name: Full company name.
        sector: Sector classification (e.g., 'Technology', 'Healthcare').
        base_price: Starting stock price at Day 0 (must be > 0).
        volatility: Daily return volatility (e.g., 0.013 to 0.032).
    """

    ticker: str
    name: str
    sector: str
    base_price: float
    volatility: float

    def __post_init__(self) -> None:
        """Validate financial values."""
        if not isinstance(self.base_price, (int, float)) or not math.isfinite(self.base_price) or self.base_price <= 0.0:
            raise ValueError(
                f"base_price for {self.ticker} must be a finite positive number, got {self.base_price}"
            )
        if not isinstance(self.volatility, (int, float)) or not math.isfinite(self.volatility) or self.volatility <= 0.0:
            raise ValueError(
                f"volatility for {self.ticker} must be a finite positive number, got {self.volatility}"
            )


@dataclass
class ComponentAttribution:
    """Decomposition of daily log-returns for explainability and risk intelligence.

    For each day t in {1, ..., T}:
        return[t] = drift[t] + market[t] + sector[t] + company[t]
    """

    drift: np.ndarray  # Shape: (T,)
    market: np.ndarray  # Shape: (T,)
    sector: np.ndarray  # Shape: (T,)
    company: np.ndarray  # Shape: (T,)

    def to_dict(self) -> Dict[str, List[float]]:
        """Convert component series to list format for JSON serialization."""
        return {
            "drift": self.drift.tolist(),
            "market": self.market.tolist(),
            "sector": self.sector.tolist(),
            "company": self.company.tolist(),
        }


@dataclass
class CompanySimulationResult:
    """Simulation results for an individual company.

    Attributes:
        profile: CompanyProfile metadata.
        prices: Daily price series starting at base_price (shape: (T + 1,)).
        log_returns: Daily log-return series r_t = ln(P_t / P_{t-1}) (shape: (T,)).
        attribution: Component-wise return decomposition (drift, market, sector, company).
    """

    profile: CompanyProfile
    prices: np.ndarray
    log_returns: np.ndarray
    attribution: ComponentAttribution

    def to_dict(self) -> Dict[str, Any]:
        """Convert company simulation result to a serializable dictionary."""
        return {
            "ticker": self.profile.ticker,
            "name": self.profile.name,
            "sector": self.profile.sector,
            "base_price": self.profile.base_price,
            "daily_volatility": self.profile.volatility,
            "prices": self.prices.tolist(),
            "log_returns": self.log_returns.tolist(),
            "attribution": self.attribution.to_dict(),
        }


@dataclass
class SimulationOutput:
    """Complete synthetic market simulation output.

    Attributes:
        config: MarketConfig used for this simulation.
        trading_days: Number of simulated trading days (T).
        market_factors: Daily market factor path F_{m, t} (shape: (T,)).
        sector_factors: Mapping of sector name to daily factor path F_{s, t} (shape: (T,)).
        companies: Mapping of ticker symbol to CompanySimulationResult.
    """

    config: MarketConfig
    trading_days: int
    market_factors: np.ndarray
    sector_factors: Dict[str, np.ndarray]
    companies: Dict[str, CompanySimulationResult]

    def to_dict(self) -> Dict[str, Any]:
        """Export full simulation result to a JSON-serializable dictionary."""
        return {
            "metadata": {
                "trading_days": self.trading_days,
                "random_seed": self.config.random_seed,
                "annual_log_drift": self.config.annual_log_drift,
                "factor_weights": {
                    "market": self.config.weight_market,
                    "sector": self.config.weight_sector,
                    "company": self.config.weight_company,
                },
                "active_companies_count": len(self.companies),
                "sectors": sorted(list(self.sector_factors.keys())),
            },
            "market_factors": self.market_factors.tolist(),
            "sector_factors": {
                sector: series.tolist()
                for sector, series in self.sector_factors.items()
            },
            "companies": {
                ticker: res.to_dict() for ticker, res in self.companies.items()
            },
        }

    def get_price_matrix(self) -> np.ndarray:
        """Return 2D array of prices with shape (num_companies, T + 1)."""
        return np.array([res.prices for res in self.companies.values()], dtype=np.float64)

    def get_return_matrix(self) -> np.ndarray:
        """Return 2D array of log-returns with shape (num_companies, T)."""
        return np.array([res.log_returns for res in self.companies.values()], dtype=np.float64)

    def get_tickers(self) -> List[str]:
        """Return list of company tickers in consistent order."""
        return list(self.companies.keys())


class MarketSimulationEngine:
    """Synthetic stock market simulation engine using vectorized multi-factor dynamics.

    Simulates daily price and return trajectories using three independent components:
      1. Market factor: shared macroeconomic innovation across all companies.
      2. Sector factor: industry-specific innovation shared by companies within the same sector.
      3. Company factor: idiosyncratic company-specific shock (Student's t, df=5).

    The magnitude of price movements is governed directly by each company's daily volatility.
    """

    # Scale factor to standardize Student's t(df=5) to unit variance:
    # Var(T_5) = 5 / (5 - 2) = 5/3  =>  scale = sqrt(3/5) ~= 0.7745966692414834
    _STUDENT_T_DF = 5
    _STUDENT_T_SCALE = math.sqrt(3.0 / 5.0)

    def __init__(
        self,
        companies: List[CompanyProfile],
        config: Optional[MarketConfig] = None,
    ) -> None:
        """Initialize the simulation engine.

        Args:
            companies: List of CompanyProfile objects to simulate.
            config: MarketConfig instance. If None, default configuration is used.
        """
        if not companies:
            raise ValueError("companies list cannot be empty")

        # Verify ticker uniqueness
        tickers = [c.ticker for c in companies]
        if len(tickers) != len(set(tickers)):
            duplicates = [t for t in tickers if tickers.count(t) > 1]
            raise ValueError(f"Duplicate company tickers detected: {set(duplicates)}")

        self.companies = companies
        self.config = config or MarketConfig()

        # Dynamically discover unique sectors in alphabetical order
        self.sectors = sorted(list({c.sector for c in self.companies}))
        self.sector_to_idx = {sector: idx for idx, sector in enumerate(self.sectors)}

        # Pre-construct company property vectors for vectorized NumPy broadcasting
        self.num_companies = len(self.companies)
        self.base_prices = np.array(
            [c.base_price for c in self.companies], dtype=np.float64
        )  # Shape: (N,)
        self.volatilities = np.array(
            [c.volatility for c in self.companies], dtype=np.float64
        )  # Shape: (N,)
        self.sector_indices = np.array(
            [self.sector_to_idx[c.sector] for c in self.companies], dtype=np.int32
        )  # Shape: (N,)

    def _generate_stationary_ar1(
        self,
        rng: np.random.Generator,
        phi: float,
        num_series: int,
        length: int,
    ) -> np.ndarray:
        """Generate stationary AR(1) series with zero startup bias.

        Dynamic equation:
            F_0 ~ Normal(0, 1)
            F_t = phi * F_{t-1} + sqrt(1 - phi^2) * eta_t,  eta_t ~ Normal(0, 1)

        Because F_0 is sampled from the stationary distribution Normal(0, 1),
        E[F_t] = 0 and Var(F_t) = 1.0 identically for all t in {0, ..., length - 1}.

        Args:
            rng: NumPy random Generator.
            phi: Autoregressive persistence parameter (-1 < phi < 1).
            num_series: Number of independent series to generate (e.g. 1 for market, K for sectors).
            length: Number of time steps (T).

        Returns:
            Array of shape (num_series, length) with unit variance AR(1) paths.
        """
        series = np.empty((num_series, length), dtype=np.float64)
        innovations = rng.standard_normal((num_series, length))

        # Stationary initialization: variance is 1.0 at day 0
        series[:, 0] = rng.standard_normal(num_series)

        scale = math.sqrt(1.0 - phi * phi)
        for t in range(1, length):
            series[:, t] = phi * series[:, t - 1] + scale * innovations[:, t]

        return series

    def simulate(self) -> SimulationOutput:
        """Execute the multi-factor market simulation.

        Returns:
            SimulationOutput containing full price paths, return series, factor paths,
            and component attributions.
        """
        rng = np.random.default_rng(self.config.random_seed)
        t_days = self.config.trading_days
        num_comp = self.num_companies
        num_sectors = len(self.sectors)

        # -------------------------------------------------------------------------
        # 1. Generate Latent Market & Sector Factors
        # -------------------------------------------------------------------------
        # Market factor: 1 series of length T, shape: (1, T) -> flattened to (T,)
        market_series = self._generate_stationary_ar1(
            rng=rng,
            phi=self.config.phi_market,
            num_series=1,
            length=t_days,
        )[0]  # Shape: (T,)

        # Sector factors: K series of length T, shape: (num_sectors, T)
        sector_matrix = self._generate_stationary_ar1(
            rng=rng,
            phi=self.config.phi_sector,
            num_series=num_sectors,
            length=t_days,
        )  # Shape: (K, T)

        # -------------------------------------------------------------------------
        # 2. Generate Standardized Student's t Idiosyncratic Shocks
        # -------------------------------------------------------------------------
        # Raw Student's t with df=5 has variance 5/3 ~= 1.6667.
        # Multiplying by sqrt(3/5) normalizes empirical variance to 1.0 while
        # preserving fat tails and realistic earnings surprises.
        raw_t_shocks = rng.standard_t(
            df=self._STUDENT_T_DF, size=(num_comp, t_days)
        )
        company_shocks = raw_t_shocks * self._STUDENT_T_SCALE  # Shape: (N, T)

        # -------------------------------------------------------------------------
        # 3. Vectorized Factor Mapping & Return Attribution
        # -------------------------------------------------------------------------
        # Broadcast company daily volatility to shape (N, 1)
        sigma = self.volatilities[:, np.newaxis]  # Shape: (N, 1)

        # Map each company's sector to its corresponding row in sector_matrix
        # sector_indices has shape (N,) => mapped_sector_factors has shape (N, T)
        mapped_sector_factors = sector_matrix[self.sector_indices, :]

        # Broadcast market factor across all companies: shape (1, T) -> (N, T)
        broadcast_market = market_series[np.newaxis, :]

        # Component contributions:
        # Each factor's contribution is: sigma_i * beta_k * Factor_{k, t}
        # This makes volatility directly scale the magnitude of price movements.
        contrib_market = sigma * (self.config.beta_market * broadcast_market)
        contrib_sector = sigma * (self.config.beta_sector * mapped_sector_factors)
        contrib_company = sigma * (self.config.beta_company * company_shocks)
        contrib_drift = np.full((num_comp, t_days), self.config.daily_drift, dtype=np.float64)

        # Total daily log-returns:
        # r_{i, t} = drift + contrib_market + contrib_sector + contrib_company
        log_returns = contrib_drift + contrib_market + contrib_sector + contrib_company

        # -------------------------------------------------------------------------
        # 4. Price Evolution via Exponential Compounding
        # -------------------------------------------------------------------------
        # Cumulative sum of daily log-returns: shape (N, T)
        cumulative_log_returns = np.cumsum(log_returns, axis=1)

        # Prepend day 0 (zeros) so that index 0 corresponds to Day 0 (base_price)
        cum_returns_with_day0 = np.hstack(
            [np.zeros((num_comp, 1), dtype=np.float64), cumulative_log_returns]
        )  # Shape: (N, T + 1)

        # P_{i, t} = P_{i, 0} * exp(sum_{k=1}^t r_{i, k})
        # Strictly positive for all t because exp(x) > 0.
        prices = self.base_prices[:, np.newaxis] * np.exp(cum_returns_with_day0)

        # -------------------------------------------------------------------------
        # 5. Assemble Structured Results
        # -------------------------------------------------------------------------
        sector_factor_dict: Dict[str, np.ndarray] = {
            sector: sector_matrix[idx, :]
            for sector, idx in self.sector_to_idx.items()
        }

        company_results: Dict[str, CompanySimulationResult] = {}
        for idx, comp in enumerate(self.companies):
            attribution = ComponentAttribution(
                drift=contrib_drift[idx],
                market=contrib_market[idx],
                sector=contrib_sector[idx],
                company=contrib_company[idx],
            )
            company_results[comp.ticker] = CompanySimulationResult(
                profile=comp,
                prices=prices[idx],
                log_returns=log_returns[idx],
                attribution=attribution,
            )

        return SimulationOutput(
            config=self.config,
            trading_days=t_days,
            market_factors=market_series,
            sector_factors=sector_factor_dict,
            companies=company_results,
        )

