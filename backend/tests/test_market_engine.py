"""Unit tests for FinSight's synthetic market simulation engine.

Tests all mathematical invariants, dimension properties, seed determinism,
factor attribution accuracy, sector mapping integrity, and input validation.
"""

import math
import numpy as np
import pytest

from app.market_engine import (
    CompanyProfile,
    MarketConfig,
    MarketSimulationEngine,
)


@pytest.fixture
def sample_companies() -> list[CompanyProfile]:
    """Provide a standard diverse set of mock companies across sectors."""
    return [
        CompanyProfile(
            ticker="TECH1",
            name="NovaTech Systems",
            sector="Technology",
            base_price=120.0,
            volatility=0.025,
        ),
        CompanyProfile(
            ticker="TECH2",
            name="Apex Cloud",
            sector="Technology",
            base_price=85.0,
            volatility=0.030,
        ),
        CompanyProfile(
            ticker="HLTH1",
            name="BioCure Pharma",
            sector="Healthcare",
            base_price=64.0,
            volatility=0.015,
        ),
        CompanyProfile(
            ticker="FIN1",
            name="Vanguard Prime",
            sector="Financials",
            base_price=145.0,
            volatility=0.018,
        ),
    ]


# ---------------------------------------------------------------------------
# 1 & 2. Seed Reproducibility & Divergence Tests
# ---------------------------------------------------------------------------


def test_seed_determinism_identical_runs(sample_companies: list[CompanyProfile]) -> None:
    """Requirement 1: Same random seed produces identical prices and returns."""
    config1 = MarketConfig(trading_days=100, random_seed=42)
    config2 = MarketConfig(trading_days=100, random_seed=42)

    engine1 = MarketSimulationEngine(companies=sample_companies, config=config1)
    engine2 = MarketSimulationEngine(companies=sample_companies, config=config2)

    out1 = engine1.simulate()
    out2 = engine2.simulate()

    np.testing.assert_array_equal(out1.get_price_matrix(), out2.get_price_matrix())
    np.testing.assert_array_equal(out1.get_return_matrix(), out2.get_return_matrix())
    np.testing.assert_array_equal(out1.market_factors, out2.market_factors)

    for sector in out1.sector_factors:
        np.testing.assert_array_equal(
            out1.sector_factors[sector], out2.sector_factors[sector]
        )


def test_different_seeds_produce_divergent_runs(sample_companies: list[CompanyProfile]) -> None:
    """Requirement 2: Different seeds produce different simulations."""
    config1 = MarketConfig(trading_days=100, random_seed=42)
    config2 = MarketConfig(trading_days=100, random_seed=999)

    out1 = MarketSimulationEngine(companies=sample_companies, config=config1).simulate()
    out2 = MarketSimulationEngine(companies=sample_companies, config=config2).simulate()

    assert not np.array_equal(out1.get_price_matrix(), out2.get_price_matrix())
    assert not np.array_equal(out1.get_return_matrix(), out2.get_return_matrix())
    assert not np.array_equal(out1.market_factors, out2.market_factors)


# ---------------------------------------------------------------------------
# 3. Price Positivity & Finiteness
# ---------------------------------------------------------------------------


def test_prices_are_finite_and_strictly_positive(sample_companies: list[CompanyProfile]) -> None:
    """Requirement 3: All generated prices are finite and strictly positive."""
    # Test across multi-year horizon (504 days = 2 trading years)
    config = MarketConfig(trading_days=504, random_seed=123)
    engine = MarketSimulationEngine(companies=sample_companies, config=config)
    output = engine.simulate()

    price_matrix = output.get_price_matrix()
    return_matrix = output.get_return_matrix()

    # Strict positivity check
    assert np.all(price_matrix > 0.0), "Encountered non-positive price"

    # Finiteness check (no NaN, no Inf)
    assert np.all(np.isfinite(price_matrix)), "Encountered non-finite price"
    assert np.all(np.isfinite(return_matrix)), "Encountered non-finite return"

    # Base price integrity check at Day 0
    for comp in sample_companies:
        assert output.companies[comp.ticker].prices[0] == comp.base_price


# ---------------------------------------------------------------------------
# 4. Array Dimensions
# ---------------------------------------------------------------------------


def test_price_and_return_array_dimensions(sample_companies: list[CompanyProfile]) -> None:
    """Requirement 4: Price and return array dimensions are correct."""
    t_days = 60
    config = MarketConfig(trading_days=t_days, random_seed=7)
    engine = MarketSimulationEngine(companies=sample_companies, config=config)
    output = engine.simulate()

    assert output.trading_days == t_days
    assert output.market_factors.shape == (t_days,)

    for sector, series in output.sector_factors.items():
        assert series.shape == (t_days,), f"Sector {sector} factor wrong shape"

    for ticker, res in output.companies.items():
        assert res.prices.shape == (t_days + 1,), f"Price series for {ticker} wrong shape"
        assert res.log_returns.shape == (t_days,), f"Return series for {ticker} wrong shape"
        assert res.attribution.drift.shape == (t_days,)
        assert res.attribution.market.shape == (t_days,)
        assert res.attribution.sector.shape == (t_days,)
        assert res.attribution.company.shape == (t_days,)

    assert output.get_price_matrix().shape == (len(sample_companies), t_days + 1)
    assert output.get_return_matrix().shape == (len(sample_companies), t_days)


# ---------------------------------------------------------------------------
# 5. Component Attribution Sum Integrity
# ---------------------------------------------------------------------------


def test_daily_log_return_attribution_decomposition(sample_companies: list[CompanyProfile]) -> None:
    """Requirement 5: Daily log-return attribution sums to actual log-return within tolerance."""
    config = MarketConfig(trading_days=150, random_seed=314)
    output = MarketSimulationEngine(companies=sample_companies, config=config).simulate()

    for ticker, res in output.companies.items():
        attr = res.attribution
        reconstructed_returns = (
            attr.drift + attr.market + attr.sector + attr.company
        )
        # Check that the 4 decomposed components sum exactly to the logged return
        np.testing.assert_allclose(
            reconstructed_returns,
            res.log_returns,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Return attribution sum failed for {ticker}",
        )


# ---------------------------------------------------------------------------
# 6 & 7. Duplicate Tickers and Empty List Validation
# ---------------------------------------------------------------------------


def test_duplicate_tickers_rejected() -> None:
    """Requirement 6: Duplicate tickers are rejected with ValueError."""
    duplicates = [
        CompanyProfile("TECH1", "Tech A", "Technology", 100.0, 0.02),
        CompanyProfile("TECH1", "Tech B", "Technology", 120.0, 0.025),
    ]
    with pytest.raises(ValueError, match="Duplicate company tickers detected"):
        MarketSimulationEngine(companies=duplicates)


def test_empty_company_list_rejected() -> None:
    """Requirement 7: Empty company lists are rejected with ValueError."""
    with pytest.raises(ValueError, match="companies list cannot be empty"):
        MarketSimulationEngine(companies=[])


# ---------------------------------------------------------------------------
# 8. Invalid Company Prices & Volatilities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("invalid_price", [0.0, -10.0, float("nan"), float("inf"), float("-inf")])
def test_invalid_company_base_prices_rejected(invalid_price: float) -> None:
    """Requirement 8: Zero, negative, and non-finite base prices are rejected."""
    with pytest.raises(ValueError, match="base_price for .* must be a finite positive number"):
        CompanyProfile(
            ticker="BAD",
            name="Bad Price Corp",
            sector="Finance",
            base_price=invalid_price,
            volatility=0.02,
        )


@pytest.mark.parametrize("invalid_vol", [0.0, -0.01, float("nan"), float("inf"), float("-inf")])
def test_invalid_company_volatilities_rejected(invalid_vol: float) -> None:
    """Requirement 8: Zero, negative, and non-finite volatilities are rejected."""
    with pytest.raises(ValueError, match="volatility for .* must be a finite positive number"):
        CompanyProfile(
            ticker="BAD",
            name="Bad Vol Corp",
            sector="Finance",
            base_price=100.0,
            volatility=invalid_vol,
        )


# ---------------------------------------------------------------------------
# 9. Invalid MarketConfig Parameters
# ---------------------------------------------------------------------------


def test_invalid_trading_days_rejected() -> None:
    """Requirement 9: Invalid trading days (< 1) rejected."""
    with pytest.raises(ValueError, match="trading_days must be an integer >= 1"):
        MarketConfig(trading_days=0)

    with pytest.raises(ValueError, match="trading_days must be an integer >= 1"):
        MarketConfig(trading_days=-5)


def test_weights_not_summing_to_one_rejected() -> None:
    """Requirement 9: Factor weights not summing to 1.0 are rejected."""
    with pytest.raises(ValueError, match="Factor variance weights must sum to 1.0"):
        MarketConfig(weight_market=0.5, weight_sector=0.5, weight_company=0.5)

    with pytest.raises(ValueError, match="Factor variance weights must sum to 1.0"):
        MarketConfig(weight_market=0.2, weight_sector=0.2, weight_company=0.2)


def test_negative_factor_weights_rejected() -> None:
    """Requirement 9: Negative factor weights are rejected."""
    with pytest.raises(ValueError, match="must be non-negative"):
        MarketConfig(weight_market=-0.1, weight_sector=0.6, weight_company=0.5)


@pytest.mark.parametrize("invalid_phi", [1.0, -1.0, 1.25, -2.0])
def test_non_stationary_ar1_phi_rejected(invalid_phi: float) -> None:
    """Requirement 9: |phi| >= 1.0 rejected for AR(1) stationarity."""
    with pytest.raises(ValueError, match="must satisfy -1 < phi < 1 for stationary AR"):
        MarketConfig(phi_market=invalid_phi)

    with pytest.raises(ValueError, match="must satisfy -1 < phi < 1 for stationary AR"):
        MarketConfig(phi_sector=invalid_phi)


@pytest.mark.parametrize(
    "param_name, bad_value",
    [
        ("annual_log_drift", float("nan")),
        ("annual_log_drift", float("inf")),
        ("weight_market", float("nan")),
        ("weight_sector", float("inf")),
        ("phi_market", float("nan")),
        ("phi_sector", float("inf")),
    ],
)
def test_non_finite_config_parameters_rejected(param_name: str, bad_value: float) -> None:
    """Requirement 9: Non-finite parameters in MarketConfig are rejected."""
    kwargs = {param_name: bad_value}
    with pytest.raises(ValueError, match="must be a finite real number"):
        MarketConfig(**kwargs)


# ---------------------------------------------------------------------------
# 10. Sector Factor Mapping Integrity
# ---------------------------------------------------------------------------


def test_sector_factors_mapped_correctly_to_companies() -> None:
    """Requirement 10: Sector factors are mapped correctly to companies.

    Companies in the same sector must share the exact same sector innovation path,
    while companies in different sectors must map to different sector innovation paths.
    """
    companies = [
        CompanyProfile("TECH_A", "Tech Alpha", "Technology", 100.0, 0.02),
        CompanyProfile("TECH_B", "Tech Beta", "Technology", 80.0, 0.03),
        CompanyProfile("HLTH_A", "Health Alpha", "Healthcare", 50.0, 0.015),
    ]

    config = MarketConfig(trading_days=50, random_seed=888)
    output = MarketSimulationEngine(companies=companies, config=config).simulate()

    # Sector factor paths stored in output
    tech_factor = output.sector_factors["Technology"]
    hlth_factor = output.sector_factors["Healthcare"]

    # Technology and Healthcare sector paths must be distinct
    assert not np.array_equal(tech_factor, hlth_factor)

    # In CompanySimulationResult attribution:
    # contrib_sector = sigma_i * beta_sector * SectorFactor
    # Therefore: contrib_sector / (sigma_i * beta_sector) == SectorFactor
    beta_s = config.beta_sector

    norm_tech_a = output.companies["TECH_A"].attribution.sector / (0.02 * beta_s)
    norm_tech_b = output.companies["TECH_B"].attribution.sector / (0.03 * beta_s)
    norm_hlth_a = output.companies["HLTH_A"].attribution.sector / (0.015 * beta_s)

    # Both Tech companies must share the exact same standardized sector series
    np.testing.assert_allclose(norm_tech_a, tech_factor, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(norm_tech_b, tech_factor, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(norm_hlth_a, hlth_factor, rtol=1e-10, atol=1e-10)

