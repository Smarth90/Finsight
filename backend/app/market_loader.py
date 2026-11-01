"""FinSight MongoDB Market-Data Loader.

This module handles loading active company profiles from MongoDB, executing
simulations with guaranteed reproducible seeds, and persisting compact time-series
records to MongoDB Atlas.

Collections Managed:
--------------------
1. `companies` (read-only):
   Source collection containing company metadata and daily volatility parameters.
2. `simulation_runs` (read/write):
   Catalog of simulation runs with full configuration metadata, compact company
   profile snapshots, execution status, and audit timestamps.
3. `market_prices` (write):
   Compact daily price documents (one per company per day).
   Day 0 stores base_price with log_return=None.
   Days 1..T store closing price and daily log-return.
"""

from __future__ import annotations

from datetime import datetime, timezone
import secrets
from typing import Any, Dict, List, Optional

import numpy as np
from pymongo.database import Database

from app.market_engine import (
    CompanyProfile,
    MarketConfig,
    MarketSimulationEngine,
    SimulationOutput,
)


def generate_random_seed() -> int:
    """Generate a positive 32-bit integer random seed in [1, 2^31 - 1]."""
    return secrets.randbelow(2**31 - 1) + 1


def resolve_simulation_config(config: Optional[MarketConfig]) -> MarketConfig:
    """Ensure MarketConfig has an effective integer random seed.

    If config is None, a default MarketConfig with a generated integer seed is returned.
    If config.random_seed is None or not an integer, a new MarketConfig copy is returned
    with a generated integer seed, leaving all other parameters intact.
    """
    if config is None:
        return MarketConfig(random_seed=generate_random_seed())

    if (
        config.random_seed is not None
        and isinstance(config.random_seed, int)
        and not isinstance(config.random_seed, bool)
    ):
        return config

    return MarketConfig(
        trading_days=config.trading_days,
        random_seed=generate_random_seed(),
        annual_log_drift=config.annual_log_drift,
        weight_market=config.weight_market,
        weight_sector=config.weight_sector,
        weight_company=config.weight_company,
        phi_market=config.phi_market,
        phi_sector=config.phi_sector,
    )


def ensure_indexes(db: Database) -> None:
    """Ensure required indexes exist on MongoDB collections.

    Indexes:
    - simulation_runs: unique index on `simulation_id`
    - market_prices: unique compound index on `(simulation_id, ticker, day)`

    Args:
        db: MongoDB Database instance.
    """
    db.simulation_runs.create_index([("simulation_id", 1)], unique=True)
    db.market_prices.create_index(
        [("simulation_id", 1), ("ticker", 1), ("day", 1)], unique=True
    )


def fetch_active_companies(db: Database) -> List[CompanyProfile]:
    """Fetch all active company profiles from MongoDB `companies` collection.

    Inactive companies (is_active == False) are excluded.

    Args:
        db: MongoDB Database instance.

    Returns:
        List of validated CompanyProfile instances.

    Raises:
        ValueError: If no active companies are found or if data is missing required fields.
    """
    cursor = db.companies.find(
        {"is_active": True},
        {
            "_id": 0,
            "ticker": 1,
            "name": 1,
            "sector": 1,
            "base_price": 1,
            "volatility": 1,
        },
    )

    profiles: List[CompanyProfile] = []
    for doc in cursor:
        try:
            profile = CompanyProfile(
                ticker=doc["ticker"],
                name=doc["name"],
                sector=doc["sector"],
                base_price=float(doc["base_price"]),
                volatility=float(doc["volatility"]),
            )
            profiles.append(profile)
        except (KeyError, TypeError, ValueError) as err:
            raise ValueError(f"Invalid company document in MongoDB: {doc}. Error: {err}") from err

    if not profiles:
        raise ValueError("No active companies found in MongoDB 'companies' collection.")

    return profiles


def build_price_documents(
    simulation_id: str, output: SimulationOutput
) -> List[Dict[str, Any]]:
    """Transform simulation output into compact MongoDB market_prices documents.

    Each company produces T + 1 documents:
    - Day 0: base price, log_return is None.
    - Day 1..T: simulated price, corresponding daily log-return.

    Strict Validations:
    - len(prices) == output.trading_days + 1
    - len(log_returns) == output.trading_days
    - All prices finite and strictly positive (> 0.0)
    - All log-returns finite

    Args:
        simulation_id: Unique simulation run identifier.
        output: SimulationOutput instance from MarketSimulationEngine.

    Returns:
        List of compact dictionary documents ready for bulk insertion.

    Raises:
        ValueError: If inputs or arrays are malformed, mismatched, non-finite, or non-positive.
        TypeError: If output is not an instance of SimulationOutput.
    """
    if not isinstance(simulation_id, str) or not simulation_id.strip():
        raise ValueError(
            f"simulation_id must be a non-empty string, got {simulation_id!r}"
        )

    if not isinstance(output, SimulationOutput):
        raise TypeError(
            f"output must be an instance of SimulationOutput, got {type(output).__name__}"
        )

    if not output.companies:
        raise ValueError("output.companies cannot be empty")

    if output.trading_days < 1:
        raise ValueError(f"output.trading_days must be >= 1, got {output.trading_days}")

    expected_price_len = output.trading_days + 1
    expected_return_len = output.trading_days

    # Strict pre-validation across all company time-series arrays
    for ticker, res in output.companies.items():
        prices = np.asarray(res.prices)
        log_returns = np.asarray(res.log_returns)

        if len(prices) != expected_price_len:
            raise ValueError(
                f"Company '{ticker}' prices length mismatch: expected {expected_price_len} "
                f"(trading_days + 1), got {len(prices)}"
            )

        if len(log_returns) != expected_return_len:
            raise ValueError(
                f"Company '{ticker}' log_returns length mismatch: expected {expected_return_len} "
                f"(trading_days), got {len(log_returns)}"
            )

        if not np.all(np.isfinite(prices)):
            raise ValueError(
                f"Company '{ticker}' prices array contains non-finite values (NaN, Inf, or -Inf)"
            )

        if not np.all(prices > 0.0):
            raise ValueError(
                f"Company '{ticker}' prices array contains non-positive values (<= 0.0)"
            )

        if not np.all(np.isfinite(log_returns)):
            raise ValueError(
                f"Company '{ticker}' log_returns array contains non-finite values (NaN, Inf, or -Inf)"
            )

    documents: List[Dict[str, Any]] = []

    for ticker, res in output.companies.items():
        prices = res.prices
        log_returns = res.log_returns

        # Day 0: Base price, log-return is null
        documents.append(
            {
                "simulation_id": simulation_id,
                "ticker": ticker,
                "day": 0,
                "price": float(prices[0]),
                "log_return": None,
            }
        )

        # Days 1..T: Simulated prices and corresponding daily log-returns
        for t in range(1, len(prices)):
            documents.append(
                {
                    "simulation_id": simulation_id,
                    "ticker": ticker,
                    "day": t,
                    "price": float(prices[t]),
                    "log_return": float(log_returns[t - 1]),
                }
            )

    return documents


def save_simulation_results(
    db: Database,
    simulation_id: str,
    output: SimulationOutput,
    batch_size: int = 5000,
) -> Dict[str, Any]:
    """Persist simulation metadata, company profile snapshots, and price time series to MongoDB.

    Rejects duplicate simulation_id with ValueError. Never deletes previously
    successful simulations.

    On persistence failure:
    - Treats a failure to update status to COMPLETED as a persistence failure and deletes all inserted prices.
    - Attempts to mark the simulation run as FAILED.
    - Preserves and re-raises the original exception without silently suppressing recovery failures.

    Args:
        db: MongoDB Database instance.
        simulation_id: Unique identifier for this simulation run.
        output: SimulationOutput from MarketSimulationEngine.
        batch_size: Number of documents per insert_many chunk (>= 1).

    Returns:
        The created simulation_runs document dictionary.

    Raises:
        ValueError: For invalid inputs, missing seed, or duplicate simulation_id.
        PyMongoError: If a MongoDB operation fails.
        RuntimeError: If primary persistence failed and subsequent recovery steps also failed.
    """
    if not isinstance(simulation_id, str) or not simulation_id.strip():
        raise ValueError(
            f"simulation_id must be a non-empty string, got {simulation_id!r}"
        )

    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be an integer >= 1")

    if not isinstance(output, SimulationOutput):
        raise TypeError(
            f"output must be an instance of SimulationOutput, got {type(output).__name__}"
        )

    if not output.companies:
        raise ValueError("output.companies cannot be empty")

    if (
        output.config.random_seed is None
        or not isinstance(output.config.random_seed, int)
        or isinstance(output.config.random_seed, bool)
    ):
        raise ValueError(
            f"output.config.random_seed must be an integer, got {output.config.random_seed!r}. "
            "Ensure an effective integer random seed is configured on MarketConfig before simulation."
        )

    ensure_indexes(db)

    # Reject duplicate simulation_id
    existing_run = db.simulation_runs.find_one({"simulation_id": simulation_id})
    if existing_run is not None:
        raise ValueError(
            f"Simulation run with ID '{simulation_id}' already exists. "
            "Overwriting is not permitted."
        )

    # Build price documents and perform strict array validation before recording run
    price_docs = build_price_documents(simulation_id=simulation_id, output=output)

    # Compact snapshot of company profiles used in this simulation for full provenance
    company_snapshots = [
        {
            "ticker": res.profile.ticker,
            "sector": res.profile.sector,
            "base_price": float(res.profile.base_price),
            "volatility": float(res.profile.volatility),
        }
        for res in output.companies.values()
    ]

    # Initial metadata document with status "RUNNING"
    created_at = datetime.now(timezone.utc)
    run_doc: Dict[str, Any] = {
        "simulation_id": simulation_id,
        "random_seed": output.config.random_seed,
        "trading_days": output.trading_days,
        "company_count": len(output.companies),
        "companies": company_snapshots,
        "annual_log_drift": output.config.annual_log_drift,
        "factor_weights": {
            "market": output.config.weight_market,
            "sector": output.config.weight_sector,
            "company": output.config.weight_company,
        },
        "phi_market": output.config.phi_market,
        "phi_sector": output.config.phi_sector,
        "status": "RUNNING",
        "created_at": created_at,
        "completed_at": None,
    }

    db.simulation_runs.insert_one(dict(run_doc))

    # Batch insertion into market_prices
    try:
        total_docs = len(price_docs)
        for i in range(0, total_docs, batch_size):
            chunk = price_docs[i : i + batch_size]
            db.market_prices.insert_many(chunk, ordered=True)

        completed_at = datetime.now(timezone.utc)
        db.simulation_runs.update_one(
            {"simulation_id": simulation_id},
            {
                "$set": {
                    "status": "COMPLETED",
                    "completed_at": completed_at,
                    "price_document_count": total_docs,
                }
            },
        )
        run_doc["status"] = "COMPLETED"
        run_doc["completed_at"] = completed_at
        run_doc["price_document_count"] = total_docs
        return run_doc

    except Exception as primary_exc:
        cleanup_err: Optional[Exception] = None
        status_update_err: Optional[Exception] = None

        # 1. Attempt to delete partially inserted price documents
        try:
            db.market_prices.delete_many({"simulation_id": simulation_id})
        except Exception as c_err:
            cleanup_err = c_err

        # 2. Attempt to mark the simulation run as FAILED
        try:
            db.simulation_runs.update_one(
                {"simulation_id": simulation_id},
                {
                    "$set": {
                        "status": "FAILED",
                        "completed_at": datetime.now(timezone.utc),
                        "error_message": str(primary_exc),
                        "cleanup_error": str(cleanup_err) if cleanup_err else None,
                    }
                },
            )
        except Exception as s_err:
            status_update_err = s_err

        # 3. Handle secondary recovery/status failures without silent suppression
        secondary_failures: List[str] = []
        if cleanup_err is not None:
            secondary_failures.append(f"cleanup delete_many failed ({cleanup_err})")
        if status_update_err is not None:
            secondary_failures.append(
                f"status update to FAILED failed ({status_update_err})"
            )

        if secondary_failures:
            secondary_msg = "; ".join(secondary_failures)
            raise RuntimeError(
                f"Persistence failed for simulation '{simulation_id}': {primary_exc}. "
                f"Additional recovery failure(s): {secondary_msg}"
            ) from primary_exc

        raise


def run_and_save_simulation(
    db: Database,
    simulation_id: str,
    config: Optional[MarketConfig] = None,
    batch_size: int = 5000,
) -> SimulationOutput:
    """Run an end-to-end synthetic market simulation and persist results to MongoDB.

    1. Ensures an effective integer random seed is configured on MarketConfig.
    2. Loads active company profiles from MongoDB.
    3. Runs MarketSimulationEngine with the effective configuration.
    4. Persists simulation_runs metadata (including company snapshot) and market_prices time series.

    Args:
        db: MongoDB Database instance.
        simulation_id: Unique simulation run identifier.
        config: MarketConfig instance. If None or without a seed, an effective seed is assigned.
        batch_size: Batch size for inserting market_prices documents.

    Returns:
        The in-memory SimulationOutput object.
    """
    effective_config = resolve_simulation_config(config)
    companies = fetch_active_companies(db)
    engine = MarketSimulationEngine(companies=companies, config=effective_config)
    output = engine.simulate()
    save_simulation_results(
        db=db,
        simulation_id=simulation_id,
        output=output,
        batch_size=batch_size,
    )
    return output
