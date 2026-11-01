"""Unit tests for FinSight's MongoDB market-data loader.

Tests active-company loading, compact document transformation, strict array validation,
Day 0 base price representation, index declarations, batch insertions, duplicate rejection,
error recovery/cleanup behaviors, company profile snapshots, and effective seed reproducibility.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest
from pymongo.errors import PyMongoError

from app.market_engine import (
    CompanyProfile,
    CompanySimulationResult,
    ComponentAttribution,
    MarketConfig,
    MarketSimulationEngine,
    SimulationOutput,
)
from app.market_loader import (
    build_price_documents,
    ensure_indexes,
    fetch_active_companies,
    generate_random_seed,
    resolve_simulation_config,
    run_and_save_simulation,
    save_simulation_results,
)


@pytest.fixture
def mock_db() -> MagicMock:
    """Fixture providing a mock MongoDB Database."""
    db = MagicMock()
    db.companies = MagicMock()
    db.simulation_runs = MagicMock()
    db.market_prices = MagicMock()
    return db


@pytest.fixture
def sample_companies() -> list[CompanyProfile]:
    """Fixture providing mock company profiles."""
    return [
        CompanyProfile(
            ticker="TECH1",
            name="NovaTech Systems",
            sector="Technology",
            base_price=100.0,
            volatility=0.025,
        ),
        CompanyProfile(
            ticker="HLTH1",
            name="BioCure Pharma",
            sector="Healthcare",
            base_price=50.0,
            volatility=0.015,
        ),
    ]


@pytest.fixture
def sample_simulation_output(sample_companies: list[CompanyProfile]) -> SimulationOutput:
    """Fixture providing a completed simulation output."""
    config = MarketConfig(trading_days=5, random_seed=42)
    engine = MarketSimulationEngine(companies=sample_companies, config=config)
    return engine.simulate()


# ---------------------------------------------------------------------------
# 1. Active-Company Loading Tests
# ---------------------------------------------------------------------------


def test_fetch_active_companies_success(mock_db: MagicMock) -> None:
    """Requirement: Active companies are loaded and filtered properly."""
    mock_db.companies.find.return_value = [
        {
            "ticker": "TECH1",
            "name": "NovaTech Systems",
            "sector": "Technology",
            "base_price": 100.0,
            "volatility": 0.025,
            "is_active": True,
        },
        {
            "ticker": "FIN1",
            "name": "Apex Finance",
            "sector": "Financials",
            "base_price": 80.0,
            "volatility": 0.018,
            "is_active": True,
        },
    ]

    profiles = fetch_active_companies(mock_db)

    # Verify query filter targets active companies
    mock_db.companies.find.assert_called_once_with(
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

    assert len(profiles) == 2
    assert profiles[0].ticker == "TECH1"
    assert profiles[0].base_price == 100.0
    assert profiles[1].ticker == "FIN1"
    assert profiles[1].volatility == 0.018


def test_fetch_active_companies_empty_raises_error(mock_db: MagicMock) -> None:
    """Requirement: Empty company list raises ValueError."""
    mock_db.companies.find.return_value = []

    with pytest.raises(ValueError, match="No active companies found"):
        fetch_active_companies(mock_db)


def test_fetch_active_companies_malformed_document_raises_error(
    mock_db: MagicMock,
) -> None:
    """Requirement: Malformed company documents raise ValueError."""
    mock_db.companies.find.return_value = [
        {
            "ticker": "BAD1",
            "name": "Bad Company",
            # missing sector, base_price, volatility
        }
    ]

    with pytest.raises(ValueError, match="Invalid company document in MongoDB"):
        fetch_active_companies(mock_db)


# ---------------------------------------------------------------------------
# 2. Document Transformation & Strict Validation Tests
# ---------------------------------------------------------------------------


def test_build_price_documents_structure_and_day0(
    sample_simulation_output: SimulationOutput,
) -> None:
    """Requirement: Documents are compact and Day 0 has null log_return."""
    sim_id = "sim_test_001"
    docs = build_price_documents(sim_id, sample_simulation_output)

    # 2 companies * (5 trading days + 1 base price day) = 12 documents
    assert len(docs) == 12

    # Group by ticker
    tech_docs = [d for d in docs if d["ticker"] == "TECH1"]
    hlth_docs = [d for d in docs if d["ticker"] == "HLTH1"]

    assert len(tech_docs) == 6
    assert len(hlth_docs) == 6

    # Verify Day 0 document for TECH1
    day_0 = tech_docs[0]
    assert day_0["simulation_id"] == sim_id
    assert day_0["ticker"] == "TECH1"
    assert day_0["day"] == 0
    assert day_0["price"] == 100.0
    assert day_0["log_return"] is None

    # Verify Day 1..5 documents
    for day_idx in range(1, 6):
        d = tech_docs[day_idx]
        assert d["day"] == day_idx
        assert isinstance(d["price"], float)
        assert d["price"] > 0.0
        assert isinstance(d["log_return"], float)

        expected_price = float(sample_simulation_output.companies["TECH1"].prices[day_idx])
        expected_return = float(
            sample_simulation_output.companies["TECH1"].log_returns[day_idx - 1]
        )
        assert d["price"] == pytest.approx(expected_price)
        assert d["log_return"] == pytest.approx(expected_return)

    # Verify strict 5-field compactness (no factor arrays)
    expected_keys = {"simulation_id", "ticker", "day", "price", "log_return"}
    for doc in docs:
        assert set(doc.keys()) == expected_keys


def test_build_price_documents_validation(
    sample_simulation_output: SimulationOutput,
) -> None:
    """Requirement: Invalid simulation_id or output raises appropriate errors."""
    with pytest.raises(ValueError, match="simulation_id must be a non-empty string"):
        build_price_documents("", sample_simulation_output)

    with pytest.raises(ValueError, match="simulation_id must be a non-empty string"):
        build_price_documents("   ", sample_simulation_output)

    with pytest.raises(TypeError, match="output must be an instance of SimulationOutput"):
        build_price_documents("sim_1", {"bad": "data"})  # type: ignore


def test_build_price_documents_mismatched_price_length(
    sample_companies: list[CompanyProfile],
) -> None:
    """Requirement 1: len(prices) != trading_days + 1 raises ValueError."""
    config = MarketConfig(trading_days=5, random_seed=42)
    engine = MarketSimulationEngine(companies=sample_companies, config=config)
    output = engine.simulate()

    # Artificially alter prices length for TECH1 to 4 (expected 6)
    bad_prices = output.companies["TECH1"].prices[:4]
    output.companies["TECH1"].prices = bad_prices

    with pytest.raises(ValueError, match="prices length mismatch: expected 6 .* got 4"):
        build_price_documents("sim_bad_len", output)


def test_build_price_documents_mismatched_return_length(
    sample_companies: list[CompanyProfile],
) -> None:
    """Requirement 1: len(log_returns) != trading_days raises ValueError."""
    config = MarketConfig(trading_days=5, random_seed=42)
    engine = MarketSimulationEngine(companies=sample_companies, config=config)
    output = engine.simulate()

    # Artificially alter log_returns length for HLTH1 to 3 (expected 5)
    bad_returns = output.companies["HLTH1"].log_returns[:3]
    output.companies["HLTH1"].log_returns = bad_returns

    with pytest.raises(ValueError, match="log_returns length mismatch: expected 5 .* got 3"):
        build_price_documents("sim_bad_len", output)


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_build_price_documents_non_finite_prices(
    sample_companies: list[CompanyProfile], bad_val: float
) -> None:
    """Requirement 1: Non-finite prices raise ValueError."""
    config = MarketConfig(trading_days=5, random_seed=42)
    output = MarketSimulationEngine(companies=sample_companies, config=config).simulate()

    output.companies["TECH1"].prices[2] = bad_val
    with pytest.raises(ValueError, match="prices array contains non-finite values"):
        build_price_documents("sim_non_finite_price", output)


@pytest.mark.parametrize("bad_price", [0.0, -10.5])
def test_build_price_documents_non_positive_prices(
    sample_companies: list[CompanyProfile], bad_price: float
) -> None:
    """Requirement 1: Non-positive prices (<= 0) raise ValueError."""
    config = MarketConfig(trading_days=5, random_seed=42)
    output = MarketSimulationEngine(companies=sample_companies, config=config).simulate()

    output.companies["TECH1"].prices[3] = bad_price
    with pytest.raises(ValueError, match="prices array contains non-positive values"):
        build_price_documents("sim_non_positive_price", output)


@pytest.mark.parametrize("bad_val", [np.nan, np.inf, -np.inf])
def test_build_price_documents_non_finite_returns(
    sample_companies: list[CompanyProfile], bad_val: float
) -> None:
    """Requirement 1: Non-finite log-returns raise ValueError."""
    config = MarketConfig(trading_days=5, random_seed=42)
    output = MarketSimulationEngine(companies=sample_companies, config=config).simulate()

    output.companies["HLTH1"].log_returns[1] = bad_val
    with pytest.raises(ValueError, match="log_returns array contains non-finite values"):
        build_price_documents("sim_non_finite_ret", output)


# ---------------------------------------------------------------------------
# 3. Index Specifications
# ---------------------------------------------------------------------------


def test_ensure_indexes_creates_correct_specifications(mock_db: MagicMock) -> None:
    """Requirement: Indexes are created with unique constraints."""
    ensure_indexes(mock_db)

    # Check simulation_runs index
    mock_db.simulation_runs.create_index.assert_called_once_with(
        [("simulation_id", 1)], unique=True
    )

    # Check market_prices compound unique index
    mock_db.market_prices.create_index.assert_called_once_with(
        [("simulation_id", 1), ("ticker", 1), ("day", 1)], unique=True
    )


# ---------------------------------------------------------------------------
# 4. Batch Insertion, Metadata & Company Snapshot Tests
# ---------------------------------------------------------------------------


def test_save_simulation_results_batching_and_metadata(
    mock_db: MagicMock, sample_simulation_output: SimulationOutput
) -> None:
    """Requirement: Batching respects batch_size and records complete metadata and company snapshot."""
    mock_db.simulation_runs.find_one.return_value = None

    # Total documents = 12. With batch_size=5, expect 3 batches (5, 5, 2)
    run_doc = save_simulation_results(
        db=mock_db,
        simulation_id="sim_batch_test",
        output=sample_simulation_output,
        batch_size=5,
    )

    # Verify batch calls
    assert mock_db.market_prices.insert_many.call_count == 3
    call_args_list = mock_db.market_prices.insert_many.call_args_list
    assert len(call_args_list[0][0][0]) == 5
    assert len(call_args_list[1][0][0]) == 5
    assert len(call_args_list[2][0][0]) == 2

    # Verify initial insert_one metadata
    mock_db.simulation_runs.insert_one.assert_called_once()
    initial_insert = mock_db.simulation_runs.insert_one.call_args[0][0]
    assert initial_insert["simulation_id"] == "sim_batch_test"
    assert initial_insert["random_seed"] == 42
    assert initial_insert["trading_days"] == 5
    assert initial_insert["company_count"] == 2

    # Verify company profile snapshot
    assert "companies" in initial_insert
    assert len(initial_insert["companies"]) == 2
    tech_snap = next(c for c in initial_insert["companies"] if c["ticker"] == "TECH1")
    assert tech_snap == {
        "ticker": "TECH1",
        "sector": "Technology",
        "base_price": 100.0,
        "volatility": 0.025,
    }

    assert initial_insert["annual_log_drift"] == 0.08
    assert initial_insert["factor_weights"] == {
        "market": 0.35,
        "sector": 0.35,
        "company": 0.30,
    }
    assert initial_insert["phi_market"] == 0.10
    assert initial_insert["phi_sector"] == 0.08
    assert initial_insert["status"] == "RUNNING"
    assert initial_insert["created_at"] is not None

    # Verify update_one marking COMPLETED
    mock_db.simulation_runs.update_one.assert_called_once_with(
        {"simulation_id": "sim_batch_test"},
        {
            "$set": {
                "status": "COMPLETED",
                "completed_at": run_doc["completed_at"],
                "price_document_count": 12,
            }
        },
    )

    assert run_doc["status"] == "COMPLETED"
    assert run_doc["price_document_count"] == 12


def test_invalid_batch_sizes_rejected(
    mock_db: MagicMock, sample_simulation_output: SimulationOutput
) -> None:
    """Requirement: Invalid batch sizes (< 1, booleans, or non-integer) are rejected."""
    with pytest.raises(ValueError, match="batch_size must be an integer >= 1"):
        save_simulation_results(
            mock_db, "sim_1", sample_simulation_output, batch_size=0
        )

    with pytest.raises(ValueError, match="batch_size must be an integer >= 1"):
        save_simulation_results(
            mock_db, "sim_1", sample_simulation_output, batch_size=-10
        )

    with pytest.raises(ValueError, match="batch_size must be an integer >= 1"):
        save_simulation_results(
            mock_db, "sim_1", sample_simulation_output, batch_size="5000"  # type: ignore
        )

    with pytest.raises(ValueError, match="batch_size must be an integer >= 1"):
        save_simulation_results(
            mock_db, "sim_1", sample_simulation_output, batch_size=True  # type: ignore
        )

    with pytest.raises(ValueError, match="batch_size must be an integer >= 1"):
        save_simulation_results(
            mock_db, "sim_1", sample_simulation_output, batch_size=False  # type: ignore
        )


def test_duplicate_simulation_id_rejected(
    mock_db: MagicMock, sample_simulation_output: SimulationOutput
) -> None:
    """Requirement: Existing simulation_id is rejected with ValueError."""
    mock_db.simulation_runs.find_one.return_value = {
        "simulation_id": "existing_sim_123",
        "status": "COMPLETED",
    }

    with pytest.raises(
        ValueError, match="Simulation run with ID 'existing_sim_123' already exists"
    ):
        save_simulation_results(
            mock_db, "existing_sim_123", sample_simulation_output
        )

    mock_db.market_prices.insert_many.assert_not_called()
    mock_db.simulation_runs.insert_one.assert_not_called()


# ---------------------------------------------------------------------------
# 5. Persistence Recovery & Failure Tests
# ---------------------------------------------------------------------------


def test_partial_insertion_failure_triggers_cleanup_and_reraises(
    mock_db: MagicMock, sample_simulation_output: SimulationOutput
) -> None:
    """Requirement: On failure, partial documents are cleaned up and original error re-raised."""
    mock_db.simulation_runs.find_one.return_value = None

    original_error = PyMongoError("Connection reset by peer during insert")
    mock_db.market_prices.insert_many.side_effect = [
        MagicMock(),  # First batch succeeds
        original_error,  # Second batch fails
    ]

    with pytest.raises(PyMongoError, match="Connection reset by peer during insert"):
        save_simulation_results(
            mock_db, "sim_fail_test", sample_simulation_output, batch_size=5
        )

    # Verify cleanup attempt: delete_many called for this simulation_id
    mock_db.market_prices.delete_many.assert_called_once_with(
        {"simulation_id": "sim_fail_test"}
    )

    # Verify status marked as FAILED
    mock_db.simulation_runs.update_one.assert_called_once()
    update_call_args = mock_db.simulation_runs.update_one.call_args
    assert update_call_args[0][0] == {"simulation_id": "sim_fail_test"}
    set_fields = update_call_args[0][1]["$set"]
    assert set_fields["status"] == "FAILED"
    assert "Connection reset by peer" in set_fields["error_message"]
    assert set_fields["cleanup_error"] is None


def test_failure_during_completed_status_update_triggers_rollback(
    mock_db: MagicMock, sample_simulation_output: SimulationOutput
) -> None:
    """Requirement 2: Failure to update status to COMPLETED triggers price rollback and marks FAILED."""
    mock_db.simulation_runs.find_one.return_value = None

    update_completed_error = PyMongoError("Primary error: network dropped during status update to COMPLETED")

    # insert_many succeeds completely
    mock_db.market_prices.insert_many.return_value = MagicMock()

    # First update_one (COMPLETED) fails; second update_one (FAILED) succeeds
    mock_db.simulation_runs.update_one.side_effect = [
        update_completed_error,
        MagicMock(),
    ]

    with pytest.raises(PyMongoError, match="network dropped during status update to COMPLETED"):
        save_simulation_results(
            mock_db, "sim_completed_fail", sample_simulation_output
        )

    # Verify all inserted prices were rolled back
    mock_db.market_prices.delete_many.assert_called_once_with(
        {"simulation_id": "sim_completed_fail"}
    )

    # Verify second update_one set status to FAILED
    assert mock_db.simulation_runs.update_one.call_count == 2
    failed_call_args = mock_db.simulation_runs.update_one.call_args_list[1]
    assert failed_call_args[0][0] == {"simulation_id": "sim_completed_fail"}
    assert failed_call_args[0][1]["$set"]["status"] == "FAILED"


def test_failure_during_failed_status_update_does_not_suppress_errors(
    mock_db: MagicMock, sample_simulation_output: SimulationOutput
) -> None:
    """Requirement 2: Failure to mark the run FAILED is never silently suppressed."""
    mock_db.simulation_runs.find_one.return_value = None

    insert_error = PyMongoError("Insert error")
    status_failed_error = PyMongoError("Database down: cannot mark FAILED")

    mock_db.market_prices.insert_many.side_effect = insert_error
    mock_db.simulation_runs.update_one.side_effect = status_failed_error

    with pytest.raises(RuntimeError) as exc_info:
        save_simulation_results(
            mock_db, "sim_double_fail_status", sample_simulation_output
        )

    err_msg = str(exc_info.value)
    assert "Persistence failed for simulation 'sim_double_fail_status'" in err_msg
    assert "Insert error" in err_msg
    assert "status update to FAILED failed (Database down: cannot mark FAILED)" in err_msg
    assert exc_info.value.__cause__ is insert_error


def test_cleanup_failure_does_not_suppress_errors(
    mock_db: MagicMock, sample_simulation_output: SimulationOutput
) -> None:
    """Requirement: If cleanup also fails, RuntimeError chains both errors."""
    mock_db.simulation_runs.find_one.return_value = None

    insert_error = PyMongoError("Primary insertion error")
    cleanup_error = PyMongoError("Secondary cleanup error: network down")

    mock_db.market_prices.insert_many.side_effect = insert_error
    mock_db.market_prices.delete_many.side_effect = cleanup_error

    with pytest.raises(RuntimeError) as exc_info:
        save_simulation_results(
            mock_db, "sim_double_fail", sample_simulation_output
        )

    err_msg = str(exc_info.value)
    assert "Persistence failed for simulation 'sim_double_fail'" in err_msg
    assert "Primary insertion error" in err_msg
    assert "cleanup delete_many failed (Secondary cleanup error: network down)" in err_msg
    assert exc_info.value.__cause__ is insert_error


# ---------------------------------------------------------------------------
# 6. Reproducibility & Effective Seed Tests
# ---------------------------------------------------------------------------


def test_save_simulation_results_rejects_missing_random_seed(
    mock_db: MagicMock, sample_companies: list[CompanyProfile]
) -> None:
    """Requirement 3: save_simulation_results rejects output without an integer random_seed."""
    config = MarketConfig(trading_days=5, random_seed=None)
    engine = MarketSimulationEngine(companies=sample_companies, config=config)
    output = engine.simulate()

    with pytest.raises(ValueError, match="output.config.random_seed must be an integer"):
        save_simulation_results(mock_db, "sim_no_seed", output)


def test_run_and_save_simulation_generates_effective_seed_when_unspecified(
    mock_db: MagicMock,
) -> None:
    """Requirement 3: An effective integer seed is generated and actually used when caller provides None."""
    mock_db.companies.find.return_value = [
        {
            "ticker": "TECH1",
            "name": "NovaTech",
            "sector": "Technology",
            "base_price": 100.0,
            "volatility": 0.02,
            "is_active": True,
        }
    ]
    mock_db.simulation_runs.find_one.return_value = None

    # Call with config=None
    output = run_and_save_simulation(
        db=mock_db,
        simulation_id="sim_auto_seed_001",
        config=None,
    )

    # 1. Effective seed must be an integer > 0
    effective_seed = output.config.random_seed
    assert isinstance(effective_seed, int)
    assert effective_seed > 0

    # 2. Seed stored in simulation_runs must match
    initial_insert = mock_db.simulation_runs.insert_one.call_args[0][0]
    assert initial_insert["random_seed"] == effective_seed

    # 3. Verify reproducibility: re-simulating with that exact seed reproduces identical prices
    comp = CompanyProfile("TECH1", "NovaTech", "Technology", 100.0, 0.02)
    reproduced_engine = MarketSimulationEngine(
        companies=[comp],
        config=MarketConfig(trading_days=output.trading_days, random_seed=effective_seed),
    )
    reproduced_output = reproduced_engine.simulate()

    np.testing.assert_array_equal(
        output.get_price_matrix(), reproduced_output.get_price_matrix()
    )
    np.testing.assert_array_equal(
        output.get_return_matrix(), reproduced_output.get_return_matrix()
    )


def test_resolve_simulation_config_preserves_explicit_seed() -> None:
    """Requirement 3: Explicit seed is preserved unchanged."""
    cfg = MarketConfig(trading_days=10, random_seed=999)
    resolved = resolve_simulation_config(cfg)
    assert resolved.random_seed == 999
    assert resolved.trading_days == 10
