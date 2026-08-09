from __future__ import annotations

from src.policy.risk import RiskLevel
from src.risk import ArchitectureImpactDetector, ArchitectureThresholds, RiskFactorCode


def test_central_abstraction_change_is_architectural() -> None:
    detector = ArchitectureImpactDetector.default()
    factor = detector.detect(("src/routing/capabilities.py",), 10, ())
    assert factor is not None
    assert factor.code == RiskFactorCode.ARCHITECTURAL_CHANGE
    assert factor.risk_level == RiskLevel.R3_HIGH


def test_public_interface_change_is_architectural() -> None:
    detector = ArchitectureImpactDetector.default()
    factor = detector.detect(("src/providers/adapter.py",), 10, ())
    assert factor is not None
    assert factor.code == RiskFactorCode.ARCHITECTURAL_CHANGE
    assert factor.risk_level == RiskLevel.R3_HIGH


def test_persistence_schema_change_is_architectural() -> None:
    detector = ArchitectureImpactDetector.default()
    factor = detector.detect(("src/persistence/configuration.py",), 10, ())
    assert factor is not None
    assert factor.code == RiskFactorCode.ARCHITECTURAL_CHANGE
    assert factor.risk_level == RiskLevel.R3_HIGH


def test_cross_subsystem_change_is_architectural() -> None:
    detector = ArchitectureImpactDetector.default()
    paths = (
        "src/providers/openai/adapter.py",
        "src/routing/capabilities.py",
        "src/context/schema.py",
        "src/recovery/state_machine.py",
    )
    factor = detector.detect(paths, 100, ())
    assert factor is not None
    assert factor.code == RiskFactorCode.ARCHITECTURAL_CHANGE
    assert factor.risk_level == RiskLevel.R3_HIGH


def test_file_count_threshold_deterministic() -> None:
    detector = ArchitectureImpactDetector.default()
    paths = tuple(f"src/module/file{i}.py" for i in range(6))
    factor = detector.detect(paths, 10, ())
    assert factor is not None
    assert "changed 6 non-generated files" in factor.evidence


def test_file_count_below_threshold_no_signal() -> None:
    detector = ArchitectureImpactDetector.default()
    paths = tuple(f"src/module/file{i}.py" for i in range(5))
    factor = detector.detect(paths, 10, ())
    assert factor is None


def test_line_count_threshold_deterministic() -> None:
    detector = ArchitectureImpactDetector.default()
    paths = ("src/module/file.py",)
    factor = detector.detect(paths, 500, ())
    assert factor is not None
    assert "changed approximately 500 lines" in factor.evidence


def test_generated_only_change_avoids_architectural() -> None:
    detector = ArchitectureImpactDetector.default()
    factor = detector.detect(("src/generated/big_lock.py",), 10000, ("src/generated/big_lock.py",))
    assert factor is None


def test_mixed_generated_and_real_still_considers_real() -> None:
    detector = ArchitectureImpactDetector.default()
    paths = ("src/generated/big_lock.py", "src/routing/capabilities.py")
    factor = detector.detect(paths, 10, ("src/generated/big_lock.py",))
    assert factor is not None
    assert factor.code == RiskFactorCode.ARCHITECTURAL_CHANGE


def test_threshold_override() -> None:
    thresholds = ArchitectureThresholds(
        subsystem_risk_floor=2,
        file_count_risk_floor=3,
        line_count_risk_floor=100,
    )
    detector = ArchitectureImpactDetector(thresholds)
    paths = (
        "src/a/x.py",
        "src/b/y.py",
    )
    factor = detector.detect(paths, 50, ())
    assert factor is not None
    assert "cross-package change spans 2 subsystems" in factor.evidence


def test_architecture_level_override() -> None:
    thresholds = ArchitectureThresholds(
        central_abstraction_risk_level=RiskLevel.R4_CRITICAL_AUTHORITY,
    )
    detector = ArchitectureImpactDetector(thresholds)
    factor = detector.detect(("src/routing/capabilities.py",), 10, ())
    assert factor is not None
    assert factor.risk_level == RiskLevel.R4_CRITICAL_AUTHORITY


def test_small_central_change_is_architectural_not_size_based() -> None:
    detector = ArchitectureImpactDetector.default()
    factor = detector.detect(("src/routing/capabilities.py",), 5, ())
    assert factor is not None
    assert "central abstraction changed" in factor.evidence


def test_large_generated_lockfile_is_not_architectural() -> None:
    detector = ArchitectureImpactDetector.default()
    paths = ("uv.lock",)
    factor = detector.detect(paths, 5000, ("uv.lock",))
    assert factor is None
