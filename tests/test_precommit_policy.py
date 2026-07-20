from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_dflash_gate_is_manual_only() -> None:
    hook = (REPO_ROOT / ".githooks" / "pre-commit").read_text()

    assert "coherence-gate-dflash.sh" not in hook
    assert "HIPFIRE_FORCE_SPEC_GATE" not in hook
    assert (REPO_ROOT / "tests" / "coherence-gate-dflash.sh").is_file()
