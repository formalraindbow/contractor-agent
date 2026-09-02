from pathlib import Path

import contractor_agent


def test_package_importable() -> None:
    assert contractor_agent.__version__ == "0.1.0"


def test_snapshot_files_present(data_dir: Path) -> None:
    assert (data_dir / "contractors_audit.snapshot.json").is_file()
    assert (data_dir / "contractors_audit.snapshot_C12613591.csv").is_file()
