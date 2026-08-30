from pathlib import Path

import pytest

from bankmarketing.data.ingest import (
    EXPECTED_FILES,
    expected_files_exist,
    validate_download,
)


def create_expected_files(
    directory: Path,
) -> None:
    """Create fake non-empty Kaggle files."""
    for filename in EXPECTED_FILES:
        path = directory / filename
        path.write_text(
            "fake-data",
            encoding="utf-8",
        )


def test_expected_files_exist(
    tmp_path: Path,
) -> None:
    create_expected_files(tmp_path)

    assert expected_files_exist(tmp_path)


def test_expected_files_exist_returns_false_when_missing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "train.csv"

    path.write_text(
        "fake-data",
        encoding="utf-8",
    )

    assert not expected_files_exist(tmp_path)


def test_validate_download_success(
    tmp_path: Path,
) -> None:
    create_expected_files(tmp_path)

    validate_download(tmp_path)


def test_validate_download_fails_when_file_missing(
    tmp_path: Path,
) -> None:
    for filename in EXPECTED_FILES:
        if filename != "test.csv":
            path = tmp_path / filename

            path.write_text(
                "fake-data",
                encoding="utf-8",
            )

    with pytest.raises(
        RuntimeError,
        match="Missing expected files",
    ):
        validate_download(tmp_path)


def test_validate_download_fails_when_file_empty(
    tmp_path: Path,
) -> None:
    create_expected_files(tmp_path)

    empty_file = tmp_path / "test.csv"

    empty_file.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="Empty downloaded files",
    ):
        validate_download(tmp_path)