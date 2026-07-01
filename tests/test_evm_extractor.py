import sys
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("pyevmasm")
pytest.importorskip("sklearn")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "features"))

from evm_extractor import EVMBytecodeFeatureExtractor  # noqa: E402


def test_feature_schema_has_expected_size_and_unique_names():
    extractor = EVMBytecodeFeatureExtractor(n_workers=1)

    names = list(extractor.get_feature_names_out())

    assert len(names) == 67
    assert len(names) == len(set(names))
    assert names[0] == "total_instructions"
    assert names[-1] == "has_dos_vulnerabilities"


def test_empty_and_invalid_bytecode_return_zero_features():
    extractor = EVMBytecodeFeatureExtractor(n_workers=1)
    data = pd.DataFrame({"bytecode": ["", "0x", "not-hex"]})

    features = extractor.transform(data)

    assert features.shape == (3, 67)
    assert (features.sum(axis=1) == 0).all()


def test_simple_bytecode_extracts_basic_instruction_features():
    extractor = EVMBytecodeFeatureExtractor(n_workers=1)
    data = pd.DataFrame({"bytecode": ["0x6001600055"]})  # PUSH1 1; PUSH1 0; SSTORE

    features = extractor.transform(data)

    assert features.shape == (1, 67)
    assert features.loc[0, "total_instructions"] > 0
    assert features.loc[0, "pushes"] > 0
