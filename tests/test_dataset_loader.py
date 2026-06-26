import pytest
from src.data.dataset_loader import DatasetLoader, EvalSample


@pytest.fixture
def loader():
    return DatasetLoader()


def test_mmlu_loads_correct_count(loader):
    samples = loader.load("mmlu", sample_size=10)
    assert len(samples) == 10


def test_mmlu_sample_has_required_fields(loader):
    samples = loader.load("mmlu", sample_size=5)
    for sample in samples:
        assert isinstance(sample, EvalSample)
        assert sample.dataset == "mmlu"
        assert sample.question != ""
        assert sample.expected_answer in ["A", "B", "C", "D"]
        assert sample.choices is not None
        assert len(sample.choices) == 4
        assert sample.subject is not None


def test_truthfulqa_loads_correct_count(loader):
    samples = loader.load("truthfulqa", sample_size=10)
    assert len(samples) == 10


def test_truthfulqa_sample_has_required_fields(loader):
    samples = loader.load("truthfulqa", sample_size=5)
    for sample in samples:
        assert sample.dataset == "truthfulqa"
        assert sample.question != ""
        assert sample.expected_answer != ""


def test_humaneval_loads_correct_count(loader):
    samples = loader.load("humaneval", sample_size=10)
    assert len(samples) == 10


def test_all_samples_have_unique_ids(loader):
    """
    If IDs collide, we can't track individual results in the DB.
    This test catches that bug immediately.
    """
    samples = loader.load("mmlu", sample_size=20)
    ids = [s.id for s in samples]
    assert len(ids) == len(set(ids)), "duplicate IDs found"


def test_unsupported_dataset_raises_error(loader):
    with pytest.raises(ValueError, match="Unknown dataset"):
        loader.load("made_up_dataset", sample_size=10)


def test_sample_questions_are_printable(loader):
    """Smoke test — just make sure we can print and read a sample."""
    samples = loader.load("mmlu", sample_size=3)
    for sample in samples:
        print(f"\n--- {sample.id} ({sample.subject}) ---")
        print(sample.question)
        print(f"Expected: {sample.expected_answer}")