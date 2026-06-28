import random
from dataclasses import dataclass, field
from typing import Optional
from loguru import logger
from datasets import load_dataset


@dataclass
class EvalSample:
    """
    Standard format for any evaluation question.
    
    Why normalize into a single format?
    Every dataset has a different schema. If our evaluator knew
    about MMLU's schema specifically, we couldn't add TruthfulQA
    without changing the evaluator. Instead we normalize once
    at load time, and everything downstream is dataset-agnostic.
    
    This is the same decoupling principle as our model clients.
    """
    id: str
    dataset: str
    question: str
    expected_answer: str
    choices: Optional[list[str]] = None
    subject: Optional[str] = None
    metadata: Optional[dict] = field(default_factory=dict)


class DatasetLoader:
    """
    Loads benchmark datasets and normalizes them into EvalSamples.
    
    Each dataset gets its own private _load_X method that handles
    the messy schema differences. The public load() method is the
    single clean interface the rest of the system uses.
    """

    SUPPORTED_DATASETS = ["mmlu", "truthfulqa", "humaneval", "custom"]

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def load(
        self,
        dataset_name: str,
        sample_size: int = 50,
        seed: int = None,
        custom_path: str = None,    # ← new parameter
    ) -> list[EvalSample]:
        """
        custom_path is required when dataset_name == "custom".
        Ignored for all other datasets.
        """
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Supported: {self.SUPPORTED_DATASETS}"
            )

        if dataset_name == "custom":
            if custom_path is None:
                raise ValueError(
                    "custom_path is required when dataset_name='custom'. "
                    "Example: loader.load('custom', custom_path='data/my_eval.csv')"
                )
            return self._load_custom(custom_path, sample_size, seed)

        if seed is not None:
            random.seed(seed)

        logger.info(f"Loading {dataset_name} (sample_size={sample_size})")

        loaders = {
            "mmlu": self._load_mmlu,
            "truthfulqa": self._load_truthfulqa,
            "humaneval": self._load_humaneval,
        }

        samples = loaders[dataset_name](sample_size)
        logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
        return samples

    def _load_mmlu(self, sample_size: int) -> list[EvalSample]:
        """
        MMLU schema:
          question: str
          choices:  list[str]  (always 4 options)
          answer:   int        (index into choices: 0=A, 1=B, 2=C, 3=D)
          subject:  str
        """
        dataset = load_dataset(
            "cais/mmlu",
            "all",
            split="test"
        )

        # shuffle so we get diverse subjects, not just the first subject
        indices = random.sample(range(len(dataset)), min(sample_size, len(dataset)))

        samples = []
        for i, idx in enumerate(indices):
            row = dataset[idx]
            answer_index = row["answer"]          # 0, 1, 2, or 3
            answer_letter = ["A", "B", "C", "D"][answer_index]
            answer_text = row["choices"][answer_index]

            # build the question with choices embedded
            # this is what we'll actually send to the model
            choices_text = "\n".join([
                f"{letter}. {text}"
                for letter, text in zip("ABCD", row["choices"])
            ])
            question = (
                f"{row['question']}\n\n"
                f"{choices_text}\n\n"
                f"Answer with just the letter A, B, C, or D."
            )

            samples.append(EvalSample(
                id=f"mmlu_{idx}",
                dataset="mmlu",
                question=question,
                expected_answer=answer_letter,   # we check against "A", "B", etc
                choices=row["choices"],
                subject=row["subject"],
                metadata={"answer_text": answer_text}
            ))

        return samples

    def _load_truthfulqa(self, sample_size: int) -> list[EvalSample]:
        """
        TruthfulQA schema:
          question:         str
          correct_answers:  list[str]
          incorrect_answers: list[str]
          category:         str
        """
        dataset = load_dataset(
            "truthfulqa/truthful_qa",
            "generation",
            split="validation"
        )

        indices = self.rng.sample(range(len(dataset)), min(sample_size, len(dataset)))

        samples = []
        for idx in indices:
            row = dataset[idx]

            samples.append(EvalSample(
                id=f"truthfulqa_{idx}",
                dataset="truthfulqa",
                question=row["question"],
                # use the first correct answer as the reference
                # the metrics engine will handle fuzzy matching
                expected_answer=row["correct_answers"][0],
                metadata={
                    "all_correct": row["correct_answers"],
                    "incorrect": row["incorrect_answers"],
                    "category": row.get("category", ""),
                }
            ))

        return samples

    def _load_humaneval(self, sample_size: int) -> list[EvalSample]:
        """
        HumanEval schema:
          task_id:          str  (e.g. "HumanEval/0")
          prompt:           str  (function signature + docstring)
          canonical_solution: str
          test:             str  (test cases as runnable code)
          entry_point:      str  (function name)
        """
        dataset = load_dataset(
            "openai-community/openai_humaneval",
            split="test"
        )

        # HumanEval only has 164 problems — take all if sample_size > 164
        indices = self.rng.sample(range(len(dataset)), min(sample_size, len(dataset)))

        samples = []
        for idx in indices:
            row = dataset[idx]

            question = (
                f"Complete the following Python function. "
                f"Return ONLY the function body, no explanation:\n\n"
                f"{row['prompt']}"
            )

            samples.append(EvalSample(
                id=f"humaneval_{row['task_id'].replace('/', '_')}",
                dataset="humaneval",
                question=question,
                expected_answer=row["canonical_solution"],
                metadata={
                    "task_id": row["task_id"],
                    "test_code": row["test"],
                    "entry_point": row["entry_point"],
                }
            ))

        return samples
    
    def _load_custom(
        self,
        path: str,
        sample_size: int,
        seed: int = None,
    ) -> list[EvalSample]:
        """
        Load a user-provided CSV as EvalSamples.

        Required columns:  question, expected_answer
        Optional columns:  id, subject, choices

        Why CSV? It's the lowest friction format — any spreadsheet,
        database export, or data pipeline can produce a CSV.
        We deliberately avoid requiring JSON or YAML — those create
        barriers for non-engineers who want to use the framework.
        """
        import csv
        from pathlib import Path

        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Custom dataset not found: {path}\n"
                f"Current directory: {Path.cwd()}"
            )

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # validate required columns exist
        if not rows:
            raise ValueError(f"CSV is empty: {path}")

        required_columns = {"question", "expected_answer"}
        actual_columns = set(rows[0].keys())
        missing = required_columns - actual_columns

        if missing:
            raise ValueError(
                f"CSV missing required columns: {missing}\n"
                f"Found columns: {actual_columns}\n"
                f"Required: {required_columns}"
            )

        # sample if needed
        if seed is not None:
            random.seed(seed)

        if len(rows) > sample_size:
            rows = random.sample(rows, sample_size)

        samples = []
        for i, row in enumerate(rows):
            # auto-generate id if not provided
            sample_id = row.get("id", "").strip() or f"custom_{i:04d}"

            # handle optional choices column
            choices = None
            if "choices" in row and row["choices"].strip():
                choices = [c.strip() for c in row["choices"].split("|")]

            # build question with choices if MCQ
            question = row["question"].strip()
            if choices:
                choices_text = "\n".join([
                    f"{letter}. {text}"
                    for letter, text in zip("ABCD", choices)
                ])
                question = (
                    f"{question}\n\n{choices_text}\n\n"
                    f"Answer with just the letter A, B, C, or D."
                )

            samples.append(EvalSample(
                id=sample_id,
                dataset="custom",
                question=question,
                expected_answer=row["expected_answer"].strip(),
                choices=choices,
                subject=row.get("subject", "").strip() or None,
                metadata={
                    "source_file": str(csv_path.name),
                    "original_row": i,
                }
            ))

        logger.info(
            f"Loaded {len(samples)} samples from custom dataset: {csv_path.name}"
        )
        return samples