"""
RAG Evaluation Experiment Runner for LangSmith

Executes evaluation across the curated benchmark dataset in LangSmith:
1. Ensures benchmark dataset exists and is populated in LangSmith.
2. Evaluates retrieval quality and generation quality independently:
   - Retrieval Relevance
   - Context Precision
   - Context Recall
   - Faithfulness
   - Answer Correctness
   - Operational Health & Latency
3. Generates experiment report with links to LangSmith evaluation dashboard.
"""

import json
import os
import sys
from typing import Dict, List
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

from langsmith import Client
from langsmith.evaluation import evaluate
from app.services.rag.matching_service import match_resume_to_job
from evaluation.evaluators import (
    eval_answer_correctness,
    eval_context_precision,
    eval_context_recall,
    eval_faithfulness,
    eval_operational_metrics,
    eval_retrieval_relevance,
)

DATASET_NAME = "resume-rag-eval-benchmark"
DATASET_DESC = "Benchmark dataset for evaluating RAG retrieval and generation in resume-job matching."


def load_local_benchmark_dataset() -> List[Dict]:
    """Load benchmark dataset from local JSON file."""
    dataset_path = os.path.join(os.path.dirname(__file__), "benchmark_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def setup_langsmith_dataset(client: Client, examples: List[Dict]) -> str:
    """Ensure benchmark dataset exists and contains expected examples in LangSmith."""
    if not client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Creating LangSmith dataset: '{DATASET_NAME}'...")
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=DATASET_DESC,
        )
        inputs = [ex["inputs"] for ex in examples]
        outputs = [ex["outputs"] for ex in examples]
        client.create_examples(
            inputs=inputs,
            outputs=outputs,
            dataset_id=dataset.id,
        )
        print(f"Successfully uploaded {len(examples)} examples to dataset '{DATASET_NAME}'.")
    else:
        print(f"LangSmith dataset '{DATASET_NAME}' already exists.")

    return DATASET_NAME


def target_system(inputs: Dict) -> Dict:
    """Target function wrapped for LangSmith evaluation runner."""
    resume_text = inputs.get("resume_text", "")
    job_description = inputs.get("job_description", "")
    response = match_resume_to_job(
        resume_text=resume_text,
        job_description=job_description,
    )
    return response.model_dump()


def run_rag_evaluation():
    print("=" * 75)
    print("RAG EVALUATION WORKFLOW: LANGSMITH EXPERIMENT RUNNER")
    print("=" * 75)

    client = Client()
    benchmark_examples = load_local_benchmark_dataset()
    print(f"Loaded {len(benchmark_examples)} local benchmark scenarios.")

    dataset_name = setup_langsmith_dataset(client, benchmark_examples)

    evaluators = [
        eval_retrieval_relevance,
        eval_context_precision,
        eval_context_recall,
        eval_faithfulness,
        eval_answer_correctness,
        eval_operational_metrics,
    ]

    print(f"\nLaunching LangSmith evaluation experiment on dataset '{dataset_name}'...")
    print("Active Evaluators:")
    for ev in evaluators:
        print(f"  • {ev.__name__}")

    results = evaluate(
        target_system,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="rag-eval-suite",
        description="Comprehensive evaluation of FAISS retrieval and Groq LLM matching insights.",
        max_concurrency=1,
    )

    print("\n" + "=" * 75)
    print("EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 75)

    # Collect metric statistics across examples
    metric_scores: Dict[str, List[float]] = {}
    for result_item in results:
        eval_results = result_item.get("evaluation_results", {})
        for res in eval_results.get("results", []):
            if res.score is not None:
                metric_scores.setdefault(res.key, []).append(float(res.score))

    print("\n### RAG Performance Summary\n")
    print(f"| {'Metric':<25} | {'Average Score':<15} | {'Interpretation':<35} |")
    print(f"|{'-' * 27}|{'-' * 17}|{'-' * 37}|")

    interpretations = {
        "retrieval_relevance": "Passage relevance & absence of unevidenced drift",
        "context_precision": "Proportion of high-signal supporting passages",
        "context_recall": "Coverage of expected reference requirements",
        "faithfulness": "Freedom from hallucinations or false skill claims",
        "answer_correctness": "Agreement with reference scores and skills",
        "operational_health": "Error-free pipeline execution rate",
    }

    for metric_name, scores in sorted(metric_scores.items()):
        avg = round(sum(scores) / len(scores), 3) if scores else 0.0
        interp = interpretations.get(metric_name, "Evaluated quality aspect")
        score_display = f"{avg * 100:.1f}%" if avg <= 1.0 else f"{avg}"
        print(f"| {metric_name:<25} | {score_display:<15} | {interp:<35} |")

    print("\n" + "=" * 75)
    print("View detailed comparison, charts, and per-example traces in LangSmith:")
    print("https://smith.langchain.com -> Datasets & Testing -> 'resume-rag-eval-benchmark'")
    print("=" * 75)


if __name__ == "__main__":
    run_rag_evaluation()
