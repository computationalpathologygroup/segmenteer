"""Public evaluator component entry point."""

from segmenteer.benchmark.evaluation import EvaluationSummary, evaluate_output_directory, main

__all__ = ["EvaluationSummary", "evaluate_output_directory", "main"]

if __name__ == "__main__":
    main()
