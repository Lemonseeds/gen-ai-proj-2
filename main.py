"""
Main CLI Entrypoint for Foundations of Generative AI (ICT4403) — Assignment 3:
Neural Machine Translation: Comparing Encoder-Decoder (Basic Transformer, mT5)
and Decoder-Only (LLaMA, Gemma) Architectures on English -> French OPUS Books.

Usage:
    python main.py
"""
import argparse
import os
from src.train_and_eval import run_complete_experiment


def main():
    parser = argparse.ArgumentParser(
        description="ICT4403 Assignment 3: Neural Machine Translation Benchmark"
    )
    parser.add_argument("--basic-epochs", type=int, default=12)
    parser.add_argument("--mt5-epochs", type=int, default=3)
    parser.add_argument("--llama-epochs", type=int, default=3)
    parser.add_argument("--gemma-epochs", type=int, default=2)
    parser.add_argument("--eval-size", type=int, default=50)
    parser.add_argument("--temp-size", type=int, default=20)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    run_complete_experiment(
        basic_epochs=args.basic_epochs,
        mt5_epochs=args.mt5_epochs,
        llama_epochs=args.llama_epochs,
        gemma_epochs=args.gemma_epochs,
        eval_test_size=args.eval_size,
        temp_eval_size=args.temp_size,
        threads=args.threads,
    )
    print(f"\nAll models, plots, and CSV/MD artifacts saved to: {os.path.abspath('results')}")


if __name__ == "__main__":
    main()

