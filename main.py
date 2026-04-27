import argparse
import platform
import sys
from pathlib import Path

import torch

from benchmark import Benchmark
from core_utils import get_available_models
from logger import logger


def log_environment_info(args: argparse.Namespace, device: str) -> None:
    logger.debug("=" * 50)
    logger.debug("SRBenchmark Launch Info")
    logger.debug("=" * 50)
    logger.debug(f"OS: {platform.system()} {platform.release()}")
    logger.debug(f"Python: {platform.python_version()}")
    logger.debug(f"PyTorch: {torch.__version__}")
    logger.debug(f"Device: {device}")

    if device == "cuda" and torch.cuda.is_available():
        logger.debug(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.debug("Arguments:")
    for key, value in vars(args).items():
        if value is not None:
            logger.debug(f"   {key}: {value}")

    logger.debug("=" * 50)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models_dir = Path("models")
    models = get_available_models(models_dir)

    parser = argparse.ArgumentParser(
        description="SRBenchmark: A utility for testing and comparing Super-Resolution models.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "task",
        nargs="+",
        choices=["benchmark", "upscale", "compare"],
        help="Tasks to perform. Multiple tasks can be specified separated by spaces.",
    )

    parser.add_argument(
        "-m",
        "--models",
        nargs="+",
        choices=models + ["all"],
        default=["all"],
        help=f"Models to use. Available: {', '.join(models)}. Default: all",
    )

    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        type=Path,
        help="Paths to dataset folders (required for 'benchmark' task)",
    )

    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        type=Path,
        help="Paths to input (HR) images (required for 'upscale' and 'compare')",
    )

    parser.add_argument(
        "-o",
        "--output",
        nargs="+",
        type=Path,
        help="Paths to save results (images or CSV)",
    )

    parser.add_argument(
        "-lri",
        "--lr-input",
        nargs="+",
        type=lambda p: Path(p) if p.lower() != "none" else None,
        help="Paths to LR images for 'compare'. Use 'none' to generate on the fly.",
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        help="Crop size for creating collages in 'compare' task",
    )

    parser.add_argument(
        "--downscale",
        action="store_true",
        help="Downscale image before upscaling (for 'upscale' task)",
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="",
    )

    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=32,
        help="",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="",
    )

    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=4,
        help="",
    )

    args = parser.parse_args()

    log_environment_info(args, device)

    if not models:
        logger.error("No models with config.yaml found in 'models/' directory.")
        return

    selected_models = models if "all" in args.models else args.models
    config_paths = [models_dir / model / "config.yaml" for model in selected_models]
    logger.info(f"Loaded models: {', '.join(selected_models)}")

    benchmark_app = Benchmark(
        device=device,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
    )

    if "benchmark" in args.task:
        if not args.datasets:
            parser.error("The --datasets argument is required for 'benchmark' task")

        save_csv_path = args.output[0] if args.output else Path("results/benchmark.csv")
        logger.info(f"Starting benchmark on datasets: {[d.name for d in args.datasets]}")
        benchmark_app.evaluate(config_paths, args.datasets, save_csv_path)

    if "upscale" in args.task:
        if not args.input or not args.output:
            parser.error("The --input and --output arguments are required for 'upscale' task")

        logger.info("Starting upscale...")
        benchmark_app.upscale(config_paths, args.input, args.output, args.downscale)

    if "compare" in args.task:
        if not args.input or not args.output:
            parser.error("The --input (HR) and --output arguments are required for 'compare' task")

        logger.info("Starting model comparison...")
        benchmark_app.compare(config_paths, args.input, args.output, args.lr_input, args.crop_size)


if __name__ == "__main__":
    main()
