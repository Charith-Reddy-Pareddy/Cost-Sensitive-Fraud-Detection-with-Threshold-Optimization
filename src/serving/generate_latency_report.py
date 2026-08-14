"""Hits a running instance of the inference service with a real replay batch over HTTP, then
pulls the recorded per-prediction latencies and saves a histogram for the README.

Usage: start the service (`uvicorn src.serving.app:app`), then run this against it:
    python -m src.serving.generate_latency_report --url http://127.0.0.1:8000 --n 2000
"""

import argparse
from pathlib import Path

import httpx
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--delay-ms", type=float, default=1.0)
    args = parser.parse_args()

    with httpx.Client(timeout=60.0) as client:
        client.post(f"{args.url}/replay", params={"n": args.n, "delay_ms": args.delay_ms})
        stats = client.get(f"{args.url}/latency").json()
        samples = client.get(f"{args.url}/latency/raw").json()["samples_ms"]

    print(f"count={stats['count']} p50_ms={stats['p50_ms']:.3f} p95_ms={stats['p95_ms']:.3f}")

    # a cold-start first request or GC pause can be an order of magnitude slower than steady
    # state and otherwise squashes the histogram's x-axis into illegibility; clip the display
    # range to the 99.5th percentile (the tail is still fully reflected in p50/p95 above)
    display_cap = np.percentile(samples, 99.5)

    plt.figure(figsize=(7, 4))
    plt.hist(samples, bins=50, range=(0, display_cap))
    plt.axvline(stats["p50_ms"], color="green", linestyle="--", label=f"p50 = {stats['p50_ms']:.2f}ms")
    plt.axvline(stats["p95_ms"], color="red", linestyle="--", label=f"p95 = {stats['p95_ms']:.2f}ms")
    plt.xlabel("prediction latency (ms)")
    plt.ylabel("count")
    plt.title(f"Inference latency distribution (n={stats['count']}, clipped at p99.5)")
    plt.legend()
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "latency_histogram.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
