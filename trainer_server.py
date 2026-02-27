#!/usr/bin/env python
import argparse
import sys

from flask import Flask, request, jsonify

from shinrai.config import (
    Config,
    DataConfig,
    ModelConfig,
    TrainConfig,
    CheckpointConfig,
    GenerateConfig,
)
from shinrai.logging import log
from shinrai.trainer import run_training


def dict_to_config(d: dict) -> Config:
    """Reconstruct a Config from a dict representation.

    This assumes the payload structure mirrors ``asdict(Config)`` and that
    every sub-dictionary has keys matching the dataclass fields.
    """
    return Config(
        data=DataConfig(**d.get("data", {})),
        model=ModelConfig(**d.get("model", {})),
        train=TrainConfig(**d.get("train", {})),
        checkpoint=CheckpointConfig(**d.get("checkpoint", {})),
        generate=GenerateConfig(**d.get("generate", {})),
    )


app = Flask(__name__)


@app.route("/train", methods=["POST"])
def train_endpoint():
    payload = request.get_json(force=True)
    try:
        cfg = dict_to_config(payload)
    except Exception as exc:
        log.error(f"Invalid configuration: {exc}")
        return jsonify({"error": str(exc)}), 400

    try:
        model, vocab = run_training(cfg)
    except Exception as exc:
        log.error(f"Training failed: {exc}")
        return jsonify({"error": str(exc)}), 500

    return jsonify(
        {
            "status": "ok",
            "vocab_size": len(vocab),
            "params": model.num_parameters,
        }
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="trainer_server",
        description="Run remote shinrai training service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="Host/IP to bind the server")
    p.add_argument("--port", type=int, default=8000,
                   help="Port to listen on")
    return p


def main() -> None:
    args = build_parser().parse_args()
    log.banner("shinrai — trainer server")
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
