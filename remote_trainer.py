#!/usr/bin/env python
"""remote_trainer.py — submit a training job to a shinrai trainer server.

This script exposes the same command‑line flags as ``train.py`` with the
addition of ``--server``.  All arguments are converted into a
:class:`~shinrai.config.Config` object, serialized to JSON, and sent to the
remote host via HTTP.  The server is expected to speak the lightweight API
implemented by :mod:`trainer_server`.

Usage examples:

    # send a job to a locally running server
    python remote_trainer.py --server http://localhost:8000 \
        --use_seed_articles --epochs 30 --save_checkpoint model.pt

    # you can still specify text/crawl/model options exactly as you would
    # with the normal ``train.py`` frontend.
"""

import argparse
import sys
from dataclasses import asdict
from typing import Any, Dict

import requests

from shinrai.config import (
    add_checkpoint_args,
    add_data_args,
    add_generate_args,
    add_model_args,
    add_train_args,
    config_from_namespace,
)
from shinrai.logging import log


class RemoteTrainerClient:
    """Simple HTTP client for talking to a remote trainer server."""

    def __init__(self, server_url: str) -> None:
        self.server_url = server_url.rstrip("/")

    def submit(self, cfg: "shinrai.config.Config") -> Dict[str, Any]:
        """Post the configuration to ``<server>/train`` and return the JSON
        response.  Raises ``requests.HTTPError`` on failure.
        """
        payload = asdict(cfg)
        resp = requests.post(f"{self.server_url}/train", json=payload)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# CLI glue
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="remote_trainer",
        description="Submit shinrai training jobs to a remote server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--server",
        required=True,
        help="Base URL of the trainer server (e.g. http://host:8000)",
    )
    add_data_args(p)
    add_model_args(p)
    add_train_args(p)
    add_checkpoint_args(p)
    add_generate_args(p)
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = config_from_namespace(args)

    log.banner("shinrai — remote trainer client")
    client = RemoteTrainerClient(args.server)
    try:
        result = client.submit(cfg)
        log.success(f"Server response: {result}")
    except requests.RequestException as exc:
        log.error(f"Failed to contact server: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
