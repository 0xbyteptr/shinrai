# shinrai — Character-level RNN trainer

A modular, production-quality character-level language model trainer supporting LSTM and GRU backends.

## Project layout

```
shinrai/
├── shinrai/
│   ├── __init__.py
│   ├── config.py        # Dataclass-based configuration
│   ├── data.py          # Text fetching, crawling, dataset
│   ├── model.py         # CharRNN (LSTM / GRU), weight tying
│   ├── trainer.py       # Training loop, checkpointing, early stopping
│   ├── generate.py      # Nucleus + temperature sampling
│   └── logging.py       # Rich/plain console helpers
├── train.py             # Entry point: train a model
├── generate.py          # Entry point: sample from a saved checkpoint
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# Train on a Wikipedia article
python train.py --url https://en.wikipedia.org/wiki/Philosophy

# Train on the curated seed article set
python train.py --use_seed_articles --epochs 30 --save_checkpoint model.pt

# Resume training
python train.py --use_seed_articles --save_checkpoint model.pt --epochs 50

# Train on a local file
python train.py --text_file my_corpus.txt --hidden_size 512 --num_layers 3

# Generate text from a saved checkpoint
python generate.py --checkpoint model.pt --seed "the meaning of" --length 500
```

## Remote training

You can run model training on a dedicated host by pairing the
``trainer_server.py`` helper with the ``remote_trainer.py`` client script.
The remote server exposes a simple HTTP `/train` endpoint that accepts the
same command‑line flags as ``train.py`` (encoded as JSON) and returns a
status object when the run finishes.  Example:

```bash
# start the server on the remote machine
python trainer_server.py --host 0.0.0.0 --port 8000

# from your laptop submit a job using identical arguments to `train.py`
python remote_trainer.py --server http://server:8000 \
    --use_seed_articles --epochs 30 --save_checkpoint model.pt
```

Both scripts are lightweight wrappers around the core :mod:`shinrai` APIs,
so you can easily integrate them into your own orchestration system.

## Key options

| Flag | Default | Description |
|---|---|---|
| `--cell_type` | `lstm` | `lstm` or `gru` |
| `--embed_size` | `128` | Embedding dimensions |
| `--hidden_size` | `512` | RNN hidden size |
| `--num_layers` | `2` | Number of stacked RNN layers |
| `--dropout` | `0.3` | Dropout probability |
| `--seq_length` | `120` | Context window length |
| `--epochs` | `20` | Training epochs |
| `--lr` | `0.002` | Initial learning rate |
| `--grad_clip` | `5.0` | Gradient clipping max norm |
| `--val_split` | `0.05` | Validation fraction |
| `--patience` | `5` | Early stopping patience (0 = off) |
| `--temperature` | `0.8` | Sampling temperature |
| `--top_p` | `0.9` | Nucleus sampling p (1.0 = off) |
| `--save_every` | `1` | Periodically write checkpoint every N epochs |
