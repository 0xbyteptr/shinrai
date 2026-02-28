# shinrai — A vibe coded Character-level RNN trainer

A modular, production-quality character-level language model trainer supporting LSTM and GRU backends.

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```bash
# Train on a Wikipedia article
python train.py --url https://en.wikipedia.org/wiki/Philosophy

# crawl starting page plus 10 levels deep (use multiple workers)
# progress will be shown via a tqdm bar when available
python train.py --crawl 10 --crawl_workers 8 --url https://en.wikipedia.org/wiki/Philosophy

# Resume from saved data (skip acquisition/encoding)
python train.py --continue_from data.pt --epochs 10  # file must contain encoded/chars

# Train on the curated seed article set
python train.py --use_seed_articles --epochs 30 --save_checkpoint model.pt

# or crawl with explicit depth
python train.py --crawl 5 --max_pages 50 --save_checkpoint model.pt

# Resume training
python train.py --use_seed_articles --save_checkpoint model.pt --epochs 50

# Train on a local file
python train.py --text_file my_corpus.txt --hidden_size 512 --num_layers 3
# For better GPU utilization you can compile the model and parallelize
# (requires Triton; install with `pip install triton` or omit --use_compile
# to fall back automatically)
python train.py --text_file my_corpus.txt --use_compile --data_parallel \
    --accumulate_steps 4 --use_amp
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
python trainer_server.py --host 0.0.0.0 --port 8000 --save_checkpoint /path/to/remote.pt

# from your laptop submit a job using identical arguments to `train.py`
python remote_trainer.py --server http://server:8000 \
    --use_seed_articles --epochs 30 --save_checkpoint model.pt

# or continue from preprocessed data
python remote_trainer.py --server http://server:8000 \
    --continue_from data.pt --epochs 20  # use checkpoint with --load_checkpoint instead
```

Both scripts are lightweight wrappers around the core :mod:`shinrai` APIs,
so you can easily integrate them into your own orchestration system.

The server accepts an optional ``--save_checkpoint`` argument; any job
payloads received will have their ``checkpoint.save_checkpoint`` field
overwritten with this value, ensuring all runs write to a consistent
location regardless of what the client submits.

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
| `--crawl` | `0` | Crawl depth (0 = disabled; `--crawl N` sets depth) |
| `--fetch_workers` | `4` | Parallel downloads when fetching seed articles |
| `--crawl_workers` | `4` | Concurrent page fetches during crawling |
| `--top_p` | `0.9` | Nucleus sampling p (1.0 = off) |
| `--save_every` | `1` | Periodically write checkpoint every N epochs |
| `--num_workers` | `0` | DataLoader worker processes (set >0 for faster loading) |
| `--use_amp` | (off) | Enable mixed precision training on CUDA GPUs |
| `--use_compile` | (off) | Compile model with torch.compile() for potential speedup |
| `--accumulate_steps` | `1` | Accumulate gradients to simulate larger batch size |
| `--data_parallel` | (off) | Wrap model in DataParallel when multiple GPUs available |

> **Note:** Generated text is now post-processed to strip HTML tags,
> collapse excessive whitespace, remove non-printable characters, and purge
> common web/CSS fragments such as ``foo:bar;`` declarations and anything
> inside ``{}``.  This makes outputs far less likely to contain junk
> extracted from noisy HTML/CSS training data.  
