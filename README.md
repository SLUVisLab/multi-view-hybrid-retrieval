# Multi-view hybrid hotel retrieval

Train and evaluate multi-view hotel recognition on
[OpenHotels-Updated](https://huggingface.co/datasets/imagingforgood/OpenHotels-Updated).
This repository extends MV-HFMD, the method introduced in the WACV 2024 paper
[Multi-View Classification Using Hybrid Fusion and Mutual Distillation](https://openaccess.thecvf.com/content/WACV2024/papers/Black_Multi-View_Classification_Using_Hybrid_Fusion_and_Mutual_Distillation_WACV_2024_paper.pdf),
with OpenHotels data loaders and a retrieval fine-tuning pipeline.

The pipeline has two phases:

| Phase | Entry point | Training objective | Evaluation |
| --- | --- | --- | --- |
| 1. Classification pretraining | `main.py` | Hotel classification for individual images and fused collections, with mutual distillation | Closed-set single-image and collection classification accuracy |
| 2. Retrieval fine-tuning | `retrieval_main.py` | Easy Positive Semi-Hard Negative (EPSHN) mining between multi-view queries and individual gallery images | Grouped multi-view hotel retrieval and the official single-image retrieval protocol |

Only source code, configuration, tests, and documentation belong in this
repository. Dataset files, checkpoints, embedding caches, logs, and evaluation
outputs stay local and are excluded by `.gitignore`.

## Installation

```bash
git clone https://github.com/SLUVisLab/multi-view-hybrid-retrieval.git
cd multi-view-hybrid-retrieval
conda env create -f environment.yml
conda activate multi-view-hybrid
```

`environment.yml` specifies Python 3.11, PyTorch 2.5.1, torchvision 0.20.1,
CUDA 12.4, and the remaining dependencies. Alternatively, install into an
existing Python environment with `python -m pip install -r requirements.txt`;
use a PyTorch build compatible with your hardware. GPU training uses BF16
autocast. CPU smoke tests and unit tests do not require CUDA.

## Obtain OpenHotels-Updated

Open the [Hugging Face dataset page](https://huggingface.co/datasets/imagingforgood/OpenHotels-Updated),
review and accept its access conditions, and sign in with the same account when
downloading. The dataset is gated, so an unauthenticated download will fail.

```bash
python -c "from huggingface_hub import login; login()"
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="imagingforgood/OpenHotels-Updated",
    repo_type="dataset",
    local_dir="data/OpenHotels-Updated",
    allow_patterns=[
        "metadata_gallery.json",
        "metadata_test_room.json",
        "metadata_test_object.json",
        "shards/*.tar",
    ],
)
PY
export DATA_DIR="$PWD/data/OpenHotels-Updated"
```

Both phases expect this layout:

```text
data/OpenHotels-Updated/
├── metadata_gallery.json
├── metadata_test_room.json
├── metadata_test_object.json
└── shards/
    ├── gallery-*.tar
    ├── test_room-*.tar
    └── test_object-*.tar
```

`metadata_hotels.json` is optional and is not used by these loaders. Images can
remain in tar shards: no extraction is required. If you extract a shard, put its
members under `images/<shard-stem>/` inside the dataset directory, preserving
each member's path. For example, a row from `shards/gallery-00000.tar` with
`path="images/gallery/example.jpg"` resolves to
`$DATA_DIR/images/gallery-00000/images/gallery/example.jpg` after extraction.

Use the complete gallery metadata for a consistent hotel-to-class mapping.
Retrieval defaults to failing on unavailable shards. Classification commands
below also enforce this with `--skip_missing false`. Partial-data experiments
can use classification `--skip_missing true` or retrieval `--skip_missing`, but
their evaluation does not cover the full benchmark.

## Download the retrieval checkpoint

Download the [public retrieval checkpoint from Google Drive](https://drive.google.com/file/d/1z46sxAIqsLmo6HMoS07f67SE0Cjg37Yu/view?usp=sharing)
and save it as `checkpoints/openhotels-retrieval-best.pth`:

```bash
mkdir -p checkpoints
# Save the file from the link above as:
# checkpoints/openhotels-retrieval-best.pth
```

This checkpoint can be used directly with the retrieval evaluation command
below. It uses `vit_small_r26_s32_224`, 224-pixel inputs, and four learned view
slots. Keep those settings and the full OpenHotels gallery metadata when loading
it. Checkpoint loading is strict; changing resolution or the number of model
slots does not automatically interpolate positional embeddings. Newly produced
checkpoints record their class ordering and model configuration and validate
these on load; legacy checkpoints require the matching metadata and settings.

## Phase 1: classification pretraining and evaluation

Train a four-view MV-HFMD classifier with cross-entropy on both prediction
branches and mutual distillation:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset openhotels \
  --data_dir "$DATA_DIR" \
  --save_dir output/openhotels-classification \
  --architecture vit_small_r26_s32_224 \
  --num_images 4 --image_size 224 \
  --batch_size 16 --grad_accum_steps 4 \
  --eval_batch_size 8 --num_epochs 50 --eval_every 5 \
  --use_mutual_distillation_loss true \
  --skip_missing false
```

The batch size counts collections. Sixteen four-view collections contain 64
source images; accumulating four batches gives 64 collections per optimizer
step. Images use random training crops and deterministic evaluation crops.
The initial backbone weights are downloaded by timm; pass
`--pretrained_weights false` to train without them.

Classification validation deterministically holds out gallery images within
each hotel. The default validation cap is one collection per hotel; use
`--max_val_collections_per_class -1` for all validation collections.
`best.pth` is selected by collection top-1 accuracy, and `last.pth` records the
latest training state. Training then evaluates the best model on `test_room`
and `test_object` separately.

To evaluate a classification checkpoint independently:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --dataset openhotels --data_dir "$DATA_DIR" \
  --eval_only \
  --checkpoint output/openhotels-classification/best.pth \
  --save_dir output/openhotels-classification-eval \
  --num_images 4 --image_size 224 \
  --eval_batch_size 8 --skip_missing false
```

This reports single-image and fused-collection top-1/2/5/10/100 classification
accuracy and writes `classification_test_metrics.json`. Only query hotels
represented by the classifier's gallery classes are included. Classification
collections use a fixed view count and may repeat images when a hotel has fewer
images; these metrics are distinct from the retrieval protocols below.

The equivalent training launcher is
`GPU_ID=0 bash scripts/run_openhotels_gpu1.sh --skip_missing false`. The historical filename is
retained; its default GPU is 1, and `GPU_ID` overrides it.

## Phase 2: retrieval fine-tuning and evaluation

Retrieval keeps the hybrid-fusion transformer and produces normalized
collection and individual-view embeddings. Every gallery image has its own
descriptor, allowing a hotel's different visual appearances to remain separate.

Each training batch samples distinct hotel identities. For each hotel, it
samples a coherent query of 2–4 images from one upload, identified by
`(hotel_id, room)`, plus up to four positive gallery images. Query and positive
gallery paths never overlap, and another upload is preferred for positives.
Training applies asymmetric EPSHN to the fused query and an auxiliary EPSHN
loss to its individual views. The small classification loss decays to zero over
the first five epochs; mutual distillation is not used in retrieval fine-tuning.
The mining method follows
[Improved Embeddings with Easy Positive Triplet Mining](https://arxiv.org/abs/1904.04370).

Start from the classification checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 python retrieval_main.py \
  --data_dir "$DATA_DIR" \
  --checkpoint output/openhotels-classification/best.pth \
  --save_dir output/openhotels-retrieval-epshn \
  --device cuda:0 \
  --model_max_views 4 --min_query_views 2 --max_query_views 4 \
  --positive_gallery_size 4 --hotels_per_batch 24 \
  --loss epshn --temperature 0.1 --view_loss_weight 0.25 \
  --num_epochs 20 --eval_every 5 \
  --embedding_cache_dir cache/openhotels-retrieval
```

The corresponding launcher is `bash scripts/run_openhotels_retrieval.sh`.
It defaults to physical GPU 0 and 24 hotels per batch. Set
`HOTELS_PER_BATCH=48` for a larger mining batch if your GPU has room, or reduce
it if needed. Gradient accumulation does not add negatives across batches:
`--hotels_per_batch` controls how many hotel identities participate in each
mining operation.

Retrieval validation holds out complete uploads from the training gallery.
No image from a validation query's upload is present in its reference gallery.
Model selection uses multi-view **hotel Recall@1 with max aggregation and
hybrid scoring**. Tune using this validation set; reserve the official test
sets for the final evaluation.

Training saves `best.pth`, `last.pth`, `retrieval.log`, and
`validation-epoch-*.json` under `--save_dir`. By default it then evaluates the
best checkpoint on both official test subsets. Use `--skip_test` to defer that
evaluation during training.

### Evaluate a retrieval checkpoint independently

```bash
CUDA_VISIBLE_DEVICES=0 python retrieval_main.py \
  --data_dir "$DATA_DIR" \
  --eval_only \
  --checkpoint checkpoints/openhotels-retrieval-best.pth \
  --save_dir output/openhotels-retrieval-eval \
  --device cuda:0 \
  --model_max_views 4 --max_query_views 4 --test_min_views 2 \
  --gallery_batch_size 128 --query_batch_size 8 \
  --embedding_cache_dir cache/openhotels-retrieval-eval \
  --reuse_embedding_cache
```

The gallery contains all official gallery images, including distractor hotels.
`test_room` and `test_object` are evaluated separately, with two protocols saved
in `retrieval_test_metrics.json`:

- **`multi_view`** groups queries by hotel and room/upload. Groups with two or
  three images use those images; groups with at least four use a deterministic,
  category-diverse selection of four. Singleton uploads are excluded from this
  protocol. Each group contributes one collection by default. Hotel Recall@1/5/10/100
  is reported for the learned joint descriptor, late-interaction view scoring,
  and their hybrid, using max and top-3-mean hotel aggregation.
- **`official_single_image`** evaluates every official test image independently,
  including singleton uploads and images not selected for grouped queries.
  It reports image-gallery Recall@1/5/10/100: success means a correctly labelled
  gallery image appears among the top K retrieved images. This preserves the
  original single-image benchmark protocol.

Grouped evaluation uses a fixed, reproducible selection, while training randomly
samples the number of query views. Padded positions are masked rather than
treated as additional images. Hybrid scoring uses `--hybrid_alpha` (default
0.5); `--hotel_top_m 1 3` requests max and the mean of up to three best images.

Full-gallery scoring is exact and can take time. Reusable embedding caches are
keyed by checkpoint contents and evaluation settings. Reduce
`--retrieval_query_chunk_size` or `--retrieval_gallery_chunk_size` if scoring
memory is limited; reduce extraction batch sizes if embedding extraction runs
out of memory. `--max_test_queries` and `--max_test_image_queries` can cap
debugging runs; leave both at their default 0 for full evaluation.

### Resume training or add a larger-batch phase

`--checkpoint` initializes model weights for a new training run. `--resume`
restores a retrieval model, optimizer momentum, completed epochs, and best
validation score, continuing the learning-rate schedule. For example, resume
an interrupted 20-epoch run:

```bash
CUDA_VISIBLE_DEVICES=0 python retrieval_main.py \
  --data_dir "$DATA_DIR" \
  --resume output/openhotels-retrieval-epshn/last.pth \
  --save_dir output/openhotels-retrieval-epshn \
  --hotels_per_batch 24 --num_epochs 20
```

To start a fresh learning-rate cycle for 20 additional epochs with 48 hotels per
batch, retain the previous run and launch:

```bash
GPU_ID=0 HOTELS_PER_BATCH=48 ADDITIONAL_EPOCHS=20 \
  bash scripts/run_openhotels_retrieval_continuation.sh
```

This reads `output/openhotels-retrieval-epshn/last.pth`, restores the model and
momentum, and writes to `output/openhotels-retrieval-epshn-batch48-continuation`.
It preserves the source run's validated best checkpoint as the comparison
baseline. If the source completed epoch 20, the new phase runs epochs 21–40;
classification-loss decay still uses global epoch numbers. Override
`SOURCE_RUN_DIR`, `RESUME_CHECKPOINT`, or `RUN_DIR` for other runs.

The equivalent CLI option is `--resume PATH --additional_epochs 20`. To resume
an interrupted additional phase, omit `--additional_epochs` and set
`--num_epochs` to the phase's final global epoch (40 in this example), keeping
its output directory. Otherwise you would start another fresh cycle.

## Launchers and background monitoring

Scripts locate the checkout themselves and use `python` from the active
environment. Common environment overrides include `PYTHON_BIN`, `DATA_DIR`,
`GPU_ID`, `RUN_DIR`, `NUM_EPOCHS`, and `NUM_WORKERS`. Retrieval also accepts
`CLASSIFICATION_CHECKPOINT`, `HOTELS_PER_BATCH`, and `EMBEDDING_CACHE_DIR`.
Additional CLI arguments are forwarded to the corresponding Python entry point.
`GPU_ID` selects the physical GPU through `CUDA_VISIBLE_DEVICES`; within that
process it is addressed as `cuda:0`.

With GNU Screen installed, launch a monitored retrieval run:

```bash
screen -dmS openhotels-retrieval \
  bash scripts/run_openhotels_retrieval_monitored.sh
screen -r openhotels-retrieval
```

Detach with `Ctrl-A`, then `D`. The wrapper captures console output and samples
the training PID, recent progress, and GPU memory/utilization every 30 seconds.
It records the final exit code, with 0 indicating success. Inspect the latest
run without attaching:

```bash
cat output/openhotels-retrieval-epshn/monitor.status
tail -f output/openhotels-retrieval-epshn/monitor.console.log
```

`monitor.log` keeps the snapshots, and timestamped run files retain earlier
invocations. The continuation launcher already uses this monitor; it can also
be launched inside Screen.

## Smoke tests and unit tests

Unit tests build tiny models without downloading pretrained weights and use
temporary synthetic data. Run them on CPU:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -m unittest discover -s tests -v
```

Tests cover upload-disjoint sampling, view masking, exact chunked retrieval,
checkpoint compatibility, resume schedules, and finite EPSHN forward/backward
under BF16 autocast. The CUDA-specific test is skipped when CUDA is unavailable;
EPSHN similarities and mining are computed in FP32 under mixed precision.

After downloading the dataset, run real-image CPU smoke tests without a
checkpoint or pretrained-weight download:

```bash
python scripts/smoke_test.py \
  --data_dir "$DATA_DIR" --device cpu \
  --architecture vit_tiny_patch16_224 --image_size 32
python scripts/retrieval_smoke_test.py \
  --data_dir "$DATA_DIR" --device cpu --no-check_checkpoint
```

To also check strict loading and a production 224-pixel embedding, replace
`--no-check_checkpoint` with
`--checkpoint checkpoints/openhotels-retrieval-best.pth`.

## Original MV-HFMD and citation

The original Hotels-8k classifier remains available with
`python main.py --dataset hotels8k --data_dir PATH --num_images 4` using the
[original dataset release](https://files.vidarlab.net/s/EeHzz9MqggSTZMc).
Model output branches are `single` and `mv_collection`.

Please cite the original MV-HFMD paper when using this method:

```bibtex
@inproceedings{black2024multi,
  title={Multi-View Classification Using Hybrid Fusion and Mutual Distillation},
  author={Black, Samuel and Souvenir, Richard},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={270--280},
  year={2024}
}
```
