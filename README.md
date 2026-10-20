# ScaleMamba

**Multi-Scale Graph State Space Learning for Traffic Forecasting in Urban IoT Networks**

Jiashan Wan, Minhao Ling, Jingyang Li, Ziheng Jian, and Kai Wang

[Repository](https://github.com/wanjiashan/ScaleMamba) · [Report an issue](https://github.com/wanjiashan/ScaleMamba/issues)

ScaleMamba is a traffic forecasting framework that connects frequency-guided temporal representations, graph-aware state estimation, selective state evolution, and nonlinear cross-scale fusion. It is designed to model temporal variations and dependencies among traffic sensors while avoiding dense temporal self-attention.

The manuscript evaluates ScaleMamba on **PEMS03, PEMS04, PEMS07, PEMS08, PEMS-BAY, and METR-LA**, with comparisons against Transformer-based, GNN-based, and SSM-based forecasting methods.

## Contents

- [Method overview](#method-overview)
- [Repository contents](#repository-contents)
- [Installation](#installation)
- [Datasets](#datasets)
- [Training and evaluation](#training-and-evaluation)
- [Experimental settings](#experimental-settings)
- [Results reported in the manuscript](#results-reported-in-the-manuscript)
- [Computational profiling](#computational-profiling)
- [Scope](#scope)
- [Citation](#citation)
- [Contact](#contact)

## Method overview

The framework described in the manuscript comprises four stages:

| Component | Role |
| --- | --- |
| Multi-scale decoupling | Identifies dominant frequency components and constructs temporal representations at different scales. |
| KFGNN | Models self-dependency and neighbor-dependency patterns and combines complementary representations using feature-variance statistics. |
| Graph Mamba Module (GMM) | Uses graph-informed features to condition selective state evolution and capture temporal dependencies. |
| Scale-aware KAN fusion (ScaKAN) | Integrates representations across scales through learnable nonlinear mappings before prediction. |

KFGNN draws inspiration from Kalman filtering through variance-guided feature fusion. Its variance statistics are feature summaries, rather than calibrated posterior uncertainty estimates.

With graph size, scale count, and feature and state dimensions fixed, the selective temporal scan has linear complexity with respect to sequence length. Graph processing and cross-scale fusion contribute additional computational costs.

## Repository contents

| Path | Description |
| --- | --- |
| [`main.py`](main.py) | Command-line entry point for example training and testing. |
| [`prepare.py`](prepare.py) | Sliding-window construction, preprocessing, and data loaders. |
| [`ScaleMamba.py`](ScaleMamba.py) | `KFGN`, `KFGN_Mamba`, selective state space operations, and model arguments. |
| [`modules.py`](modules.py) | Graph filtering components. |
| [`train_ScaleMamba.py`](train_ScaleMamba.py) | Training, checkpoint selection, and evaluation routines. |
| [`profile_stats.py`](profile_stats.py) | Parameter, operation-count, and runtime profiling utilities. |
| [`sigkan/`](sigkan/) | KAN-related building blocks. |
| [`imgs/`](imgs/) | Figures used to illustrate the experiments. |

**Implementation note:** The current `main.py` training path instantiates `KFGN_Mamba`; it does not invoke the KAN components in `sigkan/`. The method description and manuscript results below describe the paper, while the usage instructions document the available scripts. The example runner is not a frozen reproduction configuration for every manuscript experiment.

## Installation

Use a Python environment with a CUDA-enabled PyTorch installation. Python 3.10 or later is recommended for the profiling utilities. The current training entry uses CUDA directly.

```bash
git clone https://github.com/wanjiashan/ScaleMamba.git
cd ScaleMamba

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install a PyTorch build compatible with your CUDA environment using the [official installation instructions](https://pytorch.org/get-started/locally/), then install the remaining dependencies:

```bash
python -m pip install numpy pandas matplotlib einops torch-geometric
```

Optional dependencies:

- `mamba-ssm`: enables the fused CUDA selective scan when available and compatible with the environment; otherwise the model uses its local scan implementation.
- `thop`: used by the profiling script for operation-count estimation.
- `iisignature`: required when importing the modules in `sigkan/`.

## Datasets

All six benchmark datasets use a sampling interval of **5 minutes**. Consequently, forecasting steps 3, 6, and 12 correspond to 15, 30, and 60 minutes.

| Dataset | Sensors | Train / validation / test split in the manuscript | `main.py` dataset argument |
| --- | ---: | --- | --- |
| PEMS03 | 358 | 60% / 20% / 20% | `PEMS03` |
| PEMS04 | 307 | 60% / 20% / 20% | `pems04` |
| PEMS07 | 883 | 60% / 20% / 20% | `PEMS07` |
| PEMS08 | 170 | 60% / 20% / 20% | `PEMS08` |
| PEMS-BAY | 325 | 70% / 10% / 20% | `PEMSBY` |
| METR-LA | 207 | 70% / 10% / 20% | `metr-la` |

### Expected data files

The following paths are those read by `main.py`. Dataset arguments and filenames are case-sensitive.

| Dataset | Observation CSV | Graph matrix |
| --- | --- | --- |
| PEMS03 | `PEMS03/pems03_flow.csv` | `PEMS03/PEMS03-dtw-288-1-.npy` |
| PEMS04 | `PEMS04/pems04_flow.csv` | `PEMS04/pems04_adj.npy` |
| PEMS07 | `PEMS07/pems07_flow.csv` | `PEMS07/PEMS07-dtw-288-1-.npy` |
| PEMS08 | `PEMS08/pems08_flow.csv` | `PEMS08/PEMS08-dtw-288-1-.npy` |
| PEMS-BAY | `PEMSBY/pemsby_flow.csv` | `PEMSBY/pemsby-dtw-288-1-.npy` |
| METR-LA | `metr-la/metr-la_flow.csv` | `metr-la/metr-la-dtw-288-1-.npy` |

Each observation CSV should contain a numeric matrix with time steps as rows and sensors as columns. Remove timestamp or index columns from the model inputs. The adjacency matrix must have shape `N × N`, with sensor order matching the CSV columns.

For PEMS-BAY, the repository stores an archive under `pemby/`, whereas `main.py` reads from `PEMSBY/`. Extract it into the expected training directory:

```bash
python -m zipfile -e pemby/pemby.zip PEMSBY
```

Supply any missing dataset files before running their corresponding experiments. The PEMS07 graph file in the repository has a different filename from the loader path above; align the filename or update the path in `main.py`.

## Training and evaluation

### 1. Match the sensor dimensions

Set `-mamba_features` to the number of sensors in the selected dataset. In the example runner, this argument refers to the sensor-variable dimension; it is not the hidden width of 64 reported in the manuscript.

Also update the input-width clipping in `PrepareDataset` in `prepare.py`, which currently uses a hard-coded value of 307. Both occurrences of that value must match the selected sensor count. For example, for PEMS-BAY:

```python
if speed_sequences.shape[2] > 325:
    speed_sequences = speed_sequences[:, :, :325]
```

Keep the input width, label width, adjacency size, and `-mamba_features` consistent.

### 2. Set input and prediction lengths

The current `PrepareDataset` defaults are `seq_len=5` and `pred_len=5`. To use 12 input steps and 12 prediction steps, pass those values explicitly at the data preparation call in `main.py`. For example, the PEMS04 split is configured as follows:

```python
train_dataloader, valid_dataloader, test_dataloader, max_value = PrepareDataset(
    speed_matrix,
    BATCH_SIZE=48,
    seq_len=12,
    pred_len=12,
    train_propotion=0.6,
    valid_propotion=0.2,
)
```

For PEMS-BAY and METR-LA, use `train_propotion=0.7` and `valid_propotion=0.1`. The keyword spelling `propotion` matches the current function signature.

### 3. Run an experiment

Run commands from the repository root after preparing the data and matching the dimensions above.

```bash
# PEMS04
python main.py -dataset=pems04 -model=ScaleMamba -mamba_features=307

# PEMS-BAY
python main.py -dataset=PEMSBY -model=ScaleMamba -mamba_features=325

# METR-LA
python main.py -dataset=metr-la -model=ScaleMamba -mamba_features=207
```

The remaining dataset commands are:

```bash
python main.py -dataset=PEMS03 -model=ScaleMamba -mamba_features=358
python main.py -dataset=PEMS07 -model=ScaleMamba -mamba_features=883
python main.py -dataset=PEMS08 -model=ScaleMamba -mamba_features=170
```

The training routine saves a checkpoint to `best_model.pth`, reloads it, and evaluates the model on the test loader. Preserve this file between experiments if you need to retain multiple checkpoints; the default path is shared across runs.

### Configuration and reporting notes

- Input length, prediction length, and split ratios are configured in the data preparation call; they are not exposed as `main.py` command-line arguments.
- The current runner uses min–max preprocessing, passes a maximum of 200 epochs, and uses a patience of 50. Its checkpoint selection uses the last recorded validation-batch loss. These differ from the manuscript protocol below.
- In the public training path, `d_model` is derived from the input width. The `K` passed to `TrainScaleMamba` controls the graph diffusion branches; frequency-feature selection is configured separately in `prepare.py`.
- The `std_MAE` printed by `TestScaleMamba` is calculated across test batches. It is not the standard deviation across five independent runs. Compute run-level metrics separately when reporting repeated experiments.

Changing the window lengths and split ratios alone does not align all preprocessing, architecture, and evaluation details with the manuscript protocol.

## Experimental settings

The following settings describe the **main accuracy experiments in the revised manuscript**.

| Setting | Value |
| --- | --- |
| Historical input length | 12 steps |
| Prediction length | 12 steps |
| Hidden feature width | 64 |
| SSM state dimension | 16 |
| Mamba layers | 4 |
| Frequency configuration | Three nonzero frequency components, with the DC component retained |
| Graph diffusion order | 3 |
| Optimizer | AdamW |
| Initial learning rate | `1e-3` |
| Weight decay | `1e-4` |
| Learning-rate schedule | Cosine annealing, minimum `1e-5` |
| Training batch size | 48 |
| Maximum epochs | 100 |
| Early stopping | 20 consecutive epochs without improvement in mean validation loss |
| Checkpoint selection | Lowest mean loss on the complete validation set |
| Repeated runs | 5 |

The manuscript uses Z-score normalization. On PEMS03/04/07/08, metrics are averaged over all 12 prediction steps within each run, then summarized as **mean ± standard deviation across five runs**. On PEMS-BAY and METR-LA, results are reported at prediction steps 3, 6, and 12.

The number of selected temporal frequencies and the graph diffusion order are distinct settings, even when both are set to 3.

## Results reported in the manuscript

These tables reproduce the manuscript results. They are reference results, not measurements generated by the example commands above. Lower values are better for all three metrics.

### PEMS03, PEMS04, PEMS07, and PEMS08

Results are averaged over the 12 forecasting steps and reported as **mean ± standard deviation over five runs**.

| Dataset | MAE | RMSE | MAPE (%) |
| --- | ---: | ---: | ---: |
| PEMS03 | 14.59 ± 0.15 | 24.27 ± 0.29 | 14.12 ± 0.13 |
| PEMS04 | 18.07 ± 0.20 | 29.52 ± 0.41 | 12.01 ± 0.14 |
| PEMS07 | 18.38 ± 0.15 | 30.01 ± 0.42 | 9.23 ± 0.13 |
| PEMS08 | 13.42 ± 0.15 | 23.02 ± 0.23 | 9.25 ± 0.06 |

### PEMS-BAY and METR-LA

| Dataset | Forecast horizon | MAE | RMSE | MAPE (%) |
| --- | --- | ---: | ---: | ---: |
| PEMS-BAY | 15 min (step 3) | 1.24 | 2.74 | 2.58 |
| PEMS-BAY | 30 min (step 6) | 1.52 | 3.57 | 3.39 |
| PEMS-BAY | 60 min (step 12) | 1.84 | 4.36 | 4.42 |
| METR-LA | 15 min (step 3) | 2.58 | 4.99 | 6.58 |
| METR-LA | 30 min (step 6) | 3.02 | 5.92 | 7.90 |
| METR-LA | 60 min (step 12) | 3.27 | 7.01 | 9.61 |

ScaleMamba achieves the lowest mean error in 7 of the 12 dataset–metric comparisons on the four PEMS datasets, including the lowest RMSE on all four. It achieves the best results in 9 of the 18 dataset–horizon–metric comparisons on PEMS-BAY and METR-LA. Performance advantages vary across datasets, horizons, and metrics; see the manuscript for the complete baseline comparisons and ablations.

## Computational profiling

`profile_stats.py` supports profiling ScaleMamba and a Transformer-based replacement variant. For example:

```bash
python -m pip install thop

python profile_stats.py \
    --datasets PEMS-BAY METR-LA \
    --variants ScaleMamba ScaleMamba-Transformer \
    --batch-size 1 \
    --seq-len 96 \
    --pred-len 12 \
    --device cuda \
    --warmup-batches 20 \
    --max-train-batches 1 \
    --max-inference-batches 100 \
    --output-prefix profile_results/scalemamba
```

The script writes CSV and JSON summaries. It reads PEMS-BAY data from `pemby/` or `pemby/pemby.zip`, whereas `main.py` uses `PEMSBY/`. Its `--forecast-head auto` setting adds a temporal projection when input and prediction lengths differ.

**Timing scope:** The current profiling utility reports aggregate inference time over the measured batches, with device transfer inside the timed loop. This output is not directly equivalent to the per-sample forward latency reported in the manuscript. Operation-count comparisons also require an explicit MAC/FLOP convention and consistent coverage of custom operators.

### Efficiency protocol in the manuscript

- Hardware: NVIDIA RTX A6000 GPU.
- Batch size: 1.
- Input / prediction lengths: 96 / 12 steps.
- Measurement: mean forward-pass latency per sample after warm-up, with gradient computation disabled and CUDA synchronization before and after timing.
- Excluded: data loading, data transfer, and post-processing.

| Dataset | FLOPs (G) | Forward latency (s/sample) |
| --- | ---: | ---: |
| PEMS-BAY | 3.33 | 0.75 |
| METR-LA | 1.89 | 1.06 |

These are manuscript-reported results under the specified protocol. ScaleMamba does not achieve the lowest operation count or shortest runtime among all baselines.

## Scope

The study evaluates **offline, server-side traffic forecasting with fixed sensor sets**. The urban IoT setting refers to the source of traffic observations and the forecasting application. Adaptation to evolving graph topology, continual distribution shifts, and edge-device deployment are directions for future work. Edge-device memory, energy consumption, and end-to-end streaming throughput are not evaluated.

## Citation

If you use this work, please cite the ScaleMamba manuscript. The following entry identifies the manuscript and code repository; publication metadata can be updated when available.

```bibtex
@misc{wan2026scalemamba,
  title        = {{ScaleMamba}: Multi-Scale Graph State Space Learning
                  for Traffic Forecasting in Urban {IoT} Networks},
  author       = {Wan, Jiashan and Ling, Minhao and Li, Jingyang
                  and Jian, Ziheng and Wang, Kai},
  year         = {2026},
  howpublished = {Manuscript and source code},
  url          = {https://github.com/wanjiashan/ScaleMamba}
}
```

## Contact

For implementation questions, please open a [GitHub issue](https://github.com/wanjiashan/ScaleMamba/issues) and include the dataset, command, environment, and relevant error message.

For research inquiries, contact **Jiashan Wan** at [jswanwo@gmail.com](mailto:jswanwo@gmail.com).
