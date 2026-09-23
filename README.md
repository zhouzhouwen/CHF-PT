# CHF-PT — a physics-anchored pre-trained Transformer for critical heat flux

CHF-PT predicts the critical heat flux (CHF) of a boiling system from its operating
conditions. It reads one experimental record as a **variable-length sequence of key-value
tokens** — a common physics layer that every configuration defines, a geometry-specific
layer that only some define, and closed-form correlation anchors — and regresses the
**logarithmic residual to the Zuber hydrodynamic-instability limit** instead of CHF itself.
A feature a record does not define simply emits no token, so a single model spans vertical
tubes, rod bundles, annuli, plates, helical coils, pin-fin surfaces, sprays, enhanced fusion
tubes and microgravity channels, and can describe a geometry it has never seen.

![The CHF-PT web service predicting a held-out test record](docs/web_demo.png)

*The web service in `demo/`: an operating point in, a CHF estimate with its 90% conformal
bounds and an out-of-distribution score out. Here it predicts 2,298 kW m⁻² for a test record
measured at 2,238 kW m⁻², a record no checkpoint has seen.*

This repository releases the proposed model: the consolidated database, the CHF-PT training
script, the five trained CHF-PT models, the evaluation script and the web service. The four
baselines of the paper, an ANN, a plain Transformer, XGBoost and LightGBM, are described in the
paper and are not distributed here.

```
data/    CHF_dataset_v1.1.xlsx     32,271 records × 68 columns, with the split and provenance
train/   train_chfpt.py            training, self-contained, no import from this repository
model/   chfpt/chfpt_seed{0..4}.pt the five trained models of the paper, 68 MB each
test/    test_chfpt.py             evaluation, writes test/results/chfpt/
demo/    server.py, index.html, physics.py                    the web service of Appendix B
docs/    web_demo.png                                         screenshot of that service
```

Both scripts are standalone: they import only third-party packages, never another file of this
repository, so reading one of them end to end shows the entire method.

## The database

| | |
|---|---|
| Records | **32,271**, consolidated from 15 experimental sources |
| Coverage | 7 fluids, 15 geometries, **19 fluid-geometry domains**, CHF from **12 to 41,900 kW m⁻²**, 12 International Space Station microgravity points |
| Columns | 68: `row_id`, `split`, 4 categorical identities, **27 common numeric features**, **24 + 3 geometry-specific features**, **3 physics anchors**, measured `CHF_kW_m2`, a data-quality tag and 3 provenance columns |
| Sheets | `CHF_data`, `Dictionary` (per-column definition, unit, coverage), `Sources`, `Domains`, `Notes` |

**Split.** The `split` column carries the partition used in the paper: group-stratified
7:1:1:1 inside every fluid-geometry domain, with near-duplicate operating points bound to a
single subset — 22,600 train / 3,227 validation / 3,225 calibration / 3,219 test. It ships
with the data so that held-out records can never be confused with training records. The
calibration subset is used only to calibrate conformal intervals and never enters a point
prediction.

**Leakage.** 1,224 records had flow or heated-length features completed during compilation
with a surrogate derived from the measured CHF. `flow_feature_imputation` tags every one of
them, and all scripts blank those cells by default (`--leakage_policy mask`). Outlet quality
and the CHF regime label are post-hoc quantities and are not part of the released features.

**Anchors.** The three physics anchors are closed-form quantities, not measurements. The
Zuber limit is the base of the CHF-PT target. The Bowring (1972) water-tube correlation
(`anchor_bowring_kW_m2`) and the Bo-We flow-scaling token (`anchor_bowe_kW_m2`, renamed from
`anchor_kandlikar_kW_m2` in September 2026 without any change of value) enter as additional
input tokens where their inputs exist. The Bo-We column is an engineered boiling-number
scaling and not a published CHF correlation: it exceeds the measured CHF by a median factor
of 26 and is read by the network as a standardized logarithm, so its prefactor and bias drop
out. A fourth schema column, the convectively scaled Zuber feature, equals the Zuber limit
for every record and is materialised by the scripts. See Appendix A of the paper. Baselines
never see any of them.

## Install

```bash
pip install numpy pandas openpyxl torch scikit-learn joblib
pip install optuna          # only for the --optuna flag
pip install flask waitress  # only for the web service in demo/
```

Python ≥ 3.9. A GPU is optional: evaluation runs on a CPU in minutes, training CHF-PT takes
about 20 minutes per seed on one modern GPU.

## Quick start

```bash
python test/test_chfpt.py                 # evaluate the five released models on the test split
python test/test_chfpt.py --split heldout # the 9,671 records that never entered training
python test/test_chfpt.py --ood           # add the Mahalanobis novelty score

python train/train_chfpt.py               # train one model, seed 0
python train/train_chfpt.py --all_seeds   # reproduce the five released seeds
python train/train_chfpt.py --optuna      # hyper-parameter search, then train
python train/train_chfpt.py --smoke       # three-epoch pipeline check
```

Evaluation writes per-seed metrics, a mean ± s.d. summary, a per-domain breakdown and one
prediction file per seed to `test/results/chfpt/`; the five files of the paper are already
there. Training writes one self-describing checkpoint per seed to `model/chfpt/` and a loss
log to `train/logs/chfpt/`.

## Released models and results

Test split, 3,219 held-out records, mean ± s.d. over the five seeds. The CHF-PT row is
reproduced by `python test/test_chfpt.py`; the baseline rows are quoted from the paper, where
those models are specified in full. **macro-MAPE** averages the error over the 19
fluid-geometry domains, so a rare domain counts as much as the water-tube bulk; **worst** is
the largest domain error.

| Model | Parameters | micro-MAPE | macro-MAPE | worst domain | within ±20% | 90% coverage |
|---|---|---|---|---|---|---|
| **CHF-PT** | 17.81 M | **7.54 ± 1.12** | **19.59 ± 5.54** | **69.4 ± 18.1** | **95.6%** | **0.903 ± 0.004** |
| ANN | 1.27 M | 11.05 ± 3.21 | 25.93 ± 9.06 | 178.6 ± 114.3 | 92.4% | — |
| Transformer | 20.05 M | 14.36 ± 2.46 | 28.57 ± 1.36 | 168.5 ± 39.2 | 91.0% | — |
| XGBoost | trees | 12.88 ± 0.40 | 40.29 ± 0.60 | 196.3 ± 10.6 | 92.6% | — |
| LightGBM | trees | 11.77 ± 0.53 | 40.36 ± 4.67 | 235.2 ± 82.5 | 94.5% | — |

CHF-PT is the only model that returns calibrated intervals. All four baselines regress raw
CHF with plain squared error and never see the physics anchor, so the comparison isolates
what the anchor and the token schema contribute.

Each CHF-PT checkpoint is 68 MB and carries its own configuration, the frozen preprocessor and
the conformal calibration, **341 MB for the five seeds**. They are committed to the repository
directly, so a plain `git clone` brings them; add `--depth 1` if you only need the current
state.

```bash
git clone --depth 1 https://github.com/zhouzhouwen/CHF-PT.git
```

## The web service

`demo/` is the deployed service described in Appendix B of the paper: a single-page front end
and a Flask back end that turn what an experiment actually reports into the full input the
network reads, so no database and no training environment are needed.

```bash
pip install flask waitress
python demo/server.py                 # http://localhost:5000, add --no-tunnel to stay local
```

The service finds the five CHF-PT checkpoints in `model/chfpt/` automatically; set
`CHFPT_MODEL_DIR` to point somewhere else. A browser form collects the fluid, the pressure,
the inlet temperature or subcooling, the mass flux, the heated length and the two or three
dimensions of the test section. It returns the four outputs of the paper, the prediction, the
two conformal bounds and the out-of-distribution score, together with every derived quantity
that produced them.

The screenshot at the top of this page shows one such prediction. Five further panels support design work: a parametric sweep along one input, a two-dimensional
operating map, margin sizing on the lower bound rather than on the prediction, token attribution
that withholds one token at a time, and batch evaluation of a CSV file. Any test section may be
combined with any fluid, including a fluid supplied as its own saturation properties, and a
condition outside the pre-training range is answered rather than refused, with the departure
reported next to the out-of-distribution score.

The same request can be made without a browser:

```bash
curl -s -X POST http://localhost:5000/api/predict -H 'Content-Type: application/json' \
  -d '{"fluid":"water","geometry_type":"tube","orientation":"vertical_upflow",
       "P_kPa":14700,"G_kg_m2s":2190,"T_in_C":189.68,"heated_length_m":1.57,
       "geometry":{"tube_Di_mm":8.0},"alpha":0.10}'
```

## Three things to know before you rely on this

1. **The released weights were trained on the pre-audit build of the database**, which held
   291 duplicate records more (0.89%, all water in vertical tubes, removed by a
   six-significant-figure audit). Each duplicate sat in the same subset as the record it
   copies, so the partition of the surviving records is unchanged, and re-evaluating the
   weights on the audited test split moves the micro-MAPE of every method by at most 0.03
   percentage points and leaves every worst-domain error identical. The table above reports
   the audited numbers.
2. **`fluid`, `fluid_family`, `geometry_type` and `orientation` are model inputs, not labels.**
   Every model embeds them, the mixture of experts routes on the fluid family, and the split
   and the macro-metric are grouped by fluid × geometry. Removing them from the data would
   silently change every prediction.
3. **The trained schema carries a fourth anchor token**, the convectively scaled Zuber
   feature. Its clamp selects unity for every record, so it equals `anchor_zuber_kW_m2`
   throughout the database, adds no information and is not stored as a column; the scripts
   materialise it on load, with a comment at the definition, so a released checkpoint sees
   exactly the schema it was trained with.

## Citation

> Zhou, W., Miwa, S., Wang, K., et al. *CHF-PT: A physics-anchored pre-trained Transformer
> for transferable critical heat flux prediction.* (under review)

Please also cite the 15 original experimental sources listed in the `Sources` sheet of the
data file when you use the database.

## License

[to be filled in — code and data may carry different licenses]
