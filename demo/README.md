# CHF-PT demo — the five released checkpoints behind a web page

A self-contained deployment of CHF-PT. You type what an experimenter reads off the rig — the
fluid, the pressure, the inlet temperature, the mass flux, the heated length and the two or three
numbers that describe the test section — and the server reconstructs the full model input,
predicts the critical heat flux, and returns a calibrated interval and an out-of-distribution
score. It also sweeps, maps, sizes for a margin, and runs a whole table of operating points.

Copy this folder to any machine with Python and a few packages and it behaves identically. The
32,271-record database is **not** needed at run time.

```
index.html          the page: one file, no CDN, no build step, works offline
server.py           Flask + waitress, the five seeds, the JSON API, the public link
physics.py          form entry -> full record: saturation properties, derived groups, three anchors
chfpt_core.py       schema, preprocessor and network, copied verbatim from the release
prepare_assets.py   one-off builder for assets/, and the verification described below
assets/             fluid_props.json, calibration.npz, schema.json, examples.json   (6.8 MB)
models/             chfpt_seed0..4.pt                                (340 MB)
requirements.txt
```

---

## Run

```bash
python server.py                        # local link + a temporary public link
python server.py --tunnel-name chfpt    # local link + a permanent public link
python server.py --no-tunnel            # local network only
python server.py --port 8080 --device cpu
```

Two addresses are printed at start:

```
==========================================================================
  Local     http://localhost:5000
            http://10.8.142.34:5000   (anyone on this network)
  Public    https://xxxx-yyyy.trycloudflare.com   (temporary — a new address at every start)
==========================================================================
```

The **local** address works for anyone on the same network for as long as the process runs. The
**public** one is a Cloudflare tunnel, and by default an anonymous *quick* tunnel: free, no
account, but a different address at every start and gone when the process stops.

A **permanent** address needs a named tunnel — a one-time setup on a Cloudflare account with a
domain:

```bash
cloudflared tunnel login                              # once, in a browser
cloudflared tunnel create chfpt                       # once
cloudflared tunnel route dns chfpt chf.example.com    # once
python server.py --tunnel-name chfpt --public-url https://chf.example.com
```

After that the same URL works at every start. A tunnel created in the Zero Trust dashboard works
too: pass its token with `--tunnel-token`, or set `CHFPT_TUNNEL_NAME`, `CHFPT_TUNNEL_TOKEN` and
`CHFPT_PUBLIC_URL` in the environment and just run `python server.py`.

Any environment with `torch`, `numpy`, `pandas`, `flask` and `waitress` will do — on this machine
`/home/user/anaconda3/envs/Aerosol_DT/bin/python` has all five. A GPU is optional: one prediction
takes about 20 ms on a GPU and well under a second on a CPU. Without `cloudflared` on `PATH` the
server says so and serves the local address only.

---

## Using the page

### 1. Configuration

Pick a **test section** — fifteen of them, grouped as channels, enhanced surfaces and the six
fusion high-heat-flux mock-ups — and a **fluid**. Any section may be combined with any fluid; the
corpus never saw most of those pairs, and the page says so rather than forbidding it.

For a fluid the corpus does not contain at all, choose **custom fluid**. Nine saturation
properties appear, pre-filled from the current state and ready to be overwritten. Unless the name
you give is one of the seven pre-trained fluids, no fluid-identity token is emitted and the model
works from the properties, the fluid family (set by the latent heat) and the geometry alone.
Custom properties are held as entered and do not follow the pressure.

*Fill from a real record* draws a record of that section at random from the corpus — a different
one at every click — and shows the measured CHF beside the prediction until you change an input.
The pool covers all four subsets and the page names the one each record came from, because a
**training** record is one the checkpoints have already seen: only the validation, calibration and
test subsets say anything about generalisation.

### 2. Operating point and dimensions

The form asks only for the quantities that section defines: pressure, inlet temperature **or**
subcooling (each follows from the other), mass flux, heated length, and the one to three
dimensions of the test section. Configurations without a channel flow — the pin-fin surfaces and
the sprays — ask for no mass flux at all, because the flow tokens are not emitted for them.

**Inputs may leave the corpus.** The prediction is still returned; the field is outlined in amber
and one line above the result names every input that left the pre-trained range. Weigh that
against the OOD score.

### 3. The four outputs

- **CHF**, the mean of the five pre-trained seeds;
- **the lower and upper bound** of a conformal interval at 80, 90 or 95%, whose correction was
  fitted on the calibration split that never enters a point prediction. Empirical coverage on the
  3,219 held-out records, measured by `prepare_assets.py`: 0.793 / 0.905 / 0.953 against the three
  nominal levels;
- **the OOD score**, the Mahalanobis distance of the pooled latent to 8,000 pre-training records,
  read against the held-out percentiles (p50 = 3.6, p90 = 7.2, p99 = 16.4).

A fourth tile, the **margin**, appears when you enter an operating heat flux.

### 4. Parametric sweep and operating map

*One input* walks a single variable over a range and draws the prediction with its interval
against the Zuber limit. *Operating map* varies two at once and draws CHF over the plane, with
the point you entered ringed. Both redraw themselves after every prediction.

### 5. Design margin

Enter an operating heat flux and this panel answers the two questions a designer actually asks:

- **the allowable operating flux** at a chosen safety factor — the *lower* conformal bound divided
  by the factor, not the point prediction, so the model's own uncertainty is already subtracted;
- **how far one input has to move** to keep that factor: "raise the mass flux to 2,165 kg m⁻² s⁻¹",
  or, when the target is already met, how far the input may drift before it is lost.

The margin curve is scanned rather than assumed monotone, and the answer is the boundary of the
safe region nearest to your operating point, so a non-monotone response cannot produce a
misleading single number. This is a screening aid, not a licensing basis.

### 6. Computed inputs

Every intermediate: the fluid state, the derived groups, the three anchors and the exact record
sent to the network. It also answers *which inputs the prediction leans on*: each defined token is
withheld in turn — exactly as a record that never measured it would arrive — and the change in the
prediction is reported. The feature layer is deliberately over-complete, so these are marginal
effects rather than an additive split, but they show at a glance whether a prediction rests on the
mass flux, on the geometry or on a correlation anchor.

### 7. Batch table

A CSV of operating points in, predictions out. Up to 2,000 rows per file.

- **Columns are named like the API fields**: `P_kPa`, `T_in_C`, `DT_sub_K`, `G_kg_m2s`,
  `heated_length_m`, `fluid`, `geometry_type`, `orientation`, `q_op_kW_m2`, any geometry column
  (`tube_Di_mm`, `annulus_Dh_mm`, `fin_height_um`, …), `char_length_mm`, and `custom_*` for a
  fluid you define (`custom_rho_l`, `custom_hfg`, …). An `id`, `row_id`, `label` or `name` column
  is carried through untouched.
- **A column you leave out keeps the value from the form**, so a study that varies two quantities
  needs only those two columns. Naming a different `geometry_type` drops the form's dimensions, so
  mixed-section files work.
- Give either `T_in_C` or `DT_sub_K`; give both and both are used exactly as written.
- **Units in a batch file are always canonical** — `kPa` and `kW m⁻²` — whatever the display is
  set to.

*Download a template* writes the right columns for the current section, filled with the current
values, as a starting point. The result file echoes your columns and appends `chf_kW_m2`,
`lower_kW_m2`, `upper_kW_m2`, `ood_score`, `ood_level`, the three anchors, `DNBR` and
`DNBR_lower` where an operating flux was given, `outside_corpus` (how many inputs left the
pre-trained range), `outside` (which) and `error` for a row that could not be built.

### 8. Units

The selector at the top switches the display between kW m⁻², MW m⁻² and W cm⁻², and between kPa,
MPa and bar. It changes **only what is shown**: every number that reaches the model, the API and a
batch file stays in kPa and kW m⁻², so no arithmetic ever crosses a unit boundary.

---

## What you type and what the server computes

| You supply | The server derives |
|---|---|
| fluid, test section, configuration | `fluid_family` (by latent heat, the routing key of the mixture of experts) |
| pressure | `T_sat`, `rho_l`, `rho_v`, `h_fg`, `sigma`, `mu_l`, `cp_l`, `k_l`, `P_red`, `rho_l/rho_v`, `Pr_l`, capillary length |
| inlet temperature **or** subcooling | the other one, `dH_in = cp_l dT_sub`, `dH_in/h_fg` |
| mass flux, heated length | `Re_l`, `We_l`, `L/D` |
| one or two section dimensions | the channel scale, `Bond`, the confinement number, the Dean number of a coil |
| — | the Zuber correlation, the Bo–We flow-scaling token and the Bowring token, each emitted only where it is computable |

Configurations without a channel mass flux emit no flow token at all, and their channel scale is
the capillary length — exactly as in the released database. Every gravity-dependent group is
evaluated at the reference 9.81 m s⁻², including for a microgravity record, whose gravity level
enters only through the three explicit gravity features.

---

## Fidelity to the released evaluation

`python prepare_assets.py --verify` rebuilds every held-out record from its primitive inputs, the
way the form does, and compares it with the row the released `test/test_chfpt.py` reads:

```
3,136 of the 3,219 held-out records rebuilt from their primitive inputs
53 of 55 numeric columns agree to better than 1e-3 (worst 2.7e-04)
prediction from the form vs from the database row: identical to 0.1% on 87.9% of records
ensemble MAPE against the measurement: 6.98% from the form, 6.63% from the database rows
```

Three things account for the difference, and all three are understood:

1. The saturation tables are sampled from a **newer CoolProp** than the compilation used, which
   moves the properties by at most 3 × 10⁻⁴ relative — far below anything physical.
2. The released **Bowring** column was evaluated on an upstream table whose inlet enthalpy was
   corrupted for the OECD Phase-2 records (it reads ~1.5 × 10⁶ kJ kg⁻¹, giving anchors near
   10⁶ kW m⁻²) and was left empty wherever that table had no diameter. The demo evaluates the
   correlation from the inputs you supply, so 157 of 3,136 held-out records receive a different
   Bowring token and 228 receive one where the release had none.
3. The released **Bo–We** column (`anchor_bowe_kW_m2`, formerly `anchor_kandlikar_kW_m2`) used the
   diameter recorded at compilation for every helical coil, and the
   heated equivalent diameter rather than the hydraulic diameter for annuli. The annulus
   convention is reproduced (both diameters are inputs); the coil one is not, so 16 helical
   records move.

Nothing else differs: the network, the frozen preprocessor, the leakage policy and the conformal
procedure are the released ones, and `chfpt_core.py` is a byte-identical copy of the release code.

These numbers describe the **five-seed ensemble this page serves**, which is not what the paper
tabulates: the paper reports per-seed mean ± s.d. (7.54 ± 1.12 micro-MAPE, 19.59 macro), and
averaging five seeds naturally beats the average single seed. To keep two different correct
numbers from circulating, the page itself quotes no accuracy figure.

---

## Rebuilding `assets/`

Only needed if the release changes.

```bash
python prepare_assets.py                       # expects ../CHF-PT_release
python prepare_assets.py --release /path/to/CHF-PT_release
python prepare_assets.py --schema-only         # labels and ranges only, keeps the calibration
```

The corpus it reads is `CHF-PT_release/data/CHF_dataset_v1.1.xlsx`, the deduplicated final build
(identical record for record to `data/CHF_TwoLayer_Final最终去重版本.dedup6sig.xlsx`: 32,271 rows,
same CHF, pressure and mass-flux totals). This step needs `openpyxl` and `CoolProp` in addition to
the run-time packages; the server never does.

---

## API

Everything the page does is available as JSON, so the models can be driven from a script.

```bash
curl -s localhost:5000/api/schema | jq '.geometries[].key'

curl -s localhost:5000/api/predict -H 'content-type: application/json' -d '{
  "fluid": "water", "geometry_type": "tube", "orientation": "vertical_upflow",
  "P_kPa": 9800, "subcooling_mode": "DT_sub", "DT_sub_K": 112.4,
  "G_kg_m2s": 1601, "heated_length_m": 2.44,
  "geometry": {"tube_Di_mm": 8.0}, "alpha": 0.10 }' | jq '{chf_kW_m2, interval, novelty}'
```

To predict for a fluid of your own, replace `"fluid"` with any name and add

```json
"custom_fluid": {"T_sat_C": 49, "P_crit_kPa": 1869, "rho_l": 1600, "rho_v": 13.4,
                 "hfg": 88, "sigma": 0.0108, "mu_l": 0.00064, "cp_l": 1.103, "k_l": 0.059}
```

| Endpoint | Body beyond the record | Returns |
|---|---|---|
| `POST /api/predict` | `alpha`, optional `q_op_kW_m2` | the four outputs, the derived record, the anchors |
| `POST /api/derive` | — | the derived quantities, no network run |
| `POST /api/sweep` | `variable`, `min`, `max`, `n` | the curve with its interval and the Zuber line |
| `POST /api/map` | `x`, `x_min`, `x_max`, `x_n`, and the same four for `y` | CHF over the grid |
| `POST /api/design` | `q_op_kW_m2`, `safety_factor`, `x`, `x_min`, `x_max`, `x_n` | allowable flux, required value, margin curve |
| `POST /api/attribution` | — | the change in the prediction when each token is withheld |
| `POST /api/batch` | `base`, `rows` | one result per row, up to 2,000 |
| `GET /api/example` | `?geometry=<key>&avoid=<row_id>` | a real record of that section, drawn at random |
| `GET /api/schema` | — | sections, fluids, ranges, examples, the sweep variables |
| `GET /api/health` | — | device, seeds, parameter count |

`subcooling_mode` is `T_in`, `DT_sub` or `both`; a geometry-specific field is omitted or `null`
when the configuration does not define it. Every value in and out is canonical: kPa, kW m⁻², SI.

---

## Notes

- The five checkpoints are the released ones, trained on the pre-audit build of the database; see
  the release README for what that changes (at most 0.03 percentage points).
- Inference is serialised by one lock, so concurrent visitors queue rather than compete for the
  GPU. At about 20 ms per prediction that is invisible below a few dozen users. A batch of 2,000
  rows takes a few seconds and is chunked internally so memory cannot spike.
- `trycloudflare.com` links are anonymous and ephemeral: they last as long as the process. Anyone
  with the link can use the model, so do not post one you do not want shared.
