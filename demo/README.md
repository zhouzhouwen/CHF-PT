# CHF-PT web service

A self-contained web service for the five trained CHF-PT models. The user enters the fluid, the
pressure, the inlet temperature or subcooling, the mass flux, the heated length and the dimensions
of the test section. The server derives the full model input, predicts the critical heat flux and
returns the conformal interval and the out-of-distribution (OOD) score. Parametric sweeps,
operating maps, margin sizing, token attribution and batch evaluation use the same service. The
database is not needed at run time.

```
index.html          the page (single file, no external resources)
server.py           Flask/waitress server and JSON API
physics.py          derivation of the model input from the form entries
chfpt_core.py       schema, preprocessor and network, as defined in test/test_chfpt.py
prepare_assets.py   builds assets/ and runs the reconstruction check
assets/             fluid_props.json, calibration.npz, schema.json, examples.json
models/             optional; by default the models are read from ../model/chfpt
```

## Run

```bash
python server.py --no-tunnel              # local network only
python server.py                          # also opens a temporary public link (needs cloudflared)
python server.py --tunnel-name chfpt      # permanent public link through a named Cloudflare tunnel
python server.py --port 8080 --device cpu
```

The server needs `torch`, `numpy`, `pandas`, `flask` and `waitress`. A GPU is optional: one
prediction takes about 20 ms on a GPU and less than a second on a CPU. Without `cloudflared` on
`PATH`, only the local address is served. Set `CHFPT_MODEL_DIR` to read the models from another
folder.

The public link uses a Cloudflare tunnel. A quick tunnel needs no account, but its address changes
at every start and it closes when the server stops. A permanent address needs a named tunnel on a
Cloudflare account with a domain:

```bash
cloudflared tunnel login
cloudflared tunnel create chfpt
cloudflared tunnel route dns chfpt chf.example.com
python server.py --tunnel-name chfpt --public-url https://chf.example.com
```

A tunnel created in the Zero Trust dashboard can be used with `--tunnel-token`, or through the
environment variables `CHFPT_TUNNEL_NAME`, `CHFPT_TUNNEL_TOKEN` and `CHFPT_PUBLIC_URL`. Anyone who
has a public link can use the service while it runs.

## Functions of the page

| Panel | Content |
|---|---|
| Configuration | Any of the 15 test sections with any of the 7 fluids, or a custom fluid given by its saturation properties. *Fill from a real record* loads a random data point of the selected section and shows its split; only validation, calibration and test data points were not seen in training. |
| Operating point | Only the inputs that the selected section defines. Inputs outside the training range are accepted and listed above the result. |
| Outputs | CHF (mean of the five models), the lower and upper bounds of the 80, 90 or 95% conformal interval, and the OOD score, the Mahalanobis distance of the latent representation to 8,000 training data points. |
| Parametric sweep and operating map | CHF with its interval along one input, or CHF over two inputs. |
| Design margin | The allowable operating heat flux at a chosen safety factor, computed from the lower bound of the interval, and the change in one input needed to keep that factor. |
| Computed inputs | Every derived quantity sent to the network, and the change in the prediction when each token is withheld in turn. |
| Batch table | A CSV file of up to 2,000 operating points. Columns are named like the API fields (`P_kPa`, `T_in_C`, `DT_sub_K`, `G_kg_m2s`, `heated_length_m`, geometry columns such as `tube_Di_mm`, and `custom_*` for a custom fluid); a missing column takes the value in the form. The result file adds to each row the prediction, the bounds, the OOD score, the three anchors, the inputs outside the training range and, where an operating heat flux is given, the DNBR. |
| Units | kW m⁻², MW m⁻² or W cm⁻², and kPa, MPa or bar, for display only; the model, the API and batch files use kPa and kW m⁻². |

## Derived inputs

| Entered | Derived by the server |
|---|---|
| fluid, test section, configuration | `fluid_family` (by latent heat; the routing key of the mixture of experts) |
| pressure | `T_sat`, `rho_l`, `rho_v`, `h_fg`, `sigma`, `mu_l`, `cp_l`, `k_l`, `P_red`, `rho_l/rho_v`, `Pr_l`, capillary length |
| inlet temperature or subcooling | the other one, `dH_in = cp_l dT_sub`, `dH_in/h_fg` |
| mass flux, heated length | `Re_l`, `We_l`, `L/D` |
| one or two section dimensions | the channel scale, `Bond`, the confinement number, the Dean number of a coil |
| (from the entries above) | the Zuber correlation, the Bo–We flow-scaling token and the Bowring token, each where its inputs are available |

Configurations without a channel mass flux (pin-fin surfaces and sprays) have no flow tokens, and
their channel scale is the capillary length, as in the database. Gravity-dependent groups are
evaluated at 9.81 m s⁻²; the gravity level enters through the three gravity features.

## Reconstruction check

`python prepare_assets.py --verify` rebuilds the held-out data points from their primitive inputs,
in the same way as the form, and compares every derived feature and every prediction with the
database. With `data/CHF_dataset_v1.1.xlsx`, 3,134 of the 3,219 held-out data points can be
rebuilt; the others have a heated length masked by the leakage policy or no recorded inlet
subcooling. 53 of the 55 numeric columns agree to better than 10⁻³ (at most 2.5 × 10⁻⁴, because
the property tables were sampled with a newer CoolProp version than the one used for the
compilation). The two columns that differ are the two evidence tokens:

- Bowring: the service evaluates the correlation wherever its inputs are given, including 228
  data points for which the database leaves the token empty. Where both exist, they agree.
- Bo–We: the database used the diameter recorded at compilation for every helical coil. 16 data
  points differ by more than 1%, and 8 differ in whether the token is present.

The prediction from the form agrees with the prediction from the database row to within 0.1% for
92.8% of the rebuilt data points.

## API

The functions of the page are available as JSON.

```bash
curl -s localhost:5000/api/schema | jq '.geometries[].key'

curl -s localhost:5000/api/predict -H 'content-type: application/json' -d '{
  "fluid": "water", "geometry_type": "tube", "orientation": "vertical_upflow",
  "P_kPa": 9800, "subcooling_mode": "DT_sub", "DT_sub_K": 112.4,
  "G_kg_m2s": 1601, "heated_length_m": 2.44,
  "geometry": {"tube_Di_mm": 8.0}, "alpha": 0.10 }' | jq '{chf_kW_m2, interval, novelty}'
```

For a custom fluid, give any fluid name and add

```json
"custom_fluid": {"T_sat_C": 49, "P_crit_kPa": 1869, "rho_l": 1600, "rho_v": 13.4,
                 "hfg": 88, "sigma": 0.0108, "mu_l": 0.00064, "cp_l": 1.103, "k_l": 0.059}
```

| Endpoint | Fields besides the record | Returns |
|---|---|---|
| `POST /api/predict` | `alpha`, optional `q_op_kW_m2` | the four outputs, the derived record, the anchors |
| `POST /api/derive` | none | the derived quantities, without running the network |
| `POST /api/sweep` | `variable`, `min`, `max`, `n` | the curve with its interval and the Zuber line |
| `POST /api/map` | `x`, `x_min`, `x_max`, `x_n`, and the same four for `y` | CHF over the grid |
| `POST /api/design` | `q_op_kW_m2`, `safety_factor`, `x`, `x_min`, `x_max`, `x_n` | allowable flux, required value, margin curve |
| `POST /api/attribution` | none | the change in the prediction when each token is withheld |
| `POST /api/batch` | `base`, `rows` | one result per row, up to 2,000 |
| `GET /api/example` | `?geometry=<key>&avoid=<row_id>` | a random data point of that section |
| `GET /api/schema` | none | sections, fluids, ranges, examples, sweep variables |
| `GET /api/health` | none | device, number of models, parameter count |

`subcooling_mode` is `T_in`, `DT_sub` or `both`. A geometry-specific field is omitted or `null`
when the configuration does not define it. All values are in kPa, kW m⁻² and SI units.

## Notes

- The five models are the released models of the paper, trained on an earlier build of the
  database.
- Requests are processed one at a time; a batch of 2,000 rows takes a few seconds.
