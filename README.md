# DataWorks Challenge 2026 - ML-Assisted History Matching and 20-Year Production Forecast

**Team:** 404 Team Not Found  |  **University:** UNIVERSITI KEBANGSAAN MALAYSIA

## 1. Problem
An oil field has 10 years of production history (1998-2008). Four subsurface properties are uncertain:
fault transmissibility, porosity multiplier, permeability multiplier and aquifer pore volume.
100 reservoir-simulator runs with different values are provided. We (1) build a fast ML proxy of the simulator,
(2) find the parameter values that reproduce the observed history, (3) forecast oil, gas and water production
to 2028 for the 10 best cases, and (4) estimate whether the field makes money.

## 2. Approach
| Step | Notebook | Method |
| --- | --- | --- |
| Data preparation | `00_load_data` | Reads all Excel files; fixes US dates, duplicate report dates, blanks and -999.25 nulls; summarises each grid |
| Exploration | `01_explore` | Common quarterly time grid; sensitivity analysis; data checks |
| Proxy model | `02_train_proxy` | PCA + regression (Gaussian Process vs polynomial vs random forest); chosen on validation cases 71-85, scored on test cases 86-100, retrained on all 100 |
| History matching | `03_history_match` | Normalised least-squares misfit; Sobol scan (16,384 sets) + differential evolution; 10 diverse best sets via K-means |
| Forecast | `04_forecast_submit` | Arps decline of liquid rate x logistic water cut vs cumulative oil; validated by a 2005-2008 hindcast; anchored to observed 2008 totals; gas = oil x GOR |
| Economics | `04_forecast_submit` | Yearly cash flow and NPV at 10% discount (assumed prices) |
| Packaging | `05_make_submission` | Writes this README and the ZIP |

Shared functions are in `notebooks/dwc_utils.py`.

## 3. Results
| Item | Value |
| --- | --- |
| Proxy method | Gaussian Process |
| Proxy test error at 2008 (MAPE) | oil 0.05%, gas 0.05%, water 0.10% |
| Proxy test R2 | oil 0.9997, gas 0.9998, water 0.9998 |
| Best-match parameters | fault transmissibility 0.1500, porosity multiplier 1.3810, permeability multiplier 1.0836, aquifer pore volume 188.02 |
| Forecast method (hindcast median error 2005-2008) | exponential decline: oil 4.18%, water 4.23% |
| Cumulative oil by 2028-01-01, P10 / P50 / P90 | 57,252,919 / 57,501,995 / 57,680,593 STB |
| Cumulative gas by 2028-01-01, P10 / P50 / P90 | 20,798,214 / 20,888,684 / 20,953,555 MSCF |
| Cumulative water by 2028-01-01, P10 / P50 / P90 | 44,673,047 / 44,811,906 / 44,911,040 STB |
| NPV 2008-2028, P10 / P50 / P90 (assumed prices, notebook 04) | 151 / 160 / 166 million USD |

**Submission file:** `outputs/DWC26_Deliverable.xlsx` (one sheet per case in the template's order; the first case sheet = best match).
**Trained model:** `processed/proxy.joblib`.

## 4. Folder structure
```
DWC26/
  README.md                 this file
  requirements.txt          Python packages and versions used
  notebooks/                source code: 00-05 notebooks (with outputs) + dwc_utils.py
  processed/                trained model (proxy.joblib) and intermediate data
  outputs/                  DWC26_Deliverable.xlsx, result tables (CSV), figures/
  raw/                      template file; put the competition data files here to re-run notebook 00
```

## 5. How to run
**Option A - view only:** open the notebooks; all outputs (tables, charts) are saved inside them.

**Option B - re-run notebooks 01-04 without the raw data:** the intermediate files in `processed/` are included.
1. Upload the `DWC26` folder to Google Drive so that it is at `My Drive/DWC26`.
2. Open a notebook in Google Colab and choose Runtime -> Run all. Run 01 -> 02 -> 03 -> 04 in order.

**Option C - full run from the raw data:** put the 9 competition Excel files in `raw/` and run 00 -> 04.

**Running locally instead of Colab:** Python 3.10+, `pip install -r requirements.txt`, set the environment
variable `DWC_BASE` to the full path of the `DWC26` folder, then start Jupyter and run the notebooks in order.

## 6. Key settings and assumptions
- Train / validation / test split: cases 1-70 / 71-85 / 86-100.
- Misfit: oil, gas and water cumulative, allowed error 3% of each series' maximum.
- Economics (assumptions, not provided by the organisers): oil USD 70/bbl, gas USD 3/MSCF,
  water handling USD 2/bbl, fixed operating cost USD 60 million/year, discount rate 10%.
- Random seeds are fixed, so results are reproducible.

## 7. Team
| Member | Contribution |
| --- | --- |
| Farid | Data preparation and exploration (notebooks 00-01) |
| Alya | Proxy model (notebook 02) |
| Aqil | History matching and forecast (notebooks 03-04) |
| Iman | Power BI dashboard, documentation and video |
