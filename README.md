# Vehicle Recall Early Warning from Consumer Complaints

> **Question:** Can consumer complaints filed with the National Highway Traffic Safety Administration (NHTSA) flag a vehicle safety defect before a recall is issued — and which recalls can they actually catch?

Every automaker runs some version of this analysis on its own warranty data. This project asks what the *public* complaint record can do, measures how far that signal reaches before building any model, and then checks whether each improvement actually moved the result.

## Key findings

**1. Public complaints can only warn about a small share of recalls, and which share is predictable.** Only **13.6%** of manufacturer-initiated recalls since 2017 had five or more prior complaints about the same system on the same vehicle. For recalls initiated by NHTSA's Office of Defects Investigation (ODI), it was **64%**. That ceiling was measured before any modeling, and it shaped how every later result is read.

**2. The model doubles what a spreadsheet would do.** Sorting vehicles by complaint volume reaches PR-AUC 0.019; LightGBM reaches **0.038**. Reviewing the top 25 alerts per month, **8.7%** precede a real recall — an 8.4x lift over random selection, though still mostly false alarms on a problem with a 1% base rate.

**3. Most predictive power is structural, but complaints decide the top of the list.** Make, system, and vehicle age alone reach ROC-AUC 0.780; adding complaint features raises it only to 0.809. But at 25 alerts per month, adding complaints **nearly doubles the recalls caught (16 → 28)**. ROC-AUC averages over the whole ranking and hides the part a safety team would actually use.

**4. That gain lands almost entirely on ODI recalls.** Adding complaint data more than doubles detection of ODI recalls (**7.6% → 17.4%**) but improves manufacturer recalls by only 20%. This matches how recalls originate: ODI opens investigations partly *because* of complaints, while manufacturers mostly act on internal data the public never sees. The ceiling predicted it, the ablation showed where it happens, and this breakdown shows who it reaches.

**5. A better classifier didn't make a better system.** A fine-tuned DistilBERT model classified complaint narratives with **15% fewer errors** than a strong keyword baseline, and recovered 20,666 complaints that had been filed as "unknown." Feeding those back into the pipeline left recall detection essentially unchanged.

## The leakage traps

Several design decisions exist only to keep the model from seeing the future:

- **Post-recall complaint spikes.** Owners file complaints after receiving recall notices. Any model counting those would be detecting recalls, not predicting them. Every vehicle-system's history stops at its first recall.
- **Received date, not incident date.** A complaint is timestamped when NHTSA receives it, since that's when a live system would know about it.
- **Recall-generated complaints.** NHTSA's complaint type `RC` is produced by recall investigations themselves and is excluded.
- **Historical rows built from past activity only.** Rows exist only where a complaint arrived in the prior 12 months. Filtering on all-time totals would quietly use the future.
- **An embargo year.** A November 2022 row's label depends on recalls through May 2023. Training stops at 2021 and testing starts in 2023, so no training label depends on test-period events.
- **Right-censoring.** The final six months are dropped, since their outcome isn't knowable yet.

## The taxonomy problem

NHTSA renamed its component categories at some point, and complaints and recalls sit on different sides of the change. Complaints mostly say `ENGINE`, `SERVICE BRAKES`, `FUEL/PROPULSION SYSTEM`; older recalls say `ENGINE AND ENGINE COOLING`, `SERVICE BRAKES, HYDRAULIC`, `FUEL SYSTEM, GASOLINE`.

| System | Matchable recalls before normalizing | After |
|---|---|---|
| Engine | 132 | 719 |
| Service brakes | 105 | 1,051 |
| Fuel/propulsion system | 2 | 822 |
| Visibility/wiper | 20 | 491 |

Nothing errored — the join just quietly returned far fewer matches. Without the fix, the model would have concluded fuel-system complaints never precede a recall.

## Results

**Panel:** 2.36M vehicle-system-months (make, model, model year, system), 0.875% positive, covering 1,667 recalls.
**Split:** train 2016–2021, 2022 embargoed, test 2023 onward (648 recalls).

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Base rate | 0.500 | 0.010 |
| Sort by complaint volume | 0.613 | 0.019 |
| Sibling model year recalled | 0.576 | 0.012 |
| Complaint signals only | 0.705 | 0.025 |
| Structural only (make, system, age) | 0.780 | 0.032 |
| **LightGBM, all features** | **0.811** | **0.038** |

### Alerts per month

| Alerts/month | Precision | Lift | Recalls flagged in advance |
|---|---|---|---|
| 25 | 8.7% | 8.4x | 28 |
| 50 | 8.2% | 8.0x | 42 |
| 100 | 7.4% | 7.1x | 76 |
| 250 | 6.5% | 6.3x | 143 |

### Who gets caught (100 alerts/month)

| Initiated by | Recalls | Structural only | Full model |
|---|---|---|---|
| ODI | 132 | 10 (7.6%) | 23 (17.4%) |
| Manufacturer | 507 | 44 (8.7%) | 53 (10.5%) |

Flagged recalls received a median of about **4 months** of warning. Lead time is capped near six months by design, since the target only looks six months ahead.

**Is it just predicting recall expansions?** No. Recalls often extend to more model years, and a "sibling year already recalled" feature captures that. Removing it barely changed performance (0.811 → 0.808), and it scored only 0.663 alone.

## Fine-tuning on complaint narratives

**52,846 complaints** were filed only as "unknown or other," and the pipeline discarded them. A fine-tuned transformer reads each narrative and assigns a vehicle system.

**Model:** `distilbert-base-uncased`, fine-tuned in PyTorch with Hugging Face Transformers on a Colab T4 GPU (about 10 minutes). Uncased because NHTSA narratives are all capitals. Trained on 44,724 complaints filed before 2023, tested on 12,000 from 2024 onward.

| Model | Accuracy | Macro F1 |
|---|---|---|
| TF-IDF + logistic regression | 0.838 | 0.804 |
| **Fine-tuned DistilBERT** | **0.862** | **0.824** |

The baseline was strong, since words like "airbag" and "brake pedal" give most answers away. DistilBERT's gains landed where context matters most: electrical system F1 rose from 0.735 to 0.798, the vaguest category ("dashboard lights flicker and the car won't start").

**Better models don't fix ambiguous labels.** Vehicle speed control stayed broken in both models (F1 0.355 → 0.372). Owners file surging and acceleration problems under engine, power train, and speed control interchangeably, so no model can learn a clean boundary.

**An overall threshold hid a bad class.** At confidence ≥ 0.9, predictions were 94.8% accurate overall — but vehicle speed control was only **57.6%** accurate at that same threshold. Recovered labels were kept only for systems at 90%+ precision, leaving 20,666 complaints.

**Did it help?** The panel was rebuilt with the recovered complaints using identical SQL, and the original panel was confirmed to reproduce exactly before comparing.

| | Original | With recovered complaints |
|---|---|---|
| ROC-AUC | 0.8105 | 0.8099 |
| PR-AUC | 0.0383 | 0.0390 |
| Recalls caught at 100 alerts | 76 | 80 |

Eleven recalls became newly visible, but detection was essentially flat. The panel already held 731,730 categorized complaints; adding 2.6% more wasn't enough to change what ranks highest. The classifier was a clear success as a component and did not improve the system — and the only way to know that was to measure the system.

## Data

- **Source:** NHTSA Office of Defects Investigation flat files ([nhtsa.gov/nhtsa-datasets-and-apis](https://www.nhtsa.gov/nhtsa-datasets-and-apis))
- **Complaints:** 1,106,034 vehicle complaints received 2015–2026 (787,344 distinct; one complaint can span several components)
- **Recalls:** 13,738 vehicle recall campaigns since 2010
- Files are tab-delimited with no header row; column names come from NHTSA's data dictionaries

Personal information in the raw files (VINs, cities, dealer contacts, operator names) is dropped at load time and never enters a processed table.

## Limitations

- **Only first recalls are modeled.** A vehicle-system's history stops at its first recall since 2010, so later recalls on the same vehicle-system aren't predicted. A cooldown window would allow repeat recalls.
- **Small counts for the headline comparison.** The ODI result rests on 23 versus 10 flagged recalls; the direction is clear, but the exact percentages carry wide uncertainty.
- **Coverage is overstated.** Evaluation denominators only include recalls with recent complaint activity. Recalls on vehicles with no complaints never enter the evaluation.
- **Label noise caps classifier accuracy.** Owners choose complaint categories themselves.
- **Recovered-label accuracy is likely lower than tested.** Unknown complaints are vaguer than the test set: 42% cleared the confidence threshold versus 76% of test complaints.
- **NHTSA data revisions.** A May 2021 system update retroactively changed make, model, and component names in some records.

## Next steps

- Model repeat recalls with a cooldown window instead of a hard stop at the first recall
- Join NHTSA investigations (preliminary evaluations, engineering analyses) as features, since they often precede ODI recalls
- Use narrative embeddings directly as panel features, rather than only to recover missing labels
- Report bootstrap confidence intervals on recall-level detection

## Repository

```
├── sql/
│   └── 02_panel.sql              # monthly panel with leakage controls
├── notebooks/
│   ├── 01_load.ipynb             # parsing, quality checks, taxonomy normalization, ceiling
│   ├── 02_panel.ipynb            # panel construction
│   ├── 03_models.ipynb           # baselines, LightGBM, ablations, lead time
│   ├── 04_text_prep.ipynb        # classifier training data, TF-IDF baseline
│   ├── 05_augment.ipynb          # recovered complaints → rebuilt panel → comparison
│   └── colab_finetune.ipynb      # DistilBERT fine-tuning (run on a Colab GPU)
├── results/                      # model comparisons, ablations, lead-time tables
├── notes.md                      # full analysis log and decisions
└── requirements.txt
```

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

1. Download the complaint files (`COMPLAINTS_RECEIVED_2015-2019`, `2020-2024`, `2025-2026`), `FLAT_RCL_POST_2010.zip`, and the dictionaries `CMPL.txt` and `RCL.txt` into `data/raw/`, then unzip.
2. Run notebooks `01` through `04` in order.
3. Upload the four `text_*.parquet` files to Colab, enable a T4 GPU, and run `colab_finetune.ipynb`. Download `unknown_predictions.parquet` into `data/processed/`.
4. Run notebook `05`.

## Author

[Your name] · [LinkedIn] · [email]