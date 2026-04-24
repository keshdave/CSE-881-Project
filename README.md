# QB Transfer PPA Predictor
### CSE 881 — Data Mining

A data-driven prototype tool that predicts a transfer quarterback's **Predicted Points Added (PPA)** at a destination team using an XGBoost regression model. Built to help front office personnel make evidence-based decisions when evaluating QB targets from the transfer portal.

---

## 🏈 Live App!

> Click here: **[QB PPA Predictor]([https://your-streamlit-link-here](https://qb-transfer-final-prototype.streamlit.app/))**  
> Enter a QB's previous season stats and destination team info to generate a predicted Post-PPA score.

---

## Repo Structure

```
├── app.py                        # Streamlit web application
├── transfer_model.ipynb          # Main modeling notebook (EDA, feature selection, XGBoost, SHAP)
├── cfb_stats_pull_all.R          # R script for pulling raw CFB data via cfbfastR
├── qb_xgb_model.pkl              # Trained XGBoost model (loaded by app.py)
├── feature_importance.png        # Feature importance chart (used in app)
├── shap_summary.png              # SHAP summary plot (used in app)
├── requirements.txt              # Python dependencies
└── data/
    ├── qb_transfer_master.csv        # Core modeling dataset — QB transfers with pre/post PPA
    ├── post_transfer_ppa_raw.csv     # Player-level PPA stats post-transfer
    ├── cfbd_player_stats.csv         # All CFB player stats (2018–2024), 75K+ entries
    ├── cfbd_ppa_stats.csv            # Team-level PPA and efficiency metrics per season
    ├── cfbd_team_adv_stats.csv       # Team-level advanced stats per season
    ├── cfbd_sp_ratings.csv           # Team SP+ ratings per season
    ├── cfbd_qb_transfers_raw.csv     # Raw transfer portal records for all positions
    ├── cfbd_qb_rosters.csv           # CFB roster records with physical attributes
    └── cfbd_qb_recruits.csv          # QB recruit profiles with rankings and committed schools
```

---

## Features Used in Model

| Feature | Description |
|---|---|
| `prev_pass_td` | QB's passing touchdowns from previous season |
| `prev_pass_yds` | QB's passing yards from previous season |
| `prev_avg_ppa` | QB's average PPA from previous season |
| `dest_off_ppa` | Destination team's offensive PPA from previous season |
| `dest_sp_offense` | Destination team's Offensive SP+ Rating |
| `is_transfer` | Whether the QB is a transfer (1) or not (0) |
| `years_in_college` | Number of seasons the QB has played in college |

---

## Key Findings

- **Transfer Penalty**: The model applies a measurable penalty to QBs who transfer, predicting a lower PPA compared to a non-transfer QB with identical stats.
- **Age is not a penalty**: `years_in_college` had a negligible effect on predicted PPA, suggesting programs should not penalize older QBs in their evaluations.
- **Top predictors**: Previous passing TDs, destination team offensive PPA, and previous passing yards were the strongest drivers of model output.

---

## Data Collection

Raw data was pulled using the [`cfbfastR`](https://cfbfastr.sportsdataverse.org/) R package via `cfb_stats_pull_all.R`, covering CFB seasons **2018–2024**. Data includes player stats, team advanced metrics, SP+ ratings, transfer portal records, roster data, and recruiting profiles.

---

## Group Members
_(Michigan State - Department of Statistics and Probability - MSDS)_
- Archisha Bhatt
- Keshavi Dave
- Joey Larabee
- Ethan Richards
  
*CSE 881 — Spring 2025 Semester Project*
