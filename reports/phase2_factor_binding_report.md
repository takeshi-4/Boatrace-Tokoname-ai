# Phase 2: Factor × Feature Binding Report

**Generated:** 2026-01-04 10:20:29
**Data Source:** Tokoname boatrace results

---

## Executive Summary

- **Total Rows:** 2,220
- **Total Columns:** 704
- **Date Range:** 2019-01-01 00:00:00 to 2019-12-31 00:00:00
- **Venues:** 1 unique
  - Primary: 常　滑 (1,315 races)

## Factor Coverage Overview

| Factor | Status | Coverage | Available | Missing |
|--------|--------|----------|-----------|---------|
| Race Environment Factor | ✅ AVAILABLE | 80.0% | 8 | 2 |
| Frame & Course Factor | ✅ AVAILABLE | 100.0% | 78 | 0 |
| Racer Skill Factor | ✅ AVAILABLE | 100.0% | 60 | 0 |
| Motor Performance Factor | ✅ AVAILABLE | 100.0% | 18 | 0 |
| Boat Factor | ✅ AVAILABLE | 100.0% | 30 | 0 |
| Start Timing Factor | ✅ AVAILABLE | 99.1% | 113 | 1 |
| Race Scenario Factor | ✅ AVAILABLE | 100.0% | 246 | 0 |
| Risk & Reliability Factor | ✅ AVAILABLE | 100.0% | 24 | 0 |
| Odds & Value Factor | ✅ AVAILABLE | 100.0% | 7 | 0 |

## Detailed Factor Analysis

### Race Environment Factor

**Status:** AVAILABLE

**What it means:** Weather, water, and venue conditions that affect all racers equally

**Why it matters:** Strong wind/waves favor experienced racers who can handle rough conditions. Calm weather favors raw speed.

**Available Columns (8):**

- `weather`
- `temperature`
- `water_temperature`
- `wind_speed`
- `windDir`
- `windPow`
- `wave_height`
- `waveHight`

**Missing Columns (2):**

- `weather_x`
- `weather_y`

**Impact of Missing Columns:**


---

### Frame & Course Factor

**Status:** AVAILABLE

**What it means:** Historical bias for inside/outside lanes at this venue

**Why it matters:** Tokoname has structural bias. Inside lanes (1-2) may have shorter path to first turn. This overrides racer skill in some races.

**Available Columns (78):**

- `CS_cource_1_1`
- `CS_cource_1_2`
- `CS_cource_1_3`
- `CS_cource_1_4`
- `CS_cource_1_5`
- `CS_cource_1_6`
- `CS_cource_1_7`
- `CS_cource_1_8`
- `CS_cource_1_9`
- `CS_cource_1_10`
- ... and 68 more

---

### Racer Skill Factor

**Status:** AVAILABLE

**What it means:** Racer's historical win rates, place rates, and consistency

**Why it matters:** Top-tier racers (A1 class) dominate. But venue-specific experience matters—local specialists can upset national stars.

**Available Columns (60):**

- `win_rate_national_1`
- `win_rate_national_2`
- `win_rate_national_3`
- `win_rate_national_4`
- `win_rate_national_5`
- `win_rate_national_6`
- `place2Ratio_national_1`
- `place2Ratio_national_2`
- `place2Ratio_national_3`
- `place2Ratio_national_4`
- ... and 50 more

---

### Motor Performance Factor

**Status:** AVAILABLE

**What it means:** Engine quality based on recent performance stats

**Why it matters:** Motors are randomly assigned but vary significantly. A 'hot' motor (high place2Ratio) can give a weaker racer an edge.

**Available Columns (18):**

- `motorNo_1`
- `motorNo_2`
- `motorNo_3`
- `motorNo_4`
- `motorNo_5`
- `motorNo_6`
- `motor_place2Ratio_1`
- `motor_place2Ratio_2`
- `motor_place2Ratio_3`
- `motor_place2Ratio_4`
- ... and 8 more

---

### Boat Factor

**Status:** AVAILABLE

**What it means:** Hull performance and racer-boat fit

**Why it matters:** Lighter racers get speed advantage. Propeller tilt affects turning. Boat condition varies by maintenance.

**Available Columns (30):**

- `boatNo_1`
- `boatNo_2`
- `boatNo_3`
- `boatNo_4`
- `boatNo_5`
- `boatNo_6`
- `boat_place2Ratio_1`
- `boat_place2Ratio_2`
- `boat_place2Ratio_3`
- `boat_place2Ratio_4`
- ... and 20 more

---

### Start Timing Factor

**Status:** AVAILABLE

**What it means:** Racer's starting technique and timing consistency

**Why it matters:** CRITICAL. A 0.15s difference at start can determine race outcome. Aggressive starters risk penalties but gain position.

**Available Columns (113):**

- `ave_start_time_1`
- `ave_start_time_2`
- `ave_start_time_3`
- `ave_start_time_4`
- `ave_start_time_5`
- `ave_start_time_6`
- `num_false_start_1`
- `num_false_start_2`
- `num_false_start_3`
- `num_false_start_4`
- ... and 103 more

**Missing Columns (1):**

- `exhibition_flag_6`

**Impact of Missing Columns:**


---

### Race Scenario Factor

**Status:** AVAILABLE

**What it means:** In-race positioning, turn strategy, and historical race development patterns

**Why it matters:** Early position is huge but not destiny. Some racers excel at late-race overtakes. First-turn strategy can make/break a race.

**Available Columns (246):**

- `CS_race_1_1`
- `CS_race_1_2`
- `CS_race_1_3`
- `CS_race_1_4`
- `CS_race_1_5`
- `CS_race_1_6`
- `CS_race_1_7`
- `CS_race_1_8`
- `CS_race_1_9`
- `CS_race_1_10`
- ... and 236 more

---

### Risk & Reliability Factor

**Status:** AVAILABLE

**What it means:** Racer's penalty history, disqualification risk, and consistency

**Why it matters:** High-risk racers may have great speed but unreliable results. Conservative racers finish consistently but rarely win.

**Available Columns (24):**

- `num_false_start_1`
- `num_false_start_2`
- `num_false_start_3`
- `num_false_start_4`
- `num_false_start_5`
- `num_false_start_6`
- `num_late_start_1`
- `num_late_start_2`
- `num_late_start_3`
- `num_late_start_4`
- ... and 14 more

---

### Odds & Value Factor

**Status:** AVAILABLE

**What it means:** Betting market sentiment and value opportunities from trifecta (三連単) odds

**Why it matters:** Trifecta odds reflect crowd wisdom about finish order. Low odds = high confidence, high variance = uncertain race. Our model's deviations from market consensus can identify value bets.

**Available Columns (7):**

- `1-2-3`
- `1-2-4`
- `1-2-5`
- `1-2-6`
- `1-3-2`
- `2-1-3`
- `3-1-2`

---

## Sample Derived Features

These are computed features built from raw columns:

```
     derived_wind_resistance_score  derived_rough_water_score  derived_lane_win_rate_1  derived_lane_win_rate_2  derived_lane_win_rate_3  derived_lane_win_rate_4  derived_lane_win_rate_5  derived_lane_win_rate_6  derived_skill_gap_1  derived_skill_gap_2  derived_skill_gap_3  derived_skill_gap_4  derived_skill_gap_5  derived_skill_gap_6  derived_motor_gap_1  derived_motor_gap_2  derived_motor_gap_3  derived_motor_gap_4  derived_motor_gap_5  derived_motor_gap_6  derived_start_consistency_1  derived_start_consistency_2  derived_start_consistency_3  derived_start_consistency_4  derived_start_consistency_5  derived_start_consistency_6
98                             1.8                       0.04                      5.5                      3.0                      4.0                      6.0                      4.0                      6.0             0.715000            -0.315000             1.885000             0.055000             0.835000            -3.175000            -0.036667             6.623333            -0.036667             1.863333            -6.706667            -1.706667                     0.098995                          NaN                     0.000000                          NaN                          NaN                     0.028284
99                             1.8                       0.04                      3.0                      1.0                      5.5                      2.5                      2.0                      2.5             0.346667             0.466667             0.986667            -0.363333            -1.753333             0.316667            -3.571667            13.098333           -29.211667            -3.571667            13.098333            10.158333                     0.028284                          NaN                     0.070711                     0.014142                          NaN                     0.007071
100                            1.8                       0.04                      5.0                      4.5                      6.0                      4.5                      2.5                      1.0            -0.001667             1.518333            -0.911667            -0.821667             0.408333            -0.191667            15.050000           -23.990000             5.470000             8.780000             1.720000            -7.030000                          NaN                     0.014142                          NaN                     0.021213                     0.042426                          NaN
101                            1.2                       0.02                      3.0                      1.0                      2.0                      3.0                      2.5                      1.0             1.418333            -0.291667             0.938333            -0.581667            -1.791667             0.308333            16.175000           -13.235000            19.855000            -7.355000            12.915000           -28.355000                          NaN                          NaN                          NaN                     0.049497                     0.035355                          NaN
102                            1.2                       0.02                      4.5                      3.5                      5.0                      6.0                      2.0                      1.5            -0.216667             0.453333             1.023333            -2.266667            -1.396667             2.403333            -2.895000           -25.115000             6.465000            -8.445000           -17.975000            47.965000                     0.042426                     0.021213                          NaN                     0.070711                     0.014142                     0.014142
```

_Showing 5 rows of 26 derived features_

## Next Steps: Phase 3 Prerequisites

Before proceeding to Phase 3 (modeling), ensure:

1. ✅ **No BLOCKED factors in core set (1-8)**
   - All core factors must have at least partial coverage
2. ✅ **At least 1000 rows of Tokoname data**
   - Needed for reliable training/validation split
3. ✅ **Date range covers multiple months**
   - Avoid seasonal bias from short time windows
4. ⚠️ **Review missing columns**
   - If critical columns missing, update data pipeline first

### ✅ Phase 2 Status: READY FOR PHASE 3

All prerequisites met. You may proceed to model training.

## Troubleshooting Common Issues

### Issue 1: All Factors Blocked
**Symptom:** Every factor shows 0% coverage

**Fix:**
- Check `factors_config.yaml` column names match your dataframe
- Run `print(df.columns.tolist())` to see actual column names
- Update `column_normalization.aliases` in config

### Issue 2: Low Coverage (<50%)
**Symptom:** Factor shows partial coverage

**Fix:**
- Check if data pipeline is loading all CSV files
- Verify beforeinfo/*.csv files contain expected columns
- Check for column name variations (e.g., 'weather' vs 'weather_x')

### Issue 3: No Derived Features
**Symptom:** Sample derived features section is empty

**Fix:**
- Derived features require specific raw columns to exist
- Check `factor_binder.py` logs for skip messages
- If columns exist but feature not computed, check logic in `_compute_*` methods

### Issue 4: Venue Mismatch
**Symptom:** 0 rows after filtering for Tokoname

**Fix:**
- Check venue normalization: '常　滑' vs '常滑'
- Print unique venues: `df['venue'].unique()`
- Ensure loader.py venue cleaning is working

---

**End of Report**