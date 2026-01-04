# Tokoname Factors (Interpretability-First)

Scope: Phase 2 (Factor Design) — no ML, no betting logic, no probabilities.
All factors must be explainable in one sentence, have direction (positive/neutral/negative), and expose coverage/confidence.

## Common conventions
- Venue: Tokoname only.
- Windows: use recent data first; widen only if coverage is insufficient.
- Coverage: if a factor has too few samples, surface "low-confidence" and downgrade to Neutral.
- Bands: Positive / Neutral / Negative (no numeric probability). Optionally include a small integer score (e.g., +1/0/-1) for internal sorting only.

## Factors

### 1) Course / Venue Bias
- Question: Is Tokoname currently favoring inside lanes?
- Direction: Inside-biased → Positive for lanes 1–3, Negative for 4–6; outside-biased → invert.
- Inputs: Race results (finish by lane), Tokoname only, rolling 30–60 days (fallback 90).
- Computation: Win/place rates by lane vs long-term Tokoname baseline; compute % lift or z-score.
- Coverage rule: need ≥50 races in window; if not, extend horizon to 90 days; if still <50 → Neutral/low-confidence.

### 2) Start Timing Stability (per racer, venue-aware)
- Question: Does this racer launch consistently at Tokoname?
- Direction: Low ST variance and low late/false rate → Positive; high variance/late rate → Negative.
- Inputs: Start timing (ST), late/false indicators, racer ID, Tokoname filter; window 15–30 Tokoname starts.
- Computation: Mean ST, ST stddev, late-rate vs fleet median at Tokoname.
- Coverage rule: need ≥8 Tokoname starts; otherwise Neutral with "low data" note.

### 3) Motor Performance (current meet)
- Question: Is this motor performing above the current meet average?
- Direction: Better recent placements for this motor → Positive; poor/volatile → Negative.
- Inputs: Motor ID, finishes during current meet (or last 10 days at Tokoname), race class if available.
- Computation: Average finish rank for the motor vs meet median; optionally include start success rate per motor.
- Coverage rule: need ≥6 uses in the meet; otherwise Neutral.

### 4) Racer Consistency (recent Tokoname form)
- Question: Is this racer finishing in a tight band recently at Tokoname?
- Direction: Lower variance and better median finish → Positive; erratic or back-marker → Negative.
- Inputs: Racer ID, finish positions at Tokoname; window 10–20 races.
- Computation: Median finish, IQR/variance vs Tokoname fleet distribution.
- Coverage rule: need ≥8 races; otherwise Neutral.

### 5) Weather Suitability (racer × conditions)
- Question: Do today’s conditions match this racer’s past Tokoname performance under similar weather?
- Direction: Good historical outcomes in similar wind/wave → Positive; poor outcomes → Negative; unknown → Neutral.
- Inputs: Weather (weather, windDir, windPow, waveHeight) from TXT, racer finishes in similar bins: wave (0–1, 2–3, 4+), wind speed (0–3, 4–6, 7+), wind dir (head/tail/cross bucketed per course heading).
- Computation: Conditional finish outcomes by bin vs racer’s overall Tokoname outcomes.
- Coverage rule: need ≥5 races in similar conditions; otherwise Neutral.

### 6) Field Imbalance (lane strength)
- Question: Is the field skewed toward inside or outside strength?
- Direction: If lanes 1–3 aggregate stronger (consistency + motor + start), inside favored → Positive for inside; otherwise invert.
- Inputs: Per-lane subfactors: Start Stability, Motor Performance, Racer Consistency, Course Bias.
- Computation: Normalize subfactor scores per lane; sum lanes 1–3 vs 4–6; report difference and sign.
- Coverage rule: if any lane missing key subfactor (e.g., motor uses <6), downweight that component and surface "partial data" note.

## Reporting guidance
- Show only top 3–5 factors per race.
- Always display coverage/confidence notes (e.g., "low data: ST n=6" or "extended window used").
- Keep outputs descriptive, not prescriptive; no betting suggestions.
- If a factor lacks coverage, mark Neutral and explain why.

## Next steps (when ready)
- Implement a CLI to compute these factors from the stabilized Tokoname dataset, with logging of coverage per factor.
- Keep outputs as Positive/Neutral/Negative (and optional small integer bands) plus a short explanation per factor.
- Do not introduce ML or probabilities until a later phase.
