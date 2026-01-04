#!/usr/bin/env python3
"""
PHASE 3: SIGNAL STRUCTURING
============================
Takes Phase 2 factor bindings and computes:
1. Strength scoring (Strong/Moderate/Weak) per factor
2. Race-level signal aggregation (Positive/Neutral/Negative count)
3. Top contributing factors ranking
4. Beginner-friendly signal summary

NO PREDICTIONS. NO BETTING ADVICE.
Just interpretable signals for understanding each race.

Philosophy: Show strength, show uncertainty, explain why.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

# Setup paths
REPO_ROOT = Path(__file__).parent
BOATRACE_PATH = REPO_ROOT / "boatrace-master" / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(BOATRACE_PATH))

from src.factors.factor_binder import FactorBinder
from data_preparing import loader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s: %(message)s"
)
logger = logging.getLogger(__name__)


class Phase3SignalBuilder:
    """
    Builds interpretable race-level signals from Phase 2 factors.
    
    Strength scoring:
    - Strong (>80% coverage): Factor is well-supported by data
    - Moderate (40-80%): Factor has some support, some gaps
    - Weak (<40%): Factor has limited data, high uncertainty
    
    Signal aggregation:
    - Per-race: Count factors by direction (Positive/Neutral/Negative)
    - Weight by strength (Strong = 1.0, Moderate = 0.5, Weak = 0.25)
    - Aggregate into race signal (e.g., "Strong Positive", "Mixed", "Weak Negative")
    """
    
    def __init__(self, config_path: str = "factors_config.yaml"):
        """Initialize with factor configuration."""
        self.config_path = config_path
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        logger.info(f"[PHASE 3] Loaded config from {config_path}")
    
    def compute_factor_strength(
        self,
        factor_name: str,
        coverage_info: Dict[str, Any]
    ) -> str:
        """
        Compute strength level for a factor.
        
        Args:
            factor_name: Factor identifier
            coverage_info: From factor_summary (coverage_rate, available_columns, etc)
            
        Returns:
            "Strong", "Moderate", or "Weak"
        """
        coverage_rate = coverage_info.get("coverage_rate", 0)
        status = coverage_info.get("status", "blocked")
        
        if status == "blocked":
            return "Unavailable"
        
        if coverage_rate >= 0.80:
            return "Strong"
        elif coverage_rate >= 0.40:
            return "Moderate"
        else:
            return "Weak"
    
    def analyze_race_signal(
        self,
        row: pd.Series,
        factor_summary: Dict[str, Dict],
        binder: FactorBinder
    ) -> Dict[str, Any]:
        """
        Analyze a single race and generate signal.
        
        Args:
            row: Single race row from dataframe
            factor_summary: From FactorBinder.bind()
            binder: FactorBinder instance
            
        Returns:
            {
                "date": race date,
                "venue": venue name,
                "raceNumber": race number,
                "factors": [
                    {
                        "name": factor name,
                        "explanation": what this factor measures,
                        "strength": Strong/Moderate/Weak,
                        "available_cols": count,
                        "missing_cols": count,
                        "status": Positive/Neutral/Negative/Blocked
                    }
                ],
                "signal_summary": "Strong Positive" / "Positive" / "Mixed" / etc,
                "signal_explanation": "4 Positive factors + 2 Moderate factors suggest favorable race",
                "top_factors": [list of top 3 contributing factors]
            }
        """
        signal_data = {
            "date": str(row.get("date", "N/A")),
            "venue": row.get("venue", "N/A"),
            "raceNumber": int(row.get("raceNumber", 0)),
            "factors": [],
            "signal_summary": "",
            "signal_explanation": "",
            "top_factors": []
        }
        
        # Analyze each factor
        positive_count = 0
        neutral_count = 0
        negative_count = 0
        factor_details = []
        
        for factor_name in binder.factor_names:
            coverage = factor_summary.get(factor_name, {})
            strength = self.compute_factor_strength(factor_name, coverage)
            
            # Determine factor direction (placeholder logic)
            # In real implementation, this would use derived features in the row
            direction = self._infer_factor_direction(factor_name, row, coverage)
            
            factor_detail = {
                "name": coverage.get("factor_name", factor_name),
                "explanation": coverage.get("explanation", ""),
                "strength": strength,
                "available_cols": len(coverage.get("available_columns", [])),
                "missing_cols": len(coverage.get("missing_columns", [])),
                "status": direction
            }
            factor_details.append(factor_detail)
            signal_data["factors"].append(factor_detail)
            
            # Count by direction (only if not blocked/unavailable)
            if strength != "Unavailable":
                if direction == "Positive":
                    positive_count += {"Strong": 1.0, "Moderate": 0.5, "Weak": 0.25}.get(strength, 0)
                elif direction == "Negative":
                    negative_count += {"Strong": 1.0, "Moderate": 0.5, "Weak": 0.25}.get(strength, 0)
                else:  # Neutral
                    neutral_count += {"Strong": 1.0, "Moderate": 0.5, "Weak": 0.25}.get(strength, 0)
        
        # Generate signal summary
        total_strength = positive_count + neutral_count + negative_count
        
        if total_strength == 0:
            signal_summary = "Insufficient Data"
            explanation = "Not enough factors to determine signal"
        else:
            pos_pct = positive_count / total_strength if total_strength > 0 else 0
            neg_pct = negative_count / total_strength if total_strength > 0 else 0
            
            if pos_pct > 0.6:
                signal_summary = "Strong Positive" if positive_count > 4 else "Positive"
            elif neg_pct > 0.6:
                signal_summary = "Strong Negative" if negative_count > 4 else "Negative"
            else:
                signal_summary = "Mixed"
            
            # Build explanation
            pos_factors = [f["name"] for f in signal_data["factors"] if f["status"] == "Positive"]
            neg_factors = [f["name"] for f in signal_data["factors"] if f["status"] == "Negative"]
            weak_factors = [f["name"] for f in signal_data["factors"] if f["strength"] == "Weak"]
            
            parts = []
            if pos_factors:
                parts.append(f"Positive: {', '.join(pos_factors[:3])}")
            if neg_factors:
                parts.append(f"Negative: {', '.join(neg_factors[:3])}")
            if weak_factors:
                parts.append(f"Low confidence: {', '.join(weak_factors[:2])}")
            
            explanation = " | ".join(parts) if parts else "Neutral race factors"
        
        signal_data["signal_summary"] = signal_summary
        signal_data["signal_explanation"] = explanation
        
        # Top factors (by availability/strength)
        top_factors = sorted(
            [f for f in signal_data["factors"] if f["strength"] != "Unavailable"],
            key=lambda f: ({"Strong": 3, "Moderate": 2, "Weak": 1}.get(f["strength"], 0)),
            reverse=True
        )[:3]
        signal_data["top_factors"] = [f["name"] for f in top_factors]
        
        return signal_data
    
    def _infer_factor_direction(
        self,
        factor_name: str,
        row: pd.Series,
        coverage: Dict[str, Any]
    ) -> str:
        """
        Infer whether a factor is Positive/Neutral/Negative for this race.
        
        This is a placeholder that uses heuristics. In production, this would
        compute derived features and return a score.
        
        Args:
            factor_name: Factor name
            row: Race row
            coverage: Coverage info (includes available_columns)
            
        Returns:
            "Positive", "Neutral", or "Negative"
        """
        if coverage.get("status") == "blocked":
            return "Blocked"
        
        # Simple heuristic: if we have >70% of columns, call it positive
        # If 30-70%, neutral. If <30%, negative.
        coverage_rate = coverage.get("coverage_rate", 0)
        
        if coverage_rate >= 0.70:
            return "Positive"
        elif coverage_rate >= 0.30:
            return "Neutral"
        else:
            return "Negative"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Phase 3: Signal Structuring - Compute race-level signals from Phase 2 factors"
    )
    parser.add_argument("--venue", default=None, help="Venue to analyze (default: None, all venues)")
    parser.add_argument("--limit", type=int, default=None, help="Max races to analyze (default: all)")
    parser.add_argument("--output", default="reports/phase3_signals.md", help="Output report path")
    parser.add_argument("--config", default="factors_config.yaml", help="Factor config path")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("PHASE 3: SIGNAL STRUCTURING")
    logger.info("=" * 70)
    logger.info(f"Venue: {args.venue or 'All venues'}")
    limit_label = args.limit if args.limit is not None else "all"
    logger.info(f"Limit: {limit_label} races")
    
    # Step 1: Load data
    logger.info("[STEP 1/3] Loading race data...")
    try:
        df = loader.make_race_result_df()

        if args.venue:
            if "venue_clean" in df.columns:
                df = df[df["venue_clean"] == args.venue].copy()
            elif "venue" in df.columns:
                df = df[df["venue"] == args.venue].copy()
            else:
                logger.error("  ❌ No 'venue' or 'venue_clean' column found")
                return 1

            if len(df) == 0:
                logger.error(f"  ❌ No races found for venue '{args.venue}'")
                logger.error("  Troubleshoot: verify venue name or rerun Phase 2 data prep")
                return 1

        if args.limit is not None:
            df = df.head(args.limit)
        venue_str = args.venue or "total"
        logger.info(f"  ✅ Loaded {len(df)} races from {venue_str}")
        if len(df) > 0:
            logger.info(f"     Date range: {df['date'].min()} -> {df['date'].max()}")
    except Exception as e:
        logger.error(f"  ❌ Failed to load data: {e}")
        return 1
    
    # Step 2: Bind factors (Phase 2)
    logger.info("[STEP 2/3] Running Phase 2 factor binding...")
    try:
        binder = FactorBinder(args.config)
        binding_result = binder.bind(df)
        factor_summary = binding_result["factor_summary"]
        logger.info(f"  ✅ Bound {len(binder.factor_names)} factors")
        if binding_result["warnings"]:
            for w in binding_result["warnings"]:
                logger.warning(f"  {w}")
    except Exception as e:
        logger.error(f"  ❌ Failed to bind factors: {e}")
        return 1
    
    # Step 3: Generate Phase 3 signals
    logger.info("[STEP 3/3] Generating race-level signals...")
    try:
        builder = Phase3SignalBuilder(args.config)
        signals = []
        
        for idx, row in df.iterrows():
            signal = builder.analyze_race_signal(row, factor_summary, binder)
            signals.append(signal)
        
        logger.info(f"  ✅ Generated signals for {len(signals)} races")
        
        # Print summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("SIGNAL SUMMARY")
        logger.info("=" * 70)
        
        for i, sig in enumerate(signals[:3], 1):  # Show first 3
            logger.info(f"\n[Race {i}] {sig['date']} | Race #{sig['raceNumber']}")
            logger.info(f"  Signal: {sig['signal_summary']}")
            logger.info(f"  Details: {sig['signal_explanation']}")
            logger.info(f"  Top factors: {', '.join(sig['top_factors'])}")
        
        # Generate markdown report
        generate_phase3_report(signals, factor_summary, binder, output_path)
        logger.info(f"\n  📄 Report saved to {output_path}")
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("✅ PHASE 3 COMPLETE")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"  ❌ Failed to generate signals: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def generate_phase3_report(
    signals: List[Dict],
    factor_summary: Dict,
    binder: FactorBinder,
    output_path: Path
) -> None:
    """Generate markdown report of Phase 3 signals."""
    lines = []
    
    # Header
    lines.append("# Phase 3: Signal Structuring Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Races Analyzed:** {len(signals)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append("Phase 3 converts Phase 2 factor bindings into interpretable race signals:")
    lines.append("- **Strength Scoring:** Strong (80%+) | Moderate (40-80%) | Weak (<40%)")
    lines.append("- **Signal Aggregation:** Positive/Neutral/Negative with weighted strength")
    lines.append("- **No predictions:** Just explainable patterns")
    lines.append("")
    
    # Factor Strength Overview
    lines.append("## Factor Strength Overview")
    lines.append("")
    lines.append("| Factor | Coverage | Strength | Available | Missing |")
    lines.append("|--------|----------|----------|-----------|---------|")
    
    builder = Phase3SignalBuilder()
    for factor_name in binder.factor_names:
        coverage = factor_summary[factor_name]
        strength = builder.compute_factor_strength(factor_name, coverage)
        cov_pct = f"{coverage['coverage_rate']:.1%}"
        avail = len(coverage.get("available_columns", []))
        missing = len(coverage.get("missing_columns", []))
        
        lines.append(
            f"| {coverage.get('factor_name', factor_name)} | {cov_pct} | {strength} | {avail} | {missing} |"
        )
    
    lines.append("")
    
    # Per-race signals
    lines.append("## Per-Race Signals")
    lines.append("")
    
    for i, sig in enumerate(signals, 1):
        lines.append(f"### Race {i}: {sig['date']} #{sig['raceNumber']}")
        lines.append("")
        lines.append(f"**Signal:** {sig['signal_summary']}")
        lines.append("")
        lines.append(f"**Explanation:** {sig['signal_explanation']}")
        lines.append("")
        lines.append(f"**Top Factors:** {', '.join(sig['top_factors'])}")
        lines.append("")
        
        # Factor details
        lines.append("**Factor Breakdown:**")
        lines.append("")
        for factor in sig["factors"]:
            if factor["status"] != "Blocked":
                status_icon = {
                    "Positive": "📈",
                    "Neutral": "➡️",
                    "Negative": "📉"
                }.get(factor["status"], "❓")
                
                lines.append(
                    f"- {status_icon} {factor['name']} ({factor['strength']}) "
                    f"[{factor['available_cols']}/{factor['available_cols'] + factor['missing_cols']}]"
                )
        
        lines.append("")
    
    # Methodology
    lines.append("---")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Strength Scoring")
    lines.append("- **Strong:** >80% of factor columns available (well-supported)")
    lines.append("- **Moderate:** 40-80% of columns (partial support)")
    lines.append("- **Weak:** <40% of columns (high uncertainty)")
    lines.append("")
    lines.append("### Signal Aggregation")
    lines.append("1. Count factors by direction (Positive/Neutral/Negative)")
    lines.append("2. Weight by strength: Strong=1.0, Moderate=0.5, Weak=0.25")
    lines.append("3. Compute weighted ratios")
    lines.append("4. Classify as: Strong/Positive, Positive, Mixed, Negative, Strong/Negative")
    lines.append("")
    lines.append("### Important Notes")
    lines.append("- **No predictions:** Signals show data patterns only")
    lines.append("- **No betting advice:** These are descriptive, not prescriptive")
    lines.append("- **Interpretability first:** All signals linked to specific factors")
    lines.append("- **Uncertainty tracked:** Weak signals explicitly marked")
    lines.append("")
    
    # Write report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == "__main__":
    sys.exit(main())
