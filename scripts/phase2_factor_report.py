"""
PHASE 2: Factor Binding Report Generator
==========================================
Generates a beginner-friendly markdown report showing factor coverage,
missing columns, and sample derived features.

This script validates that Phase 2 (Factor × Feature Binding) is complete
before proceeding to Phase 3 (modeling).
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "boatrace-master" / "src"))

import pandas as pd
from src.factors.factor_binder import FactorBinder
from data_preparing import loader

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def generate_report(
    df: pd.DataFrame,
    binder: FactorBinder,
    binding_result: dict,
    output_path: str
) -> None:
    """
    Generate comprehensive Phase 2 markdown report.
    
    Args:
        df: Original dataframe
        binder: FactorBinder instance
        binding_result: Result from binder.bind()
        output_path: Path to save markdown report
    """
    factor_summary = binding_result["factor_summary"]
    warnings = binding_result["warnings"]
    factor_df = binding_result["factor_df"]
    
    # Prepare report sections
    report_lines = []
    
    # Header
    report_lines.append("# Phase 2: Factor × Feature Binding Report")
    report_lines.append("")
    report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"**Data Source:** Tokoname boatrace results")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")
    report_lines.append(f"- **Total Rows:** {len(df):,}")
    report_lines.append(f"- **Total Columns:** {len(df.columns):,}")
    report_lines.append(f"- **Date Range:** {df['date'].min()} to {df['date'].max()}")
    
    # Venue breakdown
    if 'venue' in df.columns:
        venue_counts = df['venue'].value_counts()
        report_lines.append(f"- **Venues:** {len(venue_counts)} unique")
        report_lines.append(f"  - Primary: {venue_counts.index[0]} ({venue_counts.iloc[0]:,} races)")
    
    report_lines.append("")
    
    # Factor Status Overview
    report_lines.append("## Factor Coverage Overview")
    report_lines.append("")
    report_lines.append("| Factor | Status | Coverage | Available | Missing |")
    report_lines.append("|--------|--------|----------|-----------|---------|")
    
    for factor_name in binder.factor_names:
        summary = factor_summary[factor_name]
        status_icon = {
            "available": "✅",
            "partial": "⚠️",
            "blocked": "❌"
        }[summary["status"]]
        
        coverage_pct = f"{summary['coverage_rate']:.1%}"
        available_count = len(summary["available_columns"])
        missing_count = len(summary["missing_columns"])
        
        report_lines.append(
            f"| {summary['factor_name']} | {status_icon} {summary['status'].upper()} | "
            f"{coverage_pct} | {available_count} | {missing_count} |"
        )
    
    report_lines.append("")
    
    # Warnings Section
    if warnings:
        report_lines.append("## ⚠️ Warnings")
        report_lines.append("")
        for warning in warnings:
            report_lines.append(f"- {warning}")
        report_lines.append("")
    
    # Detailed Factor Analysis
    report_lines.append("## Detailed Factor Analysis")
    report_lines.append("")
    
    for factor_name in binder.factor_names:
        summary = factor_summary[factor_name]
        
        report_lines.append(f"### {summary['factor_name']}")
        report_lines.append("")
        report_lines.append(f"**Status:** {summary['status'].upper()}")
        report_lines.append("")
        report_lines.append(f"**What it means:** {summary['explanation']}")
        report_lines.append("")
        report_lines.append(f"**Why it matters:** {summary['importance']}")
        report_lines.append("")
        
        # Available columns
        if summary['available_columns']:
            report_lines.append(f"**Available Columns ({len(summary['available_columns'])}):**")
            report_lines.append("")
            # Show first 10, then "..."
            shown_cols = summary['available_columns'][:10]
            for col in shown_cols:
                report_lines.append(f"- `{col}`")
            if len(summary['available_columns']) > 10:
                report_lines.append(f"- ... and {len(summary['available_columns']) - 10} more")
            report_lines.append("")
        
        # Missing columns
        if summary['missing_columns']:
            report_lines.append(f"**Missing Columns ({len(summary['missing_columns'])}):**")
            report_lines.append("")
            shown_missing = summary['missing_columns'][:10]
            for col in shown_missing:
                report_lines.append(f"- `{col}`")
            if len(summary['missing_columns']) > 10:
                report_lines.append(f"- ... and {len(summary['missing_columns']) - 10} more")
            report_lines.append("")
        
        # Impact of missing data
        if summary['missing_columns']:
            report_lines.append("**Impact of Missing Columns:**")
            report_lines.append("")
            if summary['status'] == 'blocked':
                report_lines.append(
                    "⛔ **BLOCKED**: This factor cannot be used in modeling. "
                    "All required columns are missing."
                )
            elif summary['status'] == 'partial':
                report_lines.append(
                    f"⚠️ **DEGRADED**: Only {summary['coverage_rate']:.1%} of columns available. "
                    "Factor will have reduced predictive power."
                )
            report_lines.append("")
        
        report_lines.append("---")
        report_lines.append("")
    
    # Sample Derived Features
    report_lines.append("## Sample Derived Features")
    report_lines.append("")
    report_lines.append("These are computed features built from raw columns:")
    report_lines.append("")
    
    # Find derived columns
    derived_cols = [col for col in factor_df.columns if col.startswith("derived_")]
    
    if derived_cols:
        sample_df = factor_df[derived_cols].head(5)
        report_lines.append("```")
        report_lines.append(sample_df.to_string())
        report_lines.append("```")
        report_lines.append("")
        report_lines.append(f"_Showing 5 rows of {len(derived_cols)} derived features_")
    else:
        report_lines.append("⚠️ No derived features computed. Check data availability.")
    
    report_lines.append("")
    
    # Next Steps
    report_lines.append("## Next Steps: Phase 3 Prerequisites")
    report_lines.append("")
    report_lines.append("Before proceeding to Phase 3 (modeling), ensure:")
    report_lines.append("")
    report_lines.append("1. ✅ **No BLOCKED factors in core set (1-8)**")
    report_lines.append("   - All core factors must have at least partial coverage")
    report_lines.append("2. ✅ **At least 1000 rows of Tokoname data**")
    report_lines.append("   - Needed for reliable training/validation split")
    report_lines.append("3. ✅ **Date range covers multiple months**")
    report_lines.append("   - Avoid seasonal bias from short time windows")
    report_lines.append("4. ⚠️ **Review missing columns**")
    report_lines.append("   - If critical columns missing, update data pipeline first")
    report_lines.append("")
    
    # Current status check
    blocked_core = sum(
        1 for fname in binder.factor_names[:8]
        if factor_summary[fname]["status"] == "blocked"
    )
    
    if blocked_core == 0 and len(df) >= 1000:
        report_lines.append("### ✅ Phase 2 Status: READY FOR PHASE 3")
        report_lines.append("")
        report_lines.append("All prerequisites met. You may proceed to model training.")
    else:
        report_lines.append("### ⚠️ Phase 2 Status: NOT READY")
        report_lines.append("")
        if blocked_core > 0:
            report_lines.append(f"- {blocked_core} core factors are BLOCKED")
        if len(df) < 1000:
            report_lines.append(f"- Only {len(df)} rows (need 1000+)")
        report_lines.append("")
        report_lines.append("**Action Required:** Fix data issues before training.")
    
    report_lines.append("")
    
    # Troubleshooting
    report_lines.append("## Troubleshooting Common Issues")
    report_lines.append("")
    report_lines.append("### Issue 1: All Factors Blocked")
    report_lines.append("**Symptom:** Every factor shows 0% coverage")
    report_lines.append("")
    report_lines.append("**Fix:**")
    report_lines.append("- Check `factors_config.yaml` column names match your dataframe")
    report_lines.append("- Run `print(df.columns.tolist())` to see actual column names")
    report_lines.append("- Update `column_normalization.aliases` in config")
    report_lines.append("")
    
    report_lines.append("### Issue 2: Low Coverage (<50%)")
    report_lines.append("**Symptom:** Factor shows partial coverage")
    report_lines.append("")
    report_lines.append("**Fix:**")
    report_lines.append("- Check if data pipeline is loading all CSV files")
    report_lines.append("- Verify beforeinfo/*.csv files contain expected columns")
    report_lines.append("- Check for column name variations (e.g., 'weather' vs 'weather_x')")
    report_lines.append("")
    
    report_lines.append("### Issue 3: No Derived Features")
    report_lines.append("**Symptom:** Sample derived features section is empty")
    report_lines.append("")
    report_lines.append("**Fix:**")
    report_lines.append("- Derived features require specific raw columns to exist")
    report_lines.append("- Check `factor_binder.py` logs for skip messages")
    report_lines.append("- If columns exist but feature not computed, check logic in `_compute_*` methods")
    report_lines.append("")
    
    report_lines.append("### Issue 4: Venue Mismatch")
    report_lines.append("**Symptom:** 0 rows after filtering for Tokoname")
    report_lines.append("")
    report_lines.append("**Fix:**")
    report_lines.append("- Check venue normalization: '常　滑' vs '常滑'")
    report_lines.append("- Print unique venues: `df['venue'].unique()`")
    report_lines.append("- Ensure loader.py venue cleaning is working")
    report_lines.append("")
    
    report_lines.append("---")
    report_lines.append("")
    report_lines.append("**End of Report**")
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))
    
    logger.info(f"[REPORT] Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate Phase 2 Factor Binding Report"
    )
    parser.add_argument(
        "--venue",
        default="常滑",
        help="Target venue (default: 常滑 = Tokoname)"
    )
    parser.add_argument(
        "--config",
        default="factors_config.yaml",
        help="Path to factors config YAML"
    )
    parser.add_argument(
        "--output",
        default="reports/phase2_factor_binding_report.md",
        help="Output markdown file path"
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info("[PHASE 2] Loading race data...")
    df = loader.load_race_results()
    
    if df is None or len(df) == 0:
        logger.error("❌ No data loaded")
        sys.exit(1)
    
    # Filter for target venue
    logger.info(f"[PHASE 2] Filtering for venue: {args.venue}")
    if 'venue_clean' in df.columns:
        df = df[df['venue_clean'] == args.venue].copy()
    elif 'venue' in df.columns:
        df = df[df['venue'] == args.venue].copy()
    else:
        logger.error("❌ No 'venue' or 'venue_clean' column found")
        sys.exit(1)
    
    if len(df) == 0:
        logger.error(f"❌ No data for venue: {args.venue}")
        sys.exit(1)
    
    logger.info(f"[PHASE 2] Loaded {len(df)} rows for {args.venue}")
    
    # Initialize factor binder
    config_path = Path(project_root) / args.config
    logger.info(f"[PHASE 2] Loading factor config from {config_path}")
    
    binder = FactorBinder(str(config_path))
    
    # Bind factors
    logger.info("[PHASE 2] Binding factors to dataframe...")
    binding_result = binder.bind(df)
    
    # Print warnings to console
    if binding_result["warnings"]:
        logger.warning("[PHASE 2] Warnings detected:")
        for warning in binding_result["warnings"]:
            logger.warning(f"  {warning}")
    
    # Generate report
    logger.info("[PHASE 2] Generating markdown report...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    generate_report(
        df=df,
        binder=binder,
        binding_result=binding_result,
        output_path=args.output
    )
    
    logger.info("=" * 60)
    logger.info("[PHASE 2] ✅ Report generation complete!")
    logger.info(f"[PHASE 2] 📄 View report: {args.output}")
    logger.info("=" * 60)
    
    # Print summary to console
    factor_summary = binding_result["factor_summary"]
    print("\n" + "=" * 60)
    print("FACTOR COVERAGE SUMMARY")
    print("=" * 60)
    
    for factor_name in binder.factor_names:
        summary = factor_summary[factor_name]
        status_icon = {
            "available": "✅",
            "partial": "⚠️",
            "blocked": "❌"
        }[summary["status"]]
        
        print(
            f"{status_icon} {summary['factor_name']:30s} "
            f"({summary['coverage_rate']:>6.1%})"
        )
    
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
