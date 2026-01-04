"""
PHASE 2: Factor Binding Entry Point
=====================================
Run ONLY Phase 2: Load data, bind factors, generate report.
NO MODEL TRAINING.

This script enforces the Master Directive's phase-by-phase approach.
Phase 2 must complete successfully before Phase 3 (modeling) begins.

Usage:
    python run_phase2.py --venue 常滑
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "boatrace-master" / "src"))

import pandas as pd
from src.factors.factor_binder import FactorBinder
from data_preparing import loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Factor × Feature Binding (NO TRAINING)"
    )
    parser.add_argument(
        "--venue",
        default="常滑",
        help="Target venue (default: 常滑 = Tokoname)"
    )
    parser.add_argument(
        "--config",
        default="factors_config.yaml",
        help="Path to factors configuration YAML"
    )
    parser.add_argument(
        "--output-report",
        default="reports/phase2_factor_binding_report.md",
        help="Output path for markdown report"
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Optional: Save factor-bound dataframe to CSV"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("PHASE 2: FACTOR × FEATURE BINDING")
    print("=" * 70)
    print("Philosophy: Interpretability-first. No black boxes.")
    print("Deliverables: Factor mapping + coverage report + validation")
    print("=" * 70 + "\n")
    
    # Step 1: Load data (with CSV enrichment from beforeinfo + racelist)
    logger.info("[STEP 1/4] Loading race data (TXT + CSV enrichment)...")
    df = loader.make_race_result_df()
    
    if df is None or len(df) == 0:
        logger.error("❌ FAILED: No data loaded")
        logger.error("Troubleshooting:")
        logger.error("  1. Ensure boatrace-master/data/results_race/*.TXT files exist")
        logger.error("  2. Run: python boatrace-master/src/data_preparing/downloader_race_results.py")
        sys.exit(1)
    
    # Filter for target venue
    logger.info(f"   Filtering for venue: {args.venue}")
    if 'venue_clean' in df.columns:
        df = df[df['venue_clean'] == args.venue].copy()
    elif 'venue' in df.columns:
        df = df[df['venue'] == args.venue].copy()
    else:
        logger.error("❌ FAILED: No 'venue' or 'venue_clean' column found")
        sys.exit(1)
    
    if len(df) == 0:
        logger.error(f"❌ FAILED: No data for venue '{args.venue}'")
        logger.error("Troubleshooting:")
        logger.error("  1. Check venue name (常滑 vs 常　滑)")
        logger.error("  2. Print unique venues: df['venue_clean'].unique()")
        sys.exit(1)
    
    logger.info(f"✅ Loaded {len(df):,} rows")
    logger.info(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"   Total columns: {len(df.columns):,}")
    
    # Step 2: Initialize factor binder
    logger.info(f"[STEP 2/4] Loading factor configuration: {args.config}")
    
    config_path = project_root / args.config
    if not config_path.exists():
        logger.error(f"❌ FAILED: Config not found: {config_path}")
        sys.exit(1)
    
    binder = FactorBinder(str(config_path))
    logger.info(f"✅ Loaded {len(binder.factor_names)} factor definitions")
    
    # Step 3: Bind factors
    logger.info("[STEP 3/4] Binding raw columns to factor buckets...")
    
    try:
        binding_result = binder.bind(df)
    except Exception as e:
        logger.error(f"❌ FAILED: Factor binding error: {e}")
        logger.error("Check logs above for details")
        sys.exit(1)
    
    factor_df = binding_result["factor_df"]
    factor_summary = binding_result["factor_summary"]
    warnings = binding_result["warnings"]
    
    logger.info(f"✅ Binding complete")
    
    # Print warnings
    if warnings:
        logger.warning(f"⚠️ {len(warnings)} warnings detected:")
        for warning in warnings:
            logger.warning(f"   {warning}")
    
    # Print factor coverage summary
    print("\n" + "-" * 70)
    print("FACTOR COVERAGE SUMMARY")
    print("-" * 70)
    
    for factor_name in binder.factor_names:
        summary = factor_summary[factor_name]
        status_icon = {
            "available": "✅",
            "partial": "⚠️",
            "blocked": "❌"
        }[summary["status"]]
        
        print(
            f"{status_icon} {summary['factor_name']:35s} "
            f"{summary['coverage_rate']:>6.1%}  "
            f"({len(summary['available_columns'])}/{summary['total_columns']} cols)"
        )
    
    print("-" * 70 + "\n")
    
    # Step 4: Generate report
    logger.info(f"[STEP 4/4] Generating markdown report: {args.output_report}")
    
    # Import report generator
    from scripts.phase2_factor_report import generate_report
    
    os.makedirs(os.path.dirname(args.output_report), exist_ok=True)
    
    generate_report(
        df=df,
        binder=binder,
        binding_result=binding_result,
        output_path=args.output_report
    )
    
    logger.info(f"✅ Report saved to: {args.output_report}")
    
    # Optional: Save CSV
    if args.save_csv:
        logger.info(f"[OPTIONAL] Saving factor-bound dataframe to CSV: {args.save_csv}")
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        factor_df.to_csv(args.save_csv, index=False, encoding='utf-8-sig')
        logger.info(f"✅ CSV saved")
    
    # Final status check
    print("\n" + "=" * 70)
    print("PHASE 2 COMPLETION CHECK")
    print("=" * 70)
    
    blocked_core = sum(
        1 for fname in binder.factor_names[:8]
        if factor_summary[fname]["status"] == "blocked"
    )
    
    if blocked_core == 0 and len(df) >= 1000:
        print("✅ PHASE 2 COMPLETE: READY FOR PHASE 3")
        print("")
        print("All prerequisites met:")
        print("  ✅ No core factors blocked")
        print(f"  ✅ Sufficient data ({len(df):,} rows)")
        print(f"  ✅ Date range: {df['date'].min()} to {df['date'].max()}")
        print("")
        print("Next step: Proceed to Phase 3 (modeling)")
    else:
        print("⚠️ PHASE 2 INCOMPLETE: NOT READY FOR PHASE 3")
        print("")
        if blocked_core > 0:
            print(f"  ❌ {blocked_core} core factors are BLOCKED")
        if len(df) < 1000:
            print(f"  ❌ Insufficient data ({len(df)} rows, need 1000+)")
        print("")
        print("Action required:")
        print("  1. Review factor binding report")
        print("  2. Fix data pipeline or update factors_config.yaml")
        print("  3. Re-run Phase 2")
    
    print("=" * 70 + "\n")
    
    print(f"📄 Full report: {args.output_report}")
    print("")
    
    # Exit with appropriate code
    if blocked_core > 0 or len(df) < 1000:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
