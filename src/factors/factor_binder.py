"""
PHASE 2: Factor Binder Module
===============================
This module loads factors_config.yaml and binds raw dataframe columns
to interpretable factor buckets.

Philosophy: Fail-fast, log everything, beginner-friendly.
No black boxes. Every mapping must be explicit and traceable.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import yaml

# Opt into pandas future downcasting behavior to avoid deprecation warnings from replace
pd.set_option("future.no_silent_downcasting", True)

logger = logging.getLogger(__name__)


class FactorBinder:
    """
    Binds raw dataframe columns to factor definitions from YAML config.
    
    Usage:
        binder = FactorBinder("factors_config.yaml")
        result = binder.bind(df)
        print(result["factor_summary"])
    """
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to factors_config.yaml
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.factor_names = [
            "race_environment",
            "frame_and_course",
            "racer_skill",
            "motor_performance",
            "boat",
            "start_timing",
            "race_scenario",
            "risk_and_reliability",
            "odds_and_value"
        ]
        
    def _load_config(self) -> dict:
        """Load and validate YAML configuration."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"[FACTOR BINDER] Loaded config from {self.config_path}")
        return config
    
    def _normalize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize column names to handle variations.
        
        Rules:
        - Strip whitespace
        - Convert full-width to half-width (basic)
        - Lowercase for comparison (but preserve original for display)
        """
        # Create a mapping of normalized -> original
        self.column_mapping = {}
        for col in df.columns:
            normalized = col.strip().replace('　', ' ').replace('  ', ' ')
            self.column_mapping[normalized] = col
        
        logger.debug(f"[NORMALIZE] Mapped {len(self.column_mapping)} columns")
        return df
    
    def _resolve_column(self, column_name: str, available_columns: List[str]) -> Optional[str]:
        """
        Resolve a column name using aliases and normalization.
        
        Args:
            column_name: Target column name from config
            available_columns: List of actual dataframe columns
            
        Returns:
            Matched column name or None
        """
        # Direct match
        if column_name in available_columns:
            return column_name
        
        # Check aliases
        aliases = self.config.get("column_normalization", {}).get("aliases", {})
        if column_name in aliases:
            for alias in aliases[column_name]:
                if alias in available_columns:
                    logger.debug(f"[ALIAS] {column_name} -> {alias}")
                    return alias
        
        # Try normalized match (case-insensitive, whitespace-stripped)
        normalized_target = column_name.strip().lower()
        for col in available_columns:
            if col.strip().lower() == normalized_target:
                return col
        
        return None
    
    def _get_factor_coverage(
        self,
        factor_name: str,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze coverage for a single factor.
        
        Returns:
            {
                "factor_id": int,
                "factor_name": str,
                "explanation": str,
                "available_columns": List[str],
                "missing_columns": List[str],
                "coverage_rate": float,
                "status": "available" | "partial" | "blocked"
            }
        """
        factor_config = self.config.get(factor_name, {})
        raw_columns = factor_config.get("raw_columns", [])
        available_columns = list(df.columns)
        
        found = []
        missing = []
        
        for col in raw_columns:
            resolved = self._resolve_column(col, available_columns)
            if resolved:
                found.append(resolved)
            else:
                missing.append(col)
        
        total = len(raw_columns)
        coverage_rate = len(found) / total if total > 0 else 0.0
        
        # Determine status
        if coverage_rate == 0:
            status = "blocked"
        elif coverage_rate < 0.5:
            status = "partial"
        else:
            status = "available"
        
        return {
            "factor_id": factor_config.get("id", 0),
            "factor_name": factor_config.get("name", factor_name),
            "explanation": factor_config.get("explanation", ""),
            "importance": factor_config.get("importance", ""),
            "available_columns": found,
            "missing_columns": missing,
            "total_columns": total,
            "coverage_rate": coverage_rate,
            "status": status
        }
    
    def _compute_derived_features(
        self,
        factor_name: str,
        df: pd.DataFrame,
        available_columns: List[str]
    ) -> pd.DataFrame:
        """
        Compute derived features for a factor.
        
        Aggregates new columns then concatenates once to avoid
        repeated insertions that fragment the DataFrame.
        """
        factor_config = self.config.get(factor_name, {})
        derived_features = factor_config.get("derived_features", [])
        derived_parts: List[pd.DataFrame] = []
        
        for feature in derived_features:
            feature_name = feature.get("name", "")
            required_cols = feature.get("requires", [])
            
            # Check if all required columns are available
            if not all(self._resolve_column(col, available_columns) for col in required_cols):
                logger.warning(
                    f"[DERIVED] Skipping {feature_name} - missing required columns"
                )
                continue
            
            try:
                result: Optional[pd.DataFrame] = None
                if factor_name == "race_environment" and feature_name == "wind_resistance_score":
                    result = self._compute_wind_resistance_score(df)
                elif factor_name == "race_environment" and feature_name == "rough_water_score":
                    result = self._compute_rough_water_score(df)
                elif factor_name == "frame_and_course" and feature_name == "lane_win_rate_by_position":
                    result = self._compute_lane_win_rate_by_position(df)
                elif factor_name == "racer_skill" and feature_name == "skill_gap_vs_field":
                    result = self._compute_skill_gap(df)
                elif factor_name == "motor_performance" and feature_name == "motor_gap_vs_field":
                    result = self._compute_motor_gap(df)
                elif factor_name == "start_timing" and feature_name == "start_timing_consistency":
                    result = self._compute_start_consistency(df)
                
                if result is not None and not result.empty:
                    derived_parts.append(result)
                    logger.info(f"[DERIVED] Computed {feature_name} for {factor_name}")
            except Exception as e:
                logger.error(f"[DERIVED] Failed to compute {feature_name}: {e}")
        
        if derived_parts:
            df = pd.concat([df] + derived_parts, axis=1)
        
        return df
    
    def _compute_wind_resistance_score(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute wind resistance score (wind power * direction factor)."""
        wind_pow_col = self._resolve_column("windPow", df.columns)
        wind_dir_col = self._resolve_column("windDir", df.columns)
        
        if wind_pow_col and wind_pow_col in df.columns:
            wind_power = pd.to_numeric(df[wind_pow_col], errors='coerce').fillna(0)
            direction_factor = 1.0
            if wind_dir_col and wind_dir_col in df.columns:
                wind_dir = pd.to_numeric(df[wind_dir_col], errors='coerce').fillna(0)
                direction_factor = 1.0 + (wind_dir % 8 - 4) / 10.0
            return pd.DataFrame({"derived_wind_resistance_score": wind_power * direction_factor}, index=df.index)
        return None
    
    def _compute_rough_water_score(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Normalize wave height to [0,1] rough water score."""
        wave_col = self._resolve_column("waveHight", df.columns) or self._resolve_column("wave_height", df.columns)
        if wave_col and wave_col in df.columns:
            values = pd.to_numeric(df[wave_col], errors='coerce').fillna(0).clip(0, 50) / 50.0
            return pd.DataFrame({"derived_rough_water_score": values}, index=df.index)
        return None
    
    def _compute_skill_gap(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute racer skill gap vs field average."""
        derived = {}
        for lane in range(1, 7):
            win_col = self._resolve_column(f"win_rate_national_{lane}", df.columns)
            if win_col and win_col in df.columns:
                win_cols = [
                    self._resolve_column(f"win_rate_national_{i}", df.columns)
                    for i in range(1, 7)
                ]
                win_cols = [c for c in win_cols if c and c in df.columns]
                if len(win_cols) > 0:
                    field_avg = df[win_cols].mean(axis=1)
                    derived[f"derived_skill_gap_{lane}"] = df[win_col] - field_avg
        if derived:
            return pd.DataFrame(derived, index=df.index)
        return None
    
    def _compute_motor_gap(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute motor quality gap vs field average."""
        derived = {}
        for lane in range(1, 7):
            motor_col = self._resolve_column(f"motor_place2Ratio_{lane}", df.columns)
            if motor_col and motor_col in df.columns:
                motor_cols = [
                    self._resolve_column(f"motor_place2Ratio_{i}", df.columns)
                    for i in range(1, 7)
                ]
                motor_cols = [c for c in motor_cols if c and c in df.columns]
                if len(motor_cols) > 0:
                    field_avg = df[motor_cols].mean(axis=1)
                    derived[f"derived_motor_gap_{lane}"] = df[motor_col] - field_avg
        if derived:
            return pd.DataFrame(derived, index=df.index)
        return None
    
    def _compute_start_consistency(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute start timing standard deviation (consistency metric)."""
        derived_cols = {}
        for lane in range(1, 7):
            st_cols = [
                self._resolve_column(f"CS_ST_{lane}_{i}", df.columns)
                for i in range(1, 14)
            ]
            st_cols = [c for c in st_cols if c and c in df.columns]
            
            if len(st_cols) > 3:
                cleaned = (
                    df[st_cols]
                    .replace({"\xa0": np.nan, "": np.nan, " ": np.nan})
                    .infer_objects(copy=False)
                    .apply(pd.to_numeric, errors="coerce")
                )
                df[st_cols] = cleaned
                derived_cols[f"derived_start_consistency_{lane}"] = cleaned.std(axis=1)

        if derived_cols:
            return pd.DataFrame(derived_cols, index=df.index)
        return None

    def _compute_lane_win_rate_by_position(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Compute historical win rate tendency for each lane."""
        derived = {}
        for lane in range(1, 7):
            lane_cols = [c for c in df.columns if c.startswith(f"CS_cource_{lane}_")]
            if len(lane_cols) == 0:
                continue

            lane_values = (
                df[lane_cols]
                .replace({"\xa0": np.nan, "": np.nan, " ": np.nan})
                .infer_objects(copy=False)
                .apply(pd.to_numeric, errors="coerce")
            )

            derived[f"derived_lane_win_rate_{lane}"] = lane_values.mean(axis=1)
        if derived:
            return pd.DataFrame(derived, index=df.index)
        return None
    
    def bind(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Main binding function. Maps dataframe to factors.
        
        Args:
            df: Input dataframe with raw race data
            
        Returns:
            {
                "factor_df": DataFrame with factor-organized columns,
                "factor_summary": Dict with coverage analysis,
                "warnings": List of warning messages
            }
        """
        logger.info(f"[BIND] Starting factor binding on {len(df)} rows")
        
        # Normalize column names
        df = self._normalize_column_names(df)
        # De-fragment once up front so downstream column inserts don't trigger PerformanceWarnings
        df = df.copy()
        
        # Analyze coverage for each factor
        factor_summary = {}
        warnings = []
        
        for factor_name in self.factor_names:
            coverage = self._get_factor_coverage(factor_name, df)
            factor_summary[factor_name] = coverage
            
            if coverage["status"] == "blocked":
                warnings.append(
                    f"⚠️ Factor '{coverage['factor_name']}' has ZERO available columns!"
                )
            elif coverage["status"] == "partial":
                warnings.append(
                    f"⚠️ Factor '{coverage['factor_name']}' has low coverage "
                    f"({coverage['coverage_rate']:.1%})"
                )
        
        # Check if ALL core factors (1-8) are blocked
        core_blocked = sum(
            1 for fname in self.factor_names[:8]
            if factor_summary[fname]["status"] == "blocked"
        )
        
        if core_blocked == 8:
            top_cols = list(df.columns)[:20]
            raise RuntimeError(
                f"❌ ALL core factors are blocked! No column mapping succeeded.\n"
                f"Top dataframe columns: {top_cols}\n"
                f"Check factors_config.yaml column names."
            )
        
        # Compute derived features
        for factor_name in self.factor_names:
            if factor_summary[factor_name]["status"] != "blocked":
                available_cols = factor_summary[factor_name]["available_columns"]
                df = self._compute_derived_features(factor_name, df, available_cols)
        
        # Build factor-organized dataframe (keep all columns, add metadata)
        factor_df = df.copy()
        factor_df["_factor_binding_complete"] = True
        
        logger.info(f"[BIND] Complete. {len(warnings)} warnings.")
        
        return {
            "factor_df": factor_df,
            "factor_summary": factor_summary,
            "warnings": warnings
        }
    
    def get_factor_columns(self, factor_name: str, df: pd.DataFrame) -> List[str]:
        """
        Get list of available columns for a specific factor.
        
        Args:
            factor_name: Factor identifier (e.g., "race_environment")
            df: Dataframe to check against
            
        Returns:
            List of column names belonging to this factor
        """
        coverage = self._get_factor_coverage(factor_name, df)
        return coverage["available_columns"]
