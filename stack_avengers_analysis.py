#!/usr/bin/env python3
"""
THE STACK AVENGERS — reproducible analysis source
=================================================

One source file for the computational parts of STAC WARS PS2 and PS3, plus
a transparent planning calculator for PS4 asteroid-water mining.

Subcommands
-----------
solar      Analyze Voyage-123 solar radiation CSV, export figures and metrics.
exoplanet  Calculate transit depths, planet radii, orbital distances and rough HZ.
mining     Estimate scenario-level delivered cost per kg from user-supplied inputs.

Data-integrity note
-------------------
The supplied PS2 report says the real Voyage-123 CSV was not attached. The solar
analysis will stop with a clear error until that file is supplied. No placeholder
or synthetic results are presented as measured findings.

Expected solar CSV fields are configurable via command-line flags. Defaults:
timestamp, Solar Radiation, Temperature, Humidity, Wind Speed
Optional sunrise column: Sunrise. Column names must match the actual CSV.

Expected exoplanet CSV fields:
system,time,brightness (one observation per row; system labels e.g. STAC-1).
For transit candidates, provide --normal-brightness or the script estimates a
baseline from the upper quantile; this is a screening heuristic, not validation.

Dependencies: pandas, numpy, matplotlib, scikit-learn
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SOLAR_RADIUS_KM = 696_000.0
EARTH_RADIUS_KM = 6_371.0
AU_KM = 149_600_000.0
DAYS_PER_YEAR = 365.25

def ensure_output(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def solar_analysis(args: argparse.Namespace) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = Path(args.csv)
    if not source.exists():
        print(f"ERROR: dataset not found: {source}", file=sys.stderr)
        print("Add the official Voyage-123 CSV, then rerun this command.", file=sys.stderr)
        return 2
    df = pd.read_csv(source)
    required = [args.timestamp, args.radiation]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: required column(s) missing: {missing}", file=sys.stderr)
        print(f"Available columns: {df.columns.tolist()}", file=sys.stderr)
        return 2

    df[args.timestamp] = pd.to_datetime(df[args.timestamp], errors="coerce")
    df[args.radiation] = pd.to_numeric(df[args.radiation], errors="coerce")
    for col in [args.temperature, args.humidity, args.wind]:
        if col and col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[args.timestamp, args.radiation]).copy()
    df = df.sort_values(args.timestamp).drop_duplicates(subset=[args.timestamp], keep="first")
    if df.empty:
        print("ERROR: no usable rows remain after timestamp/radiation cleaning.", file=sys.stderr)
        return 2

    out = ensure_output(args.outdir)
    df["date"] = df[args.timestamp].dt.date
    df["hour"] = df[args.timestamp].dt.hour
    daily = df.groupby("date")[args.radiation].agg(
        daily_max_radiation="max", daily_mean_radiation="mean", observations="count"
    ).reset_index()
    peak_idx = df[args.radiation].idxmax()
    peak = df.loc[peak_idx]
    results = {
        "input_file": str(source),
        "rows_input": int(before),
        "rows_used": int(len(df)),
        "rows_removed_or_deduplicated": int(before-len(df)),
        "columns": df.columns.tolist(),
        "missing_values_after_cleaning": {k:int(v) for k,v in df.isna().sum().items()},
        "peak": {
            "timestamp": str(peak[args.timestamp]),
            "radiation": float(peak[args.radiation]),
            "conditions": {
                c: (None if c not in df.columns or pd.isna(peak[c]) else float(peak[c]))
                for c in [args.temperature, args.humidity, args.wind] if c
            }
        },
        "daily_summary": daily.to_dict(orient="records")
    }
    daily.to_csv(out/"daily_solar_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df[args.timestamp], df[args.radiation], linewidth=1)
    ax.set(title="Solar radiation over time", xlabel="Timestamp", ylabel=args.radiation)
    fig.tight_layout(); fig.savefig(out/"solar_time_series.png", dpi=160); plt.close(fig)

    for col in [args.temperature, args.humidity, args.wind]:
        if col and col in df.columns:
            plot_df=df[[col,args.radiation]].dropna()
            if not plot_df.empty:
                fig, ax = plt.subplots(figsize=(7,5))
                ax.scatter(plot_df[col], plot_df[args.radiation], alpha=.6)
                ax.set(title=f"Solar radiation vs {col}", xlabel=col, ylabel=args.radiation)
                fig.tight_layout()
                fig.savefig(out/f"radiation_vs_{safe_name(col)}.png", dpi=160)
                plt.close(fig)

    hourly=df.groupby("hour")[args.radiation].mean()
    fig, ax = plt.subplots(figsize=(8,4.5))
    hourly.plot(kind="bar", ax=ax)
    ax.set(title="Mean solar radiation by hour", xlabel="Hour of day", ylabel=args.radiation)
    fig.tight_layout(); fig.savefig(out/"hourly_mean_profile.png", dpi=160); plt.close(fig)

    numeric=df.select_dtypes(include=np.number)
    if numeric.shape[1] >= 2:
        corr=numeric.corr()
        fig, ax=plt.subplots(figsize=(8,6))
        im=ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(corr.columns)), corr.columns)
        ax.set_title("Numeric-variable correlation matrix")
        fig.colorbar(im, ax=ax, label="Pearson correlation")
        fig.tight_layout(); fig.savefig(out/"correlation_heatmap.png", dpi=160); plt.close(fig)
        if args.radiation in corr:
            results["radiation_correlations"] = corr[args.radiation].drop(args.radiation).sort_values(
                key=lambda s: s.abs(), ascending=False
            ).to_dict()

    daily.to_csv(out/"daily_solar_summary.csv", index=False)
    results["top_five_days_by_max_radiation"] = daily.nlargest(5,"daily_max_radiation").to_dict(orient="records")
    if args.sunrise and args.sunrise in df.columns:
        # Sunrise may be a time string or datetime. Compare on the same date.
        sunrise = pd.to_datetime(df[args.sunrise].astype(str), errors="coerce")
        df["sunrise_dt"] = sunrise
        missing_time = df["sunrise_dt"].isna()
        if missing_time.any():
            results["sunrise_note"] = "Sunrise values could not all be parsed; inspect source format."
        else:
            df["minutes_since_sunrise"] = (df[args.timestamp]-df["sunrise_dt"]).dt.total_seconds()/60
            peakrow=df.loc[df[args.radiation].idxmax()]
            results["minutes_from_sunrise_to_overall_peak"] = float(peakrow["minutes_since_sunrise"])
    else:
        results["sunrise_note"] = "No configured sunrise column found; sunrise-to-peak interval not calculated."

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        feature_cols=[c for c in [args.temperature,args.humidity,args.wind,"hour","day_of_year"] if c]
        df["day_of_year"]=df[args.timestamp].dt.dayofyear
        feature_cols=[c for c in feature_cols if c in df.columns]
        model_df=df[feature_cols+[args.radiation]].dropna()
        if len(model_df)>=20 and feature_cols:
            X=model_df[feature_cols]; y=model_df[args.radiation]
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
            model=RandomForestRegressor(n_estimators=300,random_state=42,n_jobs=-1)
            model.fit(X_train,y_train); pred=model.predict(X_test)
            results["random_forest_baseline"]={
                "features":feature_cols,
                "test_rows":int(len(y_test)),
                "rmse":float(np.sqrt(mean_squared_error(y_test,pred))),
                "r2":float(r2_score(y_test,pred)),
                "feature_importance":dict(sorted(
                    zip(feature_cols, map(float,model.feature_importances_)),
                    key=lambda x:x[1], reverse=True))
            }
            fig,ax=plt.subplots(figsize=(6,6))
            ax.scatter(y_test,pred,alpha=.6)
            low=float(min(y_test.min(),pred.min())); high=float(max(y_test.max(),pred.max()))
            ax.plot([low,high],[low,high],linestyle="--")
            ax.set(title="Held-out actual vs predicted",xlabel="Actual radiation",ylabel="Predicted radiation")
            fig.tight_layout(); fig.savefig(out/"actual_vs_predicted.png",dpi=160); plt.close(fig)
        else:
            results["random_forest_baseline"]={"status":"skipped","reason":"Need >=20 complete rows and at least one predictor."}
    except ImportError:
        results["random_forest_baseline"]={"status":"skipped","reason":"Install scikit-learn to run model evaluation."}

    with open(out/"solar_results.json","w",encoding="utf-8") as f:
        json.dump(results,f,indent=2,default=str)
    print(json.dumps({k:v for k,v in results.items() if k!="daily_summary"},indent=2,default=str))
    print(f"\nSaved outputs to: {out.resolve()}")
    return 0

def safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")

def exoplanet_analysis(args: argparse.Namespace) -> int:
    data = pd.read_csv(args.csv)
    needed=[args.system,args.time,args.brightness]
    missing=[c for c in needed if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns {missing}; available: {data.columns.tolist()}")
    data[args.time]=pd.to_numeric(data[args.time],errors="coerce")
    data[args.brightness]=pd.to_numeric(data[args.brightness],errors="coerce")
    data=data.dropna(subset=needed).sort_values([args.system,args.time])
    if data.empty:
        raise ValueError("No valid observations.")
    out=ensure_output(args.outdir)
    rows=[]
    for name,g in data.groupby(args.system):
        # Approximate normal flux from an explicit value, or a robust upper quantile.
        normal=float(args.normal_brightness) if args.normal_brightness is not None else float(g[args.brightness].quantile(.9))
        transit=float(g[args.brightness].quantile(.1))
        depth=max(0.0,(normal-transit)/normal) if normal else float("nan")
        times=g.loc[g[args.brightness] <= g[args.brightness].quantile(.1),args.time].to_numpy()
        diffs=np.diff(np.sort(times))
        period=float(np.median(diffs)) if len(diffs) else float("nan")
        row={"system":str(name),"observations":int(len(g)),"normal_brightness_estimate":normal,
             "transit_brightness_proxy":transit,"transit_depth_fraction":depth,
             "transit_depth_percent":depth*100,"candidate_period_days_median_gap":period}
        if args.star_radius_solar is not None:
            row["planet_radius_earth"]=(args.star_radius_solar*SOLAR_RADIUS_KM*math.sqrt(depth)/EARTH_RADIUS_KM) if depth>=0 else None
        if args.stellar_mass_solar is not None and np.isfinite(period):
            period_years=period/DAYS_PER_YEAR
            au=(args.stellar_mass_solar*period_years**2)**(1/3)
            row["orbital_distance_au_simplified"]=au
            row["orbital_distance_km"]=au*AU_KM
        rows.append(row)
    report=pd.DataFrame(rows)
    report.to_csv(out/"exoplanet_candidate_summary.csv",index=False)
    report.to_json(out/"exoplanet_candidate_summary.json",orient="records",indent=2)
    print(report.to_string(index=False))
    print("\nCaution: automated quantiles are screening estimates only. Verify event times, depth, cadence, stellar parameters and false positives.")
    print(f"Saved outputs to: {out.resolve()}")
    return 0

def mining_analysis(args: argparse.Namespace) -> int:
    if args.recovered_kg <= 0 or args.delivery_fraction <= 0 or args.delivery_fraction > 1:
        raise ValueError("recovered_kg must be > 0 and delivery_fraction must be in (0, 1].")
    costs = {
        "prospecting":args.prospecting_cost, "launch_and_transfer":args.transport_cost,
        "mining_and_processing":args.processing_cost, "storage_and_delivery":args.delivery_cost,
        "operations_and_insurance":args.operations_cost
    }
    if any(v < 0 for v in costs.values()):
        raise ValueError("All cost inputs must be non-negative.")
    accepted_kg=args.recovered_kg*args.delivery_fraction
    total=sum(costs.values())
    result={"cost_breakdown":costs,"total_mission_cost":total,
            "recovered_kg":args.recovered_kg,"delivery_fraction":args.delivery_fraction,
            "accepted_delivered_kg":accepted_kg,
            "delivered_cost_per_accepted_kg":total/accepted_kg,
            "note":"Scenario calculator only; inputs are user assumptions, not measured economics."}
    out=ensure_output(args.outdir)
    (out/"mining_cost_scenario.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))
    print(f"Saved output to: {(out/'mining_cost_scenario.json').resolve()}")
    return 0

def build_parser() -> argparse.ArgumentParser:
    parser=argparse.ArgumentParser(description="The Stack Avengers PS2/PS3 analysis and PS4 scenario calculator")
    sub=parser.add_subparsers(dest="command",required=True)
    s=sub.add_parser("solar",help="Analyze the Voyage-123 solar CSV")
    s.add_argument("--csv",default="voyage123_solar_data.csv")
    s.add_argument("--outdir",default="outputs/solar")
    s.add_argument("--timestamp",default="timestamp")
    s.add_argument("--radiation",default="Solar Radiation")
    s.add_argument("--temperature",default="Temperature")
    s.add_argument("--humidity",default="Humidity")
    s.add_argument("--wind",default="Wind Speed")
    s.add_argument("--sunrise",default="Sunrise")
    s.set_defaults(func=solar_analysis)
    e=sub.add_parser("exoplanet",help="Summarize transit-like dips from a light-curve CSV")
    e.add_argument("--csv",required=True)
    e.add_argument("--outdir",default="outputs/exoplanet")
    e.add_argument("--system",default="system")
    e.add_argument("--time",default="time")
    e.add_argument("--brightness",default="brightness")
    e.add_argument("--normal-brightness",type=float,default=None)
    e.add_argument("--star-radius-solar",type=float,default=None,help="Host star radius in solar radii")
    e.add_argument("--stellar-mass-solar",type=float,default=None,help="Host star mass in solar masses")
    e.set_defaults(func=exoplanet_analysis)
    m=sub.add_parser("mining",help="Calculate an asteroid-water mission cost scenario")
    m.add_argument("--outdir",default="outputs/mining")
    m.add_argument("--prospecting-cost",type=float,required=True)
    m.add_argument("--transport-cost",type=float,required=True)
    m.add_argument("--processing-cost",type=float,required=True)
    m.add_argument("--delivery-cost",type=float,required=True)
    m.add_argument("--operations-cost",type=float,required=True)
    m.add_argument("--recovered-kg",type=float,required=True)
    m.add_argument("--delivery-fraction",type=float,default=1.0)
    m.set_defaults(func=mining_analysis)
    return parser

def main() -> int:
    parser=build_parser()
    args=parser.parse_args()
    try:
        return args.func(args)
    except (ValueError, OSError, pd.errors.ParserError) as exc:
        print(f"ERROR: {exc}",file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
