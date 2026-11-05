"""JanusForge: reproduce Steps 19-23, without training any model.

Run from the repository root:
    python finalize_dataset.py

Requires the raw data under data/2024/races/, preprocess_all_races.py,
NumPy and pandas. Creates labeled CSV, quality-audited telemetry features,
feature_columns.json, and train/val/test CSVs under data/processed/.
"""
from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROCESSED = DATA_DIR / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "gap_seconds", "position", "follower_speed_mean", "ahead_speed_mean",
    "relative_speed_mean", "follower_speed_last", "ahead_speed_last",
    "relative_speed_last", "follower_throttle_mean", "ahead_throttle_mean",
    "follower_brake_frac", "ahead_brake_frac", "follower_rpm_mean",
    "ahead_rpm_mean", "follower_drs_active_frac", "ahead_drs_active_frac",
]
SPLITS = {
    "train": ["Bahrain", "Spain", "Austria"],
    "val": ["Australia"],
    "test": ["São Paulo"],
}


def read_utc(path: Path, date_column: str = "date") -> pd.DataFrame:
    data = pd.read_csv(path)
    data[date_column] = pd.to_datetime(
        data[date_column], format="mixed", utc=True, errors="coerce"
    )
    return data.dropna(subset=[date_column])


def remove_neutralizations(dataset: pd.DataFrame) -> pd.DataFrame:
    """Conservatively remove any [t-30, t+60] overlap with SC/VSC/red flags."""
    dataset = dataset.copy()
    dataset["time"] = pd.to_datetime(dataset["time"], format="mixed", utc=True)
    output, report = [], []

    for session_key, group in dataset.groupby("session_key", sort=False):
        control = read_utc(
            DATA_DIR / "2024" / "races" / str(session_key) / "race_control.csv"
        ).sort_values("date")
        control["msg"] = control["message"].fillna("").str.upper()

        finish = control.loc[control["msg"].eq("SESSION FINISHED"), "date"].min()
        if pd.isna(finish):
            raise ValueError(f"Official finish missing for race {session_key}")

        track_clear = control.loc[control["msg"].eq("TRACK CLEAR"), "date"]
        red_times = control.loc[control["msg"].eq("RED FLAG"), "date"]
        restart_times = control.loc[control["msg"].eq("SESSION STARTED"), "date"]

        def next_after(series: pd.Series, at: pd.Timestamp, default: pd.Timestamp):
            later = series.loc[series > at]
            return later.min() if len(later) else default

        periods = []
        for event in control.itertuples(index=False):
            if event.msg in ("VIRTUAL SAFETY CAR DEPLOYED", "SAFETY CAR DEPLOYED"):
                end = min(
                    next_after(track_clear, event.date, finish),
                    next_after(red_times, event.date, finish),
                    finish,
                )
                periods.append((event.date, end))
            elif event.msg == "RED FLAG":
                restart = next_after(restart_times, event.date, finish)
                end = min(restart + pd.Timedelta(seconds=60), finish)
                periods.append((event.date, end))

        excluded = pd.Series(False, index=group.index)
        for start, end in periods:
            excluded |= (
                (group["time"] - pd.Timedelta(seconds=30) <= end)
                & (group["time"] + pd.Timedelta(seconds=60) >= start)
            )

        remaining = group.loc[~excluded].copy()
        output.append(remaining)
        report.append({
            "race": group["race"].iloc[0],
            "before": len(group),
            "removed": int(excluded.sum()),
            "after": len(remaining),
            "positives": int(remaining["label"].sum()),
        })

    print("\nNEUTRALIZATION AUDIT")
    print(pd.DataFrame(report).to_string(index=False))
    clean = pd.concat(output, ignore_index=True)
    clean.to_csv(PROCESSED / "all_races_clean_labeled.csv", index=False)
    return clean


def window_mean(values: np.ndarray, left: np.ndarray, right: np.ndarray):
    finite = np.isfinite(values)
    prefix = np.r_[0., np.cumsum(np.where(finite, values, 0.))]
    counts = np.r_[0, np.cumsum(finite.astype(int))]
    n = counts[right] - counts[left]
    return np.divide(
        prefix[right] - prefix[left], n,
        out=np.full(len(left), np.nan), where=n > 0,
    )


def attach_telemetry(labels: pd.DataFrame) -> pd.DataFrame:
    result = labels.copy()
    metrics = [
        "speed_mean", "speed_last", "throttle_mean", "brake_frac",
        "rpm_mean", "drs_active_frac", "samples", "last_age_s", "span_s",
    ]
    for role in ("follower", "ahead"):
        for name in metrics:
            result[f"{role}_{name}"] = np.nan

    for session_key, race in result.groupby("session_key"):
        directory = DATA_DIR / "2024" / "races" / str(session_key) / "car_data"
        drivers = set(race["driver_number"]) | set(race["driver_ahead"])
        cache = {}
        for driver in drivers:
            filename = directory / f"driver_{int(driver)}.csv"
            if not filename.is_file():
                continue
            car = read_utc(filename).sort_values("date")
            if car.empty:
                continue
            drs = pd.to_numeric(car["drs"], errors="coerce")
            brake = car["brake"].replace({"True": 1, "False": 0})
            brake = pd.to_numeric(brake, errors="coerce")
            # F1 telemetry brake is 0 or 100. Correct to a 0..1 fraction.
            # Boolean 0/1 data would already be normalized.
            if brake.max() > 1:
                brake = brake / 100
            cache[int(driver)] = (
                car["date"].array.asi8,
                {
                    "speed": pd.to_numeric(car["speed"], errors="coerce").to_numpy(float),
                    "throttle": pd.to_numeric(car["throttle"], errors="coerce").to_numpy(float),
                    "brake": brake.to_numpy(float),
                    "rpm": pd.to_numeric(car["rpm"], errors="coerce").to_numpy(float),
                    "drs": np.where(drs.notna(), drs.isin([10, 12, 14]).astype(float), np.nan),
                },
            )

        for role, driver_column in (
            ("follower", "driver_number"), ("ahead", "driver_ahead")
        ):
            for driver, indexes in race.groupby(driver_column).groups.items():
                if int(driver) not in cache:
                    continue
                times, values = cache[int(driver)]
                query = result.loc[indexes, "time"].array.asi8
                left = np.searchsorted(times, query - 30_000_000_000, side="right")
                right = np.searchsorted(times, query, side="right")
                count = right - left
                present = count > 0

                age = np.full(len(query), np.nan)
                span = np.full(len(query), np.nan)
                speed_last = np.full(len(query), np.nan)
                age[present] = (query[present] - times[right[present] - 1]) / 1e9
                span[present] = (times[right[present] - 1] - times[left[present]]) / 1e9
                speed_last[present] = values["speed"][right[present] - 1]

                result.loc[indexes, f"{role}_samples"] = count
                result.loc[indexes, f"{role}_last_age_s"] = age
                result.loc[indexes, f"{role}_span_s"] = span
                result.loc[indexes, f"{role}_speed_last"] = speed_last

                for target, field in (
                    ("speed_mean", "speed"), ("throttle_mean", "throttle"),
                    ("brake_frac", "brake"), ("rpm_mean", "rpm"),
                    ("drs_active_frac", "drs"),
                ):
                    result.loc[indexes, f"{role}_{target}"] = window_mean(
                        values[field], left, right
                    )

        print(f"Telemetry extracted: {race['race'].iloc[0]}")

    quality = pd.Series(True, index=result.index)
    for role in ("follower", "ahead"):
        quality &= result[f"{role}_samples"].ge(20)
        quality &= result[f"{role}_span_s"].ge(20)
        quality &= result[f"{role}_last_age_s"].le(3)
        for field in ("speed_mean", "speed_last", "throttle_mean", "brake_frac", "rpm_mean", "drs_active_frac"):
            quality &= result[f"{role}_{field}"].notna()

    print(f"Telemetry coverage removed: {int((~quality).sum())} rows")
    result = result.loc[quality].copy()
    result["relative_speed_mean"] = result["follower_speed_mean"] - result["ahead_speed_mean"]
    result["relative_speed_last"] = result["follower_speed_last"] - result["ahead_speed_last"]

    assert not result.isna().any().any(), "Missing data in clean telemetry"
    assert not result.duplicated(["session_key", "time", "driver_number"]).any(), "Duplicate windows"
    assert result["gap_seconds"].between(0, 3, inclusive="right").all()
    assert set(result["label"].unique()) == {0, 1}
    for role in ("follower", "ahead"):
        assert result[f"{role}_brake_frac"].between(0, 1).all()
        assert result[f"{role}_drs_active_frac"].between(0, 1).all()
        assert result[f"{role}_throttle_mean"].between(0, 100).all()

    result.to_csv(PROCESSED / "all_races_features.csv", index=False)
    return result


def save_splits(dataset: pd.DataFrame) -> None:
    (PROCESSED / "feature_columns.json").write_text(
        json.dumps(FEATURE_COLUMNS, indent=2), encoding="utf-8"
    )
    for name, races in SPLITS.items():
        subset = dataset.loc[dataset["race"].isin(races)].copy()
        assert not subset.empty, f"Missing split: {name}"
        assert subset[FEATURE_COLUMNS].notna().all().all()
        subset.to_csv(PROCESSED / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(subset):5d} rows, {int(subset['label'].sum()):3d} positives / "
              f"{int((subset['label'] == 0).sum()):5d} negatives ({', '.join(races)})")


if __name__ == "__main__":
    preprocess = ROOT / "preprocess_all_races.py"
    if not preprocess.exists():
        raise FileNotFoundError(f"Missing {preprocess}")
    print("1/4 — Rebuilding provisional race labels")
    # Existing script expects DATA_DIR and display() from the earlier notebook.
    runpy.run_path(str(preprocess), init_globals={
        "DATA_DIR": DATA_DIR,
        "display": lambda frame: print(frame.to_string(index=False)),
    })
    print("2/4 — Excluding neutralized racing periods")
    provisional = pd.read_csv(PROCESSED / "all_races_labeled.csv")
    clean = remove_neutralizations(provisional)
    print("3/4 — Extracting and validating 30-second telemetry")
    final = attach_telemetry(clean)
    print("4/4 — Freezing race-separated splits")
    save_splits(final)
    print(f"\nDONE — {len(final):,} rows, {int(final['label'].sum())} positives, "
          f"{len(final.columns)} columns; saved under {PROCESSED}")
