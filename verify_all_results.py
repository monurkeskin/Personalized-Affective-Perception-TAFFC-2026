#!/usr/bin/env python3
"""Verify frozen public CSV results offline; this does not validate the robot study."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "verification" / "expected-results.json"
FILES = {"subjects": "experiment_subject_data.csv", "rounds": "experiment_round_data.csv"}
SUBJECT_COLUMNS = (
    "Subject_ID,Session_ID,Condition,Agent_Utility,Human_Utility,Agreement_Rounds,"
    "Raw_Arousal,Raw_Valence,Max_Arousal,Min_Arousal,Max_Valence,Min_Valence,"
    "Norm_Arousal,Norm_Valence,Concession_Rate,Nice_Rate,Fortunate_Rate,"
    "Unfortunate_Rate,Selfish_Rate,Silent_Rate"
).split(",")
ROUND_COLUMNS = (
    "Subject_ID,Session_ID,Round_ID,Condition,Agent_Utility,Human_Utility,"
    "Raw_Arousal,Raw_Valence,Max_Arousal,Min_Arousal,Max_Valence,Min_Valence,"
    "Bidder,Agent_Move,User_Move,Scaled_Time,Norm_Arousal,Norm_Valence"
).split(",")
TEXT_COLUMNS = {"Condition", "Bidder", "Agent_Move", "User_Move"}
OPTIONAL_ROUND = {"Agent_Utility", "Human_Utility", "Scaled_Time", "Bidder", "Agent_Move", "User_Move"}
OUTCOMES = ["Agent_Utility", "Human_Utility", "Agreement_Rounds", "Concession_Rate", "Norm_Arousal", "Norm_Valence"]
HOLM_FAMILY = ["Norm_Arousal", "Concession_Rate", "Agent_Utility", "Agreement_Rounds"]
UNVERIFIED = {
    "Nash_distance": "Reference Nash solutions and a Nash-distance column are absent from the two CSVs.",
    "questionnaire": "Item-level questionnaire observations are absent.",
    "mediation_and_bootstrap": "This bounded verifier does not reproduce mediation, bootstrap, order or LMM analyses.",
    "perception_accuracy": "CSV model-output differences do not establish accuracy against ground-truth affect labels.",
    "model_and_robot_runtime": "No models, camera, robot, speech service or experiment controller are executed.",
    "manuscript_complete": "Only the explicitly listed public-data results are checked; this is not whole-manuscript validation.",
}


def read_csv(path: Path, columns: list[str], *, rounds: bool) -> list[dict]:
    """Validate the public schema and numbers without silently dropping observations."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)) or set(headers) != set(columns):
            raise ValueError(f"{path.name}: unexpected or duplicate CSV columns")
        rows = []
        for line, raw in enumerate(reader, 2):
            if None in raw or any(value is None for value in raw.values()):
                raise ValueError(f"{path.name}:{line}: malformed CSV row")
            row = {}
            for column in columns:
                value = raw[column].strip()
                if not value:
                    if rounds and column in OPTIONAL_ROUND:
                        row[column] = None
                        continue
                    raise ValueError(f"{path.name}:{line}: missing {column}")
                if column in TEXT_COLUMNS:
                    row[column] = value
                    continue
                try:
                    number = float(value)
                except ValueError as exc:
                    raise ValueError(f"{path.name}:{line}: invalid {column}") from exc
                if not math.isfinite(number):
                    raise ValueError(f"{path.name}:{line}: non-finite {column}")
                if column.endswith("_ID") or column == "Agreement_Rounds":
                    if not number.is_integer() or number < 1:
                        raise ValueError(f"{path.name}:{line}: {column} must be a positive integer")
                    number = int(number)
                if column.endswith("_Utility") or column.endswith("_Rate"):
                    if not 0 <= number <= 1:
                        raise ValueError(f"{path.name}:{line}: {column} outside [0,1]")
                row[column] = number
            if row["Condition"] not in {"CL", "FC"} or row["Session_ID"] not in {1, 2}:
                raise ValueError(f"{path.name}:{line}: unexpected condition/session")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: empty dataset")
    key_columns = ["Subject_ID", "Session_ID"] + (["Round_ID"] if rounds else [])
    keys = [tuple(row[key] for key in key_columns) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{path.name}: duplicate observation keys")
    return sorted(rows, key=lambda row: tuple(row[key] for key in key_columns))


def validate_pairing(subjects: list[dict], rounds: list[dict]) -> None:
    by_subject = defaultdict(list)
    for row in subjects:
        by_subject[row["Subject_ID"]].append(row)
    if set(by_subject) != set(range(1, 67)):
        raise ValueError("Subject summaries must contain exactly participant IDs 1 through 66")
    for rows in by_subject.values():
        if len(rows) != 2 or {r["Condition"] for r in rows} != {"CL", "FC"} or {r["Session_ID"] for r in rows} != {1, 2}:
            raise ValueError("Each participant requires one CL and one FC summary in different sessions")
    assignments = {(r["Subject_ID"], r["Session_ID"]): r["Condition"] for r in subjects}
    grouped = defaultdict(list)
    for row in rounds:
        key = (row["Subject_ID"], row["Session_ID"])
        if assignments.get(key) != row["Condition"]:
            raise ValueError("Round participant/session/condition does not match its subject summary")
        grouped[key].append(row["Round_ID"])
    if set(grouped) != set(assignments):
        raise ValueError("Round data must cover every participant session")
    for ids in grouped.values():
        if sorted(ids) != list(range(1, len(ids) + 1)):
            raise ValueError("Round IDs must be consecutive within each participant session")


def canonical_digest(rows: list[dict]) -> str:
    """Hash normalized keyed records, permitting row/column order and numeric formatting changes."""
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def paired_estimate(cl: np.ndarray, fc: np.ndarray) -> dict:
    difference = cl - fc
    n = len(difference)
    sd = float(np.std(difference, ddof=1))
    if n < 2 or not math.isfinite(sd) or sd == 0:
        raise ValueError("Paired inference requires at least two pairs and nonzero difference SD")
    mean = float(np.mean(difference))
    se = sd / math.sqrt(n)
    t_value = mean / se
    margin = float(stats.t.ppf(0.975, n - 1)) * se
    return {
        "n_pairs": n, "df": n - 1,
        "CL_mean": float(np.mean(cl)), "CL_sample_sd": float(np.std(cl, ddof=1)),
        "FC_mean": float(np.mean(fc)), "FC_sample_sd": float(np.std(fc, ddof=1)),
        "CL_minus_FC_mean": mean, "difference_sample_sd": sd,
        "difference_ci95_low": mean - margin, "difference_ci95_high": mean + margin,
        "t": t_value, "p_two_sided": float(2 * stats.t.sf(abs(t_value), n - 1)),
        "cohens_dz": mean / sd,
    }


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    result, previous = {}, 0.0
    for rank, name in enumerate(ordered):
        previous = max(previous, min(1.0, p_values[name] * (len(ordered) - rank)))
        result[name] = previous
    return result


def calculate(subjects: list[dict], rounds: list[dict]) -> dict:
    paired = {(r["Subject_ID"], r["Condition"]): r for r in subjects}
    outcomes = {}
    for column in OUTCOMES:
        cl = np.array([paired[(sid, "CL")][column] for sid in range(1, 67)], dtype=float)
        fc = np.array([paired[(sid, "FC")][column] for sid in range(1, 67)], dtype=float)
        outcomes[column] = paired_estimate(cl, fc)
    correlations = {}
    for condition in ["CL", "FC"]:
        rows = [r for r in rounds if r["Condition"] == condition]
        pooled_r = float(stats.pearsonr([r["Norm_Arousal"] for r in rows], [r["Norm_Valence"] for r in rows]).statistic)
        fisher, excluded = [], 0
        for sid in range(1, 67):
            session = [r for r in rows if r["Subject_ID"] == sid]
            a = [r["Norm_Arousal"] for r in session]
            v = [r["Norm_Valence"] for r in session]
            if len(session) < 4 or np.std(a) == 0 or np.std(v) == 0:
                excluded += 1
                continue
            r_value = float(stats.pearsonr(a, v).statistic)
            # Exact/perfect correlations have no finite Fisher transform.
            if not math.isfinite(r_value) or abs(r_value) >= 1 - 1e-12:
                excluded += 1
                continue
            fisher.append(math.atanh(r_value))
        if not math.isfinite(pooled_r) or not fisher:
            raise ValueError("Correlation summary requires finite variable observations")
        correlations[condition] = {
            "pooled_round_r": pooled_r,
            "finite_within_session_fisher_backtransformed_mean_r": math.tanh(float(np.mean(fisher))),
            "fisher_sessions_included": len(fisher), "fisher_sessions_excluded": excluded,
        }
    return {
        "counts": {
            "participants": 66, "subject_rows": len(subjects), "round_rows": len(rounds),
            "round_rows_CL": sum(r["Condition"] == "CL" for r in rounds),
            "round_rows_FC": sum(r["Condition"] == "FC" for r in rounds),
            "CL_first": sum(r["Condition"] == "CL" and r["Session_ID"] == 1 for r in subjects),
        },
        "paired_outcomes": outcomes,
        "holm_four_outcome_family": holm_adjust({name: outcomes[name]["p_two_sided"] for name in HOLM_FAMILY}),
        "descriptive_correlations": correlations,
    }


def compare(actual, expected, checks: list[dict], path: str = "results") -> None:
    """Every reference value produces its own assertion; no metric reuses another status."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            checks.append({"id": path, "passed": False, "error": "Missing result group"})
            return
        for key, value in expected.items():
            compare(actual.get(key), value, checks, f"{path}.{key}")
        return
    numeric = isinstance(expected, (int, float)) and not isinstance(expected, bool)
    if numeric:
        passed = isinstance(actual, (int, float)) and math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10)
    else:
        passed = actual == expected
    checks.append({"id": path, "passed": bool(passed), "actual": actual, "expected": expected})


def verify(data_dir: Path, expected: dict | None = None) -> dict:
    report = {
        "status": "failed", "scope": "Frozen public CSV schema, keyed paired estimates and descriptive correlations",
        "methods": {"paired_direction": "CL minus FC", "sd_ddof": 1, "ci": "Unadjusted two-sided paired t 95% CI", "holm_family": HOLM_FAMILY,
                    "correlation": "Pooled round r is descriptive. Finite within-session Fisher means are descriptive, have different inclusion counts, and are not a paired test."},
        "checks": [], "results": {}, "coverage_unverified": UNVERIFIED,
    }
    try:
        expected = expected if expected is not None else json.loads(REFERENCE.read_text())
        subjects = read_csv(data_dir / FILES["subjects"], SUBJECT_COLUMNS, rounds=False)
        rounds = read_csv(data_dir / FILES["rounds"], ROUND_COLUMNS, rounds=True)
        validate_pairing(subjects, rounds)
        report["checks"].append({"id": "schema_and_keyed_pairing", "passed": True})
        report["input_canonical_sha256"] = {"subjects": canonical_digest(subjects), "rounds": canonical_digest(rounds)}
        compare(report["input_canonical_sha256"], expected["canonical_sha256"], report["checks"], "input_canonical_sha256")
        report["results"] = calculate(subjects, rounds)
        compare(report["results"], expected["results"], report["checks"])
        report["reference_source_commit"] = expected["source_commit"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        report["checks"].append({"id": "input_or_calculation_error", "passed": False, "error": str(exc)})
    if report["checks"] and all(check["passed"] for check in report["checks"]):
        report["status"] = "passed"
    report["check_counts"] = dict(Counter("passed" if c["passed"] else "failed" for c in report["checks"]))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "final_dataset", help="Directory containing the two public CSV files")
    parser.add_argument("--output-dir", type=Path, help="Explicit optional destination for verification-results.json; default writes only JSON to stdout")
    args = parser.parse_args(argv)
    report = verify(args.data_dir)
    if args.output_dir:
        try:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "verification-results.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        except OSError as exc:
            report["status"] = "failed"
            report["checks"].append({"id": "output_write", "passed": False, "error": str(exc)})
            report["check_counts"] = dict(Counter("passed" if c["passed"] else "failed" for c in report["checks"]))
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
