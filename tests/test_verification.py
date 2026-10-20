"""Regression and failure tests using only the two public CSV fixtures."""
from __future__ import annotations

from contextlib import redirect_stdout
import copy
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verification_cli", ROOT / "verify_all_results.py")
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class VerificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads(verifier.REFERENCE.read_text())
        cls.baseline = verifier.verify(ROOT / "final_dataset")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data = Path(self.temporary.name) / "data"
        self.data.mkdir()
        for filename in verifier.FILES.values():
            shutil.copyfile(ROOT / "final_dataset" / filename, self.data / filename)

    def edit(self, table, operation, *, reverse_columns=False):
        path = self.data / verifier.FILES[table]
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            columns, rows = reader.fieldnames, list(reader)
        operation(rows)
        if reverse_columns:
            columns = list(reversed(columns))
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def assert_failed(self, report):
        self.assertEqual(report["status"], "failed")
        self.assertTrue(any(not check["passed"] for check in report["checks"]))

    def statuses(self, report):
        return {check["id"]: check["passed"] for check in report["checks"]}

    def test_public_data_passes_explicit_bounded_checks(self):
        report = self.baseline
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["check_counts"], {"passed": 99})
        self.assertEqual(report["results"]["counts"]["participants"], 66)
        self.assertEqual(report["results"]["counts"]["round_rows"], 2114)
        result = report["results"]["paired_outcomes"]["Agent_Utility"]
        self.assertAlmostEqual(result["cohens_dz"], 0.30287103506147006, places=12)
        self.assertAlmostEqual(result["p_two_sided"], 0.016539148849266504, places=12)
        self.assertAlmostEqual(result["difference_ci95_low"], 0.004208938128466726, places=12)
        self.assertAlmostEqual(report["results"]["holm_four_outcome_family"]["Concession_Rate"], 0.03973990097807921, places=12)
        self.assertEqual(set(report["coverage_unverified"]), set(verifier.UNVERIFIED))
        self.assertNotIn("Nash_Distance", report["results"]["paired_outcomes"])

    def test_pairing_and_fingerprints_ignore_row_and_column_order(self):
        for table in verifier.FILES:
            self.edit(table, lambda rows: rows.reverse(), reverse_columns=True)
        report = verifier.verify(self.data)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["results"], self.baseline["results"])

    def test_missing_file_fails(self):
        (self.data / verifier.FILES["rounds"]).unlink()
        self.assert_failed(verifier.verify(self.data))

    def test_missing_condition_summary_fails(self):
        self.edit("subjects", lambda rows: rows.pop(0))
        self.assert_failed(verifier.verify(self.data))

    def test_duplicate_subject_and_round_keys_fail(self):
        for table in verifier.FILES:
            with self.subTest(table=table):
                self.edit(table, lambda rows: rows.append(dict(rows[0])))
                self.assert_failed(verifier.verify(self.data))
                shutil.copyfile(ROOT / "final_dataset" / verifier.FILES[table], self.data / verifier.FILES[table])

    def test_changed_condition_assignment_fails(self):
        self.edit("rounds", lambda rows: rows[0].update(Condition="FC"))
        self.assert_failed(verifier.verify(self.data))

    def test_missing_round_fails_without_silent_drop(self):
        self.edit("rounds", lambda rows: rows.pop(0))
        self.assert_failed(verifier.verify(self.data))

    def test_nan_infinity_missing_and_nonnumeric_values_fail(self):
        for value in ["nan", "inf", "-inf", "", "invalid"]:
            with self.subTest(value=value):
                self.edit("subjects", lambda rows: rows[0].update(Agent_Utility=value))
                self.assert_failed(verifier.verify(self.data))
                shutil.copyfile(ROOT / "final_dataset" / verifier.FILES["subjects"], self.data / verifier.FILES["subjects"])
        self.edit("rounds", lambda rows: rows[0].update(Norm_Arousal="nan"))
        self.assert_failed(verifier.verify(self.data))

    def test_duplicate_header_and_extra_cell_fail(self):
        path = self.data / verifier.FILES["subjects"]
        original = path.read_text()
        lines = original.splitlines()
        lines[0] = lines[0].replace("Human_Utility", "Agent_Utility")
        path.write_text("\n".join(lines) + "\n")
        self.assert_failed(verifier.verify(self.data))
        lines = original.splitlines()
        lines[1] += ",extra"
        path.write_text("\n".join(lines) + "\n")
        self.assert_failed(verifier.verify(self.data))

    def test_mean_match_cannot_mask_changed_p_value_or_effect_size(self):
        def change_variance(rows):
            cl = [row for row in rows if row["Condition"] == "CL"]
            cl[0]["Agent_Utility"] = str(float(cl[0]["Agent_Utility"]) + 0.01)
            cl[1]["Agent_Utility"] = str(float(cl[1]["Agent_Utility"]) - 0.01)
        self.edit("subjects", change_variance)
        report = verifier.verify(self.data)
        self.assert_failed(report)
        checks = self.statuses(report)
        prefix = "results.paired_outcomes.Agent_Utility."
        self.assertTrue(checks[prefix + "CL_mean"])
        self.assertFalse(checks[prefix + "p_two_sided"])
        self.assertFalse(checks[prefix + "cohens_dz"])

    def test_fc_mean_p_effect_and_ci_each_have_independent_assertions(self):
        for field in ["FC_mean", "p_two_sided", "cohens_dz", "difference_ci95_low"]:
            with self.subTest(field=field):
                reference = copy.deepcopy(self.reference)
                reference["results"]["paired_outcomes"]["Agent_Utility"][field] += 0.01
                report = verifier.verify(self.data, reference)
                self.assert_failed(report)
                failed = [check["id"] for check in report["checks"] if not check["passed"]]
                self.assertEqual(failed, ["results.paired_outcomes.Agent_Utility." + field])

    def test_correlation_magnitude_is_checked_not_only_sign(self):
        reference = copy.deepcopy(self.reference)
        reference["results"]["descriptive_correlations"]["CL"]["pooled_round_r"] = 0.1
        report = verifier.verify(self.data, reference)
        self.assert_failed(report)
        self.assertFalse(self.statuses(report)["results.descriptive_correlations.CL.pooled_round_r"])

    def test_changed_round_correlation_is_detected(self):
        self.edit("rounds", lambda rows: rows[10].update(Norm_Valence="0.9"))
        report = verifier.verify(self.data)
        self.assert_failed(report)
        checks = self.statuses(report)
        self.assertTrue(any(not checks[f"results.descriptive_correlations.{condition}.pooled_round_r"] for condition in ["CL", "FC"]))

    def test_fisher_summary_keeps_exclusions_explicit_and_descriptive(self):
        correlations = self.baseline["results"]["descriptive_correlations"]
        self.assertEqual(correlations["CL"]["fisher_sessions_included"], 63)
        self.assertEqual(correlations["CL"]["fisher_sessions_excluded"], 3)
        self.assertEqual(correlations["FC"]["fisher_sessions_included"], 66)
        self.assertNotIn("p_two_sided", correlations["CL"])
        self.assertAlmostEqual(correlations["CL"]["pooled_round_r"], 0.4540581058055483, places=12)
        self.assertAlmostEqual(correlations["CL"]["finite_within_session_fisher_backtransformed_mean_r"], 0.8648320441598559, places=12)

    def test_sample_sd_and_paired_dz_definition(self):
        result = verifier.paired_estimate(np.array([2., 4., 6.]), np.array([1., 2., 3.]))
        self.assertEqual(result["difference_sample_sd"], 1.)
        self.assertEqual(result["CL_sample_sd"], 2.)
        self.assertEqual(result["cohens_dz"], 2.)
        self.assertAlmostEqual(result["t"], 2 * math.sqrt(3))
        with self.assertRaises(ValueError):
            verifier.paired_estimate(np.array([2., 3.]), np.array([1., 2.]))

    def test_holm_is_monotone_capped_and_uses_all_four_tests(self):
        self.assertEqual(verifier.holm_adjust({"a": .01, "b": .02, "c": .03, "d": .8}), {"a": .04, "b": .06, "c": .06, "d": .8})
        self.assertEqual(verifier.holm_adjust({"a": .9, "b": .95}), {"a": 1., "b": 1.})

    def test_cli_exit_and_explicit_json_destination(self):
        destination = Path(self.temporary.name) / "report"
        success = subprocess.run([sys.executable, str(ROOT / "verify_all_results.py"), "--data-dir", str(self.data), "--output-dir", str(destination)], cwd=self.temporary.name, capture_output=True, text=True)
        self.assertEqual(success.returncode, 0, success.stderr)
        self.assertEqual(json.loads(success.stdout)["status"], "passed")
        self.assertEqual(json.loads((destination / "verification-results.json").read_text())["status"], "passed")
        (self.data / verifier.FILES["subjects"]).unlink()
        failure = subprocess.run([sys.executable, str(ROOT / "verify_all_results.py"), "--data-dir", str(self.data)], cwd=self.temporary.name, capture_output=True, text=True)
        self.assertEqual(failure.returncode, 1)
        self.assert_failed(json.loads(failure.stdout))
        self.assertFalse((Path(self.temporary.name) / "statistical_analysis").exists())

    def test_output_write_error_fails_the_aggregate(self):
        blocked = Path(self.temporary.name) / "regular-file"
        blocked.write_text("keep")
        with redirect_stdout(io.StringIO()) as output:
            code = verifier.main(["--data-dir", str(self.data), "--output-dir", str(blocked)])
        self.assertEqual(code, 1)
        report = json.loads(output.getvalue())
        self.assert_failed(report)
        self.assertFalse(self.statuses(report)["output_write"])
        self.assertEqual(blocked.read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
