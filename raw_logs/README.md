# Negotiation Session Logs

These workbooks record the sequence of offers, utilities, affect estimates and
interaction text from the study. Start with the [derived CSVs](../final_dataset/README.md)
for the paired study comparisons; use the logs to examine what was recorded within
a session.

## Sessions and identifiers

The folder contains 198 Excel workbooks, each with one worksheet:

| Worksheet | Workbooks | Role in the study |
| --- | ---: | --- |
| `Session Demo` | 66 | Grocery-domain demonstration and calibration |
| `Session Session_1` | 66 | First holiday-domain negotiation |
| `Session Session_2` | 66 | Second holiday-domain negotiation |

Participants experienced continual learning (CL) and the fixed FaceChannel (FC)
baseline in counterbalanced order. A session number therefore identifies order,
not condition. The workbooks do not contain the CSV fields `Subject_ID`,
`Session_ID` or `Condition`. The repository does not provide a
workbook-to-CSV identifier map or the original export workflow, so filenames and
spreadsheet row indices should not be used as substitutes for `Subject_ID`.

## Columns

Each row is a logged offer. The first, unnamed column is a zero-based row index
that restarts in each workbook. The remaining 15 columns retain their original
labels, including the spelling `Valance`.

| Column | Meaning |
| --- | --- |
| `Bidder` | Actor making the offer: `Human` or `Agent`. |
| `Agent Utility`, `Human Utility` | Value of the offer under the agent's and participant's utility profiles. |
| `Offer` | Serialized bid. Interpret its issue positions using the matching domain and preference profiles. |
| `Scaled Time` | Elapsed negotiation time divided by the session deadline. This is a fraction, not a clock timestamp. |
| `Move` | Bidder's move category relative to that actor's previous offer: `silent`, `nice`, `fortunate`, `unfortunate`, `concession` or `selfish`. The first offer by each actor has no preceding move. |
| `Agent Emotion` | Emotion label selected for the agent's expression, where recorded. It is not a participant affect annotation. |
| `Valance`, `Arousal` | Recorded model estimates of participant affect. |
| `Max V`, `Min V`, `Max A`, `Min A` | Exported upper and lower valence/arousal reference values. |
| `Sensitivity Class` | Historical sensitivity-class field; empty in these workbooks. |
| `Sentences` | Free-text interaction records, where recorded. |

The move definitions are implemented in
[`get_target_move_list`](../src_code/solver_cl_agent/HATN/sensitivity_calculator.py);
[`NegotiationTimeController`](../src_code/solver_cl_agent/HATN/negoTime.py) defines
the time scale. The workbooks do not document the affect normalization procedure
or the time window used for the extrema. Their affect fields should therefore not
be equated with the CSVs' `Raw_*` and `Norm_*` fields by name alone. Preserve blank
cells and the original row order when inspecting a session.

## Relation to the derived data

The [subject-level CSV](../final_dataset/experiment_subject_data.csv) summarizes
participant sessions; the [round-level CSV](../final_dataset/experiment_round_data.csv)
contains the study's exported offer positions. They carry the participant, session
and condition keys used by the [public-data verifier](../docs/public-data-verification.md).
The workbooks also include calibration sessions, and their rows are not a documented
one-to-one reconstruction of the CSV export. The verifier reads the two frozen
CSVs; it does not rebuild them from these workbooks.

## Reading and reusing the records

Open workbooks without saving over them. For read-only inspection in Python,
`openpyxl` is included in the separate `requirements-analysis.lock` environment.
This example reports a workbook's structure without printing filenames or text:

```python
from pathlib import Path
from openpyxl import load_workbook

path = next(Path("raw_logs").glob("*.xlsx"))
workbook = load_workbook(path, read_only=True, data_only=True)
try:
    sheet = workbook.worksheets[0]
    print("Worksheet:", sheet.title)
    print("Offer rows:", sheet.max_row - 1)
    print("Columns:", next(sheet.iter_rows(values_only=True)))
finally:
    workbook.close()
```

The paper reports ethics approval and written informed consent for participation.
The repository does not include the consent form or a separate data-use statement
for the free-text logs. Check the permitted reuse with the study authors before
quoting or redistributing participant text. The software's MIT notice does not
document that consent scope. The derived CSVs support numerical verification
without using the utterances.
