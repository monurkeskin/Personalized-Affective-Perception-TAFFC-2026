# Dependency Compatibility and Security

The Solver-CL companion separates public-data verification from the lab robot and
perception stack. Use this guide to choose which dependency environment to inspect
and to evaluate patches against its interpreter, model and device requirements.

The inventory below is dated 20 October 2026, 13:49 UTC. It records 543 open
Dependabot alerts and candidate compatibility checks; these notes do not change
package versions or close the findings.

## Inventory scope

The authenticated, fully paginated export contains **543 alert IDs**, covering
533 distinct advisories and 18 package names after normalization. Severity totals
are **18 critical, 193 high, 225 medium and 107 low**.

| GitHub-attributed manifest | Open alerts |
| --- | ---: |
| `src_code/solver_cl_agent/requirements.txt` | 533 |
| `src_code/solver_cl_agent/FaceChannel/requirements.txt` | 10 |

Literal manifest declarations confirm 145 findings. The other 398 are attributed
to the core manifest but name packages absent from that file: TensorFlow (382),
Keras (14), intel-tensorflow (1) and OpenCV (1). Their graph attribution remains
unresolved. GitHub labels all 543 as direct; this export does not explain that
discrepancy. It is not a reason to dismiss the findings. GitHub documents how to
[inspect dependency paths](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/explore-dependencies)
and [investigate unexpected graph results](https://docs.github.com/en/code-security/reference/supply-chain-security/troubleshoot-dependabot/dependency-graph-errors).

An import, manifest entry or successful software fixture does not establish
whether an advisory's vulnerable operation is reachable. That assessment remains
separate from the package/version classification.

## Modern lock audit

The three hashed modern locks were audited on the inventory date with
`pip-audit 2.10.1` and its PyPI service, without installing or resolving new
dependencies. Each audit completed successfully with no skipped packages.

| Lock | Packages checked | Known vulnerabilities reported |
| --- | ---: | ---: |
| `requirements.lock` | 2 | 0 |
| `requirements-test.lock` | 19 | 0 |
| `requirements-analysis.lock` | 29 | 0 |

These results apply to those lock snapshots and the advisory records available on the audit date.
They do not cover the historical robot/ML manifests, SDK binaries, container
system libraries, or every possible security defect. Keep the modern host,
historical robot and FaceChannel environments separate.

## Review priorities

Start with the critical/high findings and resolve the unexplained attribution
in parallel. The groups below partition all 543 alerts.

| Review group | Alerts | Next step |
| --- | ---: | --- |
| Core data, model and numerical dependencies | 107, including 14 critical | Review Pillow, NLTK, joblib, scikit-learn and NumPy against the actual installed environment and affected input/load operations. Use trusted synthetic fixtures before any compatibility change. |
| Unresolved graph attribution | 398, including 4 critical | Trace the reported package/version to its dependency path or submitted snapshot. Record the explanation before selecting the runtime or manifest to change. |
| Core transport/authentication dependencies | 27, including 17 high | Review Requests, urllib3, certifi, idna, RSA, pyasn1, protobuf and gRPC as a coordinated legacy client stack; test each proposed change in isolation. |
| Literal FaceChannel model-runtime dependencies | 10 | Validate the TensorFlow/Keras/interpreter/checkpoint combination before a migration. Preserve the existing scientific implementation and model inputs/outputs. |
| Core progress tooling | 1 low | Check the patched tqdm version's interpreter support and CLI behavior independently. |

The immediate operational boundary is to avoid supplying untrusted serialized
models, pickle/joblib objects, downloaded NLP artifacts or image inputs to the
historical runtime until their relevant input paths have been reviewed. This
does not classify the existing application as remotely exploitable or safe.

## Patch candidates and limits

The alert metadata lists a first patched version for **538 findings**. Five have
no first patch recorded: three NLTK findings, one Keras finding and one disputed
scikit-learn deserialization finding. A missing patch field remains unresolved;
it is not a claim that an arbitrary newer release fixes the issue.

The following are bounded metadata/range checks, **not validated upgrades**:

- [urllib3 1.26.19](https://pypi.org/project/urllib3/1.26.19/) advertises Python 2.7
  support. It falls outside 3 of the 10 reported vulnerable ranges; 7 remain.
- [Pillow 6.2.2](https://pypi.org/project/Pillow/6.2.2/) advertises Python 2.7
  support. It falls outside 6 of the 54 reported ranges; 48 remain. Crop and
  preprocessing behavior still require comparison.
- [TensorFlow 2.4.4](https://pypi.org/project/tensorflow/2.4.4/) is a same-branch
  candidate. It falls outside 196 of the 382 reported ranges; 186 remain. This
  arithmetic does not resolve the manifest attribution or establish model parity.
- The checked patch metadata requires newer interpreters for
  [joblib 1.2.0](https://pypi.org/project/joblib/1.2.0/) (Python 3.7+),
  [NLTK 3.10.3](https://pypi.org/project/nltk/3.10.3/) and
  [Requests 2.33.0](https://pypi.org/project/requests/2.33.0/) (Python 3.10+), and
  [Keras 3.15.0](https://pypi.org/project/keras/3.15.0/) (Python 3.11+).
  They cannot be treated as universal updates to the Python 2.7/legacy ML stack.

Before accepting a patch, check the advisory's affected operation, interpreter
and dependency constraints, then run a focused regression in the selected
environment. Keep any numerical/model migration separate from this maintenance
review. GitHub's [alert review guidance](https://docs.github.com/en/code-security/how-tos/manage-security-alerts/manage-dependabot-alerts/view-dependabot-alerts)
provides the advisory and version details used for that decision.

Consult the repository's Dependabot view for current advisory records; the
counts above are a dated inventory. Full per-alert exports and local audit
receipts are kept outside the submission source. [Development setup](maintenance.md)
and [tested environments](tested-environments.md) describe the corresponding
installation and model/SDK profiles.
