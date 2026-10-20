# Tested Software and SDK Environments

Choose the profile for the component you want to run: modern host/analysis,
macOS qi, legacy ALProxy or FaceChannel CPU inference. Keep these environments
separate. The versions and checks below were recorded on 20 October 2026; a
complete live study session additionally needs the matching camera, online-learning
and physical-robot integration described in [runtime setup](runtime.md).

## Modern host and optional analysis

The tested host uses Python 3.12.15 on macOS arm64 and Ubuntu 24.04 x86-64.
From the repository root, prepare its isolated environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-test.lock
.venv/bin/python -m pip check
.venv/bin/python -m pytest tests -q
.venv/bin/python -m submission_runtime doctor
.venv/bin/python -m submission_runtime smoke
.venv/bin/python verify_all_results.py
```

The modern host profile passed 83 tests and 103 subtests locally and in
[CI](https://github.com/monurkeskin/Solver-CL-Agent/actions/runs/39441201164).
The existing public-data verifier passed 99 checks. The suite includes actual
Solver decisions with artificial affect and inert output, plus separately
identified synthetic transport/startup fixtures. See [runtime setup](runtime.md)
for the headless example and the preparation/decision timeout distinction.

The optional `requirements-analysis.lock` was installed in another Python
3.12.15 arm64 environment; dependency checks and imports passed. Recorded versions
were NumPy 2.5.4, SciPy 1.18.1, pandas 2.3.3, statsmodels 0.15.0,
scikit-learn 1.9.1, Matplotlib 3.11.2, seaborn 0.13.2 and openpyxl 3.1.5.
Use that lock in its own environment. These modern locks do not install the historical
TensorFlow or robot SDK runtime.

## Native macOS qi import check

The official [qi 3.1.5 package](https://pypi.org/project/qi/3.1.5/) imported on
macOS arm64 with Python 3.12.15. A `qi.Session()` was constructed, confirmed
disconnected and closed. This API does not supply the legacy `naoqi.ALProxy`
used by the submission's Python 2.7 worker.

The tested wheel is `qi-3.1.5-cp312-cp312-macosx_13_0_arm64.whl`
(6,733,821 bytes), with SHA-256:

```text
f2130bb9fa47c618727112cc9b19d8c4d158365e9fe187198390126ef607155e
```

Obtain the matching wheel from the official package release and compare its
digest before installation. Keep the wheel and environment outside the checkout.
Set `SOLVER_QI_WHEEL` to that local file and `SOLVER_QI_ENV` to a new private
environment directory. These portable commands
follow the recorded isolated installation and disconnected probe:

```sh
shasum -a 256 "$SOLVER_QI_WHEEL"
uv venv --python python3.12 "$SOLVER_QI_ENV"
uv pip install --python "$SOLVER_QI_ENV/bin/python" --no-index --no-deps "$SOLVER_QI_WHEEL"
"$SOLVER_QI_ENV/bin/python" -c 'import qi; s = qi.Session(); assert not s.isConnected(); s.close()'
```

The official implementation is [Aldebaran libqi-python](https://github.com/aldebaran/libqi-python).
The wheel version, filename and digest identify the artifact used here.

## Legacy Linux ALProxy import check

The separate pynaoqi 2.8.7.4 SDK imported in Linux x86-64/Python 2.7.18 under
Docker on the arm64 Mac. Its ELF libraries are Linux binaries. This check
imported `qi`, `naoqi` and `ALProxy`; it constructed no proxy or connection.

The official archive is `pynaoqi-python2.7-2.8.7.4-linux64-20210819_141148.tar.gz`
at [aldebaran/nao6-doc-sdk commit fb90c6d](https://github.com/aldebaran/nao6-doc-sdk/tree/fb90c6de7dfe99198ceb5ece2fe8f555891cb832).
Its size is 208,951,852 bytes and its SHA-256 matches the official Git LFS pointer:

```text
ad60c9336376bd56d3ab2fbcbb9f249f65747bf65869624a2ce4b9112cc50cd9
```

After separately obtaining, verifying and extracting that archive outside the
checkout, set `SOLVER_NAOQI_ROOT` to its extracted SDK root. This import probe uses
the recorded image digest and library order; it supplies no robot address:

```sh
docker run --rm --platform linux/amd64 --network none --cpus 1 --memory 1g \
  --read-only --tmpfs /tmp:rw,size=64m --cap-drop ALL --security-opt no-new-privileges \
  --mount "type=bind,src=$SOLVER_NAOQI_ROOT,dst=/sdk,readonly" \
  -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/sdk/lib/python2.7/site-packages \
  -e LD_LIBRARY_PATH=/usr/local/lib:/sdk/lib \
  python@sha256:6c1ffdff499e29ea663e6e67c9b6b9a3b401d554d2c9f061f9a45344e3992363 \
  python -c 'import qi, naoqi; from naoqi import ALProxy; assert callable(ALProxy)'
```

Putting `/usr/local/lib` first fixed the observed `_PySlice_Unpack` import error
caused by the SDK's bundled libpython shadowing the interpreter library. No
binary or SDK source was changed. Import success establishes this container's
packaging/ABI compatibility; firmware compatibility, speech and behavior
delivery still require the matching device. SDK files are not redistributed here.

## FaceChannel CPU inference profile

The successful isolated Linux/amd64 profile used Python 3.6.9,
**Intel TensorFlow 2.4.0**, Keras 2.4.0, NumPy 1.19.4, h5py 2.10.0 and
OpenCV 4.5.3 (`opencv-python-headless==4.5.3.56`). Its Docker preparation began
with the official image:

```text
tensorflow/tensorflow@sha256:5e5614ec6ce3213bff3dbc48ff63130587befed61ba4426546d05175aec5cbd5
```

The recorded build installed `keras==2.4.0` and
`opencv-python-headless==4.5.3.56`, then uninstalled `tensorflow` and installed
`intel_tensorflow==2.4.0` with `--no-deps`. These legacy package steps were
version-pinned, not an independently hashed offline lock. The standard TensorFlow
2.4 CPU attempt failed because its MaxPool did not support this model's NCHW
layout. The Intel profile succeeded with the original `(2, 1, 64, 64)` input;
changing the inputs to NHWC failed the serialized model's shape contract.

The operator supplied a trusted `DimensionalFaceChannel.h5`, 18,620,352 bytes:

```text
4eb40cd61df4748984fc991112004c4f56285758dcfafb5e6110fb7909eb2573
```

Verify this identity before loading a separately supplied model. The H5 metadata
records Keras 2.2.4 and a TensorFlow backend; it does not identify the execution
runtime. No H5, SDK archive or participant image is included in this guide.

The bounded model check ran with no network, read-only source
and asset mounts, one CPU and a 3 GiB limit. It loaded the original constructor
and detector assets, applied original preprocessing to two generated BGR crops,
returned finite `arousal_output`/`valence_output` values, and ran the frozen dense
encoder with output shape `(2, 200)`. Detector forward on a camera or scene was
not exercised. The model check used a local fixture rather than a repository CLI; the profile
records its prerequisites and observed behavior, not a complete camera-session
installation recipe.

The three CAAE checkpoint files matched their Git LFS identities:

| File under `src_code/solver_cl_agent/FaceChannel/CLModel/checkpoint/` | Bytes | SHA-256 |
| --- | ---: | --- |
| `01_model.data-00000-of-00001` | 138,455,772 | `98d8e3d9f9063d5ffd7910c05ad5c85ebbd9ff9399ddc22ed96b660ded0f8924` |
| `01_model.index` | 2,302 | `50c37bd36c8e5360012c0eaba77db5474f5d84044dd8066d398bbc05702283d0` |
| `01_model.meta` | 118,686,629 | `b28af5b529ac5183070e5d3b20e38966b9c38ded58efd403a3403ed9a4b3903b` |

Listing checkpoint metadata found 65 variables. CAAE restore, generation and
training were not run. The checks also exclude live capture, calibration,
normalization, GWR execution/training, asynchronous model publication and physical
robot delivery. The original `main.py` still requires the lab interaction
adapters. See [historical runtime boundaries](maintenance.md#historical-runtime-boundaries)
before connecting these separate profiles into a study installation.
