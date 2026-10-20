# FaceChannel and CLIFER Perception Service

**TL;DR:** This service supplies affect predictions for Solver-CL, using either the static FaceChannel baseline or CLIFER's personalized continual-learning path. It exposes the session and capture operations described below.

**Keywords:** facial expression recognition, FaceChannel, CLIFER, continual learning, valence-arousal estimates

[Study overview](../../../README.md) · [Source architecture](../../ARCHITECTURE.md) · [Tested environments](../../../docs/tested-environments.md)

## Requirements

- A separate legacy TensorFlow/Keras environment. Historical TensorFlow 2.4 tested builds use Python 3.6–3.8; select the interpreter and platform together with the matching build.
- The original GPU setup specified **CUDA 11.0**. The separately tested CPU inference profile uses **Python 3.6.9**, **Intel TensorFlow 2.4.0**, **Keras 2.4.0**, **NumPy 1.19.4**, **h5py 2.10.0** and **OpenCV 4.5.3** on Linux/amd64. CUDA is not required for that CPU profile.
- The original FaceChannel model, detector assets and CAAE checkpoints, plus a compatible camera and the matching lab adapters for an interactive session.

See [tested environments](../../../docs/tested-environments.md#facechannel-cpu-inference-profile) for exact versions, asset hashes and the bounded inference check. That check loaded the model and processed generated crops; it did not validate live capture, CAAE generation, GWR training or a complete negotiation session.

## Installation

1. Prepare the perception environment separately from the modern verification/runtime host and the robot SDK. The historical version references are [Python 3.6](https://www.python.org/downloads/release/python-360/) and the [TensorFlow GPU installation guide](https://www.tensorflow.org/install/gpu); use the recorded [compatibility profiles](../../../docs/tested-environments.md) when selecting the actual interpreter and libraries.
2. Keep this directory's `requirements.txt` as the original dependency declaration. The historical command was `pip3 install -r requirements.txt`; the file includes both TensorFlow providers and unpinned packages, so it is not a validated single-step installation recipe for an arbitrary current environment. The recorded CPU profile describes the provider selection used for its inference check.
3. Place the matching CAAE checkpoint files in `CLModel/checkpoint/`. These files are already tracked in a full checkout but may be omitted by a source-focused sparse checkout. Install the matching FaceChannel model and detector assets at their expected `FaceChannel/FaceChannelV1/TrainedNetworks/` paths, verifying their identities before loading them.

`Application.py` is the Flask service entry point. Its built-in server binds to `127.0.0.1` with debug disabled. Missing FaceChannel models require an explicitly trusted installation; they are not downloaded implicitly. [Maintenance guidance](../../../docs/maintenance.md#model-assets-and-persistence) describes pinned acquisition, archive checks and trusted local GWR state. Keep captured participant images and learned session state private.

## Usage

The existing functions accept **GET** or **POST** HTTP requests and return **JSON** responses. In the URL examples below, use `<IP>=127.0.0.1` and `<PORT>=5000` for the built-in local server, and replace `<ExperimentID>` with a canonical nonnegative integer. The optional `<CameraID>` selects the local camera; its default is 0.

The same operations also accept JSON bodies at `/join`, `/start`, `/stop`, `/next` and `/exit`. Supply `{"id": 1}` and, for `/join`, optionally `"camera_id": 0`. For example, after preparing the service and camera:

```sh
curl -X POST http://127.0.0.1:5000/join \
  -H 'Content-Type: application/json' \
  -d '{"id": 1, "camera_id": 0}'
```

Local GET/POST clients retain their URL and JSON forms. Browser-origin and Host checks run before experiment side effects. If `SOLVER_API_TOKEN` is configured, send it in `X-Solver-Token` or an `Authorization: Bearer ...` header. See [local API security](../../../docs/maintenance.md#local-api-security) for explicit Host configuration and access limits.

- **Join**

  Initializes an experiment and its first helper session:

  ```text
  http://<IP>:<PORT>/join/<ExperimentID>
  ```

  Optionally, specify a camera:

  ```text
  http://<IP>:<PORT>/join/<CameraID>/<ExperimentID>
  ```

  Successful response: `{"status": true}`.

- **Start**

  Starts the current capture window. The capture code saves frames and face crops in the session's round directory:

  ```text
  http://<IP>:<PORT>/start/<ExperimentID>
  ```

  Successful response: `{"status": true}`.

- **Stop**

  Stops the current capture window, runs the selected model's predictions and queues local continual-learning work. Returns the mean affect predictions for that window:

  ```text
  http://<IP>:<PORT>/stop/<ExperimentID>
  ```

  Successful response: `{"status": true, "predictions": {"Arousal": float, "Valance": float}}`. Here `float` denotes a JSON number; `Valance` preserves the existing API spelling.

- **Next**

  Call at the end of the first helper session, after **Stop** has ended its final capture window. Closes that session, waits for its queued training work and creates the next one. The helper starts with the static **FaceChannel** baseline and uses the personalized **CL model** after this transition:

  ```text
  http://<IP>:<PORT>/next/<ExperimentID>
  ```

  Successful response: `{"status": true}`. Training is queued by **Stop**; **Next** waits during session closure rather than starting a separate fixed-duration training sequence.

- **Exit**

  Closes the active experiment session and removes it from the service's active experiments:

  ```text
  http://<IP>:<PORT>/exit/<ExperimentID>
  ```

  Successful response: `{"status": true}`.

The FC→CL order above describes this service's helper. The paper's experiment also includes demo calibration and counterbalanced condition order, which require the original lab orchestration. These endpoints alone do not implement that complete study protocol.

## Solver and Robot Integration

The optional [compatibility runtime](../../../docs/runtime.md#policy-and-perception-contract) requires separate raw and normalized `Arousal`/`Valance` dictionaries at the original round boundary, together with session, offer, capture-window, model and normalization metadata. The mean pair returned by HTTP **Stop** cannot supply both channels on its own. A matching trusted lab provider must expose those existing values while preserving condition selection, training eligibility and model publication.

Keep the perception stack separate from the robot worker. The retained `naoqi.ALProxy` bridge uses a matching **Python 2.7/NAOqi** SDK; the tested Linux import profile used **pynaoqi 2.8.7.4** and **Python 2.7.18**. The separate modern **qi 3.1.5** import check does not provide that legacy API. See [tested environments](../../../docs/tested-environments.md) and the [optional robot worker](../../../docs/runtime.md#optional-robot-worker) for the recorded boundaries. Software and model probes do not establish physical robot delivery or complete-session parity.
