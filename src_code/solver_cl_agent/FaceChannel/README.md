# FaceChannel: Facial Affect Recognition Module
Facial expression recognition module for automated negotiation

## Requirements
- CUDA 11.0
- Python 3.6 or higher

## Installation
- **Python 3.6**: [Download](https://www.python.org/downloads/release/python-360/)
- **CUDA 11.0**: [Download](https://www.tensorflow.org/install/gpu)
- To install the required Python libraries:
  ```sh
  pip3 install -r requirements.txt
  ```
- Copy checkpoint files into `CLModel/checkpoint/`

## Usage

Required functions can be called via **GET** or **POST** HTTP requests and return responses in **JSON** format.

- **Join**
  To initialize a negotiation session, invoke the Join function with an experiment ID:

  ```sh
  http://<IP>:<PORT>/join/<ExperimentID>
  ```

  Optionally, specify a Camera ID:

  ```sh
  http://<IP>:<PORT>/join/<CameraID>/<ExperimentID>
  ```

  > Note: Returns `{"status": true}`
  
- **Start**
  Starts image capture and saves raw frames to the log directory. Invoke the Start function with the experiment ID:
  ```sh
  http://<IP>:<PORT>/start/<ExperimentID>
  ```
  
  > Note: Returns `{"status": true}`

- **Stop**
  Stops image capture and triggers local Continual Learning (CL) training. Returns the average affect predictions. Invoke the Stop function with the experiment ID:
  ```sh
  http://<IP>:<PORT>/stop/<ExperimentID>
  ```
  
  > Note: Returns `{"status": true, "predictions": {"Arousal": float, "Valence": float}}`
  
- **Next**
  Must be called at the end of the first session. The first session uses the static **FaceChannel** baseline, while the second session uses the personalized **CL Model**.
  This function also initiates the **CL Model** training sequence, which may take a few minutes to complete.
  Invoke the Next function with the experiment ID:
  ```sh
  http://<IP>:<PORT>/next/<ExperimentID>
  ```
  
  > Note: Returns `{"status": true}`
  
- **Exit**
  Terminates the active experiment session. Invoke the Exit function with the experiment ID:
  ```sh
  http://<IP>:<PORT>/exit/<ExperimentID>
  ```
  
  > Note: Returns `{"status": true}`
