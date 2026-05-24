import os
import shutil
import pandas as pd
from SessionManager.Capturing import Capturing
from SessionManager.CLTrainManager import TrainManager
from FaceChannel.FaceChannelV1.FaceChannelV1 import FaceChannelV1
from CLModel.cl_model_c3_faster import predict
import numpy as np
from threading import Thread
import tensorflow as tf

SESSIONS_DIR = "sessions/"
SAVED_GWRS_DIR = "sessions/saved_gwrs/"


# Global Tensorflow Graph and Session for multi-thread operations.
global graph
graph = tf.compat.v1.get_default_graph()
global sess
# sess = tf.compat.v1.Session(graph=graph, config=tf.compat.v1.ConfigProto(allow_soft_placement=True))
sess = tf.compat.v1.InteractiveSession(graph=graph)


class Session:
    session_name: str
    round: int
    log_dir: str
    round_dir: str
    gwr_dir: str
    log_face_channel: pd.DataFrame
    log_face_channel_path: str
    log_cl: pd.DataFrame
    log_cl_path: str
    capturing: Capturing
    face_channel: FaceChannelV1
    _capturing: bool
    use_face_channel: bool
    _faces: list
    thread_capturing: Thread
    training_manager: TrainManager

    def __init__(self, session_name: str, use_face_channel: bool, camera_id: int = 0):
        """
            Constructor
        :param session_name: Session Name for Logging
        :param use_face_channel: Prediction will be done with FaceChannel, or CL Model
        :param camera_id: Camera ID for recording
        :param clean_images: Clean Face and Imagined images after training.
        """

        self.session_name = session_name
        self.use_face_channel = use_face_channel
        self.round = 0

        # Generate Paths
        self.log_dir = os.path.join(SESSIONS_DIR, self.session_name + "/")
        self.round_dir = os.path.join(self.log_dir, "round_%d/")

        experiment_id = session_name.split("_")[0]
        self.gwr_dir = os.path.join(SAVED_GWRS_DIR, "exp_%s/" % experiment_id)

        # Generate Logger
        self.log_face_channel_path = os.path.join(self.log_dir, "face_channel.csv")
        self.log_face_channel = pd.DataFrame(columns=["Round", "Arousal", "Valance"])

        self.log_cl_path = os.path.join(self.log_dir, "cl.csv")
        self.log_cl = pd.DataFrame(columns=["Round", "Arousal", "Valance"])

        # Load Camera Capturing
        self.capturing = Capturing(camera_id)

        # Load FaceChannel Model and Training Manager
        global sess
        global graph

        with graph.as_default():
            tf.compat.v1.keras.backend.set_session(sess)

            self.face_channel = FaceChannelV1(type="Dim")

            self.training_manager = TrainManager(sess, graph)

        # Clean log directory if it exists. Then, create empty folders.
        if os.path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        os.mkdir(self.log_dir)

        self.generate_round_folders()

        self._capturing = False
        self.thread_capturing = None

        self._faces = []

        # Create LOG files
        self.log_face_channel.to_csv(self.log_face_channel_path)
        self.log_cl.to_csv(self.log_cl_path)

    def __run_capturing(self):
        """
            Recording Thread
            Do not call it directly.
        :return: None
        """
        global sess
        global graph

        counter = 0

        with graph.as_default():
            tf.compat.v1.keras.backend.set_session(sess)

            while self._capturing:
                face = self.get_face(counter)

                if face is None:
                    continue

                self._faces.append(face)

                counter += 1

    def start(self):
        """
            Start Recording thread
        :return: None
        """

        self._faces = []
        self._capturing = True

        self.thread_capturing = Thread(target=self.__run_capturing)
        self.thread_capturing.start()

    def close(self):
        """
            Close session and threads
        :return: None
        """
        self.close_camera()

        self._faces = []

        self.training_manager.wait()
        self.training_manager.stop()

    def stop(self) -> list:
        """
            Stop Recording thread. Also, it runs FaceChannel and CL Model.
        :return: Prediction of the model
        """
        if not self._capturing:  # If it is not running, return empty predictions.
            return [[0.0], [0.0]]

        #  Close Thread
        self._capturing = False
        self.thread_capturing.join(timeout=0.1)

        # Current Round Folder
        round_dir = self.round_dir % self.round

        #  Training and Prediction Phrase
        global sess
        global graph

        with graph.as_default():
            tf.compat.v1.keras.backend.set_session(sess)

            if self.use_face_channel:
                predictions = self.predict_face_channel(
                    np.array(self._faces, dtype=np.float32)
                )
                self._faces = []  # Clean Faces

                # Training
                self.training_manager.enqueue(round_dir, self.gwr_dir, self.round)
            else:
                self.predict_face_channel(np.array(self._faces, dtype=np.float32))
                self._faces = []

                predictions = self.predict_cl(round_dir)

                # Training
                self.training_manager.enqueue(round_dir, self.gwr_dir, self.round)

        # Round Update
        self.round += 1
        self.generate_round_folders()

        return predictions

    def get_face(self, id: int) -> np.ndarray:
        """
            Run Session's camera and return Face as Numpy Array
        :param id: Frame Index
        :return: Face Image
        """
        face = self.capturing.run(id)

        return np.array(face, dtype=np.float32)

    def predict_face_channel(self, faces: np.ndarray) -> list:
        """
            Return the prediction of FaceChannel Model
        :param faces: Face images, Numpy Array
        :return: Predictions
        """
        predictions = self.face_channel.predict(faces, preprocess=False)

        self._log_face_channel(predictions)

        return predictions

    def predict_cl(self, log_dir: str) -> list:
        """
            Return the prediction of CL Model
        :return: Predictions
        """
        global sess
        global graph

        predictions = predict(
            log_dir=log_dir, gwrs_path=self.gwr_dir, sess=sess, graph=graph
        )

        self._log_cl(predictions)

        return predictions

    def generate_round_folders(self):
        """
            Generates round folders.
        :return: None
        """
        faces_dir = os.path.join(self.round_dir % self.round, "faces/")
        faces_cut_dir = os.path.join(self.round_dir % self.round, "faces_cut/")
        imagined_dir = os.path.join(self.round_dir % self.round, "imagined/")
        frames_dir = os.path.join(self.round_dir % self.round, "frames/")

        # Generate folders

        os.mkdir(self.round_dir % self.round)
        os.mkdir(faces_dir)
        os.mkdir(faces_cut_dir)
        os.mkdir(imagined_dir)
        os.mkdir(frames_dir)

        # Capturing Formats

        self.capturing.save_path_format = faces_dir + "image_frame_%d.jpg"
        self.capturing.save_cut_path_format = faces_cut_dir + "image_frame_%d.jpg"
        self.capturing.save_frame_path_format = frames_dir + "image_frame_%d.jpg"

    def _log_face_channel(self, predictions: list):
        """
            Log prediction results into face_channel.csv
        :param data: Predictions
        :return: None
        """
        for i in range(len(predictions[0])):
            self.log_face_channel = self.log_face_channel.append(
                {
                    "Round": self.round,
                    "Arousal": predictions[0][i][0],
                    "Valance": predictions[1][i][0],
                },
                ignore_index=True,
            )

        self.log_face_channel.to_csv(self.log_face_channel_path)

    def _log_cl(self, predictions: list):
        """
            Log prediction results into cl.csv
        :param predictions: Predictions
        :return: None
        """
        for i in range(len(predictions[0])):
            self.log_cl = self.log_cl.append(
                {
                    "Round": self.round,
                    "Arousal": predictions[0][i][0],
                    "Valance": predictions[1][i][0],
                },
                ignore_index=True,
            )

        self.log_cl.to_csv(self.log_cl_path)

    def close_camera(self):
        """
            Close camera at the end of session
        :return: None
        """

        if self._capturing:
            self._capturing = False
            self.thread_capturing.join(timeout=0.1)

        self.capturing.close()

        self._faces = []
