import numpy as np
import os
import shutil
from SessionManager.Session import Session, SESSIONS_DIR, SAVED_GWRS_DIR


class Experiment:
    experiment_id: int
    sessions: list
    session_id: int
    camera_id: int

    def __init__(self, experiment_id: int, camera_id: int = 0):
        """
            Constructor
        :param experiment_id: Experiment id, int
        :param camera_id: Camera id, int. Default: 0
        """
        self.experiment_id = experiment_id
        self.camera_id = camera_id

        # Prepare Experiment folders
        if not os.path.isdir(SESSIONS_DIR):
            os.mkdir(SESSIONS_DIR)

        if not os.path.isdir(SAVED_GWRS_DIR):
            os.mkdir(SAVED_GWRS_DIR)

        experiment_gwrs = os.path.join(SAVED_GWRS_DIR, "exp_%d/" % experiment_id)

        if os.path.isdir(experiment_gwrs):
            shutil.rmtree(experiment_gwrs)

        os.mkdir(experiment_gwrs)

        self.session_id = 0
        self.sessions = [Session("%d_1" % experiment_id, True, self.camera_id)]

    def get_session(self) -> Session:
        """
            Returns current Session object.
        :return: Current Session
        """
        return self.sessions[self.session_id]

    def next(self) -> Session:
        """
            Use it for start second session
        :return: Second Session
        """

        self.get_session().close()

        self.session_id += 1

        self.sessions.append(
            Session(
                "%d_%d" % (self.experiment_id, self.session_id + 1),
                False,
                self.camera_id,
            )
        )

        return self.get_session()

    def start(self):
        """
            It starts Current Session recording.
        :return: None
        """
        self.get_session().start()

    def stop(self) -> dict:
        """
            It stops Current Session recording and returns the average of predictions.
        :return: Prediction {"Arousal": Float, "Valance": Float}
        """
        predictions = self.get_session().stop()

        result = {
            "Arousal": float(np.mean(predictions[0])),
            "Valance": float(np.mean(predictions[1])),
        }

        return result

    def close(self):
        """
            Close Experiment and current Session
        :return: None
        """
        self.get_session().close()
