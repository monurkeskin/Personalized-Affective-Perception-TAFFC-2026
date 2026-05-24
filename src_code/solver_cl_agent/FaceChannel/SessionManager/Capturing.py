from FaceChannel.FaceChannelV1.imageProcessingUtil import imageProcessingUtil
import cv2
from PIL import Image as PILImage


IMAGE_CUT_SIZE = (96, 96)


class Capturing:
    save_path_format: str
    save_cut_path_format: str
    save_frame_path_format: str
    camera_id: int
    face_model: imageProcessingUtil
    cap: cv2.VideoCapture

    def __init__(self, camera_id: int = 0):
        self.save_path_format = ""
        self.save_cut_path_format = ""
        self.save_frame_path_format = ""

        self.camera_id = camera_id

        self.face_model = imageProcessingUtil()

        self.cap = cv2.VideoCapture(self.camera_id)

    def run(self, id: int):
        """
            Get face from camera
        :param id: Image index
        :return: Face
        """
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_id)

        ret, frame = self.cap.read()

        if not ret:
            return None

        dets, face = self.face_model.detectFace(frame)

        cv2.imwrite(self.save_path_format % id, face)
        cv2.imwrite(self.save_frame_path_format % id, frame)

        face_cut = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_cut = PILImage.fromarray(face_cut).resize(IMAGE_CUT_SIZE)
        face_cut.save(self.save_cut_path_format % id)

        return self.face_model.preProcess(face)

    def close(self):
        """
            Close the camera
        :return: None
        """
        if self.cap is not None:
            self.cap.release()

        self.cap = None
