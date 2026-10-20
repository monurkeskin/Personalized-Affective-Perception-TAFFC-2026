"""Exercise the retained capture/session methods without camera or model access."""
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest


FACE = Path(__file__).resolve().parents[1] / "src_code/solver_cl_agent/FaceChannel"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def camera_methods(monkeypatch):
    events = []

    def module(name, **attributes):
        value = types.ModuleType(name)
        value.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    class Image:
        def resize(self, size):
            events.append(("resize", size))
            return self

        def save(self, path):
            events.append(("save", path))

    module("cv2", COLOR_BGR2RGB=1, VideoCapture=object,
           imwrite=lambda path, image: events.append(("write", path)),
           cvtColor=lambda image, mode: image)
    module("PIL", Image=types.SimpleNamespace(fromarray=lambda image: Image()))
    module("FaceChannel.FaceChannelV1.imageProcessingUtil", imageProcessingUtil=object)
    capturing = load_module("camera_test_capturing", FACE / "SessionManager/Capturing.py")
    module("SessionManager.Capturing", Capturing=capturing.Capturing)
    module("SessionManager.CLTrainManager", TrainManager=object)
    module("FaceChannel.FaceChannelV1.FaceChannelV1", FaceChannelV1=object)
    module("CLModel.cl_model_c3_faster", predict=lambda **kwargs: None)
    graph = types.SimpleNamespace(as_default=nullcontext)
    v1 = types.SimpleNamespace(
        get_default_graph=lambda: graph,
        InteractiveSession=lambda **kwargs: object(),
        keras=types.SimpleNamespace(backend=types.SimpleNamespace(set_session=lambda session: None)),
    )
    module("tensorflow", compat=types.SimpleNamespace(v1=v1))
    module("pandas", DataFrame=object)
    session = load_module("camera_test_session", FACE / "SessionManager/Session.py")

    def make_capture(read, processed):
        capture = capturing.Capturing.__new__(capturing.Capturing)
        capture.cap = types.SimpleNamespace(read=read)
        capture.save_path_format = "faces/%d.jpg"
        capture.save_frame_path_format = "frames/%d.jpg"
        capture.save_cut_path_format = "faces_cut/%d.jpg"

        def detect(frame):
            events.append(("detect",))
            return (), frame

        def preprocess(face):
            events.append(("preprocess",))
            return processed

        capture.face_model = types.SimpleNamespace(detectFace=detect, preProcess=preprocess)
        return capture

    return session.Session, make_capture, events


def test_failed_camera_read_skips_detection_and_image_writes(camera_methods):
    _, make_capture, events = camera_methods
    capture = make_capture(lambda: (False, None), processed=None)
    assert capture.run(17) is None
    assert events == []


def test_session_keeps_failed_camera_read_as_none(camera_methods):
    Session, make_capture, events = camera_methods
    session = Session.__new__(Session)
    session.capturing = make_capture(lambda: (False, None), processed=None)
    assert session.get_face(17) is None
    assert events == []


@pytest.mark.parametrize("dtype", [np.uint8, np.float64])
def test_valid_face_keeps_its_values_and_shape_as_float32(camera_methods, dtype):
    Session, make_capture, events = camera_methods
    processed = np.array([[[0, 1], [2, 3]]], dtype=dtype)
    original = processed.copy()
    session = Session.__new__(Session)
    session.capturing = make_capture(lambda: (True, np.zeros((2, 2, 3), dtype=np.uint8)), processed)

    face = session.get_face(4)

    assert face.dtype == np.float32
    assert face.shape == (1, 2, 2)
    np.testing.assert_array_equal(face, [[[0, 1], [2, 3]]])
    np.testing.assert_array_equal(processed, original)
    assert events == [("detect",), ("write", "faces/4.jpg"),
                      ("write", "frames/4.jpg"), ("resize", (96, 96)),
                      ("save", "faces_cut/4.jpg"), ("preprocess",)]


def test_recording_buffer_excludes_failed_reads_and_keeps_frame_index(camera_methods):
    Session, make_capture, events = camera_methods
    session = Session.__new__(Session)
    session._capturing = True
    session._faces = []
    reads = iter([(False, None), (True, np.zeros((2, 2, 3), dtype=np.uint8)), (False, None)])
    read_count = 0

    def read():
        nonlocal read_count
        read_count += 1
        if read_count == 3:
            session._capturing = False
        return next(reads)

    processed = np.array([[[0.0, 0.25], [0.5, 1.0]]], dtype=np.float64)
    session.capturing = make_capture(read, processed)
    session._Session__run_capturing()

    assert read_count == 3
    assert len(session._faces) == 1
    assert session._faces[0].dtype == np.float32
    np.testing.assert_array_equal(session._faces[0], [[[0.0, 0.25], [0.5, 1.0]]])
    assert ("write", "faces/0.jpg") in events
    assert sum(event[0] == "preprocess" for event in events) == 1
