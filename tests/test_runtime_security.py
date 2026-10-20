"""Security boundaries with actual Flask routes and inert hardware adapters."""
from contextlib import contextmanager
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FACE = ROOT / "src_code/solver_cl_agent/FaceChannel"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def local_app(**environment):
    events = []

    class Experiment:
        def __init__(self, *args):
            events.append(("join", args))

        def start(self):
            events.append(("start",))

        def stop(self):
            events.append(("stop",))
            return {"Arousal": .3, "Valance": -.1}

        def next(self):
            events.append(("next",))

        def close(self):
            events.append(("close",))

    tf = types.ModuleType("tensorflow")
    tf.config = types.SimpleNamespace(list_physical_devices=lambda _: [])
    session = types.ModuleType("SessionManager")
    session.Experiment = Experiment
    with patch.dict(os.environ, {"SOLVER_API_TOKEN": "", "SOLVER_ALLOWED_HOSTS": "", **environment}):
        security = load_module("local_api_security", FACE / "local_api_security.py")
        with patch.dict(sys.modules, {"tensorflow": tf, "SessionManager": session, "local_api_security": security}):
            module = load_module("solver_test_application", FACE / "Application.py")
            module.application.testing = True
            yield module.application.test_client(), events


class LocalApiTests(unittest.TestCase):
    def test_legacy_get_post_and_prediction_contract_survive(self):
        with local_app() as (client, events):
            self.assertTrue(client.get("/join/12").json["status"])
            self.assertTrue(client.post("/start", json={"id": 12}).json["status"])
            self.assertEqual(client.get("/stop/12").json["predictions"], {"Arousal": .3, "Valance": -.1})
            self.assertTrue(client.post("/next", json={"id": 12}).json["status"])
            self.assertTrue(client.get("/exit/12").json["status"])
            self.assertEqual([e[0] for e in events], ["join", "start", "stop", "next", "close"])

    def test_cross_origin_requests_never_open_camera(self):
        with local_app() as (client, events):
            for headers in [{"Origin": "https://untrusted.example"}, {"Origin": "null"}, {"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "same-site"}]:
                self.assertEqual(client.get("/join/1", headers=headers).status_code, 403)
            self.assertEqual(events, [])

    def test_rebinding_host_and_unauthenticated_remote_are_rejected(self):
        with local_app() as (client, events):
            self.assertEqual(client.get("/join/1", base_url="http://untrusted.example").status_code, 400)
            self.assertEqual(client.get("/join/1", environ_base={"REMOTE_ADDR": "192.0.2.5"}).status_code, 403)
            self.assertEqual(events, [])

    def test_optional_token_is_enforced_before_any_side_effect(self):
        with local_app(SOLVER_API_TOKEN="test-only-token") as (client, events):
            self.assertEqual(client.get("/join/1").status_code, 401)
            self.assertEqual(client.get("/join/1", headers={"X-Solver-Token": "wrong"}).status_code, 401)
            self.assertEqual(events, [])
            self.assertTrue(client.get("/join/1", headers={"Authorization": "Bearer test-only-token"}).json["status"])
            self.assertEqual(len(events), 1)

    def test_canonical_id_prevents_reinitializing_an_existing_experiment(self):
        with local_app() as (client, events):
            self.assertTrue(client.post("/join", json={"id": 1}).json["status"])
            self.assertFalse(client.get("/join/001").json["status"])
            for value in [True, -1, "../2", 1.2]:
                self.assertFalse(client.post("/join", json={"id": value}).json["status"])
            self.assertEqual(len(events), 1)


class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.assets = load_module("solver_test_model_assets", FACE / "FaceChannel/FaceChannelV1/model_assets.py")

    def archive(self, path, name="TrainedNetworks/model.h5", kind=tarfile.REGTYPE):
        with tarfile.open(path, "w") as archive:
            info = tarfile.TarInfo(name)
            info.type = kind
            payload = b"synthetic model bytes, never loaded"
            if kind == tarfile.REGTYPE:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            else:
                info.linkname = "../outside"
                archive.addfile(info)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_valid_asset_roundtrip_is_byte_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); archive = root / "assets.tar"; destination = root / "install"
            destination.mkdir(); digest = self.archive(archive)
            self.assets.install_archive(archive, destination, digest)
            self.assertEqual((destination / "TrainedNetworks/model.h5").read_bytes(), b"synthetic model bytes, never loaded")
            self.assets.install_archive(archive, destination, digest)

    def test_archive_traversal_links_outside_paths_and_wrong_digest_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); archive = root / "assets.tar"; destination = root / "install"; destination.mkdir()
            for name, kind in [("../outside", tarfile.REGTYPE), ("/outside", tarfile.REGTYPE), ("TrainedNetworks/link", tarfile.SYMTYPE), ("TrainedNetworks/link", tarfile.LNKTYPE), ("Application.py", tarfile.REGTYPE)]:
                digest = self.archive(archive, name, kind)
                with self.assertRaises(ValueError):
                    self.assets.install_archive(archive, destination, digest)
                self.assertEqual(list(destination.iterdir()), [])
            self.archive(archive)
            with self.assertRaises(ValueError):
                self.assets.install_archive(archive, destination, "0" * 64)

    def test_existing_model_is_never_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); archive = root / "assets.tar"; destination = root / "install"
            model = destination / "TrainedNetworks/model.h5"; model.parent.mkdir(parents=True); model.write_bytes(b"original")
            digest = self.archive(archive)
            with self.assertRaises(ValueError):
                self.assets.install_archive(archive, destination, digest)
            self.assertEqual(model.read_bytes(), b"original")

    def test_default_acquisition_does_not_access_network(self):
        with patch.dict(os.environ, {"SOLVER_ALLOW_MODEL_DOWNLOAD": "", "SOLVER_FACECHANNEL_ARCHIVE_SHA256": ""}), patch.object(self.assets, "urlopen") as network:
            with self.assertRaises(FileNotFoundError):
                self.assets.acquire_assets("https://example.com/assets.tar", ROOT)
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
