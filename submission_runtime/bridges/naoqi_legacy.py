"""Explicit NAO/Pepper worker for an operator-prepared Python 2.7 ALProxy SDK.

No robot connection occurs on import or construction. Preflight reads runtime
and installed behaviors. Stiffness changes require a separate explicit opt-in
and request. This source has no hardware validation receipt.
"""
from __future__ import print_function

import argparse
import math
import sys

try:
    from .protocol import serve
except (ImportError, ValueError):
    from protocol import serve

try:
    string_type = basestring
except NameError:
    string_type = str


class NaoqiDriver:
    def __init__(self, host, port=9559, family="nao", behaviors=(), allow_stiffness=False):
        if not isinstance(host, string_type) or not host or family not in ("nao", "pepper"):
            raise ValueError("An explicit host and nao/pepper family are required.")
        if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
            raise ValueError("Invalid robot port.")
        self.host = host
        self.port = port
        self.family = family
        if (not isinstance(behaviors, (list, tuple))
                or any(not isinstance(item, string_type) or not item or len(item) > 500
                       or any(ord(char) < 32 for char in item) for item in behaviors)):
            raise ValueError("Robot behaviors must be an explicit bounded list of names.")
        self.behaviors = set(behaviors)
        self.allow_stiffness = bool(allow_stiffness)
        self.system = self.tts = self.manager = self.motion = None
        self.ready = False

    def call(self, operation, payload):
        if operation == "hello":
            self.ready = False
            if sys.version_info[:2] != (2, 7):
                raise RuntimeError("Use the separately prepared Python 2.7 ALProxy SDK runtime.")
            from naoqi import ALProxy
            self.system = ALProxy("ALSystem", self.host, self.port)
            self.tts = ALProxy("ALTextToSpeech", self.host, self.port)
            self.manager = ALProxy("ALBehaviorManager", self.host, self.port)
            installed = set(self.manager.getInstalledBehaviors())
            if self.behaviors - installed:
                raise ValueError("Configured robot behaviors are not installed.")
            if self.allow_stiffness:
                self.motion = ALProxy("ALMotion", self.host, self.port)
            capabilities = ["present", "speech"]
            if self.behaviors:
                capabilities.append("gesture")
            if self.allow_stiffness:
                capabilities.append("stiffness")
            runtime = str(self.system.systemVersion())
            self.ready = True
            return {"family": self.family, "runtime": runtime,
                    "capabilities": capabilities, "assets": sorted(self.behaviors),
                    "hardware_tested": False, "family_source": "operator_configuration"}
        if not self.ready:
            raise ValueError("Prepare the robot worker before issuing a command.")
        if operation == "present":
            if payload.get("face"):
                raise ValueError("The ALProxy worker has no face-display operation.")
            text = payload.get("text", "")
            gesture = payload.get("gesture")
            if not isinstance(text, string_type) or len(text) > 10000:
                raise ValueError("Presentation text must be a bounded string.")
            if gesture is not None and gesture not in self.behaviors:
                raise ValueError("Gesture is not explicitly configured for this worker.")
            if gesture:
                self.manager.runBehavior(gesture)
            if text:
                # Preserve the approved text, including its existing NAOqi markup.
                self.tts.say(text.encode("utf-8"))
            return {"delivered": True, "acknowledgement": "Selected SDK calls returned",
                    "hardware_tested": False}
        if operation == "stiffness":
            if not self.allow_stiffness or self.motion is None:
                raise ValueError("Stiffness control was not explicitly enabled.")
            value = payload.get("value")
            duration = payload.get("duration_seconds", 1.0)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or math.isnan(value) or math.isinf(value) or not 0 <= value <= 1):
                raise ValueError("Stiffness value must lie in [0, 1].")
            if (isinstance(duration, bool) or not isinstance(duration, (int, float))
                    or math.isnan(duration) or math.isinf(duration) or not 0 < duration <= 10):
                raise ValueError("Stiffness duration must lie in (0, 10].")
            self.motion.stiffnessInterpolation("Body", value, duration)
            return {"delivered": True, "acknowledgement": "Explicit stiffness SDK call returned",
                    "hardware_tested": False}
        raise ValueError("Unsupported robot operation.")

    def close(self):
        # Do not stop unrelated robot behaviors or alter autonomous-life settings.
        self.system = self.tts = self.manager = self.motion = None
        self.ready = False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=9559)
    parser.add_argument("--family", choices=("nao", "pepper"), required=True)
    parser.add_argument("--behavior", action="append", default=[])
    parser.add_argument("--allow-stiffness", action="store_true")
    args = parser.parse_args()
    serve(NaoqiDriver(args.host, args.port, args.family, args.behavior, args.allow_stiffness))


if __name__ == "__main__":
    main()
