"""
CLIFER Flask Application

REST API for face channel experiments and CL model integration.
"""
import os

# Windows-specific CUDA paths (uncomment if needed on Windows)
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin")
# os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/libnvvp")



import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import SessionManager
from flask import Flask, request, jsonify, render_template

application = Flask(__name__)

experiments = {}


@application.route("/", methods=["GET"])
def main_page():
    """
        Test Page
    :return:
    """
    return render_template("index.html")


@application.route("/join/<id>", methods=["GET", "POST"])
def join(id: int):
    try:
        if id in experiments.keys():
            return jsonify({"status": False})

        experiments[id] = SessionManager.Experiment(int(id))

        return jsonify({"status": True})
    except:
        return jsonify({"status": False})


@application.route("/join/<camera_id>/<id>", methods=["GET", "POST"])
def join_with_camera(camera_id: int, id: int):
    try:
        if id in experiments.keys():
            return jsonify({"status": False})

        experiments[id] = SessionManager.Experiment(int(id), int(camera_id))

        return jsonify({"status": True})
    except:
        return jsonify({"status": False})


@application.route("/join", methods=["GET", "POST"])
def join_with_request():
    try:
        id = request.json["id"]

        camera_id = request.json.get("camera_id", 0)

        if id in experiments.keys():
            return jsonify({"status": False})

        experiments[id] = SessionManager.Experiment(int(id), int(camera_id))

        return jsonify({"status": True})
    except:
        return jsonify({"status": False})


@application.route("/start/<id>", methods=["GET", "POST"])
def start(id: int):
    try:
        experiments[id].start()

        return jsonify({"status": True})
    except:
        return jsonify({"status": False})


@application.route("/start", methods=["GET", "POST"])
def start_with_request():
    try:
        id = request.json["id"]

        experiments[id].start()

        return jsonify({"status": True})
    except:
        return jsonify({"status": False})


@application.route("/stop/<id>", methods=["GET", "POST"])
def stop(id: int):
    predictions = experiments[id].stop()

    return jsonify({"status": True, "predictions": predictions})


@application.route("/stop", methods=["GET", "POST"])
def stop_with_request():
    try:
        id = request.json["id"]

        predictions = experiments[id].stop()

        return jsonify({"status": True, "predictions": predictions})
    except:
        return jsonify({"status": False, "predictions": {}})


@application.route("/next/<id>", methods=["GET", "POST"])
def next(id: int):
    try:
        experiments[id].next()

        return jsonify({"status": True})

    except:
        return jsonify({"status": False})


@application.route("/next", methods=["GET", "POST"])
def next_with_request():
    try:
        id = request.json["id"]

        experiments[id].next()

        return jsonify({"status": True})

    except:
        return jsonify({"status": False})


@application.route("/exit/<id>", methods=["GET", "POST"])
def exit(id: int):
    try:
        experiments[id].close()
        del experiments[id]

        return jsonify({"status": True})
    except:
        return jsonify({"status": False})


@application.route("/exit", methods=["GET", "POST"])
def exit_with_request():
    try:
        id = request.json["id"]

        experiments[id].close()
        del experiments[id]

        return jsonify({"status": True})
    except:
        return jsonify({"status": False})


if __name__ == "__main__":
    application.run(threaded=False)
