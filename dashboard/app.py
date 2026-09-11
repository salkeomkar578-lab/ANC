"""
IGARD-Net Visual Telemetry & Audio Streaming Server.
Flask + Flask-SocketIO application with decoupled audio processing,
live browser audio streaming, and high-throughput file processing.
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Any

from flask import Flask, jsonify, request, render_template, send_from_directory, send_file
from flask_socketio import SocketIO, emit

# Ensure igard_net root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import load_config
from core.state import SystemState
from core.pipeline import IgardNetPipeline
from core.telemetry import TelemetryDispatcher
from backends.backend_manager import BackendManager
from audio.audio_output import AudioOutputSink
from audio.realtime_engine import RealtimeAudioEngine
from file_mode.uploader import FileUploader, UploadValidationError
from file_mode.processor import FileProcessor
from dashboard.audio_stream import create_audio_stream_response

UPLOAD_FOLDER = PROJECT_ROOT / "dashboard" / "uploads"
OUTPUT_FOLDER = PROJECT_ROOT / "dashboard" / "outputs"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "dashboard" / "templates"),
    static_folder=str(PROJECT_ROOT / "dashboard" / "static"),
)
app.config['SECRET_KEY'] = 'igard-net-tactical-defense-2026'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# -------------------------------------------------------------
# System Initialization
# -------------------------------------------------------------
config = load_config()
state = SystemState()

backend_mgr = BackendManager(
    requested_profile=config["hardware"].get("profile", "auto"),
    device_id=config["hardware"].get("cuda_device_id", 0),
)
pipeline = IgardNetPipeline(config=config, state=state, backend=backend_mgr.backend)
telemetry_dispatcher = TelemetryDispatcher(state=state, target_fps=config["dashboard"].get("telemetry_fps", 20))
output_sink = AudioOutputSink(sample_rate=config["audio"]["sample_rate"], channels=1)

audio_engine = RealtimeAudioEngine(
    pipeline=pipeline,
    state=state,
    telemetry_dispatcher=telemetry_dispatcher,
    output_sink=output_sink,
    frame_size=config["audio"]["frame_size"],
    sample_rate=config["audio"]["sample_rate"],
)

file_uploader = FileUploader(upload_dir=UPLOAD_FOLDER)
file_processor = FileProcessor(output_dir=OUTPUT_FOLDER, sample_rate=config["audio"]["sample_rate"])

# Background thread for pushing telemetry to WebSockets
_telemetry_thread = None
_telemetry_stop_event = threading.Event()


def telemetry_broadcaster():
    """Consumes telemetry queue and emits over WebSocket."""
    last_hw_check = 0.0
    while not _telemetry_stop_event.is_set():
        packet = telemetry_dispatcher.get_telemetry(timeout=0.05)
        
        # Periodic hardware stats update
        now = time.time()
        if now - last_hw_check >= 1.0:
            last_hw_check = now
            hw = backend_mgr.get_hardware_telemetry()
            state.cpu_percent = hw["cpu_percent"]
            state.gpu_percent = hw["gpu_percent"]
            state.ram_mb = hw["ram_mb"]

        if packet:
            socketio.emit("telemetry_update", packet)


# Start telemetry broadcaster automatically
_telemetry_thread = threading.Thread(target=telemetry_broadcaster, daemon=True, name="Telemetry_Broadcaster")
_telemetry_thread.start()

# Start real-time audio engine on boot so live mode is immediately responsive
audio_engine.start()


# -------------------------------------------------------------
# HTTP Routes
# -------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    return jsonify(state.to_dict())


@app.route("/api/mode/toggle_before_after", methods=["POST"])
def api_toggle_before_after():
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled", not state.before_after)
    state.set_before_after(enabled)
    return jsonify({
        "status": "ok",
        "before_after": state.before_after,
        "mode_label": "Enhanced Audio" if state.before_after else "Raw Microphone Audio",
    })


@app.route("/api/mode/toggle_autopilot", methods=["POST"])
def api_toggle_autopilot():
    data = request.get_json(silent=True) or {}
    enabled = data.get("enabled", not state.autopilot)
    state.set_autopilot(enabled)
    return jsonify({
        "status": "ok",
        "autopilot": state.autopilot,
    })


@app.route("/api/start_live", methods=["POST"])
def api_start_live():
    audio_engine.start()
    return jsonify({"status": "ok", "mode": "realtime", "message": "Real-time audio engine streaming."})


@app.route("/api/stop", methods=["POST"])
def api_stop():
    audio_engine.stop()
    return jsonify({"status": "ok", "mode": "idle", "message": "Processing paused."})


@app.route("/api/recalibrate", methods=["POST"])
def api_recalibrate():
    pipeline.presence_gate.recalibrate()
    return jsonify({
        "status": "ok",
        "message": "Quiet baseline recalibrated. Measuring ambient floor...",
    })


@app.route("/api/stream/live_audio")
def api_stream_live_audio():
    """Browser connects here for continuous real-time audio audition."""
    return create_audio_stream_response(output_sink)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file submitted in request."}), 400
    file = request.files['file']
    if not file.filename:
        return jsonify({"status": "error", "message": "No file selected."}), 400

    try:
        saved_path, meta = file_uploader.validate_and_save(file)
        return jsonify({
            "status": "ok",
            "message": "File uploaded and validated successfully.",
            "metadata": meta,
        })
    except UploadValidationError as e:
        return jsonify({"status": "error", "message": str(e)}), 400
    except Exception as e:
        return jsonify({"status": "error", "message": f"Server error handling upload: {e}"}), 500


@app.route("/api/process_file", methods=["POST"])
def api_process_file():
    data = request.get_json(silent=True) or {}
    filename = data.get("filename")

    if not filename:
        # Default to repo sample file if available
        sample_path = PROJECT_ROOT / "sample_voice_44k.wav"
        if sample_path.exists():
            target_path = sample_path
        else:
            return jsonify({"status": "error", "message": "No filename specified."}), 400
    else:
        target_path = UPLOAD_FOLDER / Path(filename).name
        if not target_path.exists():
            return jsonify({"status": "error", "message": f"File {filename} not found."}), 404

    # Run processing asynchronously to keep HTTP responsive
    def run_file_proc():
        state.mode = "file"
        state.file_status = "Processing..."
        state.file_progress = 0.0

        def on_file_prog(prog: float, stage_msg: str = "Processing...", stage_idx: int = 1):
            state.file_progress = prog
            state.file_status = stage_msg
            socketio.emit("file_progress", {
                "progress": round(prog, 1),
                "stage": stage_msg,
                "stage_idx": stage_idx,
                "filename": target_path.name,
            })

        try:
            res = file_processor.process_file(
                filepath=target_path,
                pipeline=pipeline,
                state=state,
                progress_callback=on_file_prog,
            )
            socketio.emit("file_completed", res)
        except Exception as e:
            state.file_status = f"Error: {e}"
            socketio.emit("file_error", {"message": str(e)})

    proc_thread = threading.Thread(target=run_file_proc, daemon=True)
    proc_thread.start()

    return jsonify({
        "status": "ok",
        "message": f"File processing initiated for {target_path.name}",
        "filename": target_path.name,
    })


@app.route("/api/audio/upload/<filename>")
def serve_uploaded_file(filename):
    return send_from_directory(str(UPLOAD_FOLDER), Path(filename).name)


@app.route("/api/audio/output/<filename>")
def serve_output_file(filename):
    return send_from_directory(str(OUTPUT_FOLDER), Path(filename).name)


@app.route("/api/audio/download/<filename>")
def download_output_file(filename):
    file_path = OUTPUT_FOLDER / Path(filename).name
    if not file_path.exists():
        return jsonify({"status": "error", "message": "File not found"}), 404
    return send_file(str(file_path), as_attachment=True)


@app.route("/api/audio/sample")
def serve_sample_voice():
    sample_path = PROJECT_ROOT / "sample_voice_44k.wav"
    if sample_path.exists():
        return send_file(str(sample_path))
    return jsonify({"status": "error", "message": "Sample file not found."}), 404


@app.route("/api/devices")
def api_devices():
    devices = []
    sounddevice_ok = False
    try:
        import sounddevice as sd
        sounddevice_ok = True
        dev_list = sd.query_devices()
        for idx, d in enumerate(dev_list):
            if d.get("max_input_channels", 0) > 0:
                devices.append({
                    "index": idx,
                    "name": d.get("name", f"Device {idx}"),
                    "channels": d.get("max_input_channels", 1),
                    "default_sr": d.get("default_samplerate", 16000),
                })
    except Exception:
        sounddevice_ok = False

    if not devices:
        devices.append({
            "index": 0,
            "name": "Synthetic Tactical Scenario (Dual Mic Simulation)",
            "channels": 2,
            "default_sr": 16000,
        })

    return jsonify({
        "devices": devices,
        "sounddevice_available": sounddevice_ok,
        "active_backend": backend_mgr.backend.name,
        "hardware_profile": backend_mgr.profile,
    })


# -------------------------------------------------------------
# WebSocket Handlers
# -------------------------------------------------------------

@socketio.on("connect")
def on_ws_connect():
    emit("telemetry_update", state.to_dict())


@socketio.on("toggle_before_after")
def on_ws_toggle_before_after(data):
    enabled = bool(data.get("enabled", not state.before_after))
    state.set_before_after(enabled)
    emit("mode_update", {"before_after": state.before_after})


@socketio.on("toggle_autopilot")
def on_ws_toggle_autopilot(data):
    enabled = bool(data.get("enabled", not state.autopilot))
    state.set_autopilot(enabled)
    emit("mode_update", {"autopilot": state.autopilot})


@socketio.on("trigger_shock")
def on_ws_trigger_shock(data):
    intensity = float(data.get("intensity", 0.95))
    if isinstance(pipeline.accelerometer, MockAccelerometer):
        pipeline.accelerometer.trigger_shock(intensity)
    emit("shock_event", {"intensity": intensity})


if __name__ == "__main__":
    host = config["dashboard"].get("host", "0.0.0.0")
    port = config["dashboard"].get("port", 5000)
    print("=================================================================")
    print("  IGARD-Net Low-Latency Tactical Audio AI System Online")
    print(f"  Backend: {backend_mgr.backend.name} | Profile: {backend_mgr.profile}")
    print(f"  Dashboard: http://localhost:{port}")
    print("=================================================================")
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
