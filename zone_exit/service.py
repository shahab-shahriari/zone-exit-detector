"""HTTP service: another system starts a watch for one call on one camera, and gets the result
by webhook or by polling.

    POST   /sessions            {"call_id": "123", "camera_id": "gate1", "timeout_s": 60}
    GET    /sessions/<call_id>  state and events of that session
    DELETE /sessions/<call_id>  stop it
    GET    /health

Design points: state changes use POST/DELETE (never GET); every request except /health needs
a bearer token; a limited number of sessions can run at once; every session has a timeout;
finished sessions are purged after a retention time; stream addresses are never returned.
"""
import hmac
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

from flask import Flask, jsonify, request

from .detectors import LockedDetector, make_detector
from .pipeline import camera_config, process, sink_from_config

CALL_ID = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")


class SessionError(Exception):
    status = 400


class UnknownCamera(SessionError):
    status = 404


class NotFound(SessionError):
    status = 404


class Conflict(SessionError):
    status = 409


class Busy(SessionError):
    status = 429


class Session:
    def __init__(self, call_id: str, camera_id: str, timeout_s: float):
        self.call_id = call_id
        self.camera_id = camera_id
        self.timeout_s = timeout_s
        self.started = time.monotonic()
        self.finished: Optional[float] = None
        self.state = "running"  # running | detected | timeout | ended | stopped | error
        self.error: Optional[str] = None
        self.events: List[dict] = []
        self.stop = threading.Event()

    def to_dict(self) -> dict:
        end = self.finished if self.finished is not None else time.monotonic()
        return {
            "call_id": self.call_id,
            "camera_id": self.camera_id,
            "state": self.state,
            "seconds": round(end - self.started, 1),
            "events": self.events,
            "error": self.error,
        }


class SessionManager:
    def __init__(self, config: dict, sink, detector_factory=None):
        self.config = config
        self.sink = sink
        api = config["api"]
        self.max_sessions = int(api["max_sessions"])
        self.default_timeout_s = float(api["default_timeout_s"])
        self.retention_s = float(api["retention_s"])
        self._detector_factory = detector_factory or self._default_detector
        self._shared = None
        self._shared_lock = threading.Lock()
        self._lock = threading.Lock()
        self._sessions: Dict[str, Session] = {}

    def _default_detector(self, cfg: dict):
        det_cfg = cfg["detector"]
        if det_cfg.get("type") == "yolo":  # one model in memory, shared by all sessions
            with self._shared_lock:
                if self._shared is None:
                    self._shared = LockedDetector(make_detector(det_cfg))
                return self._shared
        return make_detector(det_cfg)  # motion detectors keep per-stream state

    def start(self, call_id: str, camera_id: str, timeout_s: Optional[float] = None) -> Session:
        if camera_id not in self.config["cameras"]:
            raise UnknownCamera(f"unknown camera: {camera_id}")
        timeout_s = float(timeout_s) if timeout_s else self.default_timeout_s
        with self._lock:
            self._purge()
            existing = self._sessions.get(call_id)
            if existing is not None and existing.state == "running":
                raise Conflict("a session for this call is already running")
            if sum(s.state == "running" for s in self._sessions.values()) >= self.max_sessions:
                raise Busy("too many sessions running")
            session = Session(call_id, camera_id, timeout_s)
            self._sessions[call_id] = session
        threading.Thread(target=self._worker, args=(session,), daemon=True).start()
        return session

    def get(self, call_id: str) -> Session:
        with self._lock:
            self._purge()
            session = self._sessions.get(call_id)
        if session is None:
            raise NotFound("no such session")
        return session

    def stop(self, call_id: str) -> Session:
        session = self.get(call_id)
        session.stop.set()
        return session

    def running_count(self) -> int:
        with self._lock:
            return sum(s.state == "running" for s in self._sessions.values())

    def _purge(self) -> None:  # call with the lock held
        now = time.monotonic()
        old = [k for k, s in self._sessions.items()
               if s.finished is not None and now - s.finished > self.retention_s]
        for k in old:
            del self._sessions[k]

    def _worker(self, s: Session) -> None:
        try:
            cfg, source = camera_config(self.config, s.camera_id)
            result = process(
                cfg, source, sink=self.sink, detector=self._detector_factory(cfg),
                stop_event=s.stop, max_seconds=s.timeout_s, stop_after_first=True,
                extra={"call_id": s.call_id, "camera_id": s.camera_id},
            )
            s.events = result.events
            if result.events:
                s.state = "detected"
            elif result.reason == "stopped":
                s.state = "stopped"
            elif result.reason == "deadline":
                s.state = "timeout"
                self.sink.emit({"event": "timeout", "call_id": s.call_id, "camera_id": s.camera_id,
                                "time": datetime.now(timezone.utc).isoformat()})
            elif result.reason == "stream_lost":
                s.state, s.error = "error", "the video stream was lost"
            else:
                s.state = "ended"
        except Exception as exc:  # report to the caller, never crash the service
            s.state, s.error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            s.finished = time.monotonic()


def create_app(config: dict, manager: Optional[SessionManager] = None, token: Optional[str] = None) -> Flask:
    """token=None turns authentication off. Do that for local experiments only."""
    app = Flask(__name__)
    manager = manager or SessionManager(config, sink_from_config(config))

    @app.before_request
    def check_token():
        if token is None or request.endpoint == "health":
            return None
        header = request.headers.get("Authorization", "")
        supplied = header[7:] if header.startswith("Bearer ") else ""
        if not hmac.compare_digest(supplied.encode(), token.encode()):
            return jsonify(error="unauthorized"), 401
        return None

    @app.errorhandler(SessionError)
    def session_error(exc):
        return jsonify(error=str(exc)), exc.status

    @app.get("/health")
    def health():
        return jsonify(status="ok", running=manager.running_count())

    @app.post("/sessions")
    def start_session():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify(error="send a JSON object"), 400
        call_id, camera_id, timeout_s = body.get("call_id"), body.get("camera_id"), body.get("timeout_s")
        if not isinstance(call_id, str) or not CALL_ID.match(call_id):
            return jsonify(error="call_id must be 1-64 letters, digits, '.', '_' or '-'"), 400
        if not isinstance(camera_id, str):
            return jsonify(error="camera_id is required"), 400
        if timeout_s is not None and (isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float))
                                      or not 0 < timeout_s <= 3600):
            return jsonify(error="timeout_s must be a number between 0 and 3600"), 400
        return jsonify(manager.start(call_id, camera_id, timeout_s).to_dict()), 202

    @app.get("/sessions/<call_id>")
    def get_session(call_id):
        return jsonify(manager.get(call_id).to_dict())

    @app.delete("/sessions/<call_id>")
    def stop_session(call_id):
        return jsonify(manager.stop(call_id).to_dict()), 202

    return app


def serve(config: dict, host: str = "127.0.0.1", port: int = 8080, no_auth: bool = False) -> None:
    if not config["cameras"]:
        raise SystemExit("config has no cameras: add a 'cameras:' section (see config.example.yaml)")
    token = os.environ.get(config["api"]["token_env"])
    if not token and not no_auth:
        raise SystemExit(f"set the {config['api']['token_env']} environment variable, "
                         "or pass --no-auth for local experiments")
    app = create_app(config, token=None if no_auth else token)
    try:
        from waitress import serve as wsgi_serve
        wsgi_serve(app, host=host, port=port)
    except ImportError:
        print("waitress is not installed; using Flask's development server")
        app.run(host=host, port=port, threaded=True)
