import os
import time
import threading
import pathlib
from flask import (Flask, render_template, Response, jsonify,
                   request, redirect, url_for, session, flash)
import cv2
from ultralytics import YOLO
from twilio.rest import Client
from dotenv import load_dotenv

# load .env
load_dotenv()

# config
SECRET_KEY = os.getenv("SECRET_KEY", "devkey")
ADMIN_USER = os.getenv("ADMIN_USER", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASS", "admin123")
TW_SID = os.getenv("TWILIO_ACCOUNT_SID")
TW_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TW_FROM = os.getenv("TWILIO_FROM")
ALERT_PHONE_TO = os.getenv("ALERT_PHONE_TO")
NGROK_URL = os.getenv("NGROK_URL", "").rstrip("/")

# Flask app
app = Flask(__name__)
app.secret_key = SECRET_KEY

# create captures dir
CAP_DIR = pathlib.Path(app.static_folder) / "captures"
CAP_DIR.mkdir(parents=True, exist_ok=True)

# Quiet logs
os.environ["YOLO_VERBOSE"] = "False"
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)
logging.getLogger().setLevel(logging.WARNING)

# YOLO model
model = YOLO("yolov8n.pt")

# Twilio client (if credentials present)
tw_client = None
if TW_SID and TW_TOKEN:
    tw_client = Client(TW_SID, TW_TOKEN)

last_sms_time = 0
SMS_COOLDOWN = 300  # 300 seconds = 5 minutes

# Shared state
latest_frame = None
latest_status = {"person": False, "ts": 0.0, "last_capture": None}
frame_lock = threading.Lock()
running = True

# helper: build public image URL for a saved capture
def make_image_url(filename):
    # filename is relative to static/captures (e.g. person_20251116-213540.jpg)
    if NGROK_URL:
        base = NGROK_URL
    else:
        # local network: use machine IP
        host = os.getenv("HOST_IP") or "127.0.0.1"
        base = f"http://{host}:5000"
    return f"{base}/static/captures/{filename}"

# send SMS/MMS via Twilio (non-blocking)
def send_alert_sms():
    if not tw_client:
        print("Twilio not configured; skipping SMS.")
        return
    try:
        body = "🚨 SafeCam Alert! Person detected. Please check your dashboard."
        tw_client.messages.create(
            to=ALERT_PHONE_TO,
            from_=TW_FROM,
            body=body
        )
        print("SMS alert sent to", ALERT_PHONE_TO)
    except Exception as e:
        print("Twilio send error:", e)


def detection_thread(camera_index=0, downscale=None, save_on_detect=True):
    global latest_frame, latest_status, running
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("ERROR: Cannot open camera.")
        running = False
        return

    last_sent_ts = 0
    cooldown_seconds = 5  # prevent SMS spam: send at most one alert per 5s

    while running:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        if downscale:
            frame = cv2.resize(frame, downscale)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb, verbose=False)

        person_found = False
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    cls = int(box.cls[0])
                    if cls == 0:
                        person_found = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = model.names[cls] if cls in model.names else str(cls)
                    color = (0,255,0) if cls==0 else (200,200,0)
                    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # if detected, save and send alert (with cooldown)
        if person_found and save_on_detect:
            ts = time.time()

            global last_sms_time
            if ts - last_sms_time > SMS_COOLDOWN:   # 5 min cooldown
                last_sms_time = ts

                timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S")
                filename = f"person_{time.strftime('%Y%m%d-%H%M%S')}.jpg"
                filepath = CAP_DIR / filename
                cv2.imwrite(str(filepath), frame)

                with frame_lock:
                    latest_status["last_capture"] = filename

                threading.Thread(target=send_alert_sms, daemon=True).start()



        # update shared frame and status
        with frame_lock:
            latest_frame = frame.copy()
            latest_status["person"] = bool(person_found)
            latest_status["ts"] = time.time()

        time.sleep(0.03)

    cap.release()

def encode_frame_jpeg(frame):
    ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ret:
        return None
    return jpeg.tobytes()

@app.route('/')
def index():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    # list captures
    files = sorted([f.name for f in CAP_DIR.iterdir() if f.suffix.lower() in ('.jpg','.png')], reverse=True)
    return render_template('dashboard.html', captures=files, status=latest_status)

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        user = request.form.get('username')
        pwd = request.form.get('password')
        if user == ADMIN_USER and pwd == ADMIN_PASS:
            session['logged_in'] = True
            flash("Logged in", "success")
            return redirect(url_for('dashboard'))
        flash("Invalid credentials", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

def mjpeg_generator():
    global latest_frame
    while running:
        with frame_lock:
            frame = None if latest_frame is None else latest_frame.copy()
        if frame is None:
            blank = 255 * (np.zeros((480,640,3), dtype='uint8'))
            data = encode_frame_jpeg(blank)
        else:
            data = encode_frame_jpeg(frame)
        if data:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')
        time.sleep(0.03)

@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    with frame_lock:
        return jsonify({
            "person": bool(latest_status.get("person", False)),
            "ts": float(latest_status.get("ts", 0.0)),
            "last_capture": latest_status.get("last_capture")
        })

if __name__ == '__main__':
    # try to guess local IP so phones on same Wi-Fi can access without ngrok
    import socket, numpy as np
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        host_ip = s.getsockname()[0]
    except Exception:
        host_ip = "127.0.0.1"
    finally:
        s.close()
    os.environ.setdefault("HOST_IP", host_ip)
    # start detection thread
    t = threading.Thread(target=detection_thread, kwargs={"camera_index": 0}, daemon=True)
    t.start()
    # hide werkzeug access logs
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
    running = False
    t.join()
