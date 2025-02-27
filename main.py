from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import mediapipe as mp
import cv2
import numpy as np
import base64
from io import BytesIO

from plank import PlankDetection
from lunge import LungeDetection
from bicep_curl import BicepCurlDetection
from squat import SquatDetection

app = FastAPI()

# Enable CORS
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change this to specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Set up MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # Lower complexity for faster CPU performance
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cv2.setNumThreads(cv2.getNumberOfCPUs())

# Initialize detection classes
plank_detector = PlankDetection()
lunge_detector = LungeDetection()
bicep_curl_detector = BicepCurlDetection()
squat_detector = SquatDetection()

# Mount static files for serving HTML, CSS, and JavaScript
app.mount("/static", StaticFiles(directory="static"), name="static")


# WebSocket for plank detection
@app.websocket("/ws/plank")
async def websocket_plank(websocket: WebSocket):
    await websocket_handler(websocket, plank_detector)


# WebSocket for lunge detection
@app.websocket("/ws/lunge")
async def websocket_lunge(websocket: WebSocket):
    await websocket_handler(websocket, lunge_detector)


# WebSocket for bicep curl detection
@app.websocket("/ws/bicep_curl")
async def websocket_bicep_curl(websocket: WebSocket):
    await websocket_handler(websocket, bicep_curl_detector)


# WebSocket for squat detection
@app.websocket("/ws/squat")
async def websocket_squat(websocket: WebSocket):
    await websocket_handler(websocket, squat_detector)


async def websocket_handler(websocket: WebSocket, detector):
    """Generic WebSocket handler for pose detection."""
    await websocket.accept()
    try:
        while True:
            # Receive the frame from the frontend (base64 encoded)
            data = await websocket.receive_text()
            image_data = base64.b64decode(data.split(',')[1])
            
            # Convert the image to a numpy array (for OpenCV)
            np_arr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            # Process the frame with MediaPipe Pose
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = pose.process(img_rgb)

            # Perform detection using the given detector
            if results.pose_landmarks:
                timestamp = 0  # For simplicity, using 0 as a timestamp
                detector.detect(results, img, timestamp)

            # Draw landmarks on the frame if detected
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            # Use BytesIO instead of writing to a file and reading back
            buffer = BytesIO()  # Create in-memory buffer
            _, buffer_array = cv2.imencode('.jpg', img)  # Encode the image to JPEG format
            buffer.write(buffer_array)  # Write the image to the buffer

            # Convert the buffer content to base64 for sending via WebSocket
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            img_base64 = f"data:image/jpeg;base64,{img_base64}"

            # Send the processed image back to the client
            await websocket.send_text(img_base64)

    except WebSocketDisconnect:
        print("Client disconnected")


# Serve the HTML page
@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open("static/index.html", "r") as html_file:
        content = html_file.read()
    return HTMLResponse(content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8011)
