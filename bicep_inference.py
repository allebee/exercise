from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import mediapipe as mp
import cv2
import numpy as np
import base64
from bicep_curl import BicepCurlDetection  # Assuming you have your BicepCurlDetection class in this module

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize Bicep Curl Detection
bicep_curl_detector = BicepCurlDetection()

# Mount static files for serving HTML, CSS, and JavaScript
app.mount("/static", StaticFiles(directory="static"), name="static")


# WebSocket route for real-time video processing with MediaPipe Pose detection
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
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

            # Perform bicep curl detection using BicepCurlDetection
            if results.pose_landmarks:
                timestamp = 0  # For simplicity, using 0 as a timestamp
                bicep_curl_detector.detect(results, img, timestamp)

            # Draw landmarks on the frame if detected
            if results.pose_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

            # Encode the processed image back to base64
            _, buffer = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
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
    uvicorn.run(app, host="0.0.0.0", port=8000)
