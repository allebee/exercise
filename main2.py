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
from concurrent.futures import ThreadPoolExecutor

# Import your detection classes
from plank import PlankDetection
from lunge import LungeDetection
from bicep_curl import BicepCurlDetection
from squat import SquatDetection

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up MediaPipe Pose with optimized settings
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # Balanced setting between speed and accuracy
    smooth_landmarks=True,  # Enable landmark smoothing
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Create a thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

# Initialize detection classes
plank_detector = PlankDetection()
lunge_detector = LungeDetection()
bicep_curl_detector = BicepCurlDetection()
squat_detector = SquatDetection()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Optimize image processing
def process_image(image_data):
    """Process image data efficiently"""
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Resize image for faster processing while maintaining aspect ratio
    height, width = img.shape[:2]
    if width > 640:
        scale = 640 / width
        img = cv2.resize(img, None, fx=scale, fy=scale)
    
    # Convert to RGB more efficiently
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb

def encode_frame(img):
    """Efficiently encode frame to base64"""
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode('utf-8')

async def websocket_handler(websocket: WebSocket, detector):
    """Optimized WebSocket handler for pose detection"""
    await websocket.accept()
    
    try:
        while True:
            # Receive the frame
            data = await websocket.receive_text()
            image_data = base64.b64decode(data.split(',')[1])
            
            # Process image in thread pool
            img, img_rgb = await app.state.loop.run_in_executor(
                executor, 
                process_image, 
                image_data
            )
            
            # Process with MediaPipe
            results = pose.process(img_rgb)
            
            if results.pose_landmarks:
                # Run detection in thread pool
                await app.state.loop.run_in_executor(
                    executor,
                    detector.detect,
                    results,
                    img,
                    0
                )
                
                # Draw landmarks efficiently
                mp.solutions.drawing_utils.draw_landmarks(
                    img,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                        color=(0, 255, 0),
                        thickness=2,
                        circle_radius=2
                    )
                )
            
            # Encode and send frame
            img_base64 = await app.state.loop.run_in_executor(
                executor,
                encode_frame,
                img
            )
            
            await websocket.send_text(f"data:image/jpeg;base64,{img_base64}")

    except WebSocketDisconnect:
        print("Client disconnected")

# WebSocket endpoints
@app.websocket("/ws/plank")
async def websocket_plank(websocket: WebSocket):
    await websocket_handler(websocket, plank_detector)

@app.websocket("/ws/lunge")
async def websocket_lunge(websocket: WebSocket):
    await websocket_handler(websocket, lunge_detector)

@app.websocket("/ws/bicep_curl")
async def websocket_bicep_curl(websocket: WebSocket):
    await websocket_handler(websocket, bicep_curl_detector)

@app.websocket("/ws/squat")
async def websocket_squat(websocket: WebSocket):
    await websocket_handler(websocket, squat_detector)

@app.on_event("startup")
async def startup_event():
    # Store the event loop for executor usage
    import asyncio
    app.state.loop = asyncio.get_event_loop()

@app.get("/", response_class=HTMLResponse)
async def get_html():
    with open("static/index.html", "r") as html_file:
        content = html_file.read()
    return HTMLResponse(content)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8011, workers=4)