const videoElement = document.getElementById('webcam');

async function setupWebcam() {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true });
    videoElement.srcObject = stream;
}

setupWebcam();
