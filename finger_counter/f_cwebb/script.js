const videoElement = document.getElementById('webcam');
const canvasElement = document.getElementById('output');
const canvasCtx = canvasElement.getContext('2d');
const fingerCountEl = document.getElementById('fingerCount');

// Load MediaPipe Hands
const hands = new Hands({
    locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
    }
});

hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.7,
    minTrackingConfidence: 0.7
});

hands.onResults(onResults);

const camera = new Camera(videoElement, {
    onFrame: async () => {
        await hands.send({image: videoElement});
    },
    width: 640,
    height: 480
});
camera.start();

function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

    if (results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];
        let count = countFingers(landmarks);

        fingerCountEl.textContent = count;
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, {color: '#00FF00', lineWidth: 2});
        drawLandmarks(canvasCtx, landmarks, {color: '#FF0000', lineWidth: 1});
    }
    canvasCtx.restore();
}

// Finger counting logic
function countFingers(landmarks) {
    let tips = [8, 12, 16, 20]; // index, middle, ring, pinky
    let count = 0;

    tips.forEach(tip => {
        if (landmarks[tip].y < landmarks[tip - 2].y) {
            count++;
        }
    });

    // Thumb
    if (landmarks[4].x > landmarks[3].x) {
        count++;
    }

    return count;
}
