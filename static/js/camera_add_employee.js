// camera_add_employee.js
const saveInfoBtn = document.getElementById("saveInfoBtn");
const startCaptureBtn = document.getElementById("startCaptureBtn");
const addEmployeeBtn = document.getElementById("addEmployeeBtn");
const video = document.getElementById("video");
const captureStatus = document.getElementById("captureStatus");
const progressBar = document.getElementById("progressBar");

let employee_id = null;
let captured = 0;
const maxImages = 30; // Reduced from 50 for significant performance gain
let images = [];
let stream = null;

document.getElementById("employeeForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(e.target);
  const res = await fetch("/add_employee", { method: "POST", body: fd });
  if (!res.ok) {
    try {
      const errJson = await res.json();
      alert("Failed: " + errJson.error);
    } catch(e) {
      alert("Failed to save employee information. Server returned " + res.status);
    }
    return;
  }
  const j = await res.json();
  employee_id = j.employee_id;
  alert("Employee record generated. Please permit Camera to register biometric facial prints.");
  startCaptureBtn.disabled = false;
});

startCaptureBtn.addEventListener("click", async () => {
  startCaptureBtn.disabled = true;
  captureStatus.innerText = "Connecting to video hardware...";
  try {
    // Request optimized 640x480 resolution from webcam to save memory/CPU
    const localStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false
    });
    stream = localStream;
    
    // Set up events BEFORE binding source
    video.onloadedmetadata = () => {
        captureStatus.innerText = "Sensor warmed. Collecting samples...";
        // Warmup and invoke
        setTimeout(() => {
            captureImagesLoop();
        }, 500);
    };
    
    video.srcObject = stream;
    await video.play().catch(e => console.log("Autoplay error: ", e));
    
  } catch (err) {
    alert("Camera Access Denied or Not Found: " + err.message);
    captureStatus.innerText = "Failed: Check hardware permissions";
    startCaptureBtn.disabled = false;
  }
});

async function captureImagesLoop() {
  const canvas = document.createElement("canvas");
  
  // Optimize: Resize captured frames to max-width 480 to speed up upload
  let targetWidth = 480;
  let targetHeight = 360;
  if (video.videoWidth && video.videoHeight) {
    const aspect = video.videoWidth / video.videoHeight;
    targetHeight = Math.round(targetWidth / aspect);
  }
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  
  const ctx = canvas.getContext("2d");

  while (captured < maxImages) {
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Compress to 80% JPEG quality to optimize payload
    const blob = await new Promise(res => canvas.toBlob(res, "image/jpeg", 0.80));
    images.push(blob);
    captured++;
    captureStatus.innerText = `Capturing Biometrics: ${captured} / ${maxImages} frames`;
    progressBar.style.width = `${(captured / maxImages) * 100}%`;
    
    // Optimized fast sampling interval
    await new Promise(r => setTimeout(r, 100));
  }

  captureStatus.innerText = "Compressing and transferring dataset...";
  
  // upload all images to employee's dataset folder
  const form = new FormData();
  form.append("employee_id", employee_id);
  images.forEach((b, i) => form.append("images[]", b, `frame_${i}.jpg`));
  
  const resp = await fetch("/upload_face", { method: "POST", body: form });
  if (resp.ok) {
    captureStatus.innerText = "Successful upload. Data acquisition complete.";
    addEmployeeBtn.disabled = false;
  } else {
    alert("Network/Storage failure uploading facial scans.");
  }

  // close stream safely
  if (stream) stream.getTracks().forEach(t => t.stop());
}

addEmployeeBtn.addEventListener("click", () => {
  alert("Profile activated and locked. Returning to dashboard.");
  window.location.href = "/";
});
