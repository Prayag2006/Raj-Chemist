// camera_mark.js
const startMarkBtn = document.getElementById("startMarkBtn");
const stopMarkBtn = document.getElementById("stopMarkBtn");
const markVideo = document.getElementById("markVideo");
const markStatus = document.getElementById("markStatus");
const recognizedList = document.getElementById("recognizedList");

let markStream = null;
let markInterval = null; // Storing either true or timeout ID
let recognizedIds = new Set();

startMarkBtn.addEventListener("click", async () => {
  startMarkBtn.disabled = true;
  stopMarkBtn.disabled = false;
  try {
    // Request optimized 640x480 resolution from webcam to save memory/CPU
    markStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 } }
    });
    
    markVideo.onloadedmetadata = () => {
        markStatus.innerText = "Scanning active...";
        markStatus.className = "mt-2 badge bg-light text-dark border fw-normal py-2 px-3";
        markInterval = true;
        captureLoop(); // Start self-paced loop
    };
    
    markVideo.srcObject = markStream;
    await markVideo.play().catch(e => console.warn("AutoPlay warning", e));
    
  } catch (err) {
    alert("Camera error: " + err.message);
    startMarkBtn.disabled = false;
    stopMarkBtn.disabled = true;
  }
});

stopMarkBtn.addEventListener("click", () => {
  stopCamera("Stopped");
});

function stopCamera(statusMessage) {
  if (markInterval) {
    if (typeof markInterval === "number" || markInterval) {
      clearTimeout(markInterval);
    }
    markInterval = null;
  }
  if (markStream) {
    markStream.getTracks().forEach(t => t.stop());
    markStream = null;
  }
  startMarkBtn.disabled = false;
  stopMarkBtn.disabled = true;
  if (statusMessage) {
    markStatus.innerText = statusMessage;
  }
}

async function captureLoop() {
  if (!markStream || !markInterval) return;
  await captureAndRecognize();
  if (markStream && markInterval) {
    // Wait 500ms before triggering the next capture to avoid request pileup
    markInterval = setTimeout(captureLoop, 500);
  }
}

async function captureAndRecognize() {
  const canvas = document.createElement("canvas");
  
  // Optimize: Limit captured snapshot size to max 480px width to speed up upload & Haar processing
  let targetWidth = 480;
  let targetHeight = 360;
  if (markVideo.videoWidth && markVideo.videoHeight) {
    const aspect = markVideo.videoWidth / markVideo.videoHeight;
    targetHeight = Math.round(targetWidth / aspect);
  }
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  
  const ctx = canvas.getContext("2d");
  ctx.drawImage(markVideo, 0, 0, canvas.width, canvas.height);
  
  // Compress slightly to 80% JPEG quality to save bandwidth
  const blob = await new Promise(r => canvas.toBlob(r, "image/jpeg", 0.80));
  const fd = new FormData();
  fd.append("image", blob, "snap.jpg");
  
  try {
    const res = await fetch("/recognize_face", { method: "POST", body: fd });
    const j = await res.json();
    
    if (j.recognized) {
      const statusText = j.status || "Recognized";
      
      if (statusText !== "Debounced (Wait 1 min)") {
          let greeting = "";
          if (statusText === "Check In" || statusText === "Late Check In") {
              greeting = `Welcome, ${j.name}! 👋`;
          } else if (statusText === "Check Out") {
              greeting = `Goodbye, ${j.name}! 🚪`;
          } else {
              greeting = `${j.name}: ${statusText}`;
          }
          
          markStatus.innerText = greeting;
          markStatus.className = "mt-2 badge bg-success text-white border fw-normal py-2 px-3";
          
          const li = document.createElement("li");
          li.className = "list-group-item d-flex justify-content-between align-items-center bg-light";
          
          const badgeColor = (statusText === "Check In" || statusText === "Late Check In") ? "bg-success" : "bg-warning text-dark";
          li.innerHTML = `
            <span><strong>${j.name}</strong> — ${new Date().toLocaleTimeString()}</span>
            <span class="badge ${badgeColor}">${statusText}</span>
          `;
          recognizedList.prepend(li);

          stopCamera(greeting);
      } else {
          const msg = `${j.name}: Already marked recently (Debounced)`;
          stopCamera(msg);
      }
    } else {
      if (j.error) {
        markStatus.innerText = `Not recognized: ${j.error}`;
      } else if (j.confidence !== undefined) {
        markStatus.innerText = `Not recognized (Low confidence: ${Math.round(j.confidence*100)}%)`;
      } else {
        markStatus.innerText = `Not recognized`;
      }
    }
  } catch (err) {
    console.error("Recognition request failed:", err);
  }
}
