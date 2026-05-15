// camera_mark.js
const startMarkBtn = document.getElementById("startMarkBtn");
const stopMarkBtn = document.getElementById("stopMarkBtn");
const markVideo = document.getElementById("markVideo");
const markStatus = document.getElementById("markStatus");
const recognizedList = document.getElementById("recognizedList");

let markStream = null;
let markInterval = null;
let recognizedIds = new Set();

startMarkBtn.addEventListener("click", async () => {
  startMarkBtn.disabled = true;
  stopMarkBtn.disabled = false;
  try {
    markStream = await navigator.mediaDevices.getUserMedia({ video: true });
    
    markVideo.onloadedmetadata = () => {
        markStatus.innerText = "Scanning active...";
        markStatus.className = "mt-2 badge bg-light text-dark border fw-normal py-2 px-3";
        markInterval = setInterval(captureAndRecognize, 1200);
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
    clearInterval(markInterval);
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

async function captureAndRecognize() {
  const canvas = document.createElement("canvas");
  canvas.width = markVideo.videoWidth || 640;
  canvas.height = markVideo.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(markVideo, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise(r => canvas.toBlob(r, "image/jpeg", 0.85));
  const fd = new FormData();
  fd.append("image", blob, "snap.jpg");
  try {
    const res = await fetch("/recognize_face", { method: "POST", body: fd });
    const j = await res.json();
    
    if (j.recognized) {
      const statusText = j.status || "Recognized";
      
      // Check if this was an actual registration event (not debounced)
      if (statusText !== "Debounced (Wait 1 min)") {
          // 1. Build Welcome / Bye message
          let greeting = "";
          if (statusText === "Check In" || statusText === "Late Check In") {
              greeting = `Welcome, ${j.name}! 👋`;
          } else if (statusText === "Check Out") {
              greeting = `Goodbye, ${j.name}! 🚪`;
          } else {
              greeting = `${j.name}: ${statusText}`;
          }
          
          // 2. Print message to logs and status
          markStatus.innerText = greeting;
          markStatus.className = "mt-2 badge bg-success text-white border fw-normal py-2 px-3"; // Give it a vibrant success badge
          
          const li = document.createElement("li");
          li.className = "list-group-item d-flex justify-content-between align-items-center bg-light";
          
          const badgeColor = (statusText === "Check In" || statusText === "Late Check In") ? "bg-success" : "bg-warning text-dark";
          li.innerHTML = `
            <span><strong>${j.name}</strong> — ${new Date().toLocaleTimeString()}</span>
            <span class="badge ${badgeColor}">${statusText}</span>
          `;
          recognizedList.prepend(li);

          // 3. Automatic shutdown camera instantly
          stopCamera(greeting);
      } else {
          // Debounced case: let them know they already did it and shut off camera to prevent continuous spam.
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
    console.error(err);
  }
}

