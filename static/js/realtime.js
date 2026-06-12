// realtime.js - Handle Server-Sent Events for real-time application updates

document.addEventListener("DOMContentLoaded", () => {
  // 1. Initialize SSE connection
  const sse = new EventSource("/stream");

  // Keep track of connection status
  sse.addEventListener("connected", () => {
    console.log("⚡ Real-time channel active: Connected to Raj Chemist server.");
  });

  sse.addEventListener("heartbeat", () => {
    // Keep-alive heartbeat received
  });

  // Handle server notice to close the SSE connection and prevent infinite reconnection loop on Vercel
  sse.addEventListener("server_notice", (event) => {
    try {
      const data = JSON.parse(event.data);
      console.log("⚠️ Server notice:", data.message);
      sse.close();
    } catch (err) {
      console.error("Failed to parse server notice:", err);
    }
  });

  // 2. Handle Attendance Marked Event
  sse.addEventListener("attendance_marked", (event) => {
    try {
      const data = JSON.parse(event.data);
      console.log("⏰ Real-time Attendance Event received:", data);

      // A. Show Premium Toast Notification
      showRealtimeToast(data);

      // B. If on Attendance Log page, dynamically insert the row
      updateAttendanceTableRealtime(data);

      // C. If on Dashboard page, update totals and refresh the Chart
      updateDashboardRealtime();

      // D. If the employee who checked in is the one currently in the sidebar, refresh sidebar stats!
      refreshSidebarIfActive(data.employee_id);

    } catch (err) {
      console.error("Failed to process real-time attendance event:", err);
    }
  });

  // 3. Handle Model Training Status Event
  sse.addEventListener("train_status", (event) => {
    try {
      const data = JSON.parse(event.data);
      console.log("🧠 Real-time Train Status received:", data);

      const trainProgress = document.getElementById("trainProgress");
      const trainMsg = document.getElementById("trainMsg");
      const trainBtn = document.getElementById("trainBtn");

      if (trainProgress && trainMsg) {
        trainProgress.style.width = data.progress + "%";
        trainProgress.innerText = data.progress + "%";
        trainMsg.innerText = data.message || "";
        
        if (data.progress >= 100) {
          if (trainBtn) trainBtn.disabled = false;
        } else if (data.running) {
          if (trainBtn) trainBtn.disabled = true;
        }
      }
    } catch (err) {
      console.error("Failed to process real-time train status:", err);
    }
  });
});

// Helper: Show custom beautiful toast notification
function showRealtimeToast(data) {
  let toastContainer = document.querySelector(".realtime-toast-container");
  if (!toastContainer) {
    toastContainer = document.createElement("div");
    toastContainer.className = "realtime-toast-container";
    document.body.appendChild(toastContainer);
  }

  let statusClass = "check-in";
  let iconClass = "fa-arrow-right-to-bracket";
  let titleText = "Member Check In";
  let msgText = `Welcome, <strong>${data.name}</strong>! 👋`;

  if (data.status === "Late Check In") {
    statusClass = "late-check-in";
    iconClass = "fa-clock";
    titleText = "Late Check In";
    msgText = `Welcome, <strong>${data.name}</strong>! ⏰`;
  } else if (data.status === "Check Out") {
    statusClass = "check-out";
    iconClass = "fa-arrow-right-from-bracket";
    titleText = "Member Check Out";
    msgText = `Goodbye, <strong>${data.name}</strong>! 🚪`;
  }

  const dateObj = new Date(data.timestamp);
  const timeText = dateObj.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true
  });

  const toast = document.createElement("div");
  toast.className = "realtime-toast";
  
  // Custom fallback image handling
  const imgSrc = `/api/employee_image/${data.employee_id}`;
  const fallbackSrc = `https://ui-avatars.com/api/?name=${encodeURIComponent(data.name)}&background=random`;

  toast.innerHTML = `
    <div class="toast-avatar-wrapper">
      <img src="${imgSrc}" onerror="this.onerror=null; this.src='${fallbackSrc}';" class="toast-avatar" alt="${data.name}">
      <span class="toast-status-badge ${statusClass}">
        <i class="fa-solid ${iconClass}"></i>
      </span>
    </div>
    <div class="toast-content">
      <h4 class="toast-title">${titleText}</h4>
      <p class="toast-message">${msgText}</p>
      <span class="toast-timestamp">${timeText}</span>
    </div>
    <button class="toast-close-btn"><i class="fa-solid fa-xmark"></i></button>
    <div class="toast-progress ${statusClass}"></div>
  `;

  toastContainer.appendChild(toast);

  // Trigger slide-in
  setTimeout(() => toast.classList.add("show"), 50);

  // Close functionality
  const closeToast = () => {
    toast.classList.remove("show");
    toast.classList.add("hide");
    setTimeout(() => toast.remove(), 500);
  };

  toast.querySelector(".toast-close-btn").addEventListener("click", closeToast);

  // Auto-close after 5 seconds (matching the CSS progress animation)
  setTimeout(() => {
    if (toast.parentNode) {
      closeToast();
    }
  }, 5000);
}

// Helper: Live-prepend row to the logs table
function updateAttendanceTableRealtime(data) {
  const tableBody = document.querySelector(".timeline-table tbody");
  if (!tableBody) return;

  // Check if we have the "No records exist" placeholder and remove it
  const emptyPlaceholder = tableBody.querySelector("tr td.text-center");
  if (emptyPlaceholder) {
    const tr = emptyPlaceholder.closest("tr");
    if (tr) tr.remove();
  }

  const dateObj = new Date(data.timestamp);
  const dateText = dateObj.toLocaleDateString("en-CA");
  const timeText = dateObj.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true
  });

  let borderCls = "border-checkout";
  let textCls = "text-checkout";
  let icon = "fa-arrow-right-from-bracket";
  
  if (data.status === "Late Check In") {
    borderCls = "border-late";
    textCls = "text-late";
    icon = "fa-clock";
  } else if (data.status === "Check In") {
    borderCls = "border-checkin";
    textCls = "text-checkin";
    icon = "fa-arrow-right-to-bracket";
  }

  const tr = document.createElement("tr");
  tr.className = "row-new"; // Triggers fade-in-up animation

  tr.innerHTML = `
    <td class="day-label">
      <div>
        <strong class="date-part">${dateText}</strong><br/>
        <span class="text-muted small time-part">${timeText}</span>
      </div>
    </td>
    <td style="width: 50%;">
      <div class="time-block-container">
        <div class="time-block ${borderCls}" style="left: 20%; width: 60%;">
          <div class="block-details">
            <span class="block-hours"><i class="fa-solid ${icon} me-1 ${textCls}"></i> ${data.status}</span>
            <span class="block-sub">Biometric Auto Verification</span>
          </div>
        </div>
      </div>
    </td>
    <td style="font-weight: 600;">
      <i class="fa-regular fa-circle-user text-muted me-1"></i> ${data.name}
    </td>
    <td class="text-end text-muted">
      #${data.employee_id}
    </td>
  `;

  // Prepend to top of logs
  tableBody.insertBefore(tr, tableBody.firstChild);
}

// Helper: Refresh dashboard statistics & charts
function updateDashboardRealtime() {
  if (document.getElementById("totalCheckins")) {
    fetch("/attendance_stats")
      .then((res) => res.json())
      .then((data) => {
        const sum = data.counts.reduce((a, b) => a + b, 0);
        document.getElementById("totalCheckins").innerText = sum + " hrs equivalent";
      });
  }

  // Trigger global dashboard chart update if available
  if (window.updateDashboardChart && typeof window.updateDashboardChart === "function") {
    window.updateDashboardChart();
  }
}

// Helper: Refresh the sidebar if the matched employee has logged in/out
function refreshSidebarIfActive(employeeId) {
  if (typeof currentSidebarEid !== "undefined" && currentSidebarEid === employeeId) {
    if (typeof changeSidebarEmployee === "function") {
      // Direct call to refresh sidebar stats via the 'current' direction logic
      changeSidebarEmployee("current");
    }
  }
}
