document.addEventListener("DOMContentLoaded", function () {
  let stressTime = 0;
  let noStressTime = 0;
  let lastUpdateTime = null;
  let isTracking = false; // Tracking flag
  let firstDetection = false;
  let previousState = null;
  let intervalId = null; // Store interval ID

  const toggleButton = document.querySelector(".toggle-btn");
  const analysisButton = document.querySelector(".analysis-btn");
  const logoutButton = document.querySelector(".logout");

  function resetBars() {
    document.getElementById("stress-bar").style.width = "0%";
    document.getElementById("no-stress-bar").style.width = "0%";
    document.getElementById("stress-value").innerText = "0%";
    document.getElementById("no-stress-value").innerText = "0%";
  }

  function updateBars(stress) {
    const stressBar = document.getElementById("stress-bar");
    const noStressBar = document.getElementById("no-stress-bar");
    const stressValue = document.getElementById("stress-value");
    const noStressValue = document.getElementById("no-stress-value");

    stressBar.style.width = stress + "%";
    noStressBar.style.width = 100 - stress + "%";
    stressValue.innerText = stress.toFixed(2) + "%";
    noStressValue.innerText = (100 - stress).toFixed(2) + "%";
  }

  function fetchStress() {
    if (!isTracking) return;

    fetch("/get_stress")
      .then((response) => response.json())
      .then((data) => {
        updateBars(data.stress);
        updateTimeData(data.stress);
      })
      .catch((error) => console.error("Error fetching stress data:", error));
  }

  function updateTimeData(stressLevel) {
    let currentTime = Date.now();
    if (lastUpdateTime === null) {
      lastUpdateTime = currentTime;
      return;
    }

    let timeDiff = (currentTime - lastUpdateTime) / 1000; // Convert to seconds
    lastUpdateTime = currentTime;

    if (!firstDetection && stressLevel !== null) {
      firstDetection = true;
      previousState = stressLevel > 50 ? "stress" : "no_stress";
    }

    if (firstDetection) {
      if (stressLevel > 50) {
        stressTime += timeDiff;
        previousState = "stress";
      } else {
        noStressTime += timeDiff;
        previousState = "no_stress";
      }
    }
  }

  function updateChart() {
    const ctx = document.getElementById("stressChart").getContext("2d");

    if (window.myChart) {
      window.myChart.destroy();
    }

    window.myChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Stress Time (s)", "No Stress Time (s)"],
        datasets: [
          {
            label: "Time Spent",
            data: [stressTime.toFixed(2), noStressTime.toFixed(2)],
            backgroundColor: ["red", "green"],
          },
        ],
      },
      options: {
        responsive: true,
        aspectRatio: 2,
        scales: {
          y: { beginAtZero: true },
        },
      },
    });
  }

  toggleButton.addEventListener("click", function () {
    if (!isTracking) {
      // Start tracking
      isTracking = true;
      lastUpdateTime = Date.now();
      firstDetection = false;
      previousState = null;
      intervalId = setInterval(fetchStress, 1000);
      toggleButton.innerText = "⏹ Stop";
    } else {
      // Stop tracking and reset everything
      isTracking = false;
      clearInterval(intervalId);
      stressTime = 0;
      noStressTime = 0;
      lastUpdateTime = null;
      resetBars();
      toggleButton.innerText = "▶️ Start";
    }
  });

  analysisButton.addEventListener("click", function () {
    if (!isTracking) {
      let chartContainer = document.getElementById("chart-container");
      chartContainer.style.display =
        chartContainer.style.display === "none" ? "block" : "none";
      updateChart();
    }
  });

  logoutButton.addEventListener("click", function () {
    fetch("/logout", { method: "GET" })
      .then((response) => {
        if (response.redirected) {
          window.location.href = response.url; // Redirect to login page
        }
      })
      .catch((error) => console.error("Logout failed:", error));
  });

  // Initialize with everything at zero
  resetBars();
});
