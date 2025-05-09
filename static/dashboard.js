document.addEventListener("DOMContentLoaded", function () {
  let stressTime = 0;
  let noStressTime = 0;
  let lastUpdateTime = null;
  let isTracking = false;
  let firstDetection = false;
  let previousState = null;
  let intervalId = null;

  const toggleButton = document.querySelector(".toggle-btn");
  const analysisButton = document.querySelector(".analysis-btn");
  const logoutButton = document.querySelector(".logout");
  const feedback = document.getElementById("feedback"); // Optional: Add a div#feedback for messages

  let currentMode = "stress"; // Default mode is stress
  let emotionTimeMap = {}; // e.g., { happy: 12.5, sad: 5.0 }

  const emotionColors = {
    happy: "gold",
    sad: "dodgerblue",
    angry: "crimson",
    surprised: "orange",
    neutral: "gray",
    disgust: "green",
    fear: "purple",
  };

  function resetBars() {
    document.getElementById("stress-bar").style.width = "0%";
    document.getElementById("no-stress-bar").style.width = "0%";
    document.getElementById("stress-value").innerText = "0%";
    document.getElementById("no-stress-value").innerText = "0%";
    document.getElementById("emotion-bar").style.width = "0%";
    document.getElementById("emotion-value").innerText = "0%";
  }

  function showFeedback(message) {
    if (feedback) {
      feedback.innerText = message;
      feedback.style.opacity = 1;
      setTimeout(() => (feedback.style.opacity = 0), 3000);
    }
  }

  function updateBars(stress, emotionData) {
    const stressBar = document.getElementById("stress-bar");
    const noStressBar = document.getElementById("no-stress-bar");
    const stressValue = document.getElementById("stress-value");
    const noStressValue = document.getElementById("no-stress-value");

    if (currentMode === "stress" && stress !== null) {
      stressBar.style.width = stress + "%";
      noStressBar.style.width = 100 - stress + "%";
      stressValue.innerText = stress.toFixed(2) + "%";
      noStressValue.innerText = (100 - stress).toFixed(2) + "%";
    }

    if (currentMode === "emotion" && emotionData) {
      const emotionBar = document.getElementById("emotion-bar");
      const emotionValue = document.getElementById("emotion-value");
      emotionBar.style.width = emotionData.confidence + "%";
      emotionValue.innerText =
        emotionData.label + " (" + emotionData.confidence.toFixed(2) + "%)";
      emotionBar.style.backgroundColor =
        emotionColors[emotionData.label.toLowerCase()] || "gray";
    }
  }

  function fetchStress() {
    if (!isTracking || currentMode !== "stress") return;

    fetch("/get_stress")
      .then((response) => response.json())
      .then((data) => {
        updateBars(data.stress, null);
        updateTimeData(data.stress);
      })
      .catch((error) => console.error("Error fetching stress data:", error));
  }

  function fetchEmotion() {
    if (!isTracking || currentMode !== "emotion") return;

    fetch("/get_emotion")
  .then((res) => res.json())
  .then((data) => {
    const topThree = data.top_emotions.slice(0, 3); // Limit to top 3 emotions
    const labels = topThree.map((e) => e.emotion);
    const values = topThree.map((e) => parseFloat(e.confidence.toFixed(2)));
    const timeSpent = topThree.map((e) => e.time_spent); // Time spent in each emotion

    // Display the emotion chart
    const ctx = document.getElementById("emotionChart").getContext("2d");
    myChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [
          {
            label: "Confidence (%)",
            data: values,
            backgroundColor: ["#f0abfc", "#c084fc", "#a78bfa"],
            borderColor: ["#be185d", "#7e22ce", "#4c1d95"],
            borderWidth: 2,
          },
        ],
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            grid: {
              color: "rgba(255, 255, 255, 0.1)",
            },
            ticks: {
              color: "rgba(255, 255, 255, 0.8)",
            },
          },
          x: {
            grid: {
              color: "rgba(255, 255, 255, 0.1)",
            },
            ticks: {
              color: "rgba(255, 255, 255, 0.8)",
            },
          },
        },
        plugins: {
          legend: {
            labels: {
              color: "rgba(255, 255, 255, 0.8)",
            },
          },
        },
      },
    });

    // Display time spent
    const timeSpentElement = document.getElementById("timeSpent");
    timeSpentElement.innerHTML = `
      <h4>Time Spent in Emotions (seconds)</h4>
      <ul>
        ${topThree
          .map(
            (e) =>
              `<li>${e.emotion}: ${e.time_spent} seconds</li>`
          )
          .join("")}
      </ul>
    `;
    
    document.getElementById("chart-title").innerText = "Top Emotions Detected";
  });

  function updateTimeData(stressLevel, emotionData) {
    let currentTime = Date.now();
    if (lastUpdateTime === null) {
      lastUpdateTime = currentTime;
      return;
    }

    let timeDiff = (currentTime - lastUpdateTime) / 1000;
    lastUpdateTime = currentTime;

    if (currentMode === "stress") {
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

    if (currentMode === "emotion" && emotionData?.label) {
      const label = emotionData.label.toLowerCase();
      if (!emotionTimeMap[label]) {
        emotionTimeMap[label] = 0;
      }
      emotionTimeMap[label] += timeDiff;
    }
  }

  function updateChart() {
    const ctx = document.getElementById("stressChart").getContext("2d");

    if (window.myChart) {
      window.myChart.destroy();
    }

    let chartData, chartLabels, chartColors;

    if (currentMode === "stress") {
      chartLabels = ["Stress Time (s)", "No Stress Time (s)"];
      chartData = [stressTime.toFixed(2), noStressTime.toFixed(2)];
      chartColors = ["#f44336", "#4caf50"];
    } else {
      chartLabels = Object.keys(emotionTimeMap);
      chartData = Object.values(emotionTimeMap).map((val) => val.toFixed(2));
      chartColors = chartLabels.map((label) => emotionColors[label] || "gray");
    }

    window.myChart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: chartLabels,
        datasets: [
          {
            label: currentMode === "stress" ? "Time Spent" : "Emotion Time (s)",
            data: chartData,
            backgroundColor: chartColors,
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
      isTracking = true;
      lastUpdateTime = Date.now();
      firstDetection = false;
      previousState = null;
      intervalId = setInterval(() => {
        currentMode === "stress" ? fetchStress() : fetchEmotion();
      }, 1000);
      toggleButton.innerText = "⏹ Stop";
      showFeedback(`Tracking started in "${currentMode}" mode.`);
    } else {
      isTracking = false;
      clearInterval(intervalId);
      stressTime = 0;
      noStressTime = 0;
      lastUpdateTime = null;
      resetBars();
      toggleButton.innerText = "▶️ Start";
      showFeedback("Tracking stopped.");
    }
  });

  analysisButton.addEventListener("click", function () {
    if (!isTracking) {
      const chartContainer = document.getElementById("chart-container");
      chartContainer.style.display =
        chartContainer.style.display === "none" ? "block" : "none";
      updateChart();
    }
  });

  logoutButton.addEventListener("click", function () {
    fetch("/logout", { method: "GET" })
      .then((response) => {
        if (response.redirected) {
          window.location.href = response.url;
        }
      })
      .catch((error) => console.error("Logout failed:", error));
  });

  document.querySelectorAll(".mode-btn").forEach((btn) =>
    btn.addEventListener("click", function () {
      document
        .querySelectorAll(".mode-btn")
        .forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      currentMode = btn.getAttribute("data-mode");
      resetBars();
      showFeedback(`Switched to "${currentMode}" mode.`);
    })
  );

  resetBars();
});
