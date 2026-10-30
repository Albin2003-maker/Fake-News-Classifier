// ============================================================
// CONFIG: change these to match your FastAPI backend
// ============================================================
const API_BASE_URL = "http://127.0.0.1:8000"; // where uvicorn is running
const PREDICT_ENDPOINT = "/predict";          // your POST route
const REQUEST_FIELD = "text";                 // JSON key your Pydantic model expects

// Label values (lowercase) that your API uses to mean "fake".
// Everything else is treated as "real".
const FAKE_LABELS = ["fake", "false", "unreliable", "0"];

// ============================================================
// Grab the elements we need from the page
// ============================================================
const form = document.getElementById("classifier-form");
const textArea = document.getElementById("news-text");
const charCount = document.getElementById("char-count");
const submitBtn = document.getElementById("submit-btn");
const submitLabel = document.getElementById("submit-label");
const spinner = document.getElementById("spinner");
const clearBtn = document.getElementById("clear-btn");
const errorBox = document.getElementById("error-box");
const resultBox = document.getElementById("result");
const resultLabel = document.getElementById("result-label");
const resultConfidenceText = document.getElementById("result-confidence-text");
const resultBar = document.getElementById("result-bar");
const apiStatus = document.getElementById("api-status");

// ============================================================
// Talking to the API
// ============================================================
async function classifyText(text) {
  // fetch() sends an HTTP request. We await it because it takes time.
  const response = await fetch(API_BASE_URL + PREDICT_ENDPOINT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },   // tell FastAPI we're sending JSON
    body: JSON.stringify({ [REQUEST_FIELD]: text }),    // JS object -> JSON string
  });

  // fetch only throws on network failure; HTTP errors (400, 422, 500) we must check ourselves.
  if (!response.ok) {
    let detail = "";
    try {
      const err = await response.json();
      detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
    } catch (_) { /* response wasn't JSON, ignore */ }
    throw new Error(`Server returned ${response.status}. ${detail}`.trim());
  }

  return response.json(); // JSON string -> JS object
}

// Your API might name fields differently, so we normalise the response here.
// Adjust if yours differs, e.g. {"prediction": "FAKE", "confidence": 0.93}
function normalizeResponse(data) {
  const rawLabel = data.prediction;

  if (rawLabel === undefined) {
    throw new Error("Unexpected response from API: " + JSON.stringify(data));
  }

  const isFake = Number(rawLabel) === 0;
  const decisionScore = Number(data.decision_score);

  if (!Number.isFinite(decisionScore)) {
    throw new Error("API did not return a valid decision score.");
  }

  return { isFake, decisionScore };
}

// ============================================================
// Updating the page
// ============================================================
function setLoading(isLoading) {
  submitBtn.disabled = isLoading;
  spinner.classList.toggle("hidden", !isLoading);
  submitLabel.textContent = isLoading ? "Analyzing…" : "Analyze";
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
  resultBox.classList.add("hidden");
}

function showResult({ isFake, decisionScore }) {
  errorBox.classList.add("hidden");
  resultBox.classList.remove("hidden", "real", "fake");
  resultBox.classList.add(isFake ? "fake" : "real");

  resultLabel.textContent = isFake
    ? "Likely FAKE news"
    : "Likely REAL news";

  resultConfidenceText.textContent =
    `Decision score: ${decisionScore.toFixed(4)} (not a probability)`;

  // Visual indicator: larger absolute scores mean a stronger decision margin.
  const strength = Math.min(Math.abs(decisionScore) / 2, 1) * 100;

  resultBar.style.width = "0%";

  requestAnimationFrame(() => {
    resultBar.style.width = strength + "%";
  });
}

function clearAll() {
  textArea.value = "";
  updateCharCount();
  errorBox.classList.add("hidden");
  resultBox.classList.add("hidden");
  textArea.focus();
}

function updateCharCount() {
  charCount.textContent = `${textArea.value.length} / ${textArea.maxLength}`;
}

// ============================================================
// Event listeners
// ============================================================
form.addEventListener("submit", async (event) => {
  event.preventDefault(); // stop the browser from reloading the page

  const text = textArea.value.trim();
  if (text.length < 20) {
    showError("Please enter at least a sentence or two (20+ characters) for a meaningful result.");
    return;
  }

  setLoading(true);
  try {
    const data = await classifyText(text);
    showResult(normalizeResponse(data));
  } catch (err) {
    // "Failed to fetch" usually means the server is down or CORS is blocking the request.
    if (err instanceof TypeError) {
      showError("Could not reach the API. Is FastAPI running, and is CORS enabled for this page?");
    } else {
      showError(err.message);
    }
    console.error(err);
  } finally {
    setLoading(false); // runs whether it succeeded or failed
  }
});

textArea.addEventListener("input", updateCharCount);
clearBtn.addEventListener("click", clearAll);

document.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    textArea.value = chip.dataset.example;
    updateCharCount();
  });
});

// ============================================================
// On page load: check whether the API is reachable
// FastAPI serves /docs by default, so we ping that.
// ============================================================
async function checkApi() {
  try {
    const res = await fetch(API_BASE_URL + "/docs", { method: "GET" });
    if (!res.ok) throw new Error();
    apiStatus.textContent = "API connected";
    apiStatus.className = "api-status ok";
  } catch (_) {
    apiStatus.textContent = "API not reachable at " + API_BASE_URL;
    apiStatus.className = "api-status down";
  }
}

updateCharCount();
checkApi();
