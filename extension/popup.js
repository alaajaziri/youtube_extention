function getCurrentTab() {
  return new Promise((resolve, reject) => {
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (!tabs || !tabs.length) {
        reject(new Error("No active tab"));
        return;
      }
      resolve(tabs[0]);
    });
  });
}

function sendRuntimeMessage(message) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(message, (resp) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }
      if (!resp?.ok) {
        reject(new Error(resp?.error || "Unknown extension error"));
        return;
      }
      resolve(resp.data);
    });
  });
}

function sendTabMessage(tabId, message) {
  return new Promise((resolve, reject) => {
    chrome.tabs.sendMessage(tabId, message, (resp) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }
      if (!resp?.ok) {
        reject(new Error(resp?.error || "Tab action failed"));
        return;
      }
      resolve(resp);
    });
  });
}

function setStatus(text) {
  document.getElementById("status").textContent = text;
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function renderResults(tabId, results) {
  const list = document.getElementById("results");
  list.innerHTML = "";

  for (const item of results) {
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="result-head">
        <span>#${item.rank} ${escapeHtml(item.time_range)}</span>
        <span>${Number(item.score).toFixed(3)}</span>
      </div>
      <button class="jump-btn" data-seconds="${item.start_seconds}">Jump</button>
    `;
    list.appendChild(li);
  }

  list.querySelectorAll(".jump-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const seconds = Number(btn.getAttribute("data-seconds") || "0");
      try {
        await sendTabMessage(tabId, { type: "SEEK_VIDEO", seconds });
        setStatus(`Jumped to ${seconds.toFixed(1)}s`);
      } catch (err) {
        setStatus(`Seek error: ${err.message}`);
      }
    });
  });
}

async function main() {
  const tab = await getCurrentTab();
  const url = tab.url || "";
  document.getElementById("videoUrl").textContent = url;

  if (!url.includes("youtube.com") && !url.includes("youtu.be")) {
    setStatus("Open a YouTube video tab first.");
    return;
  }

  document.getElementById("indexBtn").addEventListener("click", async () => {
    setStatus("Indexing video... first run may take a while.");
    try {
      const data = await sendRuntimeMessage({ type: "INDEX_VIDEO", videoUrl: url });
      setStatus(`Indexed ${data.speechIndexedWindows} speech windows, ${data.imageIndexedFrames} image frames.`);
    } catch (err) {
      setStatus(`Index error: ${err.message}`);
    }
  });

  document.getElementById("searchBtn").addEventListener("click", async () => {
    const query = document.getElementById("queryInput").value.trim();
    if (!query) {
      setStatus("Enter a query first.");
      return;
    }

    const searchType = document.querySelector('input[name="searchType"]:checked')?.value || "audio";
    const typeLabel = searchType === "speech" ? "Speech" : searchType === "audio" ? "Audio" : "Visual";
    setStatus(`Searching ${typeLabel.toLowerCase()}...`);
    try {
      const data = await sendRuntimeMessage({
        type: "SEARCH_VIDEO",
        videoUrl: url,
        query,
        searchType,
        topK: 5,
      });
      renderResults(tab.id, data.results || []);
      setStatus(`Found ${(data.results || []).length} ${typeLabel.toLowerCase()} result(s).`);
    } catch (err) {
      setStatus(`Search error: ${err.message}`);
    }
  });
}

main().catch((err) => {
  setStatus(`Init error: ${err.message}`);
});
