const API_BASE = "http://127.0.0.1:8000";

async function apiPost(path, body) {
  const controller = new AbortController();
  const timeoutMs = 120000;
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    if (err.name === "AbortError") {
      throw new Error("Request timed out after 120s. Backend may still be indexing.");
    }
    throw new Error(`Cannot reach backend at ${API_BASE}. Is uvicorn running?`);
  } finally {
    clearTimeout(timer);
  }

  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Request failed: ${res.status}`);
  }
  return data;
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  (async () => {
    try {
      if (msg.type === "INDEX_VIDEO") {
        const data = await apiPost("/index", {
          videoUrl: msg.videoUrl,
          windowSeconds: 2.0,
          hopSeconds: 1.0,
          batchSize: 16,
        });
        sendResponse({ ok: true, data });
        return;
      }

      if (msg.type === "SEARCH_AUDIO" || msg.type === "SEARCH_VIDEO") {
        const data = await apiPost("/search", {
          videoUrl: msg.videoUrl,
          query: msg.query,
          searchType: msg.searchType || "audio",
          topK: msg.topK || 5,
        });
        sendResponse({ ok: true, data });
        return;
      }

      sendResponse({ ok: false, error: "Unknown message type" });
    } catch (err) {
      sendResponse({ ok: false, error: err.message || String(err) });
    }
  })();

  return true;
});
