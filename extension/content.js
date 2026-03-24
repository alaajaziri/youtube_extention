function seekToSeconds(seconds) {
  const video = document.querySelector("video");
  if (!video) {
    return { ok: false, error: "No video element found" };
  }
  video.currentTime = Math.max(0, Number(seconds) || 0);
  video.play().catch(() => {});
  return { ok: true };
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg.type === "SEEK_VIDEO") {
    sendResponse(seekToSeconds(msg.seconds));
    return;
  }
  sendResponse({ ok: false, error: "Unknown content action" });
});
