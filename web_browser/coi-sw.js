// coi-sw.js -- service worker that makes the page cross-origin isolated on hosts that can't set headers
// (GitHub Pages). It adds Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy to this site's own
// responses, which lets ONNX Runtime Web run on several threads. Requests to other sites (the model on
// Hugging Face) pass through untouched; they are CORS requests, which COEP allows.
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (new URL(req.url).origin !== self.location.origin) return;
  if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
  e.respondWith(fetch(req).then((res) => {
    if (res.status === 0) return res;
    const headers = new Headers(res.headers);
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    headers.set("Cross-Origin-Embedder-Policy", "require-corp");
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
  }));
});
