// Minimal service worker for the Bombsurace Edge PWA.
//
// Strategy: network-first for navigation and same-origin GETs, falling back
// to the last cached copy when offline. Predictions change weekly, so a
// stale cache is far better than no page at all when you're somewhere with
// no signal -- but we always prefer the freshest data when the network is
// up. Bump CACHE_NAME any time this file or the shell it caches changes
// shape, so old clients don't get stuck serving a mismatched cache.
const CACHE_NAME = "bombsurace-edge-v2";

self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET" || new URL(req.url).origin !== self.location.origin) return;

  event.respondWith(
    fetch(req)
      .then((res) => {
        const copy = res.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(req, copy));
        return res;
      })
      .catch(() => caches.match(req).then((cached) => cached || caches.match("index.html")))
  );
});
