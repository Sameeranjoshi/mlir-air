# Model Explorer: graph stats OK but canvas looks empty

## 1. WebGL (most common on clusters / remote desktops)

The **right panel** (node count, edge count, attributes) is normal HTML. The **center** is drawn with **WebGL** (Three.js). If WebGL is blocked or broken, you still see counts but **no shapes**.

**Check:**

- Open **DevTools → Console** on the Model Explorer tab and look for WebGL / Three.js errors.
- In Chrome: `chrome://gpu` — confirm WebGL is available.
- If you use **SSH port-forward** only, that is fine; the problem is usually **GPU / software rendering** on the machine where Chrome runs (e.g. thin X11, no GPU).

**Try:**

- Run Chrome on a **local laptop** with GPU (forward `localhost:8085` to that machine).
- Enable **hardware acceleration** in browser settings.
- Try **Firefox** vs **Chrome** on the same machine.

## 2. Hierarchical layers (namespaces)

Nodes under `namespace: "air.herd"` appear inside a **collapsed layer**. You must **click / double-click the layer** (e.g. `air.herd`) to expand and see op nodes.

**Try a flat export (no namespaces):**

```bash
python -m air_explorer.json_export combined.dot -o flat.json --flatten-namespaces
model-explorer flat.json --port 8085
```

## 3. Zoom / fit

After load, try **mouse wheel zoom**, **middle-button drag** to pan, or any **fit / reset view** control in the toolbar if present.

## 4. JSON shape

Loader must get `{ "label": "...", "graphs": [ ... ] }`. See `json_export.py` docstring.
