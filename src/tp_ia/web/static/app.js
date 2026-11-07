/*
 * Laboratorio de árboles de decisión: C4.5 propio contra Random Forest.
 *
 * GUÍA DE LECTURA
 * ---------------
 * Es un módulo de anywidget: exporta { render({ model, el }) }. Todo el cálculo se hace en Python; acá solo se piden
 * cosas con model.send({ id, method, params }) y se dibuja lo que vuelve ({ id, ok, result } o { id, ok: false, error }).
 * No usa ninguna librería externa: los gráficos y el árbol son SVG armado a mano, así que funciona sin conexión.
 *
 * Partes: utilidades de DOM y de formato -> comunicación con Python -> controles -> gráficos -> una vista por pestaña
 * (Datos, Árbol, Comparar, Laboratorio) -> armazón de la aplicación.
 */

const SVGNS = "http://www.w3.org/2000/svg";
const LOCALE = "es-AR";
const DEFAULT_PARAMS = {
  criterion: "gain_ratio", max_depth: null, min_samples_leaf: 2, min_gain: 0, use_avg_gain: true,
  threshold_mode: "c45", numeric_gain_correction: false, prune: false, confidence: 0.25,
};

/* ============================ utilidades de DOM y de formato ============================ */
function applyAttrs(el, attrs) {
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2).toLowerCase(), value);
    else if (key === "style" && typeof value === "object") Object.assign(el.style, value);
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, value);
  }
}
function append(el, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
}
function h(tag, attrs = {}, ...kids) { const el = document.createElement(tag); applyAttrs(el, attrs); append(el, kids); return el; }
function s(tag, attrs = {}, ...kids) { const el = document.createElementNS(SVGNS, tag); applyAttrs(el, attrs); append(el, kids); return el; }
function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }

const isNil = (x) => x === null || x === undefined || Number.isNaN(x);
const num = (x, d = 3) => (isNil(x) ? "—" : Number(x).toLocaleString(LOCALE, { minimumFractionDigits: 0, maximumFractionDigits: d }));
const fix = (x, d = 3) => (isNil(x) ? "—" : Number(x).toLocaleString(LOCALE, { minimumFractionDigits: d, maximumFractionDigits: d }));
const pct = (x, d = 1) => (isNil(x) ? "—" : `${fix(x * 100, d)} %`);
const secs = (x) => (isNil(x) ? "—" : x < 1 ? `${fix(x * 1000, 0)} ms` : `${fix(x, 2)} s`);
const classColor = (i) => `var(--c${i % 7})`;
const sum = (xs) => xs.reduce((a, b) => a + b, 0);
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
function debounce(fn, ms) { let t; return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); }; }
function shorten(text, n) { text = String(text); return text.length > n ? `${text.slice(0, n - 1)}…` : text; }

/* ============================ comunicación con Python ============================ */
function createRpc(model) {
  let seq = 0;
  const pending = new Map();
  const onMessage = (msg) => {
    const waiting = pending.get(msg && msg.id);
    if (!waiting) return;
    pending.delete(msg.id);
    if (msg.ok) waiting.resolve(msg.result);
    else waiting.reject(new Error(msg.error || "Error desconocido"));
  };
  model.on("msg:custom", onMessage);
  const call = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++seq;
    pending.set(id, { resolve, reject });
    model.send({ id, method, params });
  });
  call.dispose = () => model.off("msg:custom", onMessage);
  return call;
}

/* ============================ controles ============================ */
function slider({ label, help, min, max, step, value, format = (v) => num(v), disabled = false, onInput, onChange }) {
  const out = h("output", {}, format(value));
  const input = h("input", { type: "range", min, max, step, value, disabled, "aria-label": label });
  input.addEventListener("input", () => { out.textContent = format(Number(input.value)); if (onInput) onInput(Number(input.value)); });
  input.addEventListener("change", () => { if (onChange) onChange(Number(input.value)); });
  const el = h("div", { class: "tl-param" },
    h("label", { class: "tl-lab" }, h("span", {}, label), out), input, help ? h("p", { class: "tl-help" }, help) : null);
  return { el, input, set(v) { input.value = v; out.textContent = format(Number(v)); }, setDisabled(d) { input.disabled = d; } };
}
function checkbox(label, checked, onChange, help) {
  const input = h("input", { type: "checkbox", checked });
  input.addEventListener("change", () => onChange(input.checked));
  return { el: h("div", { class: "tl-param" }, h("label", { class: "tl-check" }, input, h("span", {}, label)),
    help ? h("p", { class: "tl-help" }, help) : null), input };
}
function segmented(options, value, onChange) {
  const buttons = options.map(([key, text]) => {
    const b = h("button", { type: "button", "aria-pressed": String(key === value) }, text);
    b.addEventListener("click", () => { set(key); onChange(key); });
    return [key, b];
  });
  function set(key) { for (const [k, b] of buttons) b.setAttribute("aria-pressed", String(k === key)); }
  return { el: h("div", { class: "tl-seg", role: "group" }, buttons.map(([, b]) => b)), set };
}
function selectBox(options, value, onChange, attrs = {}) {
  const el = h("select", { class: "tl-select", ...attrs }, options.map(([key, text]) =>
    h("option", { value: key, selected: String(key) === String(value) }, text)));
  el.addEventListener("change", () => onChange(el.value));
  return el;
}
function button(text, onClick, kind = "") {
  const b = h("button", { type: "button", class: `tl-btn ${kind}`.trim() }, text);
  b.addEventListener("click", onClick);
  return b;
}
function field(label, control, help) {
  return h("div", { class: "tl-param" }, h("label", { class: "tl-lab" }, h("span", {}, label)), control,
    help ? h("p", { class: "tl-help" }, help) : null);
}

/* ============================ gráficos pequeños ============================ */
function stackedBar(counts, { height = 10, gap = 1 } = {}) {
  const total = sum(counts) || 1;
  return h("div", { class: "tl-pbar", style: { height: `${height}px`, gap: `${gap}px` } },
    counts.map((c, i) => c > 0 ? h("div", { style: { flex: `${c / total} 1 0`, background: classColor(i), minWidth: "2px" },
      title: `${num(c, 2)}` }) : null));
}
function legend(classes) {
  return h("div", { class: "tl-legend" }, classes.map((c, i) =>
    h("span", {}, h("i", { style: { background: classColor(i) } }), c.name + (c.count !== undefined ? ` (${num(c.count, 0)})` : ""))));
}
function miniHistogram(feature, classes, width = 260, height = 52) {
  const edges = feature.hist.edges, bins = edges.length - 1;
  const stacks = Array.from({ length: bins }, (_, b) => classes.map((c) => (feature.hist.by_class[String(c.value)] || [])[b] || 0));
  const top = Math.max(...stacks.map(sum), 1), bw = width / bins;
  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, width: "100%", height, role: "img",
    "aria-label": `Distribución de ${feature.name} por clase` });
  stacks.forEach((stack, b) => {
    let y = height;
    stack.forEach((count, i) => {
      const hh = (count / top) * (height - 2);
      y -= hh;
      if (hh > 0) svg.append(s("rect", { x: b * bw + 0.5, y, width: Math.max(bw - 1, 1), height: hh, fill: classColor(i), opacity: 0.92 }));
    });
  });
  return svg;
}
function valueBars(feature) {
  const entries = Object.entries(feature.counts).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const top = Math.max(...entries.map((e) => e[1]), 1);
  return h("div", { class: "tl-bars" }, entries.map(([value, count]) =>
    h("div", { class: "tl-bar" }, h("span", { title: value }, shorten(value, 16)),
      h("div", { class: "tl-track" }, h("div", { class: "tl-fill", style: { width: `${(count / top) * 100}%` } })),
      h("span", {}, num(count, 0)))));
}

/* ============================ vista: Datos ============================ */
function buildDatos(ctx) {
  const { S } = ctx;
  const info = S.info;
  const classes = info.classes;
  const missingCells = sum(Object.values(info.quality.missing).map((m) => m.count));
  const root = h("div", {});

  root.append(h("div", { class: "tl-datasets", role: "group", "aria-label": "Datasets" }, S.datasets.map((d) =>
    h("button", { type: "button", class: "tl-dataset", "aria-pressed": String(d.key === info.key), onclick: () => ctx.selectDataset(d.key) },
      h("b", {}, d.title), h("span", { class: "tl-muted tl-small" }, d.description)))));

  root.append(h("section", { class: "tl-section" },
    h("h2", { class: "tl-h2" }, info.title),
    h("p", { class: "tl-muted" }, info.description, " ", h("span", { class: "tl-small" }, `Fuente: ${info.source}`)),
    info.disclaimer ? h("p", { class: "tl-warn", style: { marginTop: "8px" } }, info.disclaimer) : null,
    h("div", { class: "tl-facts", style: { marginTop: "14px" } },
      [[num(info.n_rows, 0), "casos"], [num(info.n_features, 0), "atributos"], [num(classes.length, 0), "clases"],
        [num(missingCells, 0), "valores faltantes"], [num(info.quality.duplicates, 0), "filas duplicadas"]]
        .map(([value, label]) => h("div", { class: "tl-fact" }, h("b", {}, value), h("span", {}, label)))),
    h("div", { class: "tl-classbar", role: "img", "aria-label": "Distribución de clases" }, classes.map((c, i) =>
      h("div", { style: { flex: `${c.count} 1 0`, background: classColor(i) }, title: `${c.name}: ${c.count}` }))),
    legend(classes)));

  if (info.quality.warnings.length) {
    root.append(h("div", { class: "tl-section", style: { display: "grid", gap: "8px" } },
      info.quality.warnings.map((w) => h("p", { class: "tl-warn" }, w))));
  }

  root.append(h("section", { class: "tl-section" }, h("h3", { class: "tl-h3" }, `Atributos (${info.features.length}). Los colores son las clases.`),
    h("div", { class: "tl-features" }, info.features.map((f) => h("div", { class: "tl-feature" },
      h("header", {}, h("b", {}, f.name),
        h("span", { style: { display: "flex", gap: "6px" } },
          f.missing ? h("span", { class: "tl-tag tl-tag-warn" }, `${f.missing} faltantes`) : null,
          f.kind === "numeric" && f.outliers ? h("span", { class: "tl-tag tl-tag-warn", title: "Fuera de 1,5 veces el rango intercuartil (se conservan)" }, `${f.outliers} extremos`) : null,
          h("span", { class: "tl-tag" }, f.kind === "numeric" ? "numérico" : "nominal"))),
      f.kind === "numeric"
        ? [miniHistogram(f, classes), h("p", { class: "tl-muted tl-small" }, `rango ${num(f.min)} a ${num(f.max)} · mediana ${num(f.median)}`)]
        : valueBars(f))))));

  root.append(h("section", { class: "tl-section tl-decisions" },
    h("h3", { class: "tl-h3" }, "Decisiones sobre los datos (para el informe)"),
    info.decisions.map((d) => h("details", {}, h("summary", {}, `${d.id} · ${d.title}: `, h("span", { class: "tl-muted" }, d.chosen)),
      h("div", {}, d.rationale ? h("p", {}, d.rationale) : null,
        d.alternatives.length ? h("p", {}, `Alternativas: ${d.alternatives.join("; ")}`) : null,
        Object.keys(d.evidence).length ? h("p", { class: "tl-small" }, `Evidencia: ${Object.entries(d.evidence).map(([k, v]) => `${k} = ${typeof v === "object" ? JSON.stringify(v) : v}`).join(", ")}`) : null)))));

  const keys = Object.keys(info.preview[0] || {});
  root.append(h("section", { class: "tl-section" }, h("details", {}, h("summary", { class: "tl-h3", style: { cursor: "pointer" } }, "Primeras filas"),
    h("div", { class: "tl-table-wrap" }, h("table", { class: "tl-table" },
      h("thead", {}, h("tr", {}, keys.map((k) => h("th", {}, k)))),
      h("tbody", {}, info.preview.map((row) => h("tr", {}, keys.map((k) => h("td", {}, isNil(row[k]) ? "—" : typeof row[k] === "number" ? num(row[k]) : row[k]))))))))));

  root.append(buildUpload(ctx));
  return { el: root };
}

function buildUpload(ctx) {
  const box = h("section", { class: "tl-section" }, h("h3", { class: "tl-h3" }, "Probar con tu propio archivo"),
    h("p", { class: "tl-muted tl-small" }, "Un CSV con una columna que sea la clase a predecir. Se detectan solos el separador, los decimales y los faltantes."));
  const form = h("div", { class: "tl-row", style: { marginTop: "8px" } });
  const picker = h("input", { type: "file", accept: ".csv,.tsv,.txt", "aria-label": "Archivo CSV" });
  picker.addEventListener("change", async () => {
    const file = picker.files[0];
    if (!file) return;
    const text = await file.text();
    const first = text.split(/\r?\n/)[0] || "";
    const sep = [";", "\t", "|", ","].map((c) => [c, first.split(c).length]).sort((a, b) => b[1] - a[1])[0][0];
    const columns = first.split(sep).map((c) => c.trim().replace(/^"|"$/g, "")).filter(Boolean);
    const target = selectBox(columns.map((c) => [c, c]), columns[columns.length - 1], () => {}, { "aria-label": "Columna de la clase" });
    clear(form).append(h("span", {}, `${file.name}: ¿qué columna es la clase?`), target,
      button("Usar este archivo", () => ctx.loadCsv(text, target.value, file.name.replace(/\.\w+$/, "")), "tl-btn-primary"));
  });
  form.append(picker);
  box.append(form);
  return box;
}

/* ============================ vistas pendientes (se completan más abajo) ============================ */
/* ============================ vista: Árbol ============================ */
const RIBBON_MAX = 30;

function ruleText(node, branch) {
  return node.kind === "numeric" ? `${node.feature} ${node.edges[branch]}` : `${node.feature} = ${node.edges[branch]}`;
}

/* Acomoda el árbol: las hojas en fila, cada nodo interno centrado sobre sus hijos y un nivel por profundidad. */
function layoutTree(data, { nodeW = 124, nodeH = 54, gapX = 16, gapY = 72 } = {}) {
  const nodes = [];
  let leaves = 0;
  const column = nodeW + gapX;
  function place(node, parent, branch) {
    const item = { id: node.id, data: node, parent, depth: node.depth, children: [],
      rule: parent ? ruleText(parent.data, branch) : null, path: parent ? [...parent.path, ruleText(parent.data, branch)] : [] };
    nodes.push(item);
    if (node.leaf) item.x = leaves++ * column + column / 2;
    else {
      item.children = node.children.map((child, i) => place(child, item, i));
      item.x = (item.children[0].x + item.children[item.children.length - 1].x) / 2;
    }
    item.y = node.depth * (nodeH + gapY) + nodeH / 2 + 10;
    return item;
  }
  const root = place(data, null, 0);
  const deepest = Math.max(...nodes.map((n) => n.depth));
  return { nodes, root, nodeW, nodeH, width: Math.max(leaves, 1) * column, height: (deepest + 1) * (nodeH + gapY) - gapY + 20 };
}

function argmax(xs) { let best = 0; xs.forEach((x, i) => { if (x > xs[best]) best = i; }); return best; }

/* Dibuja el árbol como flujo de casos: el ancho de cada cinta es la cantidad de casos y se apila por clase. */
function drawTree(host, layout, { classes, zoom, trace, pinnedId, onHover, onPin }) {
  const { nodes, root, nodeW, nodeH } = layout;
  const rootN = root.data.n || 1;
  const ribbon = (n) => Math.max(1.5, (n / rootN) * RIBBON_MAX);
  const svg = s("svg", { class: `tl-tree${trace ? " has-path" : ""}`, viewBox: `0 0 ${layout.width} ${layout.height}`,
    width: layout.width * zoom, height: layout.height * zoom, role: "img",
    "aria-label": "Árbol de decisión: el ancho de cada rama es la cantidad de casos que pasan por ella" });
  const linkLayer = s("g", {}), edgeLayer = s("g", {}), nodeLayer = s("g", {});
  svg.append(linkLayer, nodeLayer, edgeLayer);
  const elements = new Map();

  for (const item of nodes) {
    if (!item.children.length) continue;
    const widths = item.children.map((c) => ribbon(c.data.n));
    let cursor = item.x - sum(widths) / 2;
    item.children.forEach((child, i) => {
      const w = widths[i], sx = cursor + w / 2;
      cursor += w;
      const y1 = item.y + nodeH / 2, y2 = child.y - nodeH / 2, ym = (y1 + y2) / 2;
      const counts = child.data.counts, total = sum(counts) || 1;
      const onPath = trace && (trace[child.id] || 0) > 0;
      const group = s("g", { class: `tl-link${onPath ? " is-path" : ""}` });
      let offset = 0;
      counts.forEach((count, k) => {
        const wk = (count / total) * w;
        if (wk > 0.05) {
          const xs = sx - w / 2 + offset + wk / 2, xe = child.x - w / 2 + offset + wk / 2;
          group.append(s("path", { d: `M ${xs} ${y1} C ${xs} ${ym}, ${xe} ${ym}, ${xe} ${y2}`, stroke: classColor(k),
            "stroke-width": wk, opacity: 0.88, fill: "none" }));
        }
        offset += wk;
      });
      linkLayer.append(group);
      edgeLayer.append(s("text", { class: "tl-edge", x: (sx + child.x) / 2, y: ym, "text-anchor": "middle", "dominant-baseline": "middle" },
        shorten(item.data.edges[i], 18)));
    });
  }

  for (const item of nodes) {
    const d = item.data, weight = trace ? trace[item.id] || 0 : 0;
    const majority = argmax(d.counts);
    const left = item.x - nodeW / 2, top = item.y - nodeH / 2;
    const g = s("g", { class: `tl-node${weight > 0 && trace ? " is-path" : ""}${pinnedId === item.id ? " is-pinned" : ""}`,
      tabindex: 0, role: "button", "aria-label": d.leaf ? `Hoja ${d.label}, ${num(d.n, 1)} casos` : `Nodo ${d.feature}, ${num(d.n, 1)} casos` });
    const box = s("rect", { class: "tl-box", x: left, y: top, width: nodeW, height: nodeH, rx: 9 });
    if (d.leaf) { box.setAttribute("style", `fill: color-mix(in srgb, ${classColor(majority)} 20%, var(--surface)); stroke: ${classColor(majority)}`); }
    g.append(box);
    if (d.leaf) {
      g.append(s("text", { class: "tl-leaf-label", x: item.x, y: top + 22, "text-anchor": "middle" }, shorten(d.label, 17)),
        s("text", { class: "tl-n", x: item.x, y: top + 40, "text-anchor": "middle" },
          `n = ${num(d.n, 1)}${d.errors > 1e-9 ? ` · errores ${num(d.errors, 1)}` : ""}`));
    } else {
      g.append(s("text", { class: "tl-feat", x: item.x, y: top + 20, "text-anchor": "middle" }, shorten(d.feature, 17)),
        s("text", { class: "tl-n", x: item.x, y: top + 35, "text-anchor": "middle" }, `n = ${num(d.n, 1)}`));
      let cursor = left + 8;
      const total = sum(d.counts) || 1;
      d.counts.forEach((count, k) => {
        const w = (count / total) * (nodeW - 16);
        if (w > 0) g.append(s("rect", { x: cursor, y: top + nodeH - 11, width: w, height: 5, fill: classColor(k), rx: 1.5 }));
        cursor += w;
      });
    }
    if (trace && weight > 0 && weight < 0.9995) {
      g.append(s("text", { class: "tl-n", x: left + nodeW - 4, y: top - 4, "text-anchor": "end", style: "font-weight:640; fill: var(--ink)" }, `×${num(weight, 2)}`));
    }
    g.addEventListener("mouseenter", () => onHover(item));
    g.addEventListener("focus", () => onHover(item));
    g.addEventListener("click", () => onPin(item));
    g.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onPin(item); } });
    nodeLayer.append(g);
    elements.set(item.id, g);
  }
  clear(host).append(svg);
  return { svg, elements };
}

function confusionTable(matrix, names) {
  const rowTotals = matrix.map(sum);
  return h("table", { class: "tl-matrix" },
    h("thead", {}, h("tr", {}, h("th", {}, "real \\ predicho"), names.map((n) => h("th", {}, shorten(n, 12))))),
    h("tbody", {}, matrix.map((row, i) => h("tr", {}, h("th", {}, shorten(names[i], 12)),
      row.map((count, j) => h("td", { style: { background: `color-mix(in srgb, ${i === j ? "var(--ok)" : "var(--warn)"} ${Math.round(clamp(count / (rowTotals[i] || 1), 0, 1) * 70)}%, var(--surface-2))` } },
        num(count, 0)))))));
}

function importanceBars(items, limit = 8) {
  if (!items.length) return h("p", { class: "tl-muted" }, "El árbol no usa ningún atributo (es una sola hoja).");
  const top = items[0].value || 1;
  return h("div", { class: "tl-bars" }, items.slice(0, limit).map((it) => h("div", { class: "tl-bar" },
    h("span", { title: it.feature }, shorten(it.feature, 16)),
    h("div", { class: "tl-track" }, h("div", { class: "tl-fill", style: { width: `${(it.value / top) * 100}%` } })), h("span", {}, pct(it.value, 0)))));
}

function buildArbol(ctx) {
  const { S, rpc, run } = ctx;
  const info = S.info;
  const classes = info.classes;
  const hasNumeric = info.features.some((f) => f.kind === "numeric");
  const V = { result: null, item: null, pinned: null, zoom: null, seq: 0, prediction: null, values: {}, missing: {}, touched: false, layout: null, handles: null };
  info.features.forEach((f) => { V.values[f.name] = f.default; V.missing[f.name] = false; });

  /* ---- panel de parámetros ---- */
  const P = S.params;
  const retrain = debounce(() => train(), 180);
  const set = (key, value) => { P[key] = value; retrain(); };
  const depthSlider = slider({ label: "Profundidad máxima", min: 1, max: 12, step: 1, value: P.max_depth || 4, disabled: P.max_depth === null,
    help: "Cuántos cortes seguidos puede hacer el árbol.", onInput: (v) => set("max_depth", v) });
  const depthOut = depthSlider.el.querySelector("output");
  if (P.max_depth === null) depthOut.textContent = "sin límite";
  const unlimited = checkbox("Sin límite de profundidad", P.max_depth === null, (on) => {
    depthSlider.setDisabled(on);
    depthOut.textContent = on ? "sin límite" : num(Number(depthSlider.input.value), 0);
    set("max_depth", on ? null : Number(depthSlider.input.value));
  });
  const confidence = slider({ label: "Confianza de la poda", min: 0.01, max: 0.5, step: 0.01, value: P.confidence, disabled: !P.prune,
    help: "Más chica poda más. C4.5 usa 0,25.", onInput: (v) => set("confidence", v) });
  const pruneBox = checkbox("Podar por error pesimista", P.prune, (on) => { confidence.setDisabled(!on); set("prune", on); },
    "Reemplaza ramas que no mejoran el error estimado por una hoja.");
  const testSlider = slider({ label: "Casos de prueba", min: 0, max: 50, step: 5, value: Math.round(info.split.test_size * 100),
    format: (v) => `${num(v, 0)} %`, help: "Se guardan aparte para medir el árbol. Con 0 % se entrena con todo.",
    onChange: (v) => run("Repartiendo los casos", async () => { S.info.split = await rpc("set_split", { test_size: v / 100, seed: 0 }); await train(true); }).catch(() => {}) });
  const criterion = segmented([["gain_ratio", "Razón de ganancia"], ["gain", "Ganancia"]], P.criterion, (k) => set("criterion", k));
  const threshold = segmented([["c45", "Valor observado"], ["midpoint", "Punto medio"]], P.threshold_mode, (k) => set("threshold_mode", k));
  const rail = h("aside", { class: "tl-rail", "aria-label": "Parámetros del árbol" },
    h("h2", { class: "tl-h2" }, "Parámetros"),
    testSlider.el,
    field("Criterio de corte", criterion.el, "C4.5 usa la razón de ganancia: la ganancia pura favorece atributos con muchas ramas."),
    depthSlider.el, unlimited.el,
    slider({ label: "Mínimo de casos por hoja", min: 1, max: 20, step: 1, value: P.min_samples_leaf, help: "Al menos dos ramas deben tener este peso (parámetro -m de C4.5).", onInput: (v) => set("min_samples_leaf", v) }).el,
    slider({ label: "Ganancia mínima", min: 0, max: 0.5, step: 0.01, value: P.min_gain, help: "Un corte que gana menos que esto no se hace.", onInput: (v) => set("min_gain", v) }).el,
    checkbox("Heurística de ganancia promedio", P.use_avg_gain, (on) => set("use_avg_gain", on), "Solo compiten los atributos con ganancia al menos igual al promedio.").el,
    hasNumeric ? field("Umbral numérico", threshold.el, "Dónde se corta un atributo numérico.") : null,
    hasNumeric ? checkbox("Corrección de la versión 8", P.numeric_gain_correction, (on) => set("numeric_gain_correction", on), "Penaliza los cortes sobre atributos numéricos.").el : null,
    pruneBox.el, confidence.el,
    button("Restablecer", () => { Object.assign(P, DEFAULT_PARAMS); ctx.S.views.arbol = null; ctx.goto("arbol"); }, "tl-btn-quiet"));

  /* ---- zona principal ---- */
  const stats = h("div", { class: "tl-stats" });
  const treeHost = h("div", {});
  const tools = h("div", { class: "tl-treetools" },
    button("−", () => zoomBy(0.8)), button("+", () => zoomBy(1.25)), button("Ajustar", () => { V.zoom = null; drawNow(); }));
  tools.querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-label", ["Alejar", "Acercar", "Ajustar al ancho"][i]));
  const treeWrap = h("div", { class: "tl-treewrap" }, tools, treeHost);
  const inspector = h("div", { class: "tl-inspector" });
  const caseHost = h("div", {});
  const metricsHost = h("div", {});
  const lower = h("div", { class: "tl-lower" },
    h("section", {}, h("h3", { class: "tl-h3" }, "Nodo"), inspector),
    h("section", {}, h("h3", { class: "tl-h3" }, "Predecir un caso"), caseHost),
    h("section", {}, metricsHost));
  const main = h("div", {}, stats, treeWrap, lower);
  const el = h("div", { class: "tl-arbol" }, rail, main);

  function zoomBy(factor) { V.zoom = clamp((V.zoom || 1) * factor, 0.3, 2.5); drawNow(); }

  function drawInspector() {
    const item = V.item || V.layout.root;
    const d = item.data;
    clear(inspector);
    inspector.append(
      item.path.length ? h("p", { class: "tl-muted tl-small", style: { marginBottom: "6px" } }, item.path.join("  →  ")) : h("p", { class: "tl-muted tl-small", style: { marginBottom: "6px" } }, "Raíz: todos los casos de entrenamiento"),
      h("dl", {}, ...[
        d.leaf ? ["Es una hoja que predice", d.label] : ["Corta por", d.feature],
        ["Casos", num(d.n, 2)], ["Entropía", fix(d.entropy)],
        d.leaf ? null : ["Ganancia", fix(d.gain)], d.leaf ? null : ["Razón de ganancia", fix(d.gain_ratio)],
        ["Error si fuera hoja", num(d.errors, 2)],
      ].filter(Boolean).flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])),
      h("div", { class: "tl-bars", style: { marginTop: "10px" } }, classes.map((c, i) => h("div", { class: "tl-bar" },
        h("span", {}, h("i", { class: "tl-dot", style: { background: classColor(i) } }), shorten(c.name, 12)),
        h("div", { class: "tl-track" }, h("div", { class: "tl-fill", style: { width: `${(d.counts[i] / (d.n || 1)) * 100}%`, background: classColor(i) } })),
        h("span", {}, num(d.counts[i], 1))))));
  }

  function drawNow() {
    if (!V.result) return;
    V.layout = layoutTree(V.result.tree.root);
    const width = treeWrap.clientWidth || 760;
    const zoom = V.zoom || clamp((width - 8) / V.layout.width, 0.4, 1);
    V.handles = drawTree(treeHost, V.layout, {
      classes, zoom, trace: V.prediction ? V.prediction.dt.trace : null, pinnedId: V.pinned,
      onHover: (item) => { V.item = item; drawInspector(); },
      onPin: (item) => { V.pinned = V.pinned === item.id ? null : item.id; V.item = item; for (const [id, g] of V.handles.elements) g.classList.toggle("is-pinned", id === V.pinned); drawInspector(); },
    });
    drawInspector();
  }

  function drawStats() {
    const r = V.result, test = r.metrics.test, shown = test || r.metrics.train;
    clear(stats).append(...[
      [pct(shown.accuracy), test ? "exactitud en prueba" : "exactitud en entrenamiento"],
      [fix(shown.macro_f1), "F1 macro"], [num(r.tree.n_nodes, 0), r.tree.n_pruned ? `nodos (se podaron ${r.tree.n_pruned})` : "nodos"],
      [num(r.tree.n_leaves, 0), "hojas"], [num(r.tree.depth, 0), "profundidad"], [secs(r.fit_time), "tiempo de entrenamiento"],
    ].map(([value, label]) => h("div", { class: "tl-stat" }, h("b", {}, value), h("span", {}, label))));
  }

  function drawMetrics() {
    const r = V.result, test = r.metrics.test, names = classes.map((c) => c.name);
    const matrix = (test || r.metrics.train).confusion;
    clear(metricsHost).append(
      h("h3", { class: "tl-h3" }, "Qué atributos usa el árbol"), importanceBars(r.importances),
      h("h3", { class: "tl-h3", style: { marginTop: "16px" } }, test ? "Aciertos y errores en los casos de prueba" : "Aciertos y errores en entrenamiento"),
      h("div", { class: "tl-table-wrap" }, confusionTable(matrix, names)),
      test ? h("p", { class: "tl-muted tl-small", style: { marginTop: "8px" } }, `En entrenamiento: exactitud ${pct(r.metrics.train.accuracy)}. Si es mucho mayor que en prueba, el árbol memoriza.`) : h("p", { class: "tl-muted tl-small", style: { marginTop: "8px" } }, "Sin casos de prueba: la exactitud en entrenamiento es optimista."),
      h("details", { style: { marginTop: "12px" } }, h("summary", { class: "tl-small", style: { cursor: "pointer" } }, "Ver el árbol como texto y como Graphviz"),
        h("pre", { class: "tl-code", style: { marginTop: "8px" } }, r.text),
        h("p", { class: "tl-muted tl-small", style: { margin: "10px 0 4px" } }, "Código DOT (se pega en graphviz.org o se dibuja con el paquete graphviz):"),
        h("pre", { class: "tl-code" }, r.dot)));
  }

  /* ---- predecir un caso ---- */
  const predictSoon = debounce(() => predict(), 200);
  function buildCaseForm() {
    const rows = info.features.map((f) => {
      const missing = h("input", { type: "checkbox", "aria-label": `${f.name}: faltante` });
      const out = h("output", {}, "");
      let input;
      if (f.kind === "numeric") {
        const step = (f.max - f.min) / 200 || 0.01;
        input = h("input", { type: "range", min: f.min, max: f.max, step, value: V.values[f.name], "aria-label": f.name });
        out.textContent = num(V.values[f.name], 2);
        input.addEventListener("input", () => { V.values[f.name] = Number(input.value); out.textContent = num(V.values[f.name], 2); V.touched = true; predictSoon(); });
      } else {
        input = selectBox(f.values.map((v) => [v, v]), V.values[f.name], (v) => { V.values[f.name] = f.values.find((x) => String(x) === v); out.textContent = v; V.touched = true; predictSoon(); }, { "aria-label": f.name });
        out.textContent = String(V.values[f.name]);
      }
      missing.addEventListener("change", () => { V.missing[f.name] = missing.checked; input.disabled = missing.checked; out.textContent = missing.checked ? "faltante" : out.textContent; V.touched = true; predictSoon(); });
      return h("div", { class: "tl-case-row" }, h("label", { class: "tl-lab" }, h("span", {}, f.name), out),
        h("div", { style: { display: "flex", gap: "8px", alignItems: "center" } }, input,
          h("label", { class: "tl-check tl-small", title: "Probar el caso sin este dato" }, missing, "falta")));
    });
    return h("div", { class: "tl-case" }, rows);
  }
  const caseForm = buildCaseForm();
  const caseResult = h("div", {});
  caseHost.append(caseForm, caseResult);

  function verdict(name, cls, model, names) {
    return h("div", {}, h("div", { class: "tl-verdict" }, h("span", { class: cls }, h("i", { class: "tl-dot", style: { background: "currentColor" } }), name),
      h("b", {}, model.label)), stackedBar(model.proba, { height: 12, gap: 2 }),
      h("p", { class: "tl-muted tl-small", style: { marginTop: "3px" } }, names.map((n, i) => `${shorten(n, 14)} ${pct(model.proba[i], 0)}`).join(" · ")));
  }
  function drawPrediction() {
    clear(caseResult);
    const p = V.prediction;
    if (!p) { caseResult.append(h("p", { class: "tl-muted tl-small", style: { marginTop: "10px" } }, "Mové un valor y se calcula solo.")); return; }
    caseResult.append(h("div", { style: { marginTop: "12px" } }, verdict("Árbol propio", "tl-dtc", p.dt, p.classes), verdict("Random Forest", "tl-rfc", p.rf, p.classes),
      p.dt.rules.length ? h("ol", { class: "tl-rules" }, p.dt.rules.map((r) => h("li", {}, r))) : null,
      p.missing.length ? h("p", { class: "tl-muted tl-small", style: { marginTop: "6px" } }, `Sin dato: ${p.missing.join(", ")}. El árbol reparte el caso entre las ramas (los pesos ×0,xx del dibujo).`) : null,
      info.disclaimer ? h("p", { class: "tl-warn tl-small", style: { marginTop: "8px" } }, info.disclaimer) : null));
  }
  async function predict() {
    const row = {};
    for (const f of info.features) row[f.name] = V.missing[f.name] ? null : V.values[f.name];
    const mine = ++V.seq;
    await run("Prediciendo", async () => {
      const prediction = await rpc("predict", { row });
      if (mine !== V.seq) return;
      V.prediction = prediction;
      drawPrediction();
      drawNow();
    }).catch(() => {});
  }

  async function train(resetCase = false) {
    const mine = ++V.seq;
    await run("Entrenando el árbol", async () => {
      const result = await rpc("train", { params: P });
      if (mine !== V.seq) return;
      V.result = result; V.item = null; V.pinned = null;
      if (resetCase) V.prediction = null;
      drawStats(); drawMetrics(); drawPrediction(); drawNow();
    }).catch(() => {});
    if (V.touched && V.result) predict();
  }

  drawPrediction();
  return { el, onShow() { if (!V.result) train(); else drawNow(); } };
}
/* ============================ gráficos de líneas y de pares ============================ */
function lineChart({ width = 470, height = 260, series, labels, xs = null, yMin = null, yMax = null, markIndex = null, ariaLabel = "Gráfico" }) {
  const m = { l: 50, r: 18, t: 14, b: 40 };
  const w = width - m.l - m.r, ht = height - m.t - m.b, n = labels.length;
  const all = series.flatMap((se) => se.points.flatMap((p) => [p.y, p.lo, p.hi].filter((v) => !isNil(v))));
  let lo = yMin !== null ? yMin : Math.min(...all), hi = yMax !== null ? yMax : Math.max(...all);
  if (yMin === null) lo = Math.max(0, lo - ((hi - lo) * 0.15 || 0.05));
  if (yMax === null) hi = Math.min(1, hi + ((hi - lo) * 0.15 || 0.05));
  const x = (i) => (xs ? m.l + ((xs[i] - xs[0]) / ((xs[n - 1] - xs[0]) || 1)) * w : m.l + (n === 1 ? w / 2 : (i / (n - 1)) * w));
  const y = (v) => m.t + (1 - (v - lo) / ((hi - lo) || 1)) * ht;
  const svg = s("svg", { class: "tl-chart", viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": ariaLabel, style: { width: "100%", height: "auto", maxWidth: `${width}px` } });
  for (let k = 0; k < 5; k++) {
    const v = lo + ((hi - lo) * k) / 4;
    svg.append(s("line", { class: "tl-grid", x1: m.l, x2: width - m.r, y1: y(v), y2: y(v) }),
      s("text", { x: m.l - 8, y: y(v), "text-anchor": "end", "dominant-baseline": "middle" }, fix(v, 2)));
  }
  labels.forEach((text, i) => svg.append(s("text", { x: x(i), y: height - 14, "text-anchor": "middle" }, shorten(text, 12))));
  svg.append(s("line", { class: "tl-axis", x1: m.l, x2: width - m.r, y1: m.t + ht, y2: m.t + ht }));
  for (const se of series) {
    const pts = se.points.map((p, i) => ({ ...p, px: x(i) }));
    if (pts.every((p) => !isNil(p.lo) && !isNil(p.hi)) && pts.length > 1) {
      const top = pts.map((p) => `${p.px},${y(p.hi)}`).join(" L "), bottom = [...pts].reverse().map((p) => `${p.px},${y(p.lo)}`).join(" L ");
      svg.append(s("path", { d: `M ${top} L ${bottom} Z`, fill: se.color, opacity: 0.16 }));
    }
    svg.append(s("path", { d: `M ${pts.map((p) => `${p.px},${y(p.y)}`).join(" L ")}`, fill: "none", stroke: se.color, "stroke-width": 2.4,
      "stroke-dasharray": se.dash || null, "stroke-linejoin": "round", "stroke-linecap": "round" }));
    pts.forEach((p, i) => svg.append(s("circle", { cx: p.px, cy: y(p.y), r: 3.6, fill: se.color }, s("title", {}, `${se.name} · ${labels[i]}: ${fix(p.y, 3)}`))));
  }
  if (markIndex !== null && markIndex >= 0) {
    const first = series[0].points[markIndex];
    svg.append(s("circle", { cx: x(markIndex), cy: y(first.y), r: 9, fill: "none", stroke: "var(--ink)", "stroke-width": 1.6 }));
  }
  return svg;
}

/* Un trazo por partición entre el valor del árbol y el del bosque: se ve quién gana en cada una. */
function pairedPlot({ a, b, nameA, nameB, colorA, colorB, width = 320, height = 250 }) {
  const m = { l: 46, r: 16, t: 14, b: 38 };
  const w = width - m.l - m.r, ht = height - m.t - m.b;
  const all = [...a, ...b];
  const pad = (Math.max(...all) - Math.min(...all)) * 0.2 || 0.04;
  const lo = Math.max(0, Math.min(...all) - pad), hi = Math.min(1, Math.max(...all) + pad);
  const y = (v) => m.t + (1 - (v - lo) / ((hi - lo) || 1)) * ht;
  const xa = m.l + w * 0.25, xb = m.l + w * 0.75;
  const svg = s("svg", { class: "tl-chart", viewBox: `0 0 ${width} ${height}`, role: "img", style: { width: "100%", height: "auto", maxWidth: `${width}px` },
    "aria-label": `Exactitud por partición: ${nameA} contra ${nameB}` });
  for (let k = 0; k < 5; k++) {
    const v = lo + ((hi - lo) * k) / 4;
    svg.append(s("line", { class: "tl-grid", x1: m.l, x2: width - m.r, y1: y(v), y2: y(v) }),
      s("text", { x: m.l - 8, y: y(v), "text-anchor": "end", "dominant-baseline": "middle" }, fix(v, 2)));
  }
  a.forEach((va, i) => {
    const vb = b[i], color = Math.abs(vb - va) < 1e-9 ? "var(--muted)" : vb > va ? colorB : colorA;
    svg.append(s("line", { x1: xa, y1: y(va), x2: xb, y2: y(vb), stroke: color, "stroke-width": 1.6, opacity: 0.6 }),
      s("circle", { cx: xa, cy: y(va), r: 3, fill: colorA, opacity: 0.8 }), s("circle", { cx: xb, cy: y(vb), r: 3, fill: colorB, opacity: 0.8 }));
  });
  const meanA = sum(a) / a.length, meanB = sum(b) / b.length;
  svg.append(s("line", { x1: xa - 16, x2: xa + 16, y1: y(meanA), y2: y(meanA), stroke: colorA, "stroke-width": 4, "stroke-linecap": "round" }),
    s("line", { x1: xb - 16, x2: xb + 16, y1: y(meanB), y2: y(meanB), stroke: colorB, "stroke-width": 4, "stroke-linecap": "round" }),
    s("text", { x: xa, y: height - 14, "text-anchor": "middle" }, nameA), s("text", { x: xb, y: height - 14, "text-anchor": "middle" }, nameB));
  return svg;
}

const METRIC_LABELS = { macro_f1: "F1 macro", accuracy: "Exactitud", balanced_accuracy: "Exactitud balanceada" };

/* ============================ vista: Comparar ============================ */
function buildComparar(ctx) {
  const { S, rpc, run } = ctx;
  const V = { folds: 5, repeats: 3, trees: 100, depth: 10, unlimited: false };
  const folds = slider({ label: "Pliegues", min: 2, max: 10, step: 1, value: V.folds, format: (v) => num(v, 0), onInput: (v) => { V.folds = v; },
    help: "Los casos se reparten en partes; cada una se usa una vez como prueba." });
  const repeats = slider({ label: "Repeticiones", min: 1, max: 10, step: 1, value: V.repeats, format: (v) => num(v, 0), onInput: (v) => { V.repeats = v; },
    help: "Cuántas veces se repite el reparto al azar. Más repeticiones, resultados más estables." });
  const trees = slider({ label: "Árboles del bosque", min: 10, max: 300, step: 10, value: V.trees, format: (v) => num(v, 0), onInput: (v) => { V.trees = v; } });
  const depth = slider({ label: "Profundidad del bosque", min: 1, max: 20, step: 1, value: V.depth, format: (v) => num(v, 0), onInput: (v) => { V.depth = v; },
    help: "Valores de la lámina de la cátedra: 100 árboles y profundidad 10." });
  const noLimit = checkbox("Sin límite de profundidad", false, (on) => { V.unlimited = on; depth.setDisabled(on); });
  const result = h("div", { style: { marginTop: "22px" } }, h("p", { class: "tl-empty" }, "Elegí los valores y tocá Comparar."));

  function dtSummary() {
    const p = S.params;
    return `Árbol propio: ${p.prune ? `con poda (confianza ${num(p.confidence, 2)})` : "sin poda"}, profundidad ${p.max_depth === null ? "sin límite" : p.max_depth}, mínimo ${num(p.min_samples_leaf, 0)} casos por hoja. Se cambian en la pestaña Árbol.`;
  }
  const go = button("Comparar", () => run("Comparando (puede tardar unos segundos)", async () => {
    go.disabled = true;
    try {
      const r = await rpc("compare", { dt_params: S.params, rf_params: { n_estimators: V.trees, max_depth: V.unlimited ? null : V.depth },
        n_splits: V.folds, n_repeats: V.repeats });
      draw(r);
    } finally { go.disabled = false; }
  }).catch(() => {}), "tl-btn-primary");

  function row(label, key, formatter, r, note) {
    const cell = (m) => { const v = r.models[m].stats[key]; return v ? `${formatter(v.mean)} ± ${formatter(v.std)}` : "—"; };
    return h("tr", {}, h("td", { title: note || "" }, label), h("td", {}, cell("DT")), h("td", {}, cell("RF")));
  }

  function draw(r) {
    const d = r.paired.accuracy, dt = r.models.DT, rf = r.models.RF;
    const lead = Math.abs(d.mean_diff) < 0.0005 ? "En exactitud, el árbol y el bosque quedan empatados."
      : `En exactitud, el ${d.mean_diff > 0 ? "bosque" : "árbol"} gana por ${fix(Math.abs(d.mean_diff) * 100, 1)} puntos en promedio.`;
    clear(result).append(
      h("div", { class: "tl-note" },
        h("p", {}, h("b", {}, lead), ` El bosque gana en ${d.wins_a} de ${d.n} particiones, empatan en ${d.ties} y el árbol gana en ${d.wins_b}.`),
        h("p", { style: { marginTop: "4px" } }, `Prueba de Wilcoxon: p = ${fix(d.p_wilcoxon, 3)}. ${d.p_wilcoxon < 0.05 ? "La diferencia se repite entre particiones." : "La diferencia no se repite lo suficiente: puede ser ruido."} Es descriptiva: las particiones se solapan, así que no es una prueba formal.`)),
      h("div", { class: "tl-two tl-section" },
        h("div", {}, h("h3", { class: "tl-h3" }, `Resultados en ${r.n_partitions} particiones de prueba (media ± desvío)`),
          h("div", { class: "tl-table-wrap" }, h("table", { class: "tl-table" },
            h("thead", {}, h("tr", {}, h("th", {}, ""), h("th", { class: "tl-dtc" }, "Árbol propio"), h("th", { class: "tl-rfc" }, "Random Forest"))),
            h("tbody", {}, row("Exactitud", "accuracy", (v) => fix(v, 3), r), row("Exactitud balanceada", "balanced_accuracy", (v) => fix(v, 3), r),
              row("F1 macro", "macro_f1", (v) => fix(v, 3), r), row("Área bajo la curva ROC", "roc_auc", (v) => fix(v, 3), r),
              row("Exactitud en entrenamiento", "train_accuracy", (v) => fix(v, 3), r),
              row("Brecha entrenamiento − prueba", "gap", (v) => fix(v, 3), r, "Cuánto más acierta con lo que ya vio que con lo nuevo: es el sobreajuste"),
              row("Tiempo de entrenamiento", "fit_time", secs, r), row("Tiempo de predicción", "predict_time", secs, r)))),
          h("p", { class: "tl-muted tl-small", style: { marginTop: "8px" } }, "Los tiempos no son comparables: el árbol propio está escrito en Python puro y el bosque usa código compilado.")),
        h("div", {}, h("h3", { class: "tl-h3" }, "Exactitud en cada partición"),
          pairedPlot({ a: dt.per_split, b: rf.per_split, nameA: "Árbol propio", nameB: "Random Forest", colorA: "var(--dt)", colorB: "var(--rf)" }),
          h("p", { class: "tl-muted tl-small" }, "Cada trazo une las dos exactitudes de una misma partición. Dispersión alta del árbol y baja del bosque es lo que se espera de un bosque."))),
      h("div", { class: "tl-two tl-section" },
        ["DT", "RF"].map((m) => h("div", {}, h("h3", { class: `tl-h3 ${m === "DT" ? "tl-dtc" : "tl-rfc"}` }, `Matriz de confusión: ${m === "DT" ? "árbol propio" : "Random Forest"}`),
          h("div", { class: "tl-table-wrap" }, confusionTable(r.models[m].confusion, r.classes)),
          h("p", { class: "tl-muted tl-small" }, "Suma de todas las particiones de prueba.")))),
      h("section", { class: "tl-section" }, h("h3", { class: "tl-h3" }, "En qué se diferencian"),
        h("ul", { class: "tl-rules" },
          h("li", {}, "El árbol propio corta con la razón de ganancia de C4.5 y puede abrir una rama por cada valor de un atributo nominal; los árboles del bosque son binarios (tipo CART)."),
          h("li", {}, "Ante un valor faltante, el árbol propio reparte el caso entre las ramas; el bosque necesita imputar antes con la mediana o la moda del entrenamiento."),
          h("li", {}, "Un árbol solo es fácil de leer y de explicar; el bosque promedia muchos árboles y baja la varianza, pero deja de ser legible."))),
      r.decision ? h("details", { class: "tl-section" }, h("summary", { class: "tl-h3", style: { cursor: "pointer" } }, `${r.decision.id} · ${r.decision.title}`),
        h("p", {}, r.decision.chosen), h("p", { class: "tl-muted" }, r.decision.rationale)) : null);
  }

  const el = h("div", {},
    h("h2", { class: "tl-h2" }, "Comparar el árbol propio con el Random Forest"),
    h("p", { class: "tl-muted", style: { maxWidth: "70ch" } }, "Los dos modelos se entrenan y se prueban sobre las mismas particiones. Así la diferencia sale de los modelos y no de la suerte del reparto."),
    h("div", { class: "tl-controls tl-card", style: { marginTop: "14px" } }, folds.el, repeats.el, trees.el, h("div", {}, depth.el, noLimit.el),
      h("div", {}, h("p", { class: "tl-muted tl-small", style: { marginBottom: "8px" }, "data-role": "dt-summary" }, dtSummary()), go)),
    result);
  return { el, onShow() { const p = el.querySelector("[data-role=dt-summary]"); if (p) p.textContent = dtSummary(); } };
}

/* ============================ vista: Laboratorio ============================ */
const SWEEPS = {
  dt: { max_depth: ["Profundidad máxima", "1, 2, 3, 4, 5, 6, 8, sin límite"], min_samples_leaf: ["Mínimo de casos por hoja", "1, 2, 3, 5, 8, 12, 20"],
    min_samples_split: ["Mínimo para dividir", "2, 4, 8, 16, 32"], min_gain: ["Ganancia mínima", "0, 0.01, 0.02, 0.05, 0.1, 0.2"],
    confidence: ["Confianza de la poda", "0.01, 0.05, 0.1, 0.25, 0.4, 0.5"] },
  rf: { n_estimators: ["Cantidad de árboles", "1, 3, 10, 30, 100, 300"], max_depth: ["Profundidad", "1, 2, 3, 5, 8, 12, sin límite"],
    min_samples_leaf: ["Mínimo de casos por hoja", "1, 2, 5, 10, 20"], min_samples_split: ["Mínimo para dividir", "2, 4, 8, 16"] },
};
function parseValues(text) {
  const parts = text.split(",").map((t) => t.trim().replace(",", ".")).filter(Boolean);
  if (!parts.length) throw new Error("Escribí al menos un valor.");
  return parts.map((t) => {
    if (/^(none|null|sin l[ií]mite|infinito|∞)$/i.test(t)) return null;
    const value = Number(t);
    if (Number.isNaN(value)) throw new Error(`No entendí el valor «${t}». Usá números separados por comas.`);
    return value;
  });
}
const valueLabel = (v) => (v === null ? "sin límite" : String(v));

function buildLab(ctx) {
  const { S, rpc, run } = ctx;
  const V = { kind: "dt", param: "max_depth", metric: "macro_f1", seeds: 2 };
  const kindSeg = segmented([["dt", "Árbol propio"], ["rf", "Random Forest"]], V.kind, (k) => { V.kind = k; V.param = Object.keys(SWEEPS[k])[0]; fillParams(); });
  const paramHost = h("div", {});
  const values = h("input", { class: "tl-input", type: "text", "aria-label": "Valores a probar" });
  const metricSel = selectBox(Object.entries(METRIC_LABELS), V.metric, (v) => { V.metric = v; }, { "aria-label": "Métrica" });
  const sweepOut = h("div", { style: { marginTop: "18px" } }, h("p", { class: "tl-empty" }, "Los gráficos aparecen acá."));
  let paramSel;
  function fillParams() {
    paramSel = selectBox(Object.entries(SWEEPS[V.kind]).map(([k, [text]]) => [k, text]), V.param, (v) => { V.param = v; values.value = SWEEPS[V.kind][v][1]; }, { "aria-label": "Parámetro" });
    clear(paramHost).append(paramSel);
    values.value = SWEEPS[V.kind][V.param][1];
  }
  fillParams();

  const runSweep = button("Probar valores", () => run("Probando valores (puede tardar)", async () => {
    runSweep.disabled = true;
    try {
      const list = parseValues(values.value);
      const r = await rpc("sweep", { kind: V.kind, param: V.param, values: list, base_params: V.kind === "dt" ? S.params : {}, metric: V.metric, n_splits: 5, n_repeats: 2 });
      drawSweep(r);
    } finally { runSweep.disabled = false; }
  }).catch(() => {}), "tl-btn-primary");

  function drawSweep(r) {
    const labels = r.rows.map((row) => valueLabel(row.value)), color = r.kind === "dt" ? "var(--dt)" : "var(--rf)", name = METRIC_LABELS[r.metric];
    const best = r.rows.findIndex((row) => row.value === r.best);
    clear(sweepOut).append(
      h("div", { class: "tl-two" },
        h("div", {}, lineChart({ labels, markIndex: best, ariaLabel: `${name} según ${SWEEPS[r.kind][r.param][0]}`, series: [
          { name: `${name} en prueba`, color, points: r.rows.map((row) => ({ y: row.test_mean, lo: row.test_mean - row.test_std, hi: row.test_mean + row.test_std })) },
          { name: `${name} en entrenamiento`, color: "var(--muted)", dash: "6 5", points: r.rows.map((row) => ({ y: row.train_mean })) }] }),
          h("div", { class: "tl-legend" }, h("span", {}, h("i", { style: { background: color } }), `${name} en prueba (banda: ±1 desvío entre particiones)`),
            h("span", {}, h("i", { style: { background: "var(--muted)" } }), "en entrenamiento"))),
        h("div", {}, h("p", { class: "tl-note" }, `Mejor valor en prueba: ${valueLabel(r.best)}. Ojo con las bandas: si dos valores se solapan, la diferencia no es clara y conviene elegir el modelo más simple.`),
          h("p", { class: "tl-muted", style: { marginTop: "8px" } }, "Cuando la línea de entrenamiento se separa por arriba de la de prueba, el modelo está memorizando: esa distancia es el sobreajuste."))),
      h("div", { class: "tl-table-wrap", style: { marginTop: "12px" } }, h("table", { class: "tl-table" },
        h("thead", {}, h("tr", {}, h("th", {}, SWEEPS[r.kind][r.param][0]), h("th", {}, `${name} en prueba`), h("th", {}, "Desvío"), h("th", {}, "En entrenamiento"), h("th", {}, "Brecha"), h("th", {}, "Tiempo"))),
        h("tbody", {}, r.rows.map((row) => h("tr", {}, h("td", {}, valueLabel(row.value)), h("td", {}, fix(row.test_mean)), h("td", {}, fix(row.test_std)),
          h("td", {}, fix(row.train_mean)), h("td", {}, fix(row.gap)), h("td", {}, secs(row.fit_time))))))),
      h("p", { class: "tl-muted tl-small", style: { marginTop: "6px" } }, `${r.n_partitions} particiones de prueba por valor. Los demás parámetros del árbol son los de la pestaña Árbol${r.param === "confidence" ? " (con poda activada)" : ""}.`));
  }

  /* ---- robustez a faltantes ---- */
  const fractions = h("input", { class: "tl-input", type: "text", value: "0, 0.1, 0.2, 0.3, 0.4", "aria-label": "Fracciones de datos a borrar" });
  const seeds = slider({ label: "Repeticiones del borrado", min: 1, max: 5, step: 1, value: V.seeds, format: (v) => num(v, 0), onInput: (v) => { V.seeds = v; },
    help: "Cada una borra otras celdas al azar; se promedian." });
  const robustOut = h("div", { style: { marginTop: "18px" } });
  const runRobust = button("Medir robustez", () => run("Borrando datos y midiendo (puede tardar)", async () => {
    runRobust.disabled = true;
    try {
      const list = parseValues(fractions.value);
      if (list.some((v) => v === null || v < 0 || v >= 1)) throw new Error("Las fracciones deben estar entre 0 y 0,9.");
      const r = await rpc("robustness", { fractions: list, seeds: Array.from({ length: V.seeds }, (_, i) => i), n_splits: 5, n_repeats: 1, dt_params: S.params, rf_params: {}, metric: V.metric });
      drawRobust(r);
    } finally { runRobust.disabled = false; }
  }).catch(() => {}), "tl-btn-primary");

  function drawRobust(r) {
    const name = METRIC_LABELS[r.metric], first = r.rows[0], last = r.rows[r.rows.length - 1];
    const labels = r.rows.map((row) => pct(row.fraction, 0));
    clear(robustOut).append(
      h("div", { class: "tl-two" },
        h("div", {}, lineChart({ labels, xs: r.rows.map((row) => row.fraction), ariaLabel: `${name} según la fracción de datos borrados`, series: [
          { name: "Árbol propio", color: "var(--dt)", points: r.rows.map((row) => ({ y: row.DT, lo: row.DT - row.DT_std, hi: row.DT + row.DT_std })) },
          { name: "Random Forest", color: "var(--rf)", points: r.rows.map((row) => ({ y: row.RF, lo: row.RF - row.RF_std, hi: row.RF + row.RF_std })) }] }),
          h("div", { class: "tl-legend" }, h("span", {}, h("i", { style: { background: "var(--dt)" } }), "Árbol propio"), h("span", {}, h("i", { style: { background: "var(--rf)" } }), "Random Forest"))),
        h("div", {}, h("p", { class: "tl-note" }, `Con ${pct(last.fraction, 0)} de los datos borrados, el árbol pasa de ${fix(first.DT, 3)} a ${fix(last.DT, 3)} y el bosque de ${fix(first.RF, 3)} a ${fix(last.RF, 3)} (${name}).`),
          h("p", { class: "tl-muted", style: { marginTop: "8px" } }, "El árbol propio reparte un caso con datos faltantes entre las ramas, como hace C4.5. El bosque rellena antes con la mediana o la moda calculada en entrenamiento. Se borran valores al azar, tanto en entrenamiento como en prueba."))),
      h("p", { class: "tl-muted tl-small", style: { marginTop: "6px" } }, `${r.n_partitions} particiones por fracción, ${r.seeds.length} ${r.seeds.length === 1 ? "borrado" : "borrados"} distintos. Las bandas son ±1 desvío entre particiones.`));
  }

  return { el: h("div", {},
    h("h2", { class: "tl-h2" }, "Laboratorio de parámetros"),
    h("p", { class: "tl-muted", style: { maxWidth: "70ch" } }, "Mové un parámetro por vez y mirá cuánto cambia el resultado. Cada punto es el promedio de varias particiones de prueba."),
    h("section", { class: "tl-section" }, h("h3", { class: "tl-h3" }, "Un parámetro a la vez"),
      h("div", { class: "tl-controls tl-card" }, field("Modelo", kindSeg.el), field("Parámetro", paramHost), field("Valores (separados por comas, decimales con punto)", values), field("Métrica", metricSel), runSweep),
      sweepOut),
    h("section", { class: "tl-section" }, h("h3", { class: "tl-h3" }, "Qué pasa cuando faltan datos"),
      h("div", { class: "tl-controls tl-card" }, field("Fracciones de valores a borrar", fractions, "Entre 0 y 0,9."), seeds.el, runRobust), robustOut)) };
}

/* ============================ armazón de la aplicación ============================ */
const TABS = [["datos", "Datos"], ["arbol", "Árbol"], ["comparar", "Comparar árbol y bosque"], ["lab", "Laboratorio"]];
const BUILDERS = { datos: buildDatos, arbol: buildArbol, comparar: buildComparar, lab: buildLab };

function initialTheme() {
  try { const saved = localStorage.getItem("tl-theme"); if (saved) return saved; } catch (e) { /* sin almacenamiento */ }
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function createApp(root, rpc) {
  const S = { datasets: [], info: null, tab: "datos", views: {}, params: { ...DEFAULT_PARAMS }, busy: 0 };
  const status = h("span", { class: "tl-muted tl-small", role: "status", "aria-live": "polite" });
  const errorBox = h("div", { role: "alert", hidden: true });
  const datasetSelect = h("select", { class: "tl-select", "aria-label": "Dataset" });
  const tabBar = h("div", { class: "tl-tabs", role: "tablist" });
  const panel = h("div", { class: "tl-panel", role: "tabpanel" });

  const themeButton = button("", () => {
    const next = root.dataset.theme === "dark" ? "light" : "dark";
    root.dataset.theme = next;
    try { localStorage.setItem("tl-theme", next); } catch (e) { /* sin almacenamiento */ }
    themeButton.textContent = next === "dark" ? "Modo claro" : "Modo oscuro";
  }, "tl-btn-quiet");
  themeButton.textContent = root.dataset.theme === "dark" ? "Modo claro" : "Modo oscuro";

  root.append(
    h("header", { class: "tl-head" },
      h("div", {}, h("h1", { class: "tl-title" }, "Árbol de decisión C4.5 contra Random Forest"),
        h("p", { class: "tl-sub" }, "El árbol es implementación propia, sin librerías de árboles. El bosque es el de scikit-learn. Todo se calcula en Python y se dibuja acá.")),
      h("div", { class: "tl-head-tools" }, status, datasetSelect, themeButton)),
    tabBar, errorBox, panel);

  function setStatus(text, spinning) { clear(status); if (spinning) status.append(h("span", { class: "tl-spin" })); if (text) status.append(text); }
  function showError(message) {
    errorBox.hidden = !message;
    clear(errorBox);
    if (message) { errorBox.className = "tl-error"; errorBox.append(h("b", {}, "No se pudo completar: "), message); }
  }
  async function run(label, action) {
    S.busy += 1;
    setStatus(label, true);
    showError("");
    try { return await action(); }
    catch (error) { showError(error.message); throw error; }
    finally { S.busy -= 1; if (!S.busy) setStatus(""); }
  }

  const ctx = {
    S, rpc, run, showError,
    goto: (tab) => showTab(tab),
    selectDataset: (key) => run("Cargando datos", async () => {
      S.info = await rpc("select", { key });
      S.views = {};
      S.params = { ...DEFAULT_PARAMS };
      datasetSelect.value = key;
      showTab(S.tab);
    }).catch(() => {}),
    loadCsv: (text, target, name) => run("Leyendo el archivo", async () => {
      S.info = await rpc("load_csv", { text, target, name });
      S.datasets = (await rpc("datasets")).datasets;
      S.views = {};
      refreshDatasetSelect();
      showTab("datos");
    }).catch(() => {}),
  };

  function refreshDatasetSelect() {
    clear(datasetSelect).append(...S.datasets.map((d) => h("option", { value: d.key, selected: S.info && d.key === S.info.key }, d.title)));
  }
  datasetSelect.addEventListener("change", () => ctx.selectDataset(datasetSelect.value));

  function showTab(tab) {
    S.tab = tab;
    for (const b of tabBar.children) b.setAttribute("aria-selected", String(b.dataset.tab === tab));
    if (!S.info) { clear(panel).append(h("div", { class: "tl-empty" }, "Elegí un dataset para empezar.")); return; }
    if (!S.views[tab]) S.views[tab] = BUILDERS[tab](ctx);
    clear(panel).append(S.views[tab].el);
    if (S.views[tab].onShow) S.views[tab].onShow();
  }

  for (const [key, text] of TABS) {
    tabBar.append(h("button", { type: "button", class: "tl-tab", role: "tab", "data-tab": key, "aria-selected": "false", onclick: () => showTab(key) }, text));
  }

  return {
    async start() {
      await run("Iniciando", async () => {
        S.datasets = (await rpc("datasets")).datasets;
        refreshDatasetSelect();
        const first = S.datasets.find((d) => d.key === "iris") || S.datasets[0];
        S.info = await rpc("select", { key: first.key });
        datasetSelect.value = first.key;
        showTab("datos");
      }).catch(() => {});
    },
  };
}

/* ============================ punto de entrada de anywidget ============================ */
function render({ model, el }) {
  const rpc = createRpc(model);
  const root = h("div", { class: "tl", "data-theme": initialTheme() });
  el.append(root);
  createApp(root, rpc).start();
  return () => { rpc.dispose(); root.remove(); };
}

export default { render };
