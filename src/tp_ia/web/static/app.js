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
 * (Datos, Árbol, Comparar, Experimentos) -> armazón de la aplicación.
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
const plural = (n, one, many) => (Number(n) === 1 ? one : many);

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
/* Barras de un atributo nominal: una por valor, partida en colores según cuántos casos de cada clase lo tienen. */
function valueBars(feature) {
  const entries = Object.entries(feature.counts).sort((a, b) => b[1] - a[1]).slice(0, 6);
  const top = Math.max(...entries.map((e) => e[1]), 1);
  return h("div", { class: "tl-bars" }, entries.map(([value, count]) => {
    const split = feature.by_class && feature.by_class[value];
    const fill = split
      ? h("div", { class: "tl-fill tl-fill-split", style: { width: `${(count / top) * 100}%` } }, split.map((n, i) =>
        n > 0 ? h("div", { style: { flex: `${n} 1 0`, background: classColor(i) }, title: `${n} ${plural(n, "caso", "casos")}` }) : null))
      : h("div", { class: "tl-fill", style: { width: `${(count / top) * 100}%` } });
    return h("div", { class: "tl-bar" }, h("span", { title: value }, shorten(value, 22)), h("div", { class: "tl-track" }, fill),
      h("span", {}, num(count, 0)));
  }));
}

/* ============================ vista: Datos ============================ */
function buildDatos(ctx) {
  const { S } = ctx;
  const info = S.info;
  const classes = info.classes;
  const missingCells = sum(Object.values(info.quality.missing).map((m) => m.count));
  const root = h("div", {});

  root.append(h("section", {},
    h("h2", { class: "tl-h2" }, info.title),
    h("p", { class: "tl-muted" }, info.description, " ", h("span", { class: "tl-small" }, `Fuente: ${info.source}`)),
    info.disclaimer ? h("p", { class: "tl-warn", style: { marginTop: "8px" } }, info.disclaimer) : null,
    h("div", { class: "tl-facts", style: { marginTop: "14px" } },
      [[num(info.n_rows, 0), plural(info.n_rows, "caso", "casos")], [num(info.n_features, 0), plural(info.n_features, "atributo", "atributos")],
        [num(classes.length, 0), plural(classes.length, "clase", "clases")],
        [num(missingCells, 0), plural(missingCells, "valor faltante", "valores faltantes")],
        [num(info.quality.duplicates, 0), plural(info.quality.duplicates, "fila duplicada", "filas duplicadas")]]
        .map(([value, label]) => h("div", { class: "tl-fact" }, h("b", {}, value), h("span", {}, label)))),
    h("div", { class: "tl-classbar", role: "img", "aria-label": "Distribución de clases" }, classes.map((c, i) =>
      h("div", { style: { flex: `${c.count} 1 0`, background: classColor(i) }, title: `${c.name}: ${c.count}` }))),
    legend(classes)));

  if (info.quality.warnings.length) {
    root.append(h("div", { class: "tl-section", style: { display: "grid", gap: "8px" } },
      info.quality.warnings.map((w) => h("p", { class: "tl-warn" }, w))));
  }

  root.append(h("section", { class: "tl-section" }, h("h3", { class: "tl-h3" }, `Atributos (${info.features.length})`,
      h("span", { class: "tl-muted tl-small", style: { fontWeight: 400 } }, " · cada color representa una clase")),
    h("div", { class: "tl-features" }, info.features.map((f) => h("div", { class: "tl-feature" },
      h("header", {}, h("b", {}, f.name),
        h("span", { style: { display: "flex", gap: "6px" } },
          f.missing ? h("span", { class: "tl-tag tl-tag-warn" }, `${f.missing} ${plural(f.missing, "faltante", "faltantes")}`) : null,
          f.kind === "numeric" && f.outliers ? h("span", { class: "tl-tag tl-tag-warn", title: "Son valores que se alejan mucho del resto (más de 1,5 veces el rango intercuartil). Se conservan porque pueden ser mediciones reales." }, `${f.outliers} ${plural(f.outliers, "valor extremo", "valores extremos")}`) : null,
          h("span", { class: "tl-tag" }, f.kind === "numeric" ? "numérico" : "nominal"))),
      f.description ? h("p", { class: "tl-feature-desc" }, f.description) : null,
      f.kind === "numeric"
        ? [miniHistogram(f, classes), h("p", { class: "tl-muted tl-small" }, `Valores entre ${num(f.min)} y ${num(f.max)} · mediana ${num(f.median)}`)]
        : valueBars(f))))));

  root.append(h("section", { class: "tl-section tl-decisions" },
    h("h3", { class: "tl-h3" }, "Decisiones tomadas sobre los datos"),
    h("p", { class: "tl-muted tl-small", style: { marginBottom: "6px" } }, "Cada decisión explica qué se hizo con los datos y por qué. Están pensadas para poder citarlas en el informe."),
    info.decisions.map((d, i) => h("details", {},
      h("summary", {}, `${decisionLabel(i + 1)}: `, h("span", { class: "tl-muted" }, d.chosen), " ", h("span", { class: "tl-tag" }, d.id)),
      decisionBody(d)))));

  const keys = Object.keys(info.preview[0] || {});
  root.append(h("section", { class: "tl-section" }, h("details", {}, h("summary", { class: "tl-h3", style: { cursor: "pointer" } }, "Primeras filas"),
    h("div", { class: "tl-table-wrap" }, h("table", { class: "tl-table" },
      h("thead", {}, h("tr", {}, keys.map((k) => h("th", {}, k)))),
      h("tbody", {}, info.preview.map((row) => h("tr", {}, keys.map((k) => h("td", {}, isNil(row[k]) ? "—" : typeof row[k] === "number" ? num(row[k]) : row[k]))))))))));

  root.append(buildUpload(ctx));
  return { el: root };
}

/* Decisiones sobre los datos: el mismo formato que DecisionLog.to_markdown() en Python. */
const ORDINALS = ["Primera", "Segunda", "Tercera", "Cuarta", "Quinta", "Sexta", "Séptima", "Octava", "Novena", "Décima", "Undécima", "Duodécima"];
const decisionLabel = (n) => (n >= 1 && n <= ORDINALS.length ? `${ORDINALS[n - 1]} decisión` : `Decisión n.º ${n}`);
const joinOr = (xs) => (xs.length === 1 ? String(xs[0]) : `${xs.slice(0, -1).join(", ")} o ${xs[xs.length - 1]}`);
function friendly(value) {
  if (Array.isArray(value)) return value.length ? value.join(", ") : "ninguno";
  if (value && typeof value === "object") {
    const parts = Object.entries(value).map(([k, v]) => `${k}: ${friendly(v)}`);
    return parts.length ? parts.join(parts.some((p) => p.includes(",")) ? "; " : ", ") : "ninguno";
  }
  return String(value);
}
function decisionBody(d) {
  const evidence = Object.entries(d.evidence || {});
  return h("div", {},
    h("p", { class: "tl-small" }, `Tema: ${d.title}`),
    d.rationale ? h("p", {}, h("b", {}, "Por qué: "), d.rationale) : null,
    d.alternatives.length ? h("p", {}, h("b", {}, "Otras opciones posibles: "), `${joinOr(d.alternatives)}.`) : null,
    evidence.length ? [h("p", { style: { marginBottom: "2px" } }, h("b", {}, "En qué se basa:")),
      h("ul", { class: "tl-evidence" }, evidence.map(([k, v]) => h("li", {}, `${k[0].toUpperCase()}${k.slice(1)}: ${friendly(v)}`)))] : null);
}

function buildUpload(ctx) {
  const box = h("section", { class: "tl-section" }, h("h3", { class: "tl-h3" }, "Probar con otros datos"),
    h("p", { class: "tl-muted tl-small" }, "Podés cargar un archivo CSV propio. Tiene que incluir una columna con la clase que se quiere predecir. El separador, el formato de los decimales y los valores faltantes se detectan automáticamente."));
  const form = h("div", { class: "tl-row", style: { marginTop: "8px" } });
  const picker = h("input", { type: "file", accept: ".csv,.tsv,.txt", class: "tl-file-input", "aria-label": "Archivo CSV" });
  const pickerButton = h("label", { class: "tl-btn" }, picker, "Elegir un archivo CSV…");
  picker.addEventListener("change", async () => {
    const file = picker.files[0];
    if (!file) return;
    const text = await file.text();
    const first = text.split(/\r?\n/)[0] || "";
    const sep = [";", "\t", "|", ","].map((c) => [c, first.split(c).length]).sort((a, b) => b[1] - a[1])[0][0];
    const columns = first.split(sep).map((c) => c.trim().replace(/^"|"$/g, "")).filter(Boolean);
    const target = selectBox(columns.map((c) => [c, c]), columns[columns.length - 1], () => {}, { "aria-label": "Columna de la clase" });
    clear(form).append(h("span", {}, `¿Qué columna de «${file.name}» contiene la clase a predecir?`), target,
      button("Usar este archivo", () => ctx.loadCsv(text, target.value, file.name.replace(/\.\w+$/, "")), "tl-btn-primary"));
  });
  form.append(pickerButton);
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
    h("thead", {}, h("tr", {}, h("th", {}, "real \\ predicho"), names.map((n) => h("th", {}, shorten(n, 16))))),
    h("tbody", {}, matrix.map((row, i) => h("tr", {}, h("th", {}, shorten(names[i], 16)),
      row.map((count, j) => h("td", { style: { background: `color-mix(in srgb, ${i === j ? "var(--ok)" : "var(--warn)"} ${Math.round(clamp(count / (rowTotals[i] || 1), 0, 1) * 70)}%, var(--surface-2))` } },
        num(count, 0)))))));
}

function importanceBars(items, limit = 8) {
  if (!items.length) return h("p", { class: "tl-muted" }, "El árbol no usa ningún atributo, porque quedó formado por una sola hoja que predice siempre la misma clase.");
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
  const set = (key, value) => { P[key] = value; refreshAdvanced(); retrain(); };

  // Profundidad: el extremo derecho del slider significa "sin límite" (max_depth = null).
  const DEPTH_UNLIMITED = 13;
  const depthSlider = slider({ label: "Profundidad máxima", min: 1, max: DEPTH_UNLIMITED, step: 1, value: P.max_depth === null ? DEPTH_UNLIMITED : P.max_depth,
    format: (v) => (v >= DEPTH_UNLIMITED ? "sin límite" : num(v, 0)),
    help: "Indica cuántas preguntas seguidas puede hacer el árbol, desde la raíz hasta una hoja. En el extremo derecho, la profundidad no tiene límite.",
    onInput: (v) => set("max_depth", v >= DEPTH_UNLIMITED ? null : v) });

  // Poda: la confianza solo se muestra cuando la poda está activada.
  const confidence = slider({ label: "Confianza de la poda", min: 0.01, max: 0.5, step: 0.01, value: P.confidence,
    help: "Cuanto más bajo es este valor, más ramas se podan y más simple queda el árbol. C4.5 usa 0,25 por defecto.", onInput: (v) => set("confidence", v) });
  confidence.el.hidden = !P.prune;
  const pruneBox = checkbox("Podar por error pesimista", P.prune, (on) => { confidence.el.hidden = !on; set("prune", on); },
    "Reemplaza por una hoja las ramas que no reducen el error estimado. Así se obtiene un árbol más simple, que suele funcionar mejor con casos nuevos.");

  const testSlider = slider({ label: "Casos de prueba", min: 0, max: 50, step: 5, value: Math.round(info.split.test_size * 100),
    format: (v) => `${num(v, 0)} %`, help: "Este porcentaje de los datos se usará para probar el árbol. Si se indica 0 %, se usarán todos los datos para el entrenamiento.",
    onChange: (v) => run("Repartiendo los casos", async () => { S.info.split = await rpc("set_split", { test_size: v / 100, seed: 0 }); await train(true); }).catch(() => {}) });
  const criterion = segmented([["gain_ratio", "Razón de ganancia"], ["gain", "Ganancia"]], P.criterion, (k) => set("criterion", k));
  const threshold = segmented([["c45", "Valor observado"], ["midpoint", "Punto medio"]], P.threshold_mode, (k) => set("threshold_mode", k));

  // Opciones avanzadas: plegadas, con un contador de cuántas difieren de los valores por defecto.
  const ADVANCED = ["min_gain", "use_avg_gain", "threshold_mode", "numeric_gain_correction"];
  const advancedCount = h("span", { class: "tl-tag" });
  function refreshAdvanced() {
    const changed = ADVANCED.filter((k) => P[k] !== DEFAULT_PARAMS[k]).length;
    advancedCount.textContent = `${changed} ${plural(changed, "cambiada", "cambiadas")}`;
    advancedCount.hidden = changed === 0;
  }
  const advanced = h("details", { class: "tl-advanced" },
    h("summary", {}, "Opciones avanzadas de C4.5", advancedCount),
    h("div", { class: "tl-advanced-body" },
      slider({ label: "Ganancia mínima", min: 0, max: 0.5, step: 0.01, value: P.min_gain, help: "Si una división mejora la ganancia menos que este valor, no se realiza. Sirve para evitar divisiones que aportan muy poco.", onInput: (v) => set("min_gain", v) }).el,
      checkbox("Heurística de ganancia promedio", P.use_avg_gain, (on) => set("use_avg_gain", on), "Solo se consideran los atributos cuya ganancia de información es igual o mayor que el promedio. C4.5 lo hace para no elegir atributos con una razón de ganancia alta pero una ganancia muy baja.").el,
      hasNumeric ? field("Umbral numérico", threshold.el, "Define dónde se corta un atributo numérico: en un valor que aparece en los datos, como hace C4.5, o en el punto medio entre dos valores consecutivos.") : null,
      hasNumeric ? checkbox("Corrección de la versión 8", P.numeric_gain_correction, (on) => set("numeric_gain_correction", on), "Penaliza las divisiones sobre atributos numéricos, como propone la versión 8 de C4.5. Compensa que estos atributos ofrecen muchos puntos de corte posibles.").el : null));
  refreshAdvanced();

  const rail = h("aside", { class: "tl-rail", "aria-label": "Parámetros del árbol" },
    h("div", { class: "tl-rail-head" }, h("h2", { class: "tl-h2" }, "Parámetros"),
      button("Restablecer", () => { Object.assign(P, DEFAULT_PARAMS); ctx.S.views.arbol = null; ctx.goto("arbol"); }, "tl-btn-link")),
    testSlider.el,
    field("Criterio de corte", criterion.el, "El algoritmo C4.5 utiliza la razón de ganancia en lugar de la ganancia de información, ya que esta última tiende a favorecer atributos con muchos valores distintos, lo que genera divisiones con muchas ramas."),
    depthSlider.el,
    slider({ label: "Mínimo de casos por hoja", min: 1, max: 20, step: 1, value: P.min_samples_leaf, help: "Para que el árbol haga una división, al menos dos de las ramas resultantes deben recibir esta cantidad de casos. Equivale al parámetro -m de C4.5.", onInput: (v) => set("min_samples_leaf", v) }).el,
    h("div", { class: "tl-rail-group" }, pruneBox.el, confidence.el),
    advanced);

  /* ---- zona principal ---- */
  const stats = h("div", { class: "tl-stats" });
  const treeHost = h("div", {});
  const tools = h("div", { class: "tl-treetools" },
    button("−", () => zoomBy(0.8)), button("+", () => zoomBy(1.25)), button("Ajustar", () => { V.zoom = null; drawNow(); }));
  tools.querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-label", ["Alejar", "Acercar", "Ajustar al ancho"][i]));
  const treeWrap = h("div", { class: "tl-treewrap" }, tools, treeHost);
  const inspector = h("div", { class: "tl-inspector" });
  const caseHost = h("div", { class: "tl-predict" });
  const clearCase = button("Quitar el caso", () => { V.prediction = null; V.touched = false; drawPrediction(); drawNow(); }, "tl-btn-link");
  const importanceHost = h("div", {});
  const metricsHost = h("div", {});
  const sourceHost = h("div", {});
  const lower = h("div", { class: "tl-lower" },
    h("section", { class: "tl-card" }, h("h3", { class: "tl-h3" }, "Nodo"),
      h("p", { class: "tl-muted tl-small", style: { marginBottom: "8px" } }, "Pasá el mouse sobre un nodo del árbol para ver su detalle. Si hacés clic, el detalle queda fijo."),
      inspector),
    h("section", { class: "tl-card tl-card-accent" },
      h("div", { class: "tl-card-head" }, h("h3", { class: "tl-h3" }, "Predecir un caso"), clearCase),
      h("p", { class: "tl-muted tl-small", style: { marginBottom: "10px" } }, "Mové los valores para describir un caso. Los dos modelos lo clasifican al instante y el árbol resalta el camino que sigue."),
      caseHost));
  const lower2 = h("div", { class: "tl-lower tl-lower-even" },
    h("section", { class: "tl-card" }, h("h3", { class: "tl-h3" }, "Qué atributos usa el árbol"), importanceHost),
    h("section", { class: "tl-card" }, metricsHost));
  const main = h("div", {}, stats, treeWrap, lower, lower2, sourceHost);
  const el = h("div", { class: "tl-arbol" }, rail, main);

  function zoomBy(factor) { V.zoom = clamp((V.zoom || 1) * factor, 0.3, 2.5); drawNow(); }

  function drawInspector() {
    const item = V.item || V.layout.root;
    const d = item.data;
    clear(inspector);
    inspector.append(
      item.path.length ? h("p", { class: "tl-muted tl-small", style: { marginBottom: "6px" } }, item.path.join("  →  ")) : h("p", { class: "tl-muted tl-small", style: { marginBottom: "6px" } }, "Este es el nodo raíz, que contiene todos los casos de entrenamiento."),
      h("dl", {}, ...[
        d.leaf ? ["Predice la clase", d.label] : ["Divide según", d.feature],
        ["Casos", num(d.n, 2)], ["Entropía", fix(d.entropy)],
        d.leaf ? null : ["Ganancia de información", fix(d.gain)], d.leaf ? null : ["Razón de ganancia", fix(d.gain_ratio)],
        ["Errores si fuera una hoja", num(d.errors, 2)],
      ].filter(Boolean).flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)])),
      h("div", { class: "tl-bars", style: { marginTop: "10px" } }, classes.map((c, i) => h("div", { class: "tl-bar" },
        h("span", {}, h("i", { class: "tl-dot", style: { background: classColor(i) } }), shorten(c.name, 16)),
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
    clear(importanceHost).append(importanceBars(r.importances),
      h("p", { class: "tl-muted tl-small", style: { marginTop: "8px" } }, "Indica cuánto contribuye cada atributo a las divisiones del árbol. Se calcula sumando la ganancia de cada división que usa el atributo, ponderada por la cantidad de casos que pasan por ella."));
    clear(metricsHost).append(
      h("h3", { class: "tl-h3" }, test ? "Aciertos y errores en los casos de prueba" : "Aciertos y errores en entrenamiento"),
      h("div", { class: "tl-table-wrap" }, confusionTable(matrix, names)),
      test ? h("p", { class: "tl-muted tl-small", style: { marginTop: "8px" } }, `Con los datos de entrenamiento, la exactitud es de ${pct(r.metrics.train.accuracy)}. Si es mucho mayor que con los datos de prueba, el árbol está memorizando los casos en lugar de aprender reglas generales.`) : h("p", { class: "tl-muted tl-small", style: { marginTop: "8px" } }, "No hay casos de prueba, así que la exactitud se mide con los mismos datos de entrenamiento y resulta demasiado optimista."));
    clear(sourceHost).append(
      h("details", { class: "tl-source" }, h("summary", {}, "Ver el árbol como texto y como Graphviz"),
        h("pre", { class: "tl-code", style: { marginTop: "8px" } }, r.text),
        h("p", { class: "tl-muted tl-small", style: { margin: "10px 0 4px" } }, "Código DOT del árbol. Se puede dibujar con el paquete graphviz de Python o pegándolo en un visor de Graphviz en línea."),
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
          h("label", { class: "tl-check tl-small", title: "Probar qué pasa si falta este dato" }, missing, "falta")));
    });
    return h("div", { class: "tl-case" }, rows);
  }
  const caseForm = buildCaseForm();
  const caseResult = h("div", { class: "tl-predict-result" });
  caseHost.append(caseForm, caseResult);

  function verdict(name, cls, model, names) {
    return h("div", {}, h("div", { class: "tl-verdict" }, h("span", { class: cls }, h("i", { class: "tl-dot", style: { background: "currentColor" } }), name),
      h("b", {}, model.label)), stackedBar(model.proba, { height: 12, gap: 2 }),
      h("p", { class: "tl-muted tl-small", style: { marginTop: "3px" } }, names.map((n, i) => `${shorten(n, 14)} ${pct(model.proba[i], 0)}`).join(" · ")));
  }
  function drawPrediction() {
    clear(caseResult);
    const p = V.prediction;
    clearCase.hidden = !p;
    if (!p) { caseResult.append(h("p", { class: "tl-empty tl-small" }, "Mové cualquier valor de la izquierda para ver qué clase predice cada modelo.")); return; }
    caseResult.append(h("div", {}, verdict("Árbol propio", "tl-dtc", p.dt, p.classes), verdict("Random Forest", "tl-rfc", p.rf, p.classes),
      p.dt.rules.length ? h("ol", { class: "tl-rules" }, p.dt.rules.map((r) => h("li", {}, r))) : null,
      p.missing.length ? h("p", { class: "tl-muted tl-small", style: { marginTop: "6px" } }, `Falta el dato de ${p.missing.join(", ")}. Como no puede elegir una sola rama, el árbol reparte el caso entre todas, en proporción a los casos de entrenamiento de cada una. Los números ×0,xx del dibujo muestran ese reparto.`) : null,
      info.disclaimer ? h("p", { class: "tl-warn tl-small", style: { marginTop: "8px" } }, info.disclaimer) : null));
  }
  async function predict() {
    const row = {};
    for (const f of info.features) row[f.name] = V.missing[f.name] ? null : V.values[f.name];
    const mine = ++V.seq;
    await run("Prediciendo", async () => {
      const prediction = await rpc("predict", { row });
      if (mine !== V.seq || !V.touched) return;   // llegó tarde o se quitó el caso mientras tanto
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
  const m = { l: 50, r: 32, t: 14, b: 40 };   // margen derecho amplio para que la última etiqueta no se corte
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
    help: "Los datos se dividen en esta cantidad de partes. Cada parte se usa una vez para probar los modelos, y el resto se usa para entrenarlos." });
  const repeats = slider({ label: "Repeticiones", min: 1, max: 10, step: 1, value: V.repeats, format: (v) => num(v, 0), onInput: (v) => { V.repeats = v; },
    help: "Cantidad de veces que se repite la división al azar. Con más repeticiones, los resultados son más estables, aunque el cálculo tarda más." });
  const trees = slider({ label: "Cantidad de árboles", min: 10, max: 300, step: 10, value: V.trees, format: (v) => num(v, 0), onInput: (v) => { V.trees = v; } });
  const depth = slider({ label: "Profundidad máxima", min: 1, max: 20, step: 1, value: V.depth, format: (v) => num(v, 0), onInput: (v) => { V.depth = v; },
    help: "La cátedra propone usar 100 árboles con una profundidad máxima de 10." });
  const noLimit = checkbox("Sin límite de profundidad", false, (on) => { V.unlimited = on; depth.setDisabled(on); });
  const result = h("div", { style: { marginTop: "22px" } }, h("p", { class: "tl-empty" }, "Elegí los valores y tocá «Comparar» para entrenar y evaluar los dos modelos."));

  function dtSummary() {
    const p = S.params;
    return `${p.prune ? `Con poda (confianza ${num(p.confidence, 2)})` : "Sin poda"}, profundidad ${p.max_depth === null ? "sin límite" : p.max_depth}, mínimo ${num(p.min_samples_leaf, 0)} casos por hoja.`;
  }
  const go = button("Comparar", () => run("Comparando los modelos; puede tardar unos segundos", async () => {
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
    const lead = Math.abs(d.mean_diff) < 0.0005 ? "En exactitud, el árbol y el bosque obtienen el mismo resultado promedio."
      : `En exactitud, el ${d.mean_diff > 0 ? "bosque supera al árbol" : "árbol supera al bosque"} por ${fix(Math.abs(d.mean_diff) * 100, 1)} puntos en promedio.`;
    clear(result).append(
      h("div", { class: "tl-note" },
        h("p", {}, h("b", {}, lead), ` De las ${d.n} particiones, el bosque gana en ${d.wins_a}, el árbol gana en ${d.wins_b} y empatan en ${d.ties}.`),
        h("p", { style: { marginTop: "4px" } }, `Según la prueba de Wilcoxon (p = ${fix(d.p_wilcoxon, 3)}), ${d.p_wilcoxon < 0.05 ? "la diferencia se mantiene de forma consistente entre las particiones" : "la diferencia no es lo bastante consistente, por lo que podría deberse al azar"}. Este resultado es solo orientativo, porque las particiones comparten datos entre sí.`)),
      h("div", { class: "tl-two tl-section" },
        h("div", {}, h("h3", { class: "tl-h3" }, `Resultados promedio en ${r.n_partitions} particiones (media ± desvío estándar)`),
          h("div", { class: "tl-table-wrap" }, h("table", { class: "tl-table" },
            h("thead", {}, h("tr", {}, h("th", {}, ""), h("th", { class: "tl-dtc" }, "Árbol propio"), h("th", { class: "tl-rfc" }, "Random Forest"))),
            h("tbody", {}, row("Exactitud", "accuracy", (v) => fix(v, 3), r), row("Exactitud balanceada", "balanced_accuracy", (v) => fix(v, 3), r),
              row("F1 macro", "macro_f1", (v) => fix(v, 3), r), row("Área bajo la curva ROC", "roc_auc", (v) => fix(v, 3), r),
              row("Exactitud en entrenamiento", "train_accuracy", (v) => fix(v, 3), r),
              row("Brecha entrenamiento − prueba", "gap", (v) => fix(v, 3), r, "Diferencia entre la exactitud con los datos de entrenamiento y con los de prueba. Cuanto mayor es, más sobreajuste hay."),
              row("Tiempo de entrenamiento", "fit_time", secs, r), row("Tiempo de predicción", "predict_time", secs, r)))),
          h("p", { class: "tl-muted tl-small", style: { marginTop: "8px" } }, "Los tiempos no se pueden comparar de forma directa, porque el árbol propio está escrito en Python puro y el bosque de scikit-learn usa código compilado, que es mucho más rápido.")),
        h("div", {}, h("h3", { class: "tl-h3" }, "Exactitud en cada partición"),
          pairedPlot({ a: dt.per_split, b: rf.per_split, nameA: "Árbol propio", nameB: "Random Forest", colorA: "var(--dt)", colorB: "var(--rf)" }),
          h("p", { class: "tl-muted tl-small" }, "Cada línea une la exactitud del árbol y la del bosque en una misma partición. Es esperable que los resultados del árbol varíen más que los del bosque, porque el bosque promedia muchos árboles."))),
      h("div", { class: "tl-two tl-section" },
        ["DT", "RF"].map((m) => h("div", {}, h("h3", { class: `tl-h3 ${m === "DT" ? "tl-dtc" : "tl-rfc"}` }, `Matriz de confusión: ${m === "DT" ? "árbol propio" : "Random Forest"}`),
          h("div", { class: "tl-table-wrap" }, confusionTable(r.models[m].confusion, r.classes)),
          h("p", { class: "tl-muted tl-small" }, "Suma de los resultados de todas las particiones de prueba.")))),
      h("section", { class: "tl-section" }, h("h3", { class: "tl-h3" }, "En qué se diferencian"),
        h("ul", { class: "tl-rules" },
          h("li", {}, "El árbol propio elige las divisiones con la razón de ganancia de C4.5 y puede crear una rama por cada valor de un atributo nominal. En cambio, los árboles del bosque siempre dividen en dos ramas, como en el algoritmo CART."),
          h("li", {}, "Cuando falta un valor, el árbol propio reparte el caso entre las ramas. El bosque, en cambio, necesita completar antes el dato faltante con la mediana o el valor más frecuente del entrenamiento."),
          h("li", {}, "Un único árbol es fácil de leer y de explicar. El bosque combina muchos árboles, lo que hace sus resultados más estables, pero el modelo ya no se puede interpretar a simple vista."))),
      r.decision ? h("details", { class: "tl-section tl-decisions" },
        h("summary", {}, "Decisión sobre la evaluación: ", h("span", { class: "tl-muted" }, r.decision.chosen), " ", h("span", { class: "tl-tag" }, r.decision.id)),
        decisionBody(r.decision)) : null);
  }

  const el = h("div", {},
    h("h2", { class: "tl-h2" }, "Comparar el árbol propio con el Random Forest"),
    h("p", { class: "tl-muted", style: { maxWidth: "70ch" } }, "Los dos modelos se entrenan y se evalúan con exactamente las mismas particiones de los datos. De esta forma, las diferencias se deben a los modelos y no a la suerte en la división de los datos."),
    h("div", { class: "tl-card", style: { marginTop: "14px" } },
      h("div", { class: "tl-groups" },
        h("fieldset", { class: "tl-group" }, h("legend", {}, "Cómo se evalúa"), folds.el, repeats.el),
        h("fieldset", { class: "tl-group tl-group-rf" }, h("legend", {}, "Random Forest"), trees.el, depth.el, noLimit.el),
        h("fieldset", { class: "tl-group tl-group-dt" }, h("legend", {}, "Árbol propio"),
          h("p", { class: "tl-small", "data-role": "dt-summary" }, dtSummary()),
          h("p", { class: "tl-muted tl-small" }, "Se usan los mismos parámetros elegidos en la pestaña Árbol."),
          button("Cambiar en la pestaña Árbol", () => ctx.goto("arbol"), "tl-btn-link"))),
      h("div", { class: "tl-actions" }, go)),
    result);
  return { el, onShow() { const p = el.querySelector("[data-role=dt-summary]"); if (p) p.textContent = dtSummary(); } };
}

/* ============================ vista: Experimentos ============================ */
/* Cada parámetro tiene su nombre, los valores sugeridos (null = sin límite) y qué se espera ver al probarlos. */
const SWEEPS = {
  dt: {
    max_depth: { label: "Profundidad máxima", values: [1, 2, 3, 4, 5, 6, 8, null],
      expect: "Con poca profundidad, el árbol es demasiado simple y se equivoca tanto con los datos de entrenamiento como con los de prueba. Al aumentar la profundidad, el resultado mejora hasta cierto punto. Desde ahí, el árbol sigue acertando más en el entrenamiento, pero ya no mejora en la prueba: esa separación entre las dos líneas es el sobreajuste." },
    min_samples_leaf: { label: "Mínimo de casos por hoja", values: [1, 2, 3, 5, 8, 12, 20],
      expect: "Con un mínimo bajo, el árbol puede crear hojas con muy pocos casos y adaptarse demasiado a los datos de entrenamiento. Al subir el mínimo, el árbol se vuelve más simple. Si el mínimo es demasiado alto, el árbol pierde detalle y empieza a equivocarse más." },
    min_samples_split: { label: "Mínimo de casos para dividir", values: [2, 4, 8, 16, 32],
      expect: "Este parámetro indica cuántos casos necesita un nodo para poder dividirse. Su efecto se parece al del mínimo de casos por hoja: con valores altos, el árbol queda más simple." },
    min_gain: { label: "Ganancia mínima", values: [0, 0.01, 0.02, 0.05, 0.1, 0.2],
      expect: "Con una ganancia mínima de 0, el árbol hace cualquier división que mejore algo. Al subirla, solo se hacen las divisiones más útiles y el árbol queda más chico. Si es demasiado alta, el árbol puede quedar con muy pocas divisiones y perder exactitud." },
    confidence: { label: "Confianza de la poda", values: [0.01, 0.05, 0.1, 0.25, 0.4, 0.5],
      expect: "Con una confianza baja, la poda es más agresiva y el árbol queda más simple. Con una confianza alta, se poda menos. Suele haber un valor intermedio que logra un buen equilibrio entre simplicidad y exactitud." },
  },
  rf: {
    n_estimators: { label: "Cantidad de árboles", values: [1, 3, 10, 30, 100, 300],
      expect: "Con pocos árboles, el bosque se parece a un árbol solo y sus resultados varían mucho. Al agregar árboles, el resultado mejora y se estabiliza. A partir de cierta cantidad, agregar más árboles casi no cambia el resultado y solo hace más lento el cálculo." },
    max_depth: { label: "Profundidad máxima", values: [1, 2, 3, 5, 8, 12, null],
      expect: "Con árboles poco profundos, el bosque es demasiado simple. Al aumentar la profundidad, el resultado mejora. A diferencia de un árbol solo, el bosque suele sobreajustar menos, porque promedia muchos árboles distintos." },
    min_samples_leaf: { label: "Mínimo de casos por hoja", values: [1, 2, 5, 10, 20],
      expect: "Al subir el mínimo de casos por hoja, cada árbol del bosque queda más simple. Con valores muy altos, los árboles pierden detalle y el resultado empeora." },
    min_samples_split: { label: "Mínimo de casos para dividir", values: [2, 4, 8, 16],
      expect: "Este parámetro indica cuántos casos necesita un nodo para poder dividirse. Con valores altos, los árboles del bosque quedan más simples." },
  },
};
const METRIC_HELP = {
  macro_f1: "Promedia el F1 de cada clase, así que todas las clases pesan lo mismo, aunque algunas tengan pocos casos.",
  accuracy: "Es la proporción de casos que el modelo clasifica correctamente.",
  balanced_accuracy: "Promedia el porcentaje de aciertos de cada clase. Es útil cuando algunas clases tienen muchos más casos que otras.",
};
const FRACTIONS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];

function parseValue(text) {
  const t = text.trim().replace(",", ".");
  if (/^(none|null|sin l[ií]mite|infinito|∞)$/i.test(t)) return null;
  const value = Number(t);
  if (!t || Number.isNaN(value)) throw new Error(`No se pudo interpretar el valor «${text.trim()}». Escribí un número, por ejemplo 0.25.`);
  return value;
}
const valueLabel = (v) => (v === null ? "sin límite" : num(v, 3));
const byValue = (a, b) => (a === null ? 1 : b === null ? -1 : a - b);   // "sin límite" va al final

/* Grupo de valores que se activan o desactivan con un clic, con la opción de agregar uno propio. */
function chipPicker({ values, selected = values, format = valueLabel, allowCustom = true, label }) {
  let options = [...values];
  let chosen = new Set(selected);
  const list = h("div", { class: "tl-chips", role: "group", "aria-label": label });
  function render() {
    clear(list).append(...options.map((v) => {
      const chip = h("button", { type: "button", class: "tl-chip", "aria-pressed": String(chosen.has(v)) }, format(v));
      chip.addEventListener("click", () => { if (chosen.has(v)) chosen.delete(v); else chosen.add(v); render(); });
      return chip;
    }));
  }
  const custom = h("input", { class: "tl-input tl-chip-input", type: "text", placeholder: "Otro valor", "aria-label": `Agregar otro valor a ${label}` });
  const add = () => {
    if (!custom.value.trim()) return;
    const value = parseValue(custom.value);
    if (!options.includes(value)) options = [...options, value].sort(byValue);
    chosen.add(value);
    custom.value = "";
    render();
  };
  custom.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); try { add(); } catch (err) { custom.setCustomValidity(err.message); custom.reportValidity(); } } });
  custom.addEventListener("input", () => custom.setCustomValidity(""));
  render();
  return {
    el: h("div", {}, list, allowCustom ? h("div", { class: "tl-chip-add" }, custom, button("Agregar", () => { try { add(); } catch (err) { custom.setCustomValidity(err.message); custom.reportValidity(); } })) : null),
    get: () => options.filter((v) => chosen.has(v)),
    reset(newValues) { options = [...newValues]; chosen = new Set(newValues); render(); },
  };
}

/* Recuadro que se muestra antes de correr un experimento: cuenta qué se espera ver. */
function expectBox(text) {
  return h("div", { class: "tl-expect" }, h("p", { class: "tl-expect-title" }, "Qué se espera ver"), h("p", {}, text),
    h("p", { class: "tl-muted tl-small", style: { marginTop: "8px" } }, "Cuando el experimento termine, el gráfico y la tabla van a aparecer acá."));
}

function buildLab(ctx) {
  const { S, rpc, run } = ctx;
  const V = { kind: "dt", param: "max_depth", metric: "macro_f1", seeds: 2, sweepDone: false, robustDone: false };
  const sweep = () => SWEEPS[V.kind][V.param];

  /* ---- un parámetro a la vez ---- */
  const sweepOut = h("div", { class: "tl-lab-out" });
  const showExpect = () => { if (!V.sweepDone) clear(sweepOut).append(expectBox(sweep().expect)); };
  const chips = chipPicker({ values: sweep().values, label: "Valores a probar" });
  const paramHost = h("div", {});
  function fillParams() {
    const sel = selectBox(Object.entries(SWEEPS[V.kind]).map(([k, p]) => [k, p.label]), V.param,
      (v) => { V.param = v; chips.reset(sweep().values); V.sweepDone = false; showExpect(); }, { "aria-label": "Parámetro" });
    clear(paramHost).append(sel);
    chips.reset(sweep().values);
  }
  const kindSeg = segmented([["dt", "Árbol propio"], ["rf", "Random Forest"]], V.kind,
    (k) => { V.kind = k; V.param = Object.keys(SWEEPS[k])[0]; fillParams(); V.sweepDone = false; showExpect(); });
  fillParams();
  const metricHelp = h("p", { class: "tl-help" }, METRIC_HELP[V.metric]);
  const metricSel = selectBox(Object.entries(METRIC_LABELS), V.metric, (v) => { V.metric = v; metricHelp.textContent = METRIC_HELP[v]; }, { "aria-label": "Métrica" });

  const runSweep = button("Probar valores", () => run("Probando los valores; puede tardar unos segundos", async () => {
    runSweep.disabled = true;
    try {
      const list = chips.get();
      if (list.length < 2) throw new Error("Elegí al menos dos valores para poder compararlos.");
      const r = await rpc("sweep", { kind: V.kind, param: V.param, values: list, base_params: V.kind === "dt" ? S.params : {}, metric: V.metric, n_splits: 5, n_repeats: 2 });
      V.sweepDone = true;
      drawSweep(r);
    } finally { runSweep.disabled = false; }
  }).catch(() => {}), "tl-btn-primary");

  function drawSweep(r) {
    const labels = r.rows.map((row) => valueLabel(row.value)), color = r.kind === "dt" ? "var(--dt)" : "var(--rf)", name = METRIC_LABELS[r.metric];
    const best = r.rows.findIndex((row) => row.value === r.best), title = SWEEPS[r.kind][r.param].label;
    clear(sweepOut).append(
      h("div", { class: "tl-two" },
        h("div", {}, h("h3", { class: "tl-h3" }, `${name} según ${title.toLowerCase()}`),
          lineChart({ labels, markIndex: best, ariaLabel: `${name} según ${title}`, series: [
            { name: `${name} en prueba`, color, points: r.rows.map((row) => ({ y: row.test_mean, lo: row.test_mean - row.test_std, hi: row.test_mean + row.test_std })) },
            { name: `${name} en entrenamiento`, color: "var(--muted)", dash: "6 5", points: r.rows.map((row) => ({ y: row.train_mean })) }] }),
          h("div", { class: "tl-legend" }, h("span", {}, h("i", { style: { background: color } }), `${name} con datos de prueba (la banda muestra ±1 desvío estándar)`),
            h("span", {}, h("i", { style: { background: "var(--muted)" } }), "con datos de entrenamiento"))),
        h("div", {}, h("p", { class: "tl-note" }, `El mejor resultado con los datos de prueba se obtiene con el valor ${valueLabel(r.best)}, marcado con un círculo en el gráfico. Si las bandas de dos valores se superponen, la diferencia entre ellos no es clara; en ese caso, conviene elegir el modelo más simple.`),
          h("p", { style: { marginTop: "10px" } }, h("b", {}, "Qué se esperaba ver. "), SWEEPS[r.kind][r.param].expect),
          h("p", { class: "tl-muted", style: { marginTop: "8px" } }, "Cuando la línea de entrenamiento queda muy por encima de la de prueba, el modelo está memorizando los datos. La distancia entre las dos líneas muestra el sobreajuste."))),
      h("div", { class: "tl-table-wrap", style: { marginTop: "12px" } }, h("table", { class: "tl-table" },
        h("thead", {}, h("tr", {}, h("th", {}, title), h("th", {}, `${name} en prueba`), h("th", {}, "Desvío"), h("th", {}, "En entrenamiento"), h("th", {}, "Brecha"), h("th", {}, "Tiempo"))),
        h("tbody", {}, r.rows.map((row) => h("tr", {}, h("td", {}, valueLabel(row.value)), h("td", {}, fix(row.test_mean)), h("td", {}, fix(row.test_std)),
          h("td", {}, fix(row.train_mean)), h("td", {}, fix(row.gap)), h("td", {}, secs(row.fit_time))))))),
      h("p", { class: "tl-muted tl-small", style: { marginTop: "6px" } }, `Cada valor se evaluó en ${r.n_partitions} particiones. El resto de los parámetros del árbol son los elegidos en la pestaña Árbol${r.param === "confidence" ? ", con la poda activada" : ""}.`));
  }

  /* ---- robustez a faltantes ---- */
  const fractions = chipPicker({ values: FRACTIONS, selected: FRACTIONS.slice(0, 5), format: (v) => pct(v, 0), allowCustom: false, label: "Proporción de datos a borrar" });
  const seeds = slider({ label: "Repeticiones del borrado", min: 1, max: 5, step: 1, value: V.seeds, format: (v) => num(v, 0), onInput: (v) => { V.seeds = v; },
    help: "Cantidad de veces que se repite el borrado, eligiendo otros datos al azar cada vez. Los resultados se promedian." });
  const ROBUST_EXPECT = "Lo esperable es que los dos modelos empeoren a medida que se borran más datos. La diferencia está en cómo manejan los datos que faltan: el árbol propio reparte cada caso incompleto entre las ramas, como hace C4.5, mientras que el bosque completa los datos faltantes antes de clasificar.";
  const robustOut = h("div", { class: "tl-lab-out" }, expectBox(ROBUST_EXPECT));
  const runRobust = button("Medir robustez", () => run("Borrando datos y evaluando; puede tardar unos segundos", async () => {
    runRobust.disabled = true;
    try {
      const list = fractions.get();
      if (list.length < 2) throw new Error("Elegí al menos dos proporciones para poder compararlas.");
      const r = await rpc("robustness", { fractions: list, seeds: Array.from({ length: V.seeds }, (_, i) => i), n_splits: 5, n_repeats: 1, dt_params: S.params, rf_params: {}, metric: V.metric });
      drawRobust(r);
    } finally { runRobust.disabled = false; }
  }).catch(() => {}), "tl-btn-primary");

  function drawRobust(r) {
    const name = METRIC_LABELS[r.metric], first = r.rows[0], last = r.rows[r.rows.length - 1];
    const labels = r.rows.map((row) => pct(row.fraction, 0));
    clear(robustOut).append(
      h("div", { class: "tl-two" },
        h("div", {}, h("h3", { class: "tl-h3" }, `${name} según la proporción de datos borrados`),
          lineChart({ labels, xs: r.rows.map((row) => row.fraction), ariaLabel: `${name} según la proporción de datos borrados`, series: [
            { name: "Árbol propio", color: "var(--dt)", points: r.rows.map((row) => ({ y: row.DT, lo: row.DT - row.DT_std, hi: row.DT + row.DT_std })) },
            { name: "Random Forest", color: "var(--rf)", points: r.rows.map((row) => ({ y: row.RF, lo: row.RF - row.RF_std, hi: row.RF + row.RF_std })) }] }),
          h("div", { class: "tl-legend" }, h("span", {}, h("i", { style: { background: "var(--dt)" } }), "Árbol propio"), h("span", {}, h("i", { style: { background: "var(--rf)" } }), "Random Forest"))),
        h("div", {}, h("p", { class: "tl-note" }, `Al pasar de borrar el ${pct(first.fraction, 0)} a borrar el ${pct(last.fraction, 0)} de los datos, el ${name} del árbol pasa de ${fix(first.DT, 3)} a ${fix(last.DT, 3)}, y el del bosque, de ${fix(first.RF, 3)} a ${fix(last.RF, 3)}.`),
          h("p", { class: "tl-muted", style: { marginTop: "8px" } }, "Cuando falta un dato, el árbol propio reparte el caso entre las ramas, como hace C4.5. El bosque, en cambio, completa antes el dato con la mediana o el valor más frecuente del entrenamiento. Los valores se borran al azar, tanto en los datos de entrenamiento como en los de prueba."))),
      h("p", { class: "tl-muted tl-small", style: { marginTop: "6px" } }, `Cada proporción se evaluó en ${r.n_partitions} particiones, con ${r.seeds.length} ${r.seeds.length === 1 ? "borrado distinto" : "borrados distintos"}. Las bandas muestran ±1 desvío estándar.`));
  }

  showExpect();
  return { el: h("div", {},
    h("h2", { class: "tl-h2" }, "Experimentos"),
    h("p", { class: "tl-muted", style: { maxWidth: "75ch" } }, "En esta pestaña se puede estudiar cómo influye cada parámetro en el resultado y qué tan bien tolera cada modelo los datos faltantes. Cada punto de los gráficos es el promedio de varias particiones de prueba."),

    h("section", { class: "tl-section" },
      h("h3", { class: "tl-h3" }, "¿Cómo influye un parámetro?"),
      h("p", { class: "tl-muted tl-small", style: { marginBottom: "10px" } }, "Se entrena el modelo con cada uno de los valores elegidos y se compara el resultado con los datos de entrenamiento y con los de prueba."),
      h("div", { class: "tl-card" },
        h("div", { class: "tl-groups" },
          h("fieldset", { class: "tl-group" }, h("legend", {}, "Qué se prueba"), field("Modelo", kindSeg.el), field("Parámetro", paramHost)),
          h("fieldset", { class: "tl-group" }, h("legend", {}, "Valores a probar"), chips.el,
            h("p", { class: "tl-help" }, "Tocá un valor para incluirlo o quitarlo. También podés agregar otro valor; usá punto para los decimales.")),
          h("fieldset", { class: "tl-group" }, h("legend", {}, "Cómo se mide"), field("Métrica", metricSel), metricHelp)),
        h("div", { class: "tl-actions tl-actions-split" },
          h("p", { class: "tl-muted tl-small" }, "Cada valor se evalúa con validación cruzada de 5 pliegues, repetida 2 veces."), runSweep)),
      sweepOut),

    h("section", { class: "tl-section" },
      h("h3", { class: "tl-h3" }, "¿Qué pasa cuando faltan datos?"),
      h("p", { class: "tl-muted tl-small", style: { marginBottom: "10px" } }, "Se borra al azar una parte de los datos y se mide cómo cambia el resultado de cada modelo. Así se puede ver cuál de los dos tolera mejor los datos faltantes."),
      h("div", { class: "tl-card" },
        h("div", { class: "tl-groups" },
          h("fieldset", { class: "tl-group" }, h("legend", {}, "Datos a borrar"), fractions.el,
            h("p", { class: "tl-help" }, "Cada porcentaje indica qué parte de los valores se borra. Tocá un porcentaje para incluirlo o quitarlo.")),
          h("fieldset", { class: "tl-group" }, h("legend", {}, "Repeticiones"), seeds.el)),
        h("div", { class: "tl-actions tl-actions-split" },
          h("p", { class: "tl-muted tl-small" }, "Se usa la métrica elegida en el experimento anterior y los parámetros del árbol de la pestaña Árbol."), runRobust)),
      robustOut)) };
}

/* ============================ armazón de la aplicación ============================ */
const TABS = [["datos", "Datos"], ["arbol", "Árbol"], ["comparar", "Comparar"], ["lab", "Experimentos"]];
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
      h("h1", { class: "tl-title" }, "C4.5 ", h("span", { class: "tl-muted" }, "vs."), " Random Forest"),
      h("div", { class: "tl-head-tools" }, status,
        h("label", { class: "tl-dataset-pick" }, h("span", {}, "Dataset"), datasetSelect), themeButton)),
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

  TABS.forEach(([key, text], i) => {
    tabBar.append(h("button", { type: "button", class: "tl-tab", role: "tab", "data-tab": key, "aria-selected": "false", onclick: () => showTab(key) },
      h("span", { class: "tl-step", "aria-hidden": "true" }, String(i + 1)), text));
  });

  return {
    async start() {
      await run("Iniciando", async () => {
        S.datasets = (await rpc("datasets")).datasets;
        refreshDatasetSelect();
        const first = S.datasets[0];   // se abre con el primero de la lista (default_entries en service.py)
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
