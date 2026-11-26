// Phase 8b: "Métricas del modelo" tab — KPIs, confusion matrix and per-class table from metrics.json.
"use strict";

(() => {
  const $ = (id) => document.getElementById(id);
  const pct = (v, digits = 1) => `${(v * 100).toFixed(digits).replace(".", ",")} %`;
  const dec = (v, digits = 3) => v.toFixed(digits).replace(".", ",");
  const int = (v) => v.toLocaleString("es-CO");

  const LEVELS = {
    high: { color: "Verde", label: "confianza alta", tone: "high" },
    medium: { color: "Amarillo", label: "confianza media", tone: "medium" },
    normal: { color: "Amarillo", label: "sin signos claros", tone: "normal" },
    low: { color: "Rojo", label: "confianza baja", tone: "low" },
  };

  let loaded = false;

  // ---------- tabs ----------
  const tabs = [$("tab-analyze"), $("tab-metrics")];
  function selectTab(tab) {
    for (const t of tabs) {
      const active = t === tab;
      t.setAttribute("aria-selected", String(active));
      t.tabIndex = active ? 0 : -1;
      $(t.getAttribute("aria-controls")).hidden = !active;
    }
    if (tab.id === "tab-metrics" && !loaded) load();
  }
  tabs.forEach((tab, i) => {
    tab.addEventListener("click", () => selectTab(tab));
    tab.addEventListener("keydown", (event) => {  // arrow keys move between tabs (ARIA pattern)
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
      const next = tabs[(i + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
      next.focus();
      selectTab(next);
    });
  });

  // ---------- data ----------
  async function load() {
    loaded = true;
    try {
      const response = await fetch("metrics.json");
      if (!response.ok) throw new Error(response.status);
      render(await response.json());
    } catch {
      loaded = false;
      $("metrics-loading").textContent = "No se pudieron cargar las métricas (falta web/metrics.json).";
    }
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function render(m) {
    const k = m.kpis;
    $("metrics-intro").textContent =
      `${m.architecture} (${m.image_size} px) · medido una sola vez en ${int(m.n_images)} imágenes de test ` +
      `que el modelo nunca vio · decisiones con la política de tamizaje, igual que la app.`;

    const tiles = [
      ["Exactitud", pct(k.accuracy), "de las imágenes de test, clasificadas en la enfermedad correcta"],
      ["F1 macro", dec(k.f1_macro), "promedio del F1 de las 7 clases, todas con el mismo peso"],
      ["Sensibilidad", pct(k.sensitivity), `de los enfermos, detectados como enfermos (${int(k.missed_sick)} se escapan)`],
      ["Especificidad", pct(k.specificity), `de los sanos, reconocidos como sanos (${int(k.false_alarms)} falsas alarmas)`],
      ["Aciertos en verde", pct(k.green_accuracy), `cuando el semáforo está en verde (${pct(k.green_share, 0)} de las imágenes)`],
      ["Error de calibración", dec(k.ece), "ECE: distancia entre la confianza mostrada y el acierto real (0 = perfecto)"],
      ["Parámetros", `${dec(k.parameters / 1e6, 1)} M`, `pesos aprendidos · archivo de ${dec(k.file_mb, 0)} MB`],
      ["Tiempo por imagen", `~${int(k.cpu_ms_per_image)} ms`, "en procesador (CPU), sin GPU"],
    ];
    $("kpis").replaceChildren(...tiles.map(([label, value, help]) => {
      const tile = el("div", "kpi");
      tile.append(el("p", "kpi-label", label), el("p", "kpi-value", value), el("p", "kpi-help", help));
      return tile;
    }));

    renderConfusion(m);
    renderClassTable(m);
    renderLevelTable(m);

    const d = m.data;
    const items = [
      `Datos: ${int(d.train)} imágenes para entrenar, ${int(d.val)} para ajustar y decidir, ${int(d.test)} para esta medición final (divididas por paciente).`,
      "El error más frecuente es retinopatía diabética leve respondida como Normal: por eso la app exige más evidencia antes de decir \"Normal\", y \"Normal\" nunca sale en verde.",
      "Estos números valen para fotos parecidas a las del dataset. Con otras cámaras u hospitales el rendimiento baja: en una validación externa, la retinopatía diabética y el glaucoma se detectaron mucho peor.",
      `Modelo: ${m.model}, mejor época ${k.best_epoch} · temperatura de calibración ${dec(m.policy.temperature, 2)} · cortes del semáforo: verde ≥ ${pct(m.policy.green, 0)}, rojo < ${pct(m.policy.red, 0)} · actualizado ${m.generated}.`,
    ];
    $("limits").replaceChildren(...items.map((text) => el("li", "", text)));

    $("metrics-loading").hidden = true;
    $("metrics").hidden = false;
  }

  function renderConfusion(m) {
    const table = $("cm");
    const codes = m.classes.map((c) => c.code);
    const head = el("thead");
    const headRow = el("tr");
    const corner = el("th", "cm-corner");
    corner.append(el("span", "long-name", "Real ↓ / App →"));
    headRow.append(corner);
    for (const c of m.classes) {
      const th = el("th", "cm-col", c.code);
      th.scope = "col";
      th.title = c.name;
      headRow.append(th);
    }
    head.append(headRow);

    const body = el("tbody");
    m.confusion.forEach((row, i) => {
      const total = row.reduce((a, b) => a + b, 0);
      const tr = el("tr");
      const th = el("th", "cm-row");
      th.scope = "row";
      th.append(el("span", "", codes[i]), el("span", "long-name", ` · ${m.classes[i].name}`));
      tr.append(th);
      row.forEach((count, j) => {
        const share = total ? count / total : 0;
        const td = el("td", i === j ? "cm-cell diag" : "cm-cell");
        td.style.setProperty("--share", String(share));
        td.classList.toggle("strong", share >= 0.5);
        td.classList.toggle("zero", count === 0);
        td.append(el("span", "cm-count", int(count)));
        if (count) td.append(el("span", "cm-share", share < 0.005 ? "<1 %" : pct(share, 0)));
        td.tabIndex = 0;
        td.dataset.tip = i === j
          ? `${m.classes[i].name}: ${int(count)} de ${int(total)} acertadas (${pct(share)})`
          : `${int(count)} de ${int(total)} casos reales de ${m.classes[i].name} (${pct(share)}) respondidos como ${m.classes[j].name}`;
        td.setAttribute("aria-label", td.dataset.tip);
        tr.append(td);
      });
      body.append(tr);
    });
    table.replaceChildren(head, body);
    attachTooltip(table);
  }

  function attachTooltip(table) {
    const tip = $("cm-tooltip");
    const show = (cell) => {
      tip.textContent = cell.dataset.tip;
      tip.hidden = false;
      const r = cell.getBoundingClientRect();
      const t = tip.getBoundingClientRect();
      const left = Math.min(Math.max(8, r.left + r.width / 2 - t.width / 2), window.innerWidth - t.width - 8);
      const top = r.top - t.height - 8 < 8 ? r.bottom + 8 : r.top - t.height - 8;
      tip.style.left = `${left + window.scrollX}px`;
      tip.style.top = `${top + window.scrollY}px`;
    };
    const hide = () => { tip.hidden = true; };
    table.addEventListener("mouseover", (e) => { const c = e.target.closest(".cm-cell"); if (c) show(c); });
    table.addEventListener("mouseleave", hide);
    table.addEventListener("focusin", (e) => { const c = e.target.closest(".cm-cell"); if (c) show(c); });
    table.addEventListener("focusout", hide);
  }

  function renderClassTable(m) {
    const rows = m.classes.map((c) => {
      const tr = el("tr");
      const th = el("th", "");
      th.scope = "row";
      th.title = c.name;
      th.append(el("span", "", c.code), el("span", "long-name", ` · ${c.name}`));
      tr.append(th, el("td", "num", pct(c.precision, 0)), el("td", "num", pct(c.recall, 0)), el("td", "num", dec(c.f1, 2)),
                el("td", "num", int(c.support)));
      return tr;
    });
    $("class-table").tBodies[0].replaceChildren(...rows);
  }

  function renderLevelTable(m) {
    const rows = Object.entries(LEVELS).map(([key, info]) => {
      const level = m.levels[key];
      const tr = el("tr");
      const th = el("th", "");
      th.scope = "row";
      th.append(el("span", `dot ${info.tone}`), el("span", "long-name", `${info.color} · `),
                document.createTextNode(info.label));
      tr.append(th, el("td", "num", pct(level.share, 0)),
                el("td", "num", level.accuracy === null ? "—" : pct(level.accuracy, 0)));
      return tr;
    });
    $("level-table").tBodies[0].replaceChildren(...rows);
  }
})();
