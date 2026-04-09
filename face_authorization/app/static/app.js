(function () {
  "use strict";

  const THRESHOLD_KEY = "faceauth_threshold_v1";
  const $ = (sel, r = document) => r.querySelector(sel);

  const camVideo = $("#camVideo");
  const enrollVideo = $("#enrollVideo");
  const camShell = camVideo?.closest(".camera-shell");
  const enrollShell = enrollVideo?.closest(".camera-shell");
  const camPlaceholder = $("#camPlaceholder");
  const enrollCamPlaceholder = $("#enrollCamPlaceholder");

  let sharedStream = null;
  let authTab = "identify";
  const enrollBlobs = [];

  /** @param {string} path @param {RequestInit} [options] */
  async function api(path, options = {}) {
    const init = { ...options };
    if (init.headers == null) init.headers = {};
    const res = await fetch(path, init);
    const text = await res.text();
    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = text;
      }
    }
    if (!res.ok) {
      let msg =
        typeof data === "object" && data !== null && data.detail
          ? Array.isArray(data.detail)
            ? data.detail.map((d) => d.msg || d).join("; ")
            : String(data.detail)
          : res.statusText;
      if (res.status === 500 && msg === "Internal Server Error") {
        msg = "Błąd serwera — sprawdź logi kontenera.";
      }
      throw new Error(msg || `HTTP ${res.status}`);
    }
    return data;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /** Liczba klatek — forma do komunikatu (np. „1 klatka”, „3 klatki”, „5 klatek”). */
  function phraseKlatek(n) {
    const k = Number(n);
    if (!Number.isFinite(k) || k < 0) return String(n) + " klatek";
    if (k === 1) return "1 klatka";
    const mod10 = k % 10;
    const mod100 = k % 100;
    const word =
      mod10 >= 2 && mod10 <= 4 && (mod100 < 10 || mod100 >= 20) ? "klatki" : "klatek";
    return `${k} ${word}`;
  }

  function getThreshold() {
    const v = parseFloat($("#globalThreshold")?.value || "0.45");
    return Number.isFinite(v) ? v : 0.45;
  }

  function saveThreshold() {
    try {
      localStorage.setItem(THRESHOLD_KEY, String(getThreshold()));
    } catch (_) {}
  }

  function loadThreshold() {
    try {
      const t = localStorage.getItem(THRESHOLD_KEY);
      if (t != null && $("#globalThreshold")) {
        $("#globalThreshold").value = t;
        syncSliderFromInput();
      }
    } catch (_) {}
  }

  function syncSliderFromInput() {
    const inp = $("#globalThreshold");
    const sl = $("#expThresholdSlider");
    const out = $("#expThresholdOut");
    if (!inp || !sl || !out) return;
    const x = Math.round(parseFloat(inp.value) * 100);
    sl.value = String(Number.isFinite(x) ? Math.min(100, Math.max(0, x)) : 45);
    out.textContent = parseFloat(inp.value).toFixed(2);
  }

  function syncInputFromSlider() {
    const inp = $("#globalThreshold");
    const sl = $("#expThresholdSlider");
    const out = $("#expThresholdOut");
    if (!inp || !sl || !out) return;
    const v = parseInt(sl.value, 10) / 100;
    inp.value = v.toFixed(2);
    out.textContent = v.toFixed(2);
    saveThreshold();
  }

  async function startCamera() {
    if (sharedStream) {
      camShell?.classList.add("is-live");
      enrollShell?.classList.add("is-live");
      $("#btnScan")?.removeAttribute("disabled");
      $("#btnCaptureShot")?.removeAttribute("disabled");
      return;
    }
    sharedStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    camVideo.srcObject = sharedStream;
    enrollVideo.srcObject = sharedStream;
    camShell?.classList.add("is-live");
    enrollShell?.classList.add("is-live");
    $("#btnCamToggle").textContent = "Wyłącz kamerę";
    $("#btnEnrollCamToggle").textContent = "Wyłącz kamerę";
    $("#btnScan")?.removeAttribute("disabled");
    $("#btnCaptureShot")?.removeAttribute("disabled");
  }

  function stopCamera() {
    if (sharedStream) {
      sharedStream.getTracks().forEach((t) => t.stop());
      sharedStream = null;
    }
    camVideo.srcObject = null;
    enrollVideo.srcObject = null;
    camShell?.classList.remove("is-live");
    enrollShell?.classList.remove("is-live");
    $("#btnCamToggle").textContent = "Włącz kamerę";
    $("#btnEnrollCamToggle").textContent = "Włącz kamerę";
    $("#btnScan")?.setAttribute("disabled", "disabled");
    $("#btnCaptureShot")?.setAttribute("disabled", "disabled");
  }

  function captureBlobFromVideo(videoEl, quality = 0.92) {
    return new Promise((resolve, reject) => {
      const c = document.createElement("canvas");
      c.width = videoEl.videoWidth;
      c.height = videoEl.videoHeight;
      if (!c.width || !c.height) {
        reject(new Error("Brak obrazu z kamery — poczekaj na podgląd."));
        return;
      }
      c.getContext("2d").drawImage(videoEl, 0, 0);
      c.toBlob(
        (b) => (b ? resolve(b) : reject(new Error("Nie udało się zapisać klatki"))),
        "image/jpeg",
        quality
      );
    });
  }

  function setAuthSteps(lines) {
    const ul = $("#authSteps");
    if (!ul) return;
    ul.innerHTML = lines
      .map(
        (line, i) =>
          `<li class="${i === lines.length - 1 ? "active" : "done"}">${escapeHtml(line)}</li>`
      )
      .join("");
  }

  function setCamInlineMsg(text) {
    const el = $("#camInlineMsg");
    if (!el) return;
    el.textContent = text || "";
  }

  function syncAuthTabs() {
    document.querySelectorAll("[data-auth-tab]").forEach((b) => {
      const on = b.getAttribute("data-auth-tab") === authTab;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", on ? "true" : "false");
    });
    const panel = $("#auth-panel");
    if (panel) {
      panel.setAttribute("aria-labelledby", authTab === "verify" ? "tab-verify" : "tab-identify");
    }
  }

  function showView(targetId) {
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("view-active"));
    const el = document.getElementById(targetId);
    if (el) el.classList.add("view-active");
    document.querySelectorAll(".nav-item").forEach((b) => {
      const on = b.getAttribute("data-target") === targetId;
      b.classList.toggle("active", on);
      if (on) b.setAttribute("aria-current", "page");
      else b.removeAttribute("aria-current");
    });
    if (targetId === "view-admin-dash") loadDashboard();
    if (targetId === "view-admin-users") loadAdminUsers();
  }

  document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => showView(btn.getAttribute("data-target")));
  });

  document.querySelectorAll("[data-auth-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      authTab = btn.getAttribute("data-auth-tab");
      syncAuthTabs();
      $("#verifyUserRow")?.classList.toggle("hidden", authTab !== "verify");
      $("#authResult").hidden = true;
      setAuthSteps(["Gotowe do skanu"]);
    });
  });

  $("#btnCamToggle")?.addEventListener("click", async () => {
    try {
      if (sharedStream) {
        stopCamera();
        setCamInlineMsg("");
      } else {
        await startCamera();
        setCamInlineMsg("");
      }
    } catch (e) {
      setCamInlineMsg(
        "Nie udało się uruchomić kamery. Sprawdź uprawnienia przeglądarki i podłączenie urządzenia."
      );
    }
  });

  $("#btnEnrollCamToggle")?.addEventListener("click", async () => {
    try {
      if (sharedStream) {
        stopCamera();
        setCamInlineMsg("");
      } else {
        await startCamera();
        setCamInlineMsg("");
      }
    } catch (e) {
      setCamInlineMsg(
        "Nie udało się uruchomić kamery. Sprawdź uprawnienia przeglądarki i podłączenie urządzenia."
      );
    }
  });

  $("#globalThreshold")?.addEventListener("input", () => {
    syncSliderFromInput();
    saveThreshold();
  });

  $("#expThresholdSlider")?.addEventListener("input", syncInputFromSlider);

  $("#btnScan")?.addEventListener("click", async () => {
    const resultEl = $("#authResult");
    const scanBtn = $("#btnScan");
    resultEl.hidden = true;
    const t0 = performance.now();
    if (scanBtn) {
      scanBtn.setAttribute("aria-busy", "true");
      scanBtn.disabled = true;
    }
    try {
      const blob = await captureBlobFromVideo(camVideo);
      setAuthSteps(["Klatka zapisana", "Przetwarzanie na serwerze…", "Dopasowanie z bazą…"]);

      const th = getThreshold();
      if (authTab === "identify") {
        const fd = new FormData();
        fd.append("image", blob, "scan.jpg");
        const r = await api("/identify?top_k=3", { method: "POST", body: fd });
        const dt = ((performance.now() - t0) / 1000).toFixed(2);
        if (!r.results?.length) {
          setAuthSteps(["Brak dopasowania"]);
          resultEl.hidden = false;
          resultEl.className = "result-card-auth bad";
          resultEl.innerHTML = `<strong>Odrzucono</strong><p>Brak zapisanych użytkowników w bazie.</p><p class="hint-inline">Czas: ${dt} s</p>`;
          return;
        }
        const top = r.results[0];
        const ok = top.similarity >= th;
        setAuthSteps(["Klatka zapisana", "Przetwarzanie zakończone", "Dopasowanie zakończone"]);
        resultEl.hidden = false;
        resultEl.className = "result-card-auth " + (ok ? "ok" : "bad");
        resultEl.innerHTML = ok
          ? `<strong>Dostęp przyznany</strong><p>Identyfikator: <code>${escapeHtml(top.user_id)}</code></p><p>Podobieństwo: <strong>${top.similarity.toFixed(4)}</strong> (próg: ${th})</p><p class="hint-inline">Czas: ${dt} s</p>`
          : `<strong>Odrzucono</strong><p>Najbliższy identyfikator: <code>${escapeHtml(top.user_id)}</code> — ${top.similarity.toFixed(4)} poniżej progu (${th})</p><p class="hint-inline">Czas: ${dt} s</p>`;
      } else {
        const uid = $("#verifyUserSelect")?.value;
        if (!uid) {
          setAuthSteps(["Wybierz użytkownika z listy"]);
          return;
        }
        const fd = new FormData();
        fd.append("user_id", uid);
        fd.append("image", blob, "scan.jpg");
        const q = new URLSearchParams({ threshold: String(th) });
        const r = await api(`/verify?${q}`, { method: "POST", body: fd });
        const dt = ((performance.now() - t0) / 1000).toFixed(2);
        setAuthSteps(["Klatka zapisana", "Przetwarzanie zakończone", "Weryfikacja zakończona"]);
        resultEl.hidden = false;
        resultEl.className = "result-card-auth " + (r.accepted ? "ok" : "bad");
        resultEl.innerHTML = r.accepted
          ? `<strong>Dostęp przyznany</strong><p>${escapeHtml(r.user_id)} — podobieństwo ${r.similarity.toFixed(4)}</p><p class="hint-inline">Czas: ${dt} s</p>`
          : `<strong>Odrzucono</strong><p>${escapeHtml(r.user_id)} — podobieństwo ${r.similarity.toFixed(4)}</p><p class="hint-inline">Czas: ${dt} s</p>`;
      }
    } catch (e) {
      setAuthSteps(["Wystąpił błąd", String(e.message)]);
      resultEl.hidden = false;
      resultEl.className = "result-card-auth bad";
      resultEl.innerHTML = `<strong>Błąd</strong><p>${escapeHtml(e.message)}</p>`;
    } finally {
      if (scanBtn) {
        scanBtn.setAttribute("aria-busy", "false");
        scanBtn.disabled = !sharedStream;
      }
    }
  });

  function renderShotStrip() {
    const strip = $("#shotStrip");
    const cnt = $("#shotCount");
    const btn = $("#btnSubmitEnroll");
    if (!strip || !cnt || !btn) return;
    cnt.textContent = String(enrollBlobs.length);
    strip.querySelectorAll("img").forEach((img) => {
      if (img.src.startsWith("blob:")) URL.revokeObjectURL(img.src);
    });
    strip.innerHTML = "";
    enrollBlobs.forEach((blob) => {
      const url = URL.createObjectURL(blob);
      const wrap = document.createElement("div");
      wrap.className = "shot-thumb";
      wrap.setAttribute("role", "listitem");
      wrap.innerHTML = `<img alt="" src="${url}" /><button type="button" aria-label="Usuń klatkę z listy">×</button>`;
      wrap.querySelector("button").addEventListener("click", () => {
        const i = enrollBlobs.indexOf(blob);
        if (i >= 0) enrollBlobs.splice(i, 1);
        URL.revokeObjectURL(url);
        renderShotStrip();
      });
      strip.appendChild(wrap);
    });
    btn.disabled = enrollBlobs.length < 3;
  }

  $("#btnCaptureShot")?.addEventListener("click", async () => {
    const msg = $("#enrollMsg");
    msg.textContent = "";
    msg.className = "msg";
    try {
      if (enrollBlobs.length >= 12) {
        msg.textContent = "Możesz dodać co najwyżej 12 klatek.";
        msg.classList.add("error");
        return;
      }
      const blob = await captureBlobFromVideo(enrollVideo);
      enrollBlobs.push(blob);
      renderShotStrip();
    } catch (e) {
      msg.textContent = e.message;
      msg.classList.add("error");
    }
  });

  $("#btnClearShots")?.addEventListener("click", () => {
    enrollBlobs.length = 0;
    $("#shotStrip").innerHTML = "";
    renderShotStrip();
  });

  $("#btnSubmitEnroll")?.addEventListener("click", async () => {
    const uid = ($("#enrollUserId")?.value || "").trim();
    const msg = $("#enrollMsg");
    msg.textContent = "";
    msg.className = "msg";
    if (!uid) {
      msg.textContent = "Podaj identyfikator użytkownika.";
      msg.classList.add("error");
      return;
    }
    if (enrollBlobs.length < 3) {
      msg.textContent = "Potrzebne są co najmniej 3 klatki.";
      msg.classList.add("error");
      return;
    }
    msg.textContent = "Zapisywanie embeddingu…";
    try {
      const fd = new FormData();
      enrollBlobs.forEach((b) => fd.append("images", b, "shot.jpg"));
      await api(`/users/${encodeURIComponent(uid)}/enroll_multi`, { method: "POST", body: fd });
      msg.textContent =
        "Zapisano w bazie (" + phraseKlatek(enrollBlobs.length) + ", uśredniony wektor twarzy).";
      msg.classList.add("ok");
      enrollBlobs.length = 0;
      renderShotStrip();
      loadFooterAndUsers();
    } catch (e) {
      msg.textContent = e.message;
      msg.classList.add("error");
    }
  });

  async function loadFooterAndUsers() {
    try {
      const h = await api("/health");
      $("#sidebarMeta").textContent = `W bazie: ${h.enrolled_users} · ${h.device}`;
    } catch {
      $("#sidebarMeta").textContent = "";
    }
    const sel = $("#verifyUserSelect");
    if (!sel) return;
    try {
      const users = await api("/users");
      sel.innerHTML = "";
      const o0 = document.createElement("option");
      o0.value = "";
      o0.textContent = "Wybierz…";
      sel.appendChild(o0);
      for (const u of users) {
        const o = document.createElement("option");
        o.value = u.user_id;
        o.textContent = u.user_id;
        sel.appendChild(o);
      }
    } catch {
      sel.innerHTML = '<option value="">—</option>';
    }
  }

  async function loadDashboard() {
    const grid = $("#dashCards");
    const note = $("#dashNote");
    if (!grid) return;
    try {
      const s = await api("/admin/summary");
      grid.innerHTML = `
        <div class="dash-card"><div class="label">Użytkownicy</div><div class="value">${s.enrolled_users}</div></div>
        <div class="dash-card"><div class="label">Urządzenie</div><div class="value" style="font-size:0.95rem">${escapeHtml(s.device)}</div></div>
        <div class="dash-card"><div class="label">Wagi</div><div class="value" style="font-size:0.7rem;word-break:break-all">${escapeHtml(s.weights)}</div></div>
      `;
      if (note) note.textContent = s.note;
    } catch (e) {
      grid.innerHTML = `<p class="msg error">${escapeHtml(e.message)}</p>`;
    }
  }

  async function loadAdminUsers() {
    const tbody = $("#adminUsersBody");
    const msg = $("#adminUsersMsg");
    if (!tbody) return;
    msg.textContent = "";
    try {
      const users = await api("/users");
      tbody.innerHTML = "";
      if (!users.length) {
        tbody.innerHTML = '<tr><td colspan="4" style="color:var(--muted);padding:1rem">Brak rekordów</td></tr>';
        return;
      }
      for (const u of users) {
        const tr = document.createElement("tr");
        tr.innerHTML = `<td>${escapeHtml(u.user_id)}</td><td>${u.sample_count}</td><td>${escapeHtml((u.enrolled_at || "—").slice(0, 19))}</td><td></td>`;
        const td = tr.querySelector("td:last-child");
        const del = document.createElement("button");
        del.type = "button";
        del.className = "btn danger";
        del.textContent = "Usuń";
        del.addEventListener("click", async () => {
          if (!confirm('Usunąć użytkownika "' + u.user_id + '" z bazy?')) return;
          try {
            await api(`/users/${encodeURIComponent(u.user_id)}`, { method: "DELETE" });
            loadAdminUsers();
            loadFooterAndUsers();
          } catch (err) {
            msg.textContent = err.message;
            msg.className = "msg error";
          }
        });
        td.appendChild(del);
        tbody.appendChild(tr);
      }
    } catch (e) {
      tbody.innerHTML = "";
      msg.textContent = e.message;
      msg.className = "msg error";
    }
  }

  $("#btnAdminRefreshUsers")?.addEventListener("click", loadAdminUsers);

  $("#formCompare")?.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const form = ev.target;
    const msg = $("#compareMsg");
    msg.textContent = "";
    msg.className = "msg";
    const fa = form.image_a?.files?.[0];
    const fb = form.image_b?.files?.[0];
    if (!fa || !fb) {
      msg.textContent = "Wybierz oba obrazy.";
      msg.classList.add("error");
      return;
    }
    const fd = new FormData();
    fd.append("image_a", fa);
    fd.append("image_b", fb);
    const q = new URLSearchParams({ threshold: String(form.threshold.value) });
    try {
      const r = await api(`/compare?${q}`, { method: "POST", body: fd });
      msg.textContent = `Ta sama osoba (heurystycznie): ${r.same_person_guess ? "tak" : "nie"} — podobieństwo ${r.similarity.toFixed(4)}`;
      msg.classList.add("ok");
    } catch (e) {
      msg.textContent = e.message;
      msg.classList.add("error");
    }
  });

  syncAuthTabs();
  loadThreshold();
  syncSliderFromInput();
  loadFooterAndUsers();
  setAuthSteps(["Gotowe do skanu"]);
  document.querySelectorAll(".nav-item").forEach((b) => {
    if (b.classList.contains("active")) b.setAttribute("aria-current", "page");
  });
})();
