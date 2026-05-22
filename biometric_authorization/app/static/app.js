(function () {
  "use strict";

  const THRESHOLD_KEY_FACE = "biodesk_threshold_face_v1";
  const THRESHOLD_KEY_VOICE = "biodesk_threshold_voice_v1";
  const DEFAULT_FACE_THRESHOLD = 0.16;
  const DEFAULT_VOICE_THRESHOLD = 0.35;
  /** Domyślna długość jednej próbki nagrania (UI + auto-stop MediaRecorder). */
  const VOICE_AUTH_RECORD_MS = 9000;
  const VOICE_ENROLL_CLIP_MS = 6500;
  const VOICE_ENROLL_MIN_CLIPS = 3;
  const VOICE_ENROLL_MAX_CLIPS = 12;
  const $ = (sel, r = document) => r.querySelector(sel);

  const camVideo = $("#camVideo");
  const enrollVideo = $("#enrollVideo");
  const camShell = camVideo?.closest(".camera-shell");
  const enrollShell = enrollVideo?.closest(".camera-shell");

  let sharedStream = null;
  let voiceStreamAuth = null;
  let voiceStreamEnroll = null;
  let authTab = "identify";
  const enrollBlobs = [];
  const enrollVoiceBlobs = [];

  /** @type {{ modalities: string[], face: object, voice: object } | null} */
  let capabilities = null;
  /** @type {'face' | 'voice'} */
  let serviceModality = "face";

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

  /** Sekundy z ms — notacja PL (przecinek dziesiętny). */
  function formatSecPl(ms) {
    return (ms / 1000).toFixed(1).replace(".", ",");
  }

  /** @param {'auth' | 'enroll'} which */
  function voiceRecIds(which) {
    const p = which === "auth" ? "voiceAuth" : "voiceEnroll";
    return {
      panel: $(`#${p}RecPanel`),
      phase: $(`#${p}RecPhase`),
      countdown: $(`#${p}RecCountdown`),
      bar: $(`#${p}RecBar`),
      fill: $(`#${p}RecBarFill`),
      meta: $(`#${p}RecMeta`),
    };
  }

  /** Web Audio — wizualizacja poziomu przy nagrywaniu (tylko w przeglądarce). */
  /** @type {AudioContext | null} */
  let voiceVisAudioCtx = null;
  /** @type {MediaStreamAudioSourceNode | null} */
  let voiceVisSource = null;
  /** @type {AnalyserNode | null} */
  let voiceVisAnalyser = null;
  /** @type {Uint8Array | null} */
  let voiceVisFreqData = null;
  let voiceVisRaf = 0;
  /** @type {HTMLCanvasElement | null} */
  let voiceVisCanvas = null;

  function stopVoiceLevelVisualizer() {
    if (voiceVisRaf) {
      cancelAnimationFrame(voiceVisRaf);
      voiceVisRaf = 0;
    }
    if (voiceVisSource) {
      try {
        voiceVisSource.disconnect();
      } catch (_) {}
      voiceVisSource = null;
    }
    voiceVisAnalyser = null;
    voiceVisFreqData = null;
    voiceVisCanvas = null;
    document.querySelectorAll(".voice-vis-wrap").forEach((w) => w.classList.remove("is-active"));
    [$("#voiceAuthVisCanvas"), $("#voiceEnrollVisCanvas")].forEach((el) => {
      if (!(el instanceof HTMLCanvasElement)) return;
      const c = el.getContext("2d");
      if (c) c.clearRect(0, 0, el.width, el.height);
    });
  }

  /**
   * @param {MediaStream} stream
   * @param {'auth' | 'enroll'} which
   */
  async function startVoiceLevelVisualizer(stream, which) {
    stopVoiceLevelVisualizer();
    const canvas =
      which === "auth" ? $("#voiceAuthVisCanvas") : $("#voiceEnrollVisCanvas");
    if (!(canvas instanceof HTMLCanvasElement) || !stream?.getAudioTracks().length) return;
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      if (!voiceVisAudioCtx) voiceVisAudioCtx = new AudioCtx();
      if (voiceVisAudioCtx.state === "suspended") await voiceVisAudioCtx.resume();
      voiceVisSource = voiceVisAudioCtx.createMediaStreamSource(stream);
      voiceVisAnalyser = voiceVisAudioCtx.createAnalyser();
      voiceVisAnalyser.fftSize = 256;
      voiceVisAnalyser.smoothingTimeConstant = 0.68;
      voiceVisSource.connect(voiceVisAnalyser);
      voiceVisFreqData = new Uint8Array(voiceVisAnalyser.frequencyBinCount);
    } catch (_) {
      stopVoiceLevelVisualizer();
      return;
    }
    voiceVisCanvas = canvas;
    const wrap = canvas.closest(".voice-vis-wrap");
    if (wrap) wrap.classList.add("is-active");
    const ctx2d = canvas.getContext("2d");
    if (!ctx2d) {
      stopVoiceLevelVisualizer();
      return;
    }

    const draw = () => {
      if (!voiceVisAnalyser || !voiceVisFreqData || !voiceVisCanvas) return;
      voiceVisRaf = requestAnimationFrame(draw);
      voiceVisAnalyser.getByteFrequencyData(voiceVisFreqData);

      const cvs = voiceVisCanvas;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const cssW = cvs.clientWidth || 320;
      const cssH = 48;
      const w = Math.max(160, Math.floor(cssW * dpr));
      const h = Math.max(32, Math.floor(cssH * dpr));
      if (cvs.width !== w || cvs.height !== h) {
        cvs.width = w;
        cvs.height = h;
      }

      ctx2d.clearRect(0, 0, w, h);
      const rootStyle = getComputedStyle(document.documentElement);
      const accent = (rootStyle.getPropertyValue("--accent").trim() || "#5b8cff").replace(/\s/g, "");
      const track = (rootStyle.getPropertyValue("--border").trim() || "#3a3f4a").replace(/\s/g, "");

      const data = voiceVisFreqData;
      const binCount = data.length;
      const startBin = 2;
      const endBin = Math.min(binCount - 1, Math.floor(binCount * 0.52));
      const span = Math.max(1, endBin - startBin);
      const nBars = Math.min(72, Math.max(28, Math.floor(w / (5 * dpr))));
      const gap = Math.max(1, dpr);
      const barW = (w - (nBars - 1) * gap) / nBars;

      ctx2d.fillStyle = track;
      ctx2d.fillRect(0, h - Math.max(1, dpr), w, Math.max(1, dpr));

      for (let i = 0; i < nBars; i++) {
        const lo = startBin + Math.floor((i * span) / nBars);
        const hi = startBin + Math.floor(((i + 1) * span) / nBars);
        let peak = 0;
        for (let j = lo; j < hi; j++) peak = Math.max(peak, data[j]);
        const mag = peak / 255;
        const barH = mag * (h - 3 * dpr);
        const x = i * (barW + gap);
        const y = h - barH;
        ctx2d.fillStyle = accent;
        ctx2d.globalAlpha = 0.28 + mag * 0.72;
        ctx2d.fillRect(x, y, barW, barH);
        ctx2d.globalAlpha = 1;
      }
    };

    draw();
  }

  /** @param {'auth' | 'enroll'} which */
  function clearVoiceRecPanel(which) {
    stopVoiceLevelVisualizer();
    const u = voiceRecIds(which);
    u.panel?.classList.add("hidden");
    if (u.phase) u.phase.textContent = "";
    if (u.countdown) u.countdown.textContent = "";
    if (u.fill) u.fill.style.width = "0%";
    if (u.bar) u.bar.setAttribute("aria-valuenow", "0");
    if (u.meta) u.meta.textContent = "";
  }

  /**
   * @param {'auth' | 'enroll'} which
   * @param {{ phase: string, subline: string, elapsedMs: number, totalMs: number }} st
   */
  function setVoiceRecPanel(which, st) {
    const u = voiceRecIds(which);
    if (!u.panel || !u.phase || !u.countdown || !u.fill || !u.bar) return;
    u.panel.classList.remove("hidden");
    u.phase.textContent = st.phase;
    const rem = Math.max(0, st.totalMs - st.elapsedMs);
    u.countdown.textContent = `${formatSecPl(rem)} s do końca nagrania · łącznie ${formatSecPl(st.totalMs)} s`;
    if (u.meta) u.meta.textContent = st.subline;
    const pct = Math.min(100, Math.round((100 * st.elapsedMs) / st.totalMs));
    u.fill.style.width = `${pct}%`;
    u.bar.setAttribute("aria-valuenow", String(pct));
  }

  function hideAllVoiceRecPanels() {
    clearVoiceRecPanel("auth");
    clearVoiceRecPanel("enroll");
  }

  function syncVoiceEnrollQuotaUi() {
    const el = $("#voiceEnrollQuota");
    if (!el) return;
    const n = enrollVoiceBlobs.length;
    const need = VOICE_ENROLL_MIN_CLIPS;
    const max = VOICE_ENROLL_MAX_CLIPS;
    if (n === 0) {
      el.textContent = `Nagrano 0 z ${need} wymaganych próbek (maks. ${max}).`;
    } else if (n < need) {
      el.textContent = `Nagrano ${n} z ${need} wymaganych — brakuje jeszcze ${need - n}. Możesz dodać do ${max}.`;
    } else if (n < max) {
      el.textContent = `Nagrano ${n} próbek — minimum ${need} jest. Możesz dodać jeszcze ${max - n}, aby profil był stabilniejszy.`;
    } else {
      el.textContent = `Osiągnięto limit ${max} nagrań — usuń zbędne lub wyślij profil.`;
    }
  }

  function initVoiceDurationLabels() {
    const a = $("#voiceAuthDurationLabel");
    if (a) a.textContent = formatSecPl(VOICE_AUTH_RECORD_MS);
    const e = $("#voiceEnrollDurationLabel");
    if (e) e.textContent = formatSecPl(VOICE_ENROLL_CLIP_MS);
  }

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

  function phraseNagran(n) {
    const k = Number(n);
    if (!Number.isFinite(k) || k < 0) return String(n) + " nagrań";
    if (k === 1) return "1 nagranie";
    const mod10 = k % 10;
    const mod100 = k % 100;
    const word =
      mod10 >= 2 && mod10 <= 4 && (mod100 < 10 || mod100 >= 20) ? "nagrania" : "nagrań";
    return `${k} ${word}`;
  }

  function getThreshold() {
    const def = serviceModality === "voice" ? DEFAULT_VOICE_THRESHOLD : DEFAULT_FACE_THRESHOLD;
    const v = parseFloat($("#globalThreshold")?.value || String(def));
    return Number.isFinite(v) ? v : def;
  }

  function thresholdStorageKey() {
    return serviceModality === "voice" ? THRESHOLD_KEY_VOICE : THRESHOLD_KEY_FACE;
  }

  function saveThreshold() {
    try {
      localStorage.setItem(thresholdStorageKey(), String(getThreshold()));
    } catch (_) {}
  }

  function loadThresholdForModality() {
    const def = serviceModality === "voice" ? DEFAULT_VOICE_THRESHOLD : DEFAULT_FACE_THRESHOLD;
    const inp = $("#globalThreshold");
    if (!inp) return;
    try {
      const t = localStorage.getItem(thresholdStorageKey());
      inp.value = t != null ? t : String(def);
    } catch (_) {
      inp.value = String(def);
    }
    syncSliderFromInput();
  }

  function syncSliderFromInput() {
    const inp = $("#globalThreshold");
    const sl = $("#expThresholdSlider");
    const out = $("#expThresholdOut");
    if (!inp || !sl || !out) return;
    const raw = parseFloat(inp.value);
    const x = Math.round(raw * 100);
    const clamped = Number.isFinite(x) ? Math.min(100, Math.max(-100, x)) : Math.round(DEFAULT_FACE_THRESHOLD * 100);
    sl.value = String(clamped);
    out.textContent = Number.isFinite(raw) ? raw.toFixed(2) : String(DEFAULT_FACE_THRESHOLD);
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

  function stopVoiceStreams() {
    if (voiceStreamAuth) {
      voiceStreamAuth.getTracks().forEach((t) => t.stop());
      voiceStreamAuth = null;
    }
    if (voiceStreamEnroll) {
      voiceStreamEnroll.getTracks().forEach((t) => t.stop());
      voiceStreamEnroll = null;
    }
    const b1 = $("#btnVoiceMicAuth");
    const b2 = $("#btnVoiceMicEnroll");
    if (b1) b1.textContent = "Włącz mikrofon";
    if (b2) b2.textContent = "Włącz mikrofon";
    $("#btnVoiceRecordAuth")?.setAttribute("disabled", "disabled");
    $("#btnVoiceAddClip")?.setAttribute("disabled", "disabled");
    hideAllVoiceRecPanels();
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

  async function toggleVoiceMic(which) {
    const isAuth = which === "auth";
    const cur = isAuth ? voiceStreamAuth : voiceStreamEnroll;
    const btn = isAuth ? $("#btnVoiceMicAuth") : $("#btnVoiceMicEnroll");
    const msg = isAuth ? $("#voiceAuthMsg") : $("#enrollVoiceMsg");
    if (cur) {
      cur.getTracks().forEach((t) => t.stop());
      if (isAuth) voiceStreamAuth = null;
      else voiceStreamEnroll = null;
      if (btn) btn.textContent = "Włącz mikrofon";
      if (isAuth) $("#btnVoiceRecordAuth")?.setAttribute("disabled", "disabled");
      else $("#btnVoiceAddClip")?.setAttribute("disabled", "disabled");
      clearVoiceRecPanel(isAuth ? "auth" : "enroll");
      return;
    }
    if (msg) {
      msg.textContent = "";
      msg.className = "msg";
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      if (isAuth) voiceStreamAuth = stream;
      else voiceStreamEnroll = stream;
      if (btn) btn.textContent = "Wyłącz mikrofon";
      if (isAuth) $("#btnVoiceRecordAuth")?.removeAttribute("disabled");
      else $("#btnVoiceAddClip")?.removeAttribute("disabled");
    } catch (e) {
      if (msg) {
        msg.textContent =
          "Nie udało się uruchomić mikrofonu. Sprawdź uprawnienia przeglądarki.";
        msg.classList.add("error");
      }
    }
  }

  /**
   * @param {MediaStream} stream
   * @param {number} durationMs
   * @param {(tick: { elapsedMs: number; remainingMs: number; totalMs: number }) => void} [onTick]
   */
  function recordAudioFromStream(stream, durationMs, onTick) {
    const mime = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : "audio/webm";
    const mr = new MediaRecorder(stream, { mimeType: mime });
    const chunks = [];
    mr.addEventListener("dataavailable", (e) => {
      if (e.data.size) chunks.push(e.data);
    });
    return new Promise((resolve, reject) => {
      const t0 = Date.now();
      let tickId = null;
      const clearTick = () => {
        if (tickId !== null) {
          clearInterval(tickId);
          tickId = null;
        }
      };
      const fireTick = () => {
        if (!onTick) return;
        const elapsedMs = Date.now() - t0;
        const remainingMs = Math.max(0, durationMs - elapsedMs);
        onTick({ elapsedMs, remainingMs, totalMs: durationMs });
      };
      if (onTick) {
        fireTick();
        tickId = setInterval(fireTick, 100);
      }
      mr.addEventListener("error", () => {
        clearTick();
        reject(new Error("Błąd nagrywania"));
      });
      mr.addEventListener("stop", () => {
        clearTick();
        const type = (mr.mimeType && mr.mimeType.split(";")[0]) || "audio/webm";
        resolve(new Blob(chunks, { type }));
      });
      mr.start();
      setTimeout(() => {
        try {
          mr.stop();
        } catch (err) {
          clearTick();
          reject(err);
        }
      }, durationMs);
    });
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

  function syncModalitySegment(containerSel, m) {
    const root = document.querySelector(containerSel);
    if (!root) return;
    root.querySelectorAll("[data-modality-pick]").forEach((b) => {
      const on = b.getAttribute("data-modality-pick") === m;
      b.classList.toggle("active", on);
    });
  }

  function applyServiceModalityToUi() {
    const faceOn = capabilities?.modalities?.includes("face");
    const voiceOn = capabilities?.modalities?.includes("voice");
    if (serviceModality === "face" && !faceOn && voiceOn) serviceModality = "voice";
    if (serviceModality === "voice" && !voiceOn && faceOn) serviceModality = "face";

    syncModalitySegment("#modalitySegAuth", serviceModality);
    syncModalitySegment("#modalitySegEnroll", serviceModality);

    const isFace = serviceModality === "face";
    $("#authFaceColumn")?.classList.toggle("hidden", !isFace);
    $("#authVoiceColumn")?.classList.toggle("hidden", isFace);
    $("#enrollFaceBlock")?.classList.toggle("hidden", !isFace);
    $("#enrollVoiceBlock")?.classList.toggle("hidden", isFace);

    const tl = $("#thresholdLabelText");
    if (tl) tl.textContent = isFace ? "Próg podobieństwa (twarz)" : "Próg podobieństwa (głos)";

    loadThresholdForModality();

    $("#verifyUserRow")?.classList.toggle("hidden", authTab !== "verify");

    if (!isFace) {
      stopCamera();
      setCamInlineMsg("");
    } else {
      stopVoiceStreams();
    }

    loadVerifyUserOptions();
    const scan = $("#btnScan");
    if (scan) scan.disabled = isFace ? !sharedStream : true;
  }

  function setServiceModality(m) {
    if (!capabilities?.modalities?.includes(m)) return;
    stopVoiceStreams();
    serviceModality = m;
    applyServiceModalityToUi();
  }

  async function loadCapabilities() {
    try {
      capabilities = await api("/capabilities");
      const mods = capabilities.modalities || [];
      const multi = mods.length > 1;
      $("#modalityBarAuth")?.classList.toggle("hidden", !multi);
      $("#modalityBarEnroll")?.classList.toggle("hidden", !multi);
      if (mods.includes("face")) serviceModality = "face";
      else serviceModality = "voice";
      applyServiceModalityToUi();

      const adm = $("#adminModalitySelect");
      if (adm) {
        adm.innerHTML = "";
        for (const m of mods) {
          const o = document.createElement("option");
          o.value = m;
          o.textContent = m === "face" ? "Twarz" : "Głos";
          adm.appendChild(o);
        }
      }
    } catch {
      capabilities = { modalities: ["face"], face: {}, voice: {} };
      const mods = capabilities.modalities || [];
      $("#modalityBarAuth")?.classList.add("hidden");
      $("#modalityBarEnroll")?.classList.add("hidden");
      if (mods.includes("face")) serviceModality = "face";
      else serviceModality = "voice";
      applyServiceModalityToUi();
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

  document.querySelectorAll("#modalitySegAuth [data-modality-pick], #modalitySegEnroll [data-modality-pick]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const m = btn.getAttribute("data-modality-pick");
      if (m === "face" || m === "voice") setServiceModality(m);
    });
  });

  document.querySelectorAll("[data-auth-tab]").forEach((btn) => {
    btn.addEventListener("click", () => {
      authTab = btn.getAttribute("data-auth-tab");
      syncAuthTabs();
      $("#verifyUserRow")?.classList.toggle("hidden", authTab !== "verify");
      $("#authResult").hidden = true;
      setAuthSteps([serviceModality === "face" ? "Gotowe do skanu" : "Gotowe do nagrania"]);
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

  $("#btnVoiceMicAuth")?.addEventListener("click", () => toggleVoiceMic("auth"));
  $("#btnVoiceMicEnroll")?.addEventListener("click", () => toggleVoiceMic("enroll"));

  $("#globalThreshold")?.addEventListener("input", () => {
    syncSliderFromInput();
    saveThreshold();
  });

  $("#expThresholdSlider")?.addEventListener("input", syncInputFromSlider);

  async function loadVerifyUserOptions() {
    const sel = $("#verifyUserSelect");
    if (!sel) return;
    const m = serviceModality;
    try {
      const users = await api(`/users?modality=${encodeURIComponent(m)}`);
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

  $("#btnScan")?.addEventListener("click", async () => {
    if (serviceModality !== "face") return;
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
      const mq = "modality=face";
      if (authTab === "identify") {
        const fd = new FormData();
        fd.append("image", blob, "scan.jpg");
        const r = await api(`/identify?top_k=3&${mq}`, { method: "POST", body: fd });
        const dt = ((performance.now() - t0) / 1000).toFixed(2);
        if (!r.results?.length) {
          setAuthSteps(["Brak dopasowania"]);
          resultEl.hidden = false;
          resultEl.className = "result-card-auth bad";
          resultEl.innerHTML = `<strong>Odrzucono</strong><p>Brak zapisanych użytkowników w bazie (twarz).</p><p class="hint-inline">Czas: ${dt} s</p>`;
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
        const q = new URLSearchParams({ threshold: String(th), modality: "face" });
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

  $("#btnVoiceRecordAuth")?.addEventListener("click", async () => {
    if (serviceModality !== "voice") return;
    const resultEl = $("#authResult");
    const msg = $("#voiceAuthMsg");
    const btn = $("#btnVoiceRecordAuth");
    resultEl.hidden = true;
    if (msg) {
      msg.textContent = "";
      msg.className = "msg";
    }
    if (!voiceStreamAuth) {
      if (msg) {
        msg.textContent = "Najpierw włącz mikrofon.";
        msg.classList.add("error");
      }
      return;
    }
    const t0 = performance.now();
    if (btn) {
      btn.setAttribute("aria-busy", "true");
      btn.disabled = true;
    }
    try {
      setAuthSteps(["Nagrywanie…", "Oczekiwanie na odpowiedź serwera…"]);
      setVoiceRecPanel("auth", {
        phase: "Nagrywanie",
        subline: `Wymagana długość próbki: ${formatSecPl(VOICE_AUTH_RECORD_MS)} s — nagranie kończy się automatycznie.`,
        elapsedMs: 0,
        totalMs: VOICE_AUTH_RECORD_MS,
      });
      let blob;
      try {
        await startVoiceLevelVisualizer(voiceStreamAuth, "auth");
        blob = await recordAudioFromStream(voiceStreamAuth, VOICE_AUTH_RECORD_MS, (tick) => {
          setVoiceRecPanel("auth", {
            phase: "Nagrywanie",
            subline: "Mów naturalnie — zostań przy mikrofonie do końca odliczania.",
            elapsedMs: Math.min(tick.elapsedMs, tick.totalMs),
            totalMs: tick.totalMs,
          });
        });
      } finally {
        stopVoiceLevelVisualizer();
      }
      setVoiceRecPanel("auth", {
        phase: "Wysyłanie i przetwarzanie",
        subline: "Łączenie z serwerem i obliczanie dopasowania — zwykle kilka sekund.",
        elapsedMs: VOICE_AUTH_RECORD_MS,
        totalMs: VOICE_AUTH_RECORD_MS,
      });
      setAuthSteps(["Nagranie gotowe", "Przetwarzanie na serwerze…"]);
      const th = getThreshold();
      const qBase = { threshold: String(th), modality: "voice" };
      if (authTab === "identify") {
        const fd = new FormData();
        fd.append("audio", blob, "probe.webm");
        const r = await api(`/identify?${new URLSearchParams({ ...qBase, top_k: "3" })}`, {
          method: "POST",
          body: fd,
        });
        const dt = ((performance.now() - t0) / 1000).toFixed(2);
        if (!r.results?.length) {
          setAuthSteps(["Brak dopasowania"]);
          if (msg) {
            msg.textContent =
              "Nagranie wysłano, ale w bazie nie ma jeszcze profili głosu do identyfikacji.";
            msg.className = "msg error";
          }
          resultEl.hidden = false;
          resultEl.className = "result-card-auth bad";
          resultEl.innerHTML = `<strong>Odrzucono</strong><p>Brak zapisanych użytkowników (głos).</p><p class="hint-inline">Czas: ${dt} s</p>`;
          return;
        }
        const top = r.results[0];
        const ok = top.similarity >= th;
        setAuthSteps(["Nagranie zapisane", "Dopasowanie zakończone"]);
        if (msg) {
          if (ok) {
            msg.textContent =
              "Nagranie wysłano i przetworzono — dopasowanie pozytywne (szczegóły w polu obok).";
            msg.className = "msg ok";
          } else {
            msg.textContent =
              "Nagranie wysłano i przetworzono — dostęp nieprzyznany (najbliższy wynik poniżej progu; szczegóły obok).";
            msg.className = "msg";
          }
        }
        resultEl.hidden = false;
        resultEl.className = "result-card-auth " + (ok ? "ok" : "bad");
        resultEl.innerHTML = ok
          ? `<strong>Dostęp przyznany</strong><p>Identyfikator: <code>${escapeHtml(top.user_id)}</code></p><p>Podobieństwo: <strong>${top.similarity.toFixed(4)}</strong> (próg: ${th})</p><p class="hint-inline">Czas: ${dt} s</p>`
          : `<strong>Odrzucono</strong><p>Najbliższy: <code>${escapeHtml(top.user_id)}</code> — ${top.similarity.toFixed(4)} (próg ${th})</p><p class="hint-inline">Czas: ${dt} s</p>`;
      } else {
        const uid = $("#verifyUserSelect")?.value;
        if (!uid) {
          setAuthSteps(["Wybierz użytkownika z listy"]);
          if (msg) {
            msg.textContent =
              "Nagranie zebrano, ale nie wysłano na serwer — wybierz użytkownika z listy i ponów „Nagraj i wyślij”.";
            msg.className = "msg error";
          }
          return;
        }
        const fd = new FormData();
        fd.append("user_id", uid);
        fd.append("audio", blob, "probe.webm");
        const r = await api(`/verify?${new URLSearchParams(qBase)}`, { method: "POST", body: fd });
        const dt = ((performance.now() - t0) / 1000).toFixed(2);
        setAuthSteps(["Nagranie zapisane", "Weryfikacja zakończona"]);
        if (msg) {
          if (r.accepted) {
            msg.textContent =
              "Nagranie wysłano i przetworzono — weryfikacja pozytywna (szczegóły obok).";
            msg.className = "msg ok";
          } else {
            msg.textContent =
              "Nagranie wysłano i przetworzono — weryfikacja negatywna (szczegóły obok).";
            msg.className = "msg";
          }
        }
        resultEl.hidden = false;
        resultEl.className = "result-card-auth " + (r.accepted ? "ok" : "bad");
        resultEl.innerHTML = r.accepted
          ? `<strong>Dostęp przyznany</strong><p>${escapeHtml(r.user_id)} — ${r.similarity.toFixed(4)}</p><p class="hint-inline">Czas: ${dt} s</p>`
          : `<strong>Odrzucono</strong><p>${escapeHtml(r.user_id)} — ${r.similarity.toFixed(4)}</p><p class="hint-inline">Czas: ${dt} s</p>`;
      }
    } catch (e) {
      setAuthSteps(["Błąd", String(e.message)]);
      if (msg) {
        msg.textContent = `Nie udało się dokończyć operacji: ${e.message}`;
        msg.className = "msg error";
      }
      resultEl.hidden = false;
      resultEl.className = "result-card-auth bad";
      resultEl.innerHTML = `<strong>Błąd</strong><p>${escapeHtml(e.message)}</p>`;
    } finally {
      if (btn) {
        btn.setAttribute("aria-busy", "false");
        btn.disabled = !voiceStreamAuth;
      }
      clearVoiceRecPanel("auth");
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

  function renderVoiceClipStrip() {
    const strip = $("#voiceClipStrip");
    const cnt = $("#voiceClipCount");
    const btn = $("#btnSubmitVoiceEnroll");
    if (!strip || !cnt || !btn) return;
    cnt.textContent = String(enrollVoiceBlobs.length);
    strip.querySelectorAll("audio").forEach((a) => {
      if (a.src.startsWith("blob:")) URL.revokeObjectURL(a.src);
    });
    strip.innerHTML = "";
    enrollVoiceBlobs.forEach((blob) => {
      const url = URL.createObjectURL(blob);
      const wrap = document.createElement("div");
      wrap.className = "shot-thumb";
      wrap.setAttribute("role", "listitem");
      wrap.innerHTML = `<audio controls src="${url}" style="max-width:10rem"></audio><button type="button" aria-label="Usuń nagranie">×</button>`;
      wrap.querySelector("button").addEventListener("click", () => {
        const i = enrollVoiceBlobs.indexOf(blob);
        if (i >= 0) enrollVoiceBlobs.splice(i, 1);
        URL.revokeObjectURL(url);
        renderVoiceClipStrip();
      });
      strip.appendChild(wrap);
    });
    btn.disabled = enrollVoiceBlobs.length < VOICE_ENROLL_MIN_CLIPS;
    syncVoiceEnrollQuotaUi();
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
      await api(`/users/${encodeURIComponent(uid)}/enroll_multi?modality=face`, { method: "POST", body: fd });
      msg.textContent =
        "Zapisano w bazie (" + phraseKlatek(enrollBlobs.length) + ", uśredniony wektor twarzy).";
      msg.classList.add("ok");
      enrollBlobs.length = 0;
      renderShotStrip();
      loadFooterAndUsers();
      loadVerifyUserOptions();
    } catch (e) {
      msg.textContent = e.message;
      msg.classList.add("error");
    }
  });

  $("#btnVoiceAddClip")?.addEventListener("click", async () => {
    const msg = $("#enrollVoiceMsg");
    const addBtn = $("#btnVoiceAddClip");
    if (msg) {
      msg.textContent = "";
      msg.className = "msg";
    }
    if (!voiceStreamEnroll) {
      if (msg) {
        msg.textContent = "Najpierw włącz mikrofon.";
        msg.classList.add("error");
      }
      return;
    }
    if (enrollVoiceBlobs.length >= VOICE_ENROLL_MAX_CLIPS) {
      if (msg) {
        msg.textContent = `Możesz dodać co najwyżej ${VOICE_ENROLL_MAX_CLIPS} nagrań.`;
        msg.classList.add("error");
      }
      return;
    }
    if (addBtn) {
      addBtn.setAttribute("aria-busy", "true");
      addBtn.disabled = true;
    }
    try {
      const nextIdx = enrollVoiceBlobs.length + 1;
      setVoiceRecPanel("enroll", {
        phase: "Nagrywanie",
        subline: `Próbka ${nextIdx} — długość ${formatSecPl(VOICE_ENROLL_CLIP_MS)} s. Minimum to ${VOICE_ENROLL_MIN_CLIPS} osobnych nagrań.`,
        elapsedMs: 0,
        totalMs: VOICE_ENROLL_CLIP_MS,
      });
      let blob;
      try {
        await startVoiceLevelVisualizer(voiceStreamEnroll, "enroll");
        blob = await recordAudioFromStream(voiceStreamEnroll, VOICE_ENROLL_CLIP_MS, (tick) => {
          const after = enrollVoiceBlobs.length + 1;
          setVoiceRecPanel("enroll", {
            phase: "Nagrywanie",
            subline: `Po zapisaniu tej próbki: ${after} z min. ${VOICE_ENROLL_MIN_CLIPS} wymaganych (limit ${VOICE_ENROLL_MAX_CLIPS}).`,
            elapsedMs: Math.min(tick.elapsedMs, tick.totalMs),
            totalMs: tick.totalMs,
          });
        });
      } finally {
        stopVoiceLevelVisualizer();
      }
      enrollVoiceBlobs.push(blob);
      renderVoiceClipStrip();
    } catch (e) {
      if (msg) {
        msg.textContent = e.message;
        msg.classList.add("error");
      }
    } finally {
      if (addBtn) {
        addBtn.setAttribute("aria-busy", "false");
        addBtn.disabled =
          !voiceStreamEnroll || enrollVoiceBlobs.length >= VOICE_ENROLL_MAX_CLIPS;
      }
      clearVoiceRecPanel("enroll");
    }
  });

  $("#btnVoiceClearClips")?.addEventListener("click", () => {
    enrollVoiceBlobs.length = 0;
    $("#voiceClipStrip").innerHTML = "";
    renderVoiceClipStrip();
  });

  $("#btnSubmitVoiceEnroll")?.addEventListener("click", async () => {
    const uid = ($("#enrollUserIdVoice")?.value || "").trim();
    const msg = $("#enrollVoiceMsg");
    if (msg) {
      msg.textContent = "";
      msg.className = "msg";
    }
    if (!uid) {
      if (msg) {
        msg.textContent = "Podaj identyfikator użytkownika.";
        msg.classList.add("error");
      }
      return;
    }
    if (enrollVoiceBlobs.length < VOICE_ENROLL_MIN_CLIPS) {
      if (msg) {
        msg.textContent = `Potrzebne są co najmniej ${phraseNagran(VOICE_ENROLL_MIN_CLIPS)}.`;
        msg.classList.add("error");
      }
      return;
    }
    if (msg) msg.textContent = "Zapisywanie profilu głosu…";
    try {
      const fd = new FormData();
      enrollVoiceBlobs.forEach((b) => fd.append("audios", b, "clip.webm"));
      await api(`/users/${encodeURIComponent(uid)}/enroll_multi?modality=voice`, { method: "POST", body: fd });
      if (msg) {
        msg.textContent = "Zapisano (" + phraseNagran(enrollVoiceBlobs.length) + ", uśredniony embedding).";
        msg.classList.add("ok");
      }
      enrollVoiceBlobs.length = 0;
      renderVoiceClipStrip();
      loadFooterAndUsers();
      loadVerifyUserOptions();
    } catch (e) {
      if (msg) {
        msg.textContent = e.message;
        msg.classList.add("error");
      }
    }
  });

  async function loadFooterAndUsers() {
    try {
      const h = await api("/health");
      const parts = [`twarz ${h.enrolled_users_face}`, `głos ${h.enrolled_users_voice}`];
      $("#sidebarMeta").textContent = `${parts.join(" · ")} · ${h.device}`;
    } catch {
      $("#sidebarMeta").textContent = "";
    }
    await loadVerifyUserOptions();
  }

  async function loadDashboard() {
    const grid = $("#dashCards");
    const note = $("#dashNote");
    if (!grid) return;
    try {
      const s = await api("/admin/summary");
      grid.innerHTML = `
        <div class="dash-card"><div class="label">Użytkownicy (twarz)</div><div class="value">${s.enrolled_users_face}</div></div>
        <div class="dash-card"><div class="label">Użytkownicy (głos)</div><div class="value">${s.enrolled_users_voice}</div></div>
        <div class="dash-card"><div class="label">Urządzenie</div><div class="value" style="font-size:0.95rem">${escapeHtml(s.device)}</div></div>
        <div class="dash-card"><div class="label">Wagi twarzy</div><div class="value" style="font-size:0.65rem;word-break:break-all">${escapeHtml(s.face_weights || "—")}</div></div>
        <div class="dash-card"><div class="label">Wagi głosu</div><div class="value" style="font-size:0.65rem;word-break:break-all">${escapeHtml(s.voice_weights || "—")}</div></div>
      `;
      if (note) note.textContent = s.note;
    } catch (e) {
      grid.innerHTML = `<p class="msg error">${escapeHtml(e.message)}</p>`;
    }
  }

  async function loadAdminUsers() {
    const tbody = $("#adminUsersBody");
    const msg = $("#adminUsersMsg");
    const modSel = $("#adminModalitySelect");
    if (!tbody) return;
    msg.textContent = "";
    const modality = modSel?.value || serviceModality || "face";
    try {
      const users = await api(`/users?modality=${encodeURIComponent(modality)}`);
      tbody.innerHTML = "";
      if (!users.length) {
        tbody.innerHTML =
          '<tr><td colspan="4" style="color:var(--muted);padding:1rem">Brak rekordów</td></tr>';
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
          if (!confirm('Usunąć użytkownika "' + u.user_id + '" z bazy (' + modality + ")?")) return;
          try {
            await api(
              `/users/${encodeURIComponent(u.user_id)}?modality=${encodeURIComponent(modality)}`,
              { method: "DELETE" }
            );
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
  $("#adminModalitySelect")?.addEventListener("change", loadAdminUsers);

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

  $("#formCompareVoice")?.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const form = ev.target;
    const msg = $("#compareVoiceMsg");
    msg.textContent = "";
    msg.className = "msg";
    const fa = form.audio_a?.files?.[0];
    const fb = form.audio_b?.files?.[0];
    if (!fa || !fb) {
      msg.textContent = "Wybierz oba pliki audio.";
      msg.classList.add("error");
      return;
    }
    const fd = new FormData();
    fd.append("audio_a", fa);
    fd.append("audio_b", fb);
    const q = new URLSearchParams({ threshold: String(form.threshold.value) });
    try {
      const r = await api(`/compare_voice?${q}`, { method: "POST", body: fd });
      msg.textContent = `Ten sam mówca (heurystycznie): ${r.same_speaker_guess ? "tak" : "nie"} — podobieństwo ${r.similarity.toFixed(4)}`;
      msg.classList.add("ok");
    } catch (e) {
      msg.textContent = e.message;
      msg.classList.add("error");
    }
  });

  syncAuthTabs();
  initVoiceDurationLabels();
  syncVoiceEnrollQuotaUi();
  loadCapabilities().then(() => {
    loadThresholdForModality();
    loadFooterAndUsers();
    setAuthSteps(["Gotowe do skanu"]);
  });
  document.querySelectorAll(".nav-item").forEach((b) => {
    if (b.classList.contains("active")) b.setAttribute("aria-current", "page");
  });
})();
