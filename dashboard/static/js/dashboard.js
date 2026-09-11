/**
 * IGARD-Net Reconstructed Tactical Audio Telemetry & Live Streaming Controller.
 * Features:
 *  - Real-Time Web Audio browser streaming with instant BEFORE/AFTER toggle
 *  - Dual synchronized waveform oscilloscopes & waterfall spectrograms
 *  - Decoupled Socket.IO telemetry (15-20 FPS)
 *  - Autopilot mode control & dynamic pipeline visualization
 *  - Robust file upload, batch processing, and A/B comparison player
 */

(function () {
  'use strict';

  // --- State Variables ---
  let socket = null;
  let currentMode = 'realtime'; // 'realtime' or 'file'
  let isBeforeAfterEnhanced = true;
  let isAutopilotOn = true;
  let isAuditionActive = false;

  // Web Audio Streaming Context
  let audioCtx = null;
  let streamAbortController = null;

  // Waveform Buffers (128 points each for fast, low-overhead canvas draw)
  const WAVE_POINTS = 128;
  const inputWaveBuffer = new Float32Array(WAVE_POINTS);
  const outputWaveBuffer = new Float32Array(WAVE_POINTS);

  // Spectrogram Waterfall History
  const SPEC_HISTORY = 160; // time slices
  const SPEC_BINS = 64;     // frequency bins
  const inputSpecHistory = [];
  const outputSpecHistory = [];

  for (let i = 0; i < SPEC_HISTORY; i++) {
    inputSpecHistory.push(new Float32Array(SPEC_BINS));
    outputSpecHistory.push(new Float32Array(SPEC_BINS));
  }

  // Canvases
  const inputWaveCanvas = document.getElementById('inputWaveCanvas');
  const outputWaveCanvas = document.getElementById('outputWaveCanvas');
  const inputSpecCanvas = document.getElementById('inputSpecCanvas');
  const outputSpecCanvas = document.getElementById('outputSpecCanvas');

  const inputWaveCtx = inputWaveCanvas ? inputWaveCanvas.getContext('2d') : null;
  const outputWaveCtx = outputWaveCanvas ? outputWaveCanvas.getContext('2d') : null;
  const inputSpecCtx = inputSpecCanvas ? inputSpecCanvas.getContext('2d') : null;
  const outputSpecCtx = outputSpecCanvas ? outputSpecCanvas.getContext('2d') : null;

  // High-contrast tactical cyber colormap
  function getTacticalColormap(val) {
    val = Math.max(0, Math.min(1, val));
    if (val < 0.2) {
      const t = val / 0.2;
      return `rgb(${Math.round(8 + t * 30)}, ${Math.round(12 + t * 20)}, ${Math.round(24 + t * 80)})`;
    } else if (val < 0.5) {
      const t = (val - 0.2) / 0.3;
      return `rgb(${Math.round(38 - t * 38)}, ${Math.round(32 + t * 180)}, ${Math.round(104 + t * 110)})`;
    } else if (val < 0.8) {
      const t = (val - 0.5) / 0.3;
      return `rgb(${Math.round(t * 220)}, ${Math.round(212 + t * 35)}, ${Math.round(214 - t * 170)})`;
    } else {
      const t = (val - 0.8) / 0.2;
      return `rgb(${Math.round(220 + t * 35)}, ${Math.round(247 + t * 8)}, ${Math.round(44 + t * 211)})`;
    }
  }

  // --- Real-Time Browser Audio Audition via Web Audio API ---
  async function startBrowserAudioAudition() {
    try {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      }
      if (audioCtx.state === 'suspended') {
        await audioCtx.resume();
      }

      streamAbortController = new AbortController();
      const response = await fetch('/api/stream/live_audio', {
        signal: streamAbortController.signal
      });

      if (!response.body) {
        console.warn('Audio stream body not readable.');
        return;
      }

      const reader = response.body.getReader();
      isAuditionActive = true;
      updateAuditionUI(true);

      let nextPlayTime = audioCtx.currentTime + 0.05;

      while (isAuditionActive) {
        const { value, done } = await reader.read();
        if (done) break;

        if (value && value.byteLength >= 2) {
          // Decode raw PCM16 bytes into Float32 buffer
          const int16Array = new Int16Array(value.buffer, value.byteOffset, value.byteLength / 2);
          const floatBuffer = audioCtx.createBuffer(1, int16Array.length, 16000);
          const channelData = floatBuffer.getChannelData(0);

          for (let i = 0; i < int16Array.length; i++) {
            channelData[i] = int16Array[i] / 32768.0;
          }

          const source = audioCtx.createBufferSource();
          source.buffer = floatBuffer;
          source.connect(audioCtx.destination);

          if (nextPlayTime < audioCtx.currentTime) {
            nextPlayTime = audioCtx.currentTime;
          }
          source.start(nextPlayTime);
          nextPlayTime += floatBuffer.duration;
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.warn('Live audio audition streaming disconnected:', err);
      }
      isAuditionActive = false;
      updateAuditionUI(false);
    }
  }

  function stopBrowserAudioAudition() {
    isAuditionActive = false;
    if (streamAbortController) {
      streamAbortController.abort();
      streamAbortController = null;
    }
    updateAuditionUI(false);
  }

  function updateAuditionUI(active) {
    const btn = document.getElementById('audioAuditionBtn');
    const icon = document.getElementById('audioAuditionIcon');
    const text = document.getElementById('audioAuditionText');
    if (btn && icon && text) {
      if (active) {
        btn.classList.add('active');
        icon.textContent = '🔊';
        text.textContent = 'Browser Audio: ON';
      } else {
        btn.classList.remove('active');
        icon.textContent = '🔈';
        text.textContent = 'Browser Audio: MUTED';
      }
    }
  }

  // --- Socket.IO Telemetry ---
  function initSocket() {
    if (typeof io !== 'undefined') {
      socket = io({ credentials: true, reconnectionDelay: 500, timeout: 5000 });

      socket.on('connect', () => {
        updateStatusDot('connected');
        console.log('[IGARD-Net] Telemetry WebSocket connected.');
      });

      socket.on('disconnect', () => {
        updateStatusDot('disconnected');
        console.warn('[IGARD-Net] Telemetry WebSocket disconnected.');
      });

      socket.on('telemetry_update', (data) => {
        handleTelemetryUpdate(data);
      });

      socket.on('file_completed', (res) => {
        handleFileCompleted(res);
      });

      socket.on('file_error', (err) => {
        const s = document.getElementById('fileStatusText');
        if (s) s.textContent = `Error: ${err.message}`;
      });
    } else {
      setInterval(pollStatusHTTP, 200);
    }
  }

  async function pollStatusHTTP() {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        handleTelemetryUpdate(data);
      }
    } catch (e) {
      updateStatusDot('disconnected');
    }
  }

  function updateStatusDot(state) {
    const dot = document.getElementById('statusDot');
    if (!dot) return;
    dot.className = 'status-dot';
    if (state === 'connected') {
      dot.classList.add('pulse');
    } else if (state === 'alert') {
      dot.classList.add('red', 'pulse');
    } else if (state === 'clean') {
      dot.classList.add('cyan');
    } else {
      dot.classList.add('amber');
    }
  }

  // --- Handle Telemetry Packet ---
  function handleTelemetryUpdate(data) {
    if (!data) return;

    // 1. Status Text & Meta
    const statusEl = document.getElementById('statusText');
    if (statusEl && data.stage_status) {
      statusEl.textContent = data.stage_status;
    }

    const metaEl = document.getElementById('statusMeta');
    if (metaEl) {
      const lat = data.latency_ms !== undefined ? `${data.latency_ms.toFixed(1)} ms` : '-- ms';
      const be = data.active_backend || 'CPU';
      const prof = data.hardware_profile || 'CPU_ONLY';
      metaEl.textContent = `Total Latency: ${lat} • Frame: 256 samples (16ms @ 16 kHz) • Backend: ${be} [${prof}]`;
    }

    // 2. Before / After & Autopilot State Sync
    if (data.before_after !== undefined) {
      updateBeforeAfterUI(data.before_after);
    }
    if (data.autopilot !== undefined) {
      updateAutopilotUI(data.autopilot);
    }

    // 3. Performance Metrics
    const latEl = document.getElementById('metricLatency');
    const latSub = document.getElementById('metricLatencySub');
    const latBadge = document.getElementById('latencyBadge');
    if (latEl && data.latency_ms !== undefined) {
      latEl.textContent = `${data.latency_ms.toFixed(1)} ms`;
      if (latSub) {
        const avg = data.avg_latency_ms !== undefined ? data.avg_latency_ms.toFixed(1) : '--';
        const p95 = data.p95_latency_ms !== undefined ? data.p95_latency_ms.toFixed(1) : '--';
        latSub.textContent = `Avg: ${avg} ms | P95: ${p95} ms | Max: ${data.max_latency_ms?.toFixed(1) || '--'} ms`;
      }
      if (latBadge) {
        if (data.latency_ms < 40.0) {
          latBadge.className = 'metric-badge badge-latency';
          latBadge.textContent = 'Ideal: <40ms';
        } else if (data.latency_ms < 50.0) {
          latBadge.className = 'metric-badge badge-snr';
          latBadge.textContent = 'Target: <50ms';
        } else {
          latBadge.className = 'metric-badge';
          latBadge.textContent = '>50ms';
        }
      }
    }

    const snrEl = document.getElementById('metricSnr');
    if (snrEl && data.estimated_snr !== undefined) {
      const sign = data.estimated_snr >= 0 ? '+' : '';
      snrEl.textContent = `${sign}${data.estimated_snr.toFixed(1)} dB`;
      snrEl.style.color = data.estimated_snr >= 0 ? 'var(--accent-emerald)' : 'var(--text-muted)';
    }

    const stepEl = document.getElementById('metricStep');
    if (stepEl && data.nlms_step_size !== undefined) {
      stepEl.textContent = data.nlms_step_size.toFixed(4);
    }

    const floorEl = document.getElementById('metricFloor');
    if (floorEl && data.noise_floor !== undefined) {
      floorEl.textContent = data.noise_floor.toExponential(2);
    }

    const hwEl = document.getElementById('metricHw');
    const hwSub = document.getElementById('metricHwSub');
    const hwBadge = document.getElementById('hwProfileBadge');
    if (hwEl && data.cpu_percent !== undefined) {
      hwEl.textContent = `CPU: ${data.cpu_percent.toFixed(0)}%`;
      if (hwSub) {
        hwSub.textContent = `GPU: ${data.gpu_percent?.toFixed(0) || 0}% • RAM: ${data.ram_mb?.toFixed(0) || 0} MB`;
      }
      if (hwBadge && data.active_backend) {
        hwBadge.textContent = data.active_backend;
      }
    }

    const framesEl = document.getElementById('metricFrames');
    const droppedEl = document.getElementById('metricDropped');
    if (framesEl && data.blocks_processed !== undefined) {
      framesEl.textContent = data.blocks_processed.toLocaleString();
      if (droppedEl) {
        droppedEl.textContent = `Drops: ${data.dropped_frames || 0} • Underruns: ${data.buffer_underruns || 0}`;
      }
    }

    // 4. Dual VU Meters
    updateMeter('primaryMeterBar', 'primaryDbText', data.primary_level);
    updateMeter('outputMeterBar', 'outputDbText', data.output_level);
    updateMeter('refMeterBar', 'refDbText', data.reference_level);

    // 5. Shock Indicator
    const shockVal = document.getElementById('shockVal');
    const shockOrb = document.getElementById('shockOrb');
    const shockBadge = document.getElementById('shockBadge');
    if (shockVal && data.shock_score !== undefined) {
      const score = data.shock_score;
      shockVal.textContent = `${(score * 100).toFixed(0)}%`;
      if (data.shock_confirmed || score >= 0.5) {
        if (shockOrb) shockOrb.className = 'shock-orb active-shock';
        if (shockBadge) {
          shockBadge.className = 'shock-badge alert';
          shockBadge.textContent = 'SHOCK CONFIRMED (CROSS-MODAL)';
        }
        updateStatusDot('alert');
      } else {
        if (shockOrb) shockOrb.className = 'shock-orb';
        if (shockBadge) {
          shockBadge.className = 'shock-badge normal';
          shockBadge.textContent = 'NORMAL (NO MECHANICAL IMPACT)';
        }
      }
    }

    // 6. Pipeline Diagram Highlights
    updatePipelineDiagram(data);

    // 7. Waveform Samples
    if (Array.isArray(data.primary_samples)) {
      copyToBuffer(inputWaveBuffer, data.primary_samples);
    }
    if (Array.isArray(data.output_samples)) {
      copyToBuffer(outputWaveBuffer, data.output_samples);
    }

    // 8. Spectrogram FFT Slices
    if (Array.isArray(data.primary_fft)) {
      pushSpectrogramSlice(inputSpecHistory, data.primary_fft);
    }
    if (Array.isArray(data.output_fft)) {
      pushSpectrogramSlice(outputSpecHistory, data.output_fft);
    }

    // 9. File Progress in File Mode
    if (data.file_progress !== undefined && currentMode === 'file') {
      const bar = document.getElementById('fileProgressBar');
      const text = document.getElementById('fileProgressText');
      if (bar) bar.style.width = `${data.file_progress}%`;
      if (text) text.textContent = `${data.file_progress.toFixed(0)}%`;
    }
  }

  function updateMeter(barId, textId, level) {
    if (level === undefined) return;
    const bar = document.getElementById(barId);
    const text = document.getElementById(textId);
    const pct = Math.min(100, Math.max(0, level * 100));
    if (bar) bar.style.width = `${pct}%`;
    if (text) {
      const db = (level * 60 - 60).toFixed(1);
      text.textContent = `${db} dB`;
    }
  }

  function copyToBuffer(target, src) {
    const n = Math.min(target.length, src.length);
    for (let i = 0; i < n; i++) {
      target[i] = src[i];
    }
  }

  function pushSpectrogramSlice(history, fftArray) {
    history.shift();
    const slice = new Float32Array(SPEC_BINS);
    const n = Math.min(SPEC_BINS, fftArray.length);
    for (let i = 0; i < n; i++) {
      slice[i] = fftArray[i];
    }
    history.push(slice);
  }

  function updatePipelineDiagram(data) {
    const stages = [
      document.getElementById('stage0'),
      document.getElementById('stage1'),
      document.getElementById('stage1b'),
      document.getElementById('stage2'),
      document.getElementById('stage3'),
      document.getElementById('stage4'),
      document.getElementById('stage6'),
    ];

    stages.forEach(s => {
      if (s) s.className = 'pipeline-stage';
    });

    const noisePresent = data.noise_present;
    const conf = data.confidence || 0;
    const shock = data.shock_score || 0;
    const label = data.noise_label || 'standby';

    // Stage 0
    const s0 = stages[0];
    const s0Badge = document.getElementById('stage0Badge');
    if (s0) {
      s0.classList.add('active');
      if (!noisePresent && data.autopilot) {
        s0.classList.add('stage-clean-pass');
        if (s0Badge) s0Badge.textContent = 'QUIET BYPASS (CLEAN)';
        return;
      } else {
        if (s0Badge) s0Badge.textContent = 'NOISE ENGAGED';
      }
    }

    // Stage 1
    const s1 = stages[1];
    const s1Badge = document.getElementById('stage1Badge');
    if (s1) {
      s1.classList.add('active');
      if (s1Badge) s1Badge.textContent = `${label.toUpperCase()} (${(conf * 100).toFixed(0)}%)`;
    }

    // Stage 1b
    const s1b = stages[2];
    const s1bBadge = document.getElementById('stage1bBadge');
    if (s1b) {
      s1b.classList.add('active');
      if (data.shock_confirmed || shock >= 0.5) {
        s1b.classList.add('stage-shock');
        if (s1bBadge) s1bBadge.textContent = `SHOCK ${(shock * 100).toFixed(0)}%`;
      } else {
        if (s1bBadge) s1bBadge.textContent = `Score: ${(shock * 100).toFixed(0)}%`;
      }
    }

    // Stage 2
    const s2 = stages[3];
    const s2Badge = document.getElementById('stage2Badge');
    if (s2) {
      s2.classList.add('active');
      if (s2Badge) s2Badge.textContent = `64 Taps • μ=${data.nlms_step_size?.toFixed(4) || '0.2500'}`;
    }

    // Stage 3
    const s3 = stages[4];
    if (s3) s3.classList.add('active');

    // Stage 4/5
    const s4 = stages[5];
    const s5Badge = document.getElementById('stage5Badge');
    if (s4) {
      s4.classList.add('active');
      if (data.used_cleanup_stage) {
        s4.classList.add('stage-clean-pass');
        if (s5Badge) s5Badge.textContent = 'WIENER OLA MASK ON';
      } else {
        s4.classList.add('stage-failsafe');
        if (s5Badge) s5Badge.textContent = 'FAIL-SAFE PASS';
      }
    }

    // Stage 6
    const s6 = stages[6];
    if (s6) s6.classList.add('active');
  }

  // --- Canvas Rendering Loop ---
  function renderWaveform(ctx, canvas, buffer, strokeColor, glowColor) {
    if (!ctx || !canvas) return;
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    // Center reference line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.07)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Waveform line
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2;
    ctx.shadowBlur = 6;
    ctx.shadowColor = glowColor;

    ctx.beginPath();
    const sliceWidth = width / buffer.length;
    let x = 0;

    for (let i = 0; i < buffer.length; i++) {
      const v = buffer[i];
      const y = (0.5 - v * 0.45) * height;
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
      x += sliceWidth;
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  function renderSpectrogram(ctx, canvas, history) {
    if (!ctx || !canvas) return;
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const timeSlices = history.length;
    const sliceWidth = width / timeSlices;
    const binHeight = height / SPEC_BINS;

    for (let t = 0; t < timeSlices; t++) {
      const slice = history[t];
      const x = t * sliceWidth;
      for (let b = 0; b < SPEC_BINS; b++) {
        const y = height - (b + 1) * binHeight;
        const mag = slice[b];
        ctx.fillStyle = getTacticalColormap(mag);
        ctx.fillRect(Math.floor(x), Math.floor(y), Math.ceil(sliceWidth), Math.ceil(binHeight));
      }
    }
  }

  function animationLoop() {
    renderWaveform(inputWaveCtx, inputWaveCanvas, inputWaveBuffer, '#ff6b6b', 'rgba(255, 107, 107, 0.4)');
    renderWaveform(outputWaveCtx, outputWaveCanvas, outputWaveBuffer, '#00f0ff', 'rgba(0, 240, 255, 0.5)');

    renderSpectrogram(inputSpecCtx, inputSpecCanvas, inputSpecHistory);
    renderSpectrogram(outputSpecCtx, outputSpecCanvas, outputSpecHistory);

    requestAnimationFrame(animationLoop);
  }

  function resizeCanvases() {
    [inputWaveCanvas, outputWaveCanvas, inputSpecCanvas, outputSpecCanvas].forEach(c => {
      if (c) {
        c.width = c.clientWidth || 600;
        c.height = c.clientHeight || 130;
      }
    });
  }

  // --- UI Updates for Controls ---
  function updateBeforeAfterUI(enhanced) {
    isBeforeAfterEnhanced = enhanced;
    const btn = document.getElementById('beforeAfterBtn');
    const label = document.getElementById('baLabel');
    if (btn && label) {
      if (enhanced) {
        btn.className = 'btn btn-ba active';
        label.textContent = 'AFTER (Enhanced)';
      } else {
        btn.className = 'btn btn-ba raw-mode';
        label.textContent = 'BEFORE (Raw Mic)';
      }
    }
  }

  function updateAutopilotUI(autopilot) {
    isAutopilotOn = autopilot;
    const btn = document.getElementById('autopilotBtn');
    const label = document.getElementById('autopilotLabel');
    if (btn && label) {
      if (autopilot) {
        btn.classList.add('active');
        label.textContent = 'ON';
      } else {
        btn.classList.remove('active');
        label.textContent = 'OFF';
      }
    }
  }

  // --- Control Event Listeners ---
  function setupControls() {
    // Mode Switcher
    const realtimeBtn = document.getElementById('realtimeModeBtn');
    const fileBtn = document.getElementById('fileModeBtn');
    const rtPanel = document.getElementById('realtimePanel');
    const fPanel = document.getElementById('filePanel');
    const panelTitle = document.getElementById('panelModeTitle');
    const panelSub = document.getElementById('panelModeSub');

    if (realtimeBtn && fileBtn) {
      realtimeBtn.addEventListener('click', () => {
        currentMode = 'realtime';
        realtimeBtn.classList.add('active');
        fileBtn.classList.remove('active');
        if (rtPanel) rtPanel.style.display = 'block';
        if (fPanel) fPanel.style.display = 'none';
        if (panelTitle) panelTitle.textContent = '⚙ Real-Time Stream Controls';
        if (panelSub) panelSub.textContent = 'Dual-Mic / Simulation';
      });

      fileBtn.addEventListener('click', () => {
        currentMode = 'file';
        fileBtn.classList.add('active');
        realtimeBtn.classList.remove('active');
        if (rtPanel) rtPanel.style.display = 'none';
        if (fPanel) fPanel.style.display = 'block';
        if (panelTitle) panelTitle.textContent = '📁 File Processing Module';
        if (panelSub) panelSub.textContent = 'WAV Upload & Fast Batch Processing';
      });
    }

    // Before / After Toggle
    const baBtn = document.getElementById('beforeAfterBtn');
    if (baBtn) {
      baBtn.addEventListener('click', async () => {
        const nextState = !isBeforeAfterEnhanced;
        updateBeforeAfterUI(nextState);
        try {
          await fetch('/api/mode/toggle_before_after', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: nextState })
          });
        } catch (e) {
          console.error(e);
        }
      });
    }

    // Autopilot Toggle
    const apBtn = document.getElementById('autopilotBtn');
    if (apBtn) {
      apBtn.addEventListener('click', async () => {
        const nextState = !isAutopilotOn;
        updateAutopilotUI(nextState);
        try {
          await fetch('/api/mode/toggle_autopilot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ enabled: nextState })
          });
        } catch (e) {
          console.error(e);
        }
      });
    }

    // Browser Audio Audition Button
    const auditionBtn = document.getElementById('audioAuditionBtn');
    if (auditionBtn) {
      auditionBtn.addEventListener('click', () => {
        if (!isAuditionActive) {
          startBrowserAudioAudition();
        } else {
          stopBrowserAudioAudition();
        }
      });
    }

    // Recalibrate Button
    const recalBtn = document.getElementById('recalibrateBtn');
    if (recalBtn) {
      recalBtn.addEventListener('click', async () => {
        recalBtn.disabled = true;
        recalBtn.innerHTML = '<span>⚡</span> Recalibrating...';
        try {
          const res = await fetch('/api/recalibrate', { method: 'POST' });
          const d = await res.json();
          document.getElementById('statusText').textContent = d.message;
        } catch (e) {
          console.error(e);
        } finally {
          setTimeout(() => {
            recalBtn.disabled = false;
            recalBtn.innerHTML = '<span>⚡</span> Recalibrate Baseline';
          }, 1200);
        }
      });
    }

    // Simulate Shock Recoil Button
    const shockBtn = document.getElementById('triggerShockBtn');
    if (shockBtn) {
      shockBtn.addEventListener('click', () => {
        if (socket) {
          socket.emit('trigger_shock', { intensity: 0.95 });
        }
      });
    }

    // Start / Stop
    const startBtn = document.getElementById('startBtn');
    const stopBtn = document.getElementById('stopBtn');
    if (startBtn) {
      startBtn.addEventListener('click', () => fetch('/api/start_live', { method: 'POST' }));
    }
    if (stopBtn) {
      stopBtn.addEventListener('click', () => fetch('/api/stop', { method: 'POST' }));
    }

    // File Upload & Processing
    const dropzone = document.getElementById('uploadDropzone');
    const fileInput = document.getElementById('wavFileInput');
    const runSampleBtn = document.getElementById('runSampleBtn');

    if (dropzone && fileInput) {
      dropzone.addEventListener('click', () => fileInput.click());

      dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'var(--accent-cyan)';
      });
      dropzone.addEventListener('dragleave', () => {
        dropzone.style.borderColor = 'rgba(56, 189, 248, 0.3)';
      });
      dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.style.borderColor = 'rgba(56, 189, 248, 0.3)';
        if (e.dataTransfer.files.length > 0) {
          handleFileUpload(e.dataTransfer.files[0]);
        }
      });

      fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
          handleFileUpload(fileInput.files[0]);
        }
      });
    }

    if (runSampleBtn) {
      runSampleBtn.addEventListener('click', async () => {
        document.getElementById('fileStatusText').textContent = 'Processing repository demo sample...';
        try {
          await fetch('/api/process_file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: null })
          });
        } catch (e) {
          console.error(e);
        }
      });
    }
  }

  // --- File Upload & Batch Processing Flow ---
  async function handleFileUpload(file) {
    const statusText = document.getElementById('fileStatusText');
    if (statusText) statusText.textContent = `Uploading ${file.name}...`;

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (data.status === 'ok') {
        const meta = data.metadata;
        const metaCard = document.getElementById('fileMetaCard');
        if (metaCard) metaCard.style.display = 'block';

        const nameEl = document.getElementById('metaFileName');
        const durEl = document.getElementById('metaFileDur');
        const srEl = document.getElementById('metaFileSr');
        const chEl = document.getElementById('metaFileCh');

        if (nameEl) nameEl.textContent = meta.filename;
        if (durEl) durEl.textContent = `${meta.duration_s}s`;
        if (srEl) srEl.textContent = `${meta.sample_rate} Hz`;
        if (chEl) chEl.textContent = `${meta.channels} ch`;

        if (statusText) statusText.textContent = 'Upload validated. Initiating DSP batch pipeline...';

        // Trigger batch file processing
        await fetch('/api/process_file', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: meta.filename })
        });
      } else {
        if (statusText) statusText.textContent = `Error: ${data.message}`;
      }
    } catch (e) {
      if (statusText) statusText.textContent = `Upload failed: ${e.message}`;
    }
  }

  function handleFileCompleted(res) {
    const statusText = document.getElementById('fileStatusText');
    if (statusText) {
      statusText.textContent = `Complete in ${res.processing_time_s}s (RTF: ${res.rtf}) • SNR: +${res.snr_improvement_db} dB`;
    }

    const pBar = document.getElementById('fileProgressBar');
    const pText = document.getElementById('fileProgressText');
    if (pBar) pBar.style.width = '100%';
    if (pText) pText.textContent = '100%';

    // Setup A/B Audio Player
    const playerContainer = document.getElementById('abPlayerContainer');
    const rawPlayer = document.getElementById('rawAudioPlayer');
    const cleanPlayer = document.getElementById('cleanedAudioPlayer');
    const dlBtn = document.getElementById('downloadEnhancedBtn');

    if (playerContainer) playerContainer.style.display = 'block';
    if (rawPlayer) {
      rawPlayer.src = res.input_file === 'sample_voice_44k.wav'
        ? '/api/audio/sample'
        : `/api/audio/upload/${res.input_file}`;
    }
    if (cleanPlayer) {
      cleanPlayer.src = `/api/audio/output/${res.output_file}`;
    }
    if (dlBtn) {
      dlBtn.href = `/api/audio/download/${res.output_file}`;
    }
  }

  // --- Document Initialization ---
  window.addEventListener('DOMContentLoaded', () => {
    resizeCanvases();
    window.addEventListener('resize', resizeCanvases);
    setupControls();
    initSocket();
    requestAnimationFrame(animationLoop);
  });
})();
