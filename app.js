// Reaction Meter — カメラ映像から表情・姿勢の特徴量をブラウザ内で推定するツール。
// 映像フレームは MediaPipe に渡して数値化されるだけで、保存・送信は一切行わない。

import {
  FilesetResolver,
  FaceLandmarker,
  PoseLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const MEDIAPIPE_WASM =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const POSE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

const SAMPLE_INTERVAL_MS = 200; // 5Hz で時系列に記録
const POSE_INTERVAL_MS = 150;   // 姿勢推定は少し間引いて負荷を下げる
const CALIB_MS = 2000;          // 記録開始直後の基準値計測時間
const CHART_WINDOW_MS = 60000;

const $ = (id) => document.getElementById(id);
const els = {
  video: $("video"), overlay: $("overlay"), videoWrap: $("videoWrap"),
  placeholder: $("videoPlaceholder"), recBadge: $("recBadge"), recTime: $("recTime"),
  calibBadge: $("calibBadge"), status: $("status"),
  btnCamera: $("btnCamera"), btnFacing: $("btnFacing"),
  chkOverlay: $("chkOverlay"), chkMirror: $("chkMirror"),
  btnRecord: $("btnRecord"), btnMark: $("btnMark"),
  btnCsv: $("btnCsv"), btnSummaryCsv: $("btnSummaryCsv"),
  inpParticipant: $("inpParticipant"), inpStimulus: $("inpStimulus"),
  barSmile: $("barSmile"), valSmile: $("valSmile"),
  barFurrow: $("barFurrow"), valFurrow: $("valFurrow"),
  barRaise: $("barRaise"), valRaise: $("valRaise"),
  barEye: $("barEye"), valEye: $("valEye"),
  valenceFill: $("valenceFill"), valValence: $("valValence"),
  valEngage: $("valEngage"), valAttention: $("valAttention"),
  valPosture: $("valPosture"), valBlink: $("valBlink"), valLean: $("valLean"),
  chart: $("chart"),
  diagnosisPanel: $("diagnosisPanel"), diagnosisBody: $("diagnosisBody"),
  chkDetails: $("chkDetails"), detailsSection: $("detailsSection"),
  barDuchenne: $("barDuchenne"), valDuchenne: $("valDuchenne"),
  barLipPress: $("barLipPress"), valLipPress: $("valLipPress"),
  barSneer: $("barSneer"), valSneer: $("valSneer"),
  barPucker: $("barPucker"), valPucker: $("valPucker"),
  barMove: $("barMove"), valMove: $("valMove"),
  valNod: $("valNod"), valShake: $("valShake"), valTouch: $("valTouch"),
  valGazeZone: $("valGazeZone"), gazeGrid: $("gazeGrid"),
  selfreportPanel: $("selfreportPanel"),
  inpAuto: $("inpAuto"), inpActive: $("inpActive"), inpStable: $("inpStable"),
  btnSaveSession: $("btnSaveSession"),
  sessionsPanel: $("sessionsPanel"), sessionCount: $("sessionCount"),
  sessionsTable: $("sessionsTable"),
  btnSessionsCsv: $("btnSessionsCsv"), btnClearSessions: $("btnClearSessions"),
  selX: $("selX"), selY: $("selY"),
  corrResult: $("corrResult"), scatter: $("scatter"),
};

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

let faceLandmarker = null;
let poseLandmarker = null;
let stream = null;
let facing = "user";
let running = false;
let rafId = null;
let lastPoseAt = 0;
let lastPoseResult = null;

// 記録状態
let recording = false;
let recStartMs = 0;
let lastSampleAt = 0;
let samples = [];
let pendingMarker = "";
let markerSeq = 0;

// まばたき・笑顔イベント（生値＋ヒステリシスで検出）
let blinkTotal = 0, blinkClosed = false;
let smileEvents = 0, smileOn = false;

// 基準値（記録開始直後 CALIB_MS の中央値）: 前のめり・猫背・視線上下の判定に使う
let baseline = null;
let calibrating = false;
let calibBuf = [];

// うなずき・首振り（頭部角度の振動を検出）
let nodTotal = 0, shakeTotal = 0;

// 頭の動き量（そわそわ度）用の前フレーム位置
let prevNose = null;

// 視線ゾーンの滞留カウント（記録中のみ蓄積）
let gazeDwell = {};

// 振動検出器：ゆっくり追従する基準線からの偏差の符号反転を数える
function makeOscillationDetector(minAmp) {
  let base = null, lastSign = 0, crossings = 0, lastCrossAt = 0, lastEventAt = 0;
  return {
    update(v, now) {
      if (v == null) { base = null; crossings = 0; lastSign = 0; return false; }
      base = base == null ? v : base + 0.05 * (v - base);
      const d = v - base;
      if (now - lastCrossAt > 900) { crossings = 0; lastSign = 0; }
      if (Math.abs(d) > minAmp) {
        const s = Math.sign(d);
        if (s !== lastSign) { lastSign = s; crossings++; lastCrossAt = now; }
      }
      if (crossings >= 3 && now - lastEventAt > 1500) {
        lastEventAt = now;
        crossings = 0;
        return true;
      }
      return false;
    },
  };
}
const nodDetector = makeOscillationDetector(0.06);   // pitch の振動 = うなずき
const shakeDetector = makeOscillationDetector(0.08); // yaw の振動 = 首振り

// 表示用の平滑化値（EMA）
const smoothed = {};
const ema = (key, v, alpha = 0.35) => {
  if (v == null || Number.isNaN(v)) return smoothed[key] ?? null;
  smoothed[key] = smoothed[key] == null ? v : smoothed[key] + alpha * (v - smoothed[key]);
  return smoothed[key];
};

// チャート履歴（直近60秒）
const chartHistory = [];

// 直近の記録セッションのサマリー（自己報告と紐づけて保存する）
let lastSummary = null;

function setStatus(msg, isError = false) {
  els.status.textContent = msg;
  els.status.classList.toggle("error", isError);
}

// ---------- モデル読み込み ----------
async function loadModels() {
  setStatus("解析モデルを読み込み中…（初回は数秒かかります）");
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: FACE_MODEL, delegate: "GPU" },
    outputFaceBlendshapes: true,
    runningMode: "VIDEO",
    numFaces: 1,
  });
  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: POSE_MODEL, delegate: "GPU" },
    runningMode: "VIDEO",
    numPoses: 1,
  });
  setStatus("モデル読み込み完了");
}

// ---------- カメラ ----------
async function startCamera() {
  els.btnCamera.disabled = true;
  try {
    if (!faceLandmarker) await loadModels();
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing, width: { ideal: 960 }, height: { ideal: 720 } },
      audio: false,
    });
    els.video.srcObject = stream;
    await new Promise((res) => (els.video.onloadedmetadata = res));
    await els.video.play();
    els.overlay.width = els.video.videoWidth;
    els.overlay.height = els.video.videoHeight;
    els.placeholder.classList.add("hidden");
    running = true;
    els.btnCamera.textContent = "カメラ停止";
    els.btnCamera.classList.remove("primary");
    els.btnFacing.disabled = false;
    els.btnRecord.disabled = false;
    setStatus("計測中（記録はまだ開始されていません）");
    loop();
  } catch (err) {
    console.error(err);
    if (err.name === "NotAllowedError") {
      setStatus("カメラの使用が許可されませんでした。ブラウザの設定でカメラを許可してください。", true);
    } else if (err.name === "NotFoundError") {
      setStatus("カメラが見つかりませんでした。", true);
    } else {
      setStatus(`カメラ/モデルの初期化に失敗しました: ${err.message}`, true);
    }
  } finally {
    els.btnCamera.disabled = false;
  }
}

function stopCamera() {
  running = false;
  if (rafId) cancelAnimationFrame(rafId);
  if (recording) stopRecording();
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  els.video.srcObject = null;
  els.placeholder.classList.remove("hidden");
  els.btnCamera.textContent = "カメラ開始";
  els.btnCamera.classList.add("primary");
  els.btnFacing.disabled = true;
  els.btnRecord.disabled = true;
  const ctx = els.overlay.getContext("2d");
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  setStatus("カメラ停止中");
}

async function switchFacing() {
  facing = facing === "user" ? "environment" : "user";
  els.chkMirror.checked = facing === "user";
  applyMirror();
  if (running) {
    // ストリームだけ差し替える（記録は継続。ただし基準値はリセット）
    stream.getTracks().forEach((t) => t.stop());
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facing, width: { ideal: 960 }, height: { ideal: 720 } },
        audio: false,
      });
      els.video.srcObject = stream;
      await els.video.play();
      els.overlay.width = els.video.videoWidth;
      els.overlay.height = els.video.videoHeight;
      baseline = null;
      setStatus(facing === "user" ? "前面カメラに切替えました" : "背面カメラに切替えました");
    } catch (err) {
      setStatus(`カメラ切替に失敗しました: ${err.message}`, true);
    }
  }
}

function applyMirror() {
  els.videoWrap.classList.toggle("mirrored", els.chkMirror.checked);
}

// ---------- メインループ ----------
function loop() {
  if (!running) return;
  rafId = requestAnimationFrame(loop);
  if (els.video.readyState < 2) return;

  const now = performance.now();
  let faceResult = null;
  try {
    faceResult = faceLandmarker.detectForVideo(els.video, now);
    if (now - lastPoseAt >= POSE_INTERVAL_MS) {
      lastPoseAt = now;
      lastPoseResult = poseLandmarker.detectForVideo(els.video, now + 0.001);
    }
  } catch (err) {
    console.error(err);
    return;
  }

  const metrics = computeMetrics(faceResult, lastPoseResult, now);
  updateCalibration(metrics, now);
  updateUI(metrics, now);
  drawOverlay(faceResult, lastPoseResult);

  if (recording) {
    updateRecTimer(now);
    if (now - lastSampleAt >= SAMPLE_INTERVAL_MS) {
      lastSampleAt = now;
      pushSample(metrics, now);
    }
  }
}

// ---------- 特徴量の計算 ----------
function blendshapeMap(faceResult) {
  const cats = faceResult?.faceBlendshapes?.[0]?.categories;
  if (!cats) return null;
  const m = {};
  for (const c of cats) m[c.categoryName] = c.score;
  return m;
}

function computeMetrics(faceResult, poseResult, now) {
  const m = {
    faceDetected: false,
    smile: null, furrow: null, browRaise: null, eyeOpen: null, jawOpen: null,
    valence: null, yawProxy: null, pitchProxy: null, attention: 0,
    interocular: null, leanIn: null,
    cheekSquint: null, duchenne: null, lipPress: null, noseSneer: null, mouthPucker: null,
    gazeX: null, gazeY: null, gazeZone: "", headMove: null,
    poseDetected: false, shoulderTiltDeg: null, headOffset: null,
    neckRatio: null, postureScore: null, engagement: null, faceTouch: null,
  };

  const bs = blendshapeMap(faceResult);
  const lm = faceResult?.faceLandmarks?.[0];
  if (bs && lm) {
    m.faceDetected = true;
    m.smile = (bs.mouthSmileLeft + bs.mouthSmileRight) / 2;
    m.furrow = (bs.browDownLeft + bs.browDownRight) / 2;
    m.browRaise = bs.browInnerUp;
    const blinkRaw = (bs.eyeBlinkLeft + bs.eyeBlinkRight) / 2;
    m.eyeOpen = 1 - blinkRaw;
    m.jawOpen = bs.jawOpen;
    m.valence = m.smile - m.furrow;

    // まばたき（ヒステリシス付き立ち上がり検出）
    if (!blinkClosed && blinkRaw > 0.5) { blinkClosed = true; blinkTotal++; }
    else if (blinkClosed && blinkRaw < 0.3) blinkClosed = false;

    // 笑顔イベント
    if (!smileOn && m.smile > 0.5) { smileOn = true; smileEvents++; }
    else if (smileOn && m.smile < 0.3) smileOn = false;

    // 頭部向きの近似：目の中点に対する鼻先のずれ（landmark 1=鼻先, 33/263=目尻）
    const nose = lm[1], eyeL = lm[33], eyeR = lm[263];
    const midX = (eyeL.x + eyeR.x) / 2, midY = (eyeL.y + eyeR.y) / 2;
    const io = Math.hypot(eyeR.x - eyeL.x, eyeR.y - eyeL.y);
    m.interocular = io;
    if (io > 0.01) {
      m.yawProxy = (nose.x - midX) / io;
      m.pitchProxy = (nose.y - midY) / io;
    }

    // 注視：顔があり、正面向きで、目が開いている
    m.attention =
      Math.abs(m.yawProxy ?? 1) < 0.25 && m.eyeOpen > 0.3 ? 1 : 0;

    // 前のめり度：基準値に対する目間距離の比（近づくと拡大する）
    if (baseline?.interocular && io > 0.01) {
      m.leanIn = io / baseline.interocular - 1;
    }

    // 追加の表情AU（詳細指標）
    m.cheekSquint = (bs.cheekSquintLeft + bs.cheekSquintRight) / 2;
    m.duchenne = Math.min(m.smile, m.cheekSquint * 1.6); // 目元も動く「本物の笑顔」
    m.lipPress = (bs.mouthPressLeft + bs.mouthPressRight) / 2;
    m.noseSneer = (bs.noseSneerLeft + bs.noseSneerRight) / 2;
    m.mouthPucker = bs.mouthPucker;

    // 視線：虹彩の眼内位置＋頭の向き（キャリブレーション不要の粗い推定）
    // 虹彩中心: 468=右目, 473=左目 / 目頭・目尻: 33-133（右）, 362-263（左）
    // 上下まぶた: 159-145（右）, 386-374（左）
    if (m.eyeOpen > 0.35) {
      const ratio = (v, a, b) => (Math.abs(b - a) > 1e-6 ? (v - a) / (b - a) : 0.5);
      const hx = (ratio(lm[468].x, lm[33].x, lm[133].x) +
                  ratio(lm[473].x, lm[362].x, lm[263].x)) / 2;
      const vy = (ratio(lm[468].y, lm[159].y, lm[145].y) +
                  ratio(lm[473].y, lm[386].y, lm[374].y)) / 2;
      const eyeX = (hx - 0.5) * 2; // 正 = 画像右 = 参加者の左方向
      const eyeY = (vy - 0.5) * 2; // 正 = 下方向
      const pitchDev = baseline?.pitchProxy != null && m.pitchProxy != null
        ? m.pitchProxy - baseline.pitchProxy : 0;
      // 参加者視点で 正 = 右 になるよう水平方向は符号を反転
      m.gazeX = clamp(-(2.2 * eyeX + 1.2 * (m.yawProxy ?? 0)), -1, 1);
      m.gazeY = clamp(2.2 * eyeY + 1.5 * pitchDev, -1, 1);
      m.gazeZone = gazeZoneName(m.gazeX, m.gazeY);
    }

    // 頭の動き量（正規化座標/秒）：そわそわ・落ち着きのなさの近似
    if (prevNose && now > prevNose.t) {
      const d = Math.hypot(nose.x - prevNose.x, nose.y - prevNose.y);
      m.headMove = d / ((now - prevNose.t) / 1000);
    }
    prevNose = { x: nose.x, y: nose.y, t: now };

    // うなずき（pitch振動）・首振り（yaw振動）
    if (nodDetector.update(m.pitchProxy, now)) nodTotal++;
    if (shakeDetector.update(m.yawProxy, now)) shakeTotal++;
  } else {
    prevNose = null;
  }

  const plm = poseResult?.landmarks?.[0];
  if (plm && plm[11] && plm[12]) {
    const ls = plm[11], rs = plm[12], nose = plm[0];
    if ((ls.visibility ?? 1) > 0.5 && (rs.visibility ?? 1) > 0.5) {
      m.poseDetected = true;
      const dx = Math.abs(rs.x - ls.x), dy = Math.abs(rs.y - ls.y);
      m.shoulderTiltDeg = (Math.atan2(dy, Math.max(dx, 1e-6)) * 180) / Math.PI;
      const shoulderW = Math.max(Math.hypot(rs.x - ls.x, rs.y - ls.y), 1e-6);
      const midSX = (ls.x + rs.x) / 2, midSY = (ls.y + rs.y) / 2;
      m.headOffset = (nose.x - midSX) / shoulderW;
      m.neckRatio = (midSY - nose.y) / shoulderW; // 小さいほど頭が肩に埋もれている（猫背気味）

      let score = 100;
      score -= Math.min(40, m.shoulderTiltDeg * 2.5);
      score -= Math.min(30, Math.abs(m.headOffset) * 120);
      if (baseline?.neckRatio) {
        const slouch = (baseline.neckRatio - m.neckRatio) / baseline.neckRatio;
        if (slouch > 0.12) score -= Math.min(30, (slouch - 0.12) * 200);
      }
      m.postureScore = Math.max(0, Math.round(score));

      // 顔タッチ：手首（15=左, 16=右）が顔の近くにある（思考・不安のサイン）
      const near = (w) => w && (w.visibility ?? 1) > 0.5 &&
        Math.hypot(w.x - nose.x, w.y - nose.y) < shoulderW * 0.65;
      m.faceTouch = near(plm[15]) || near(plm[16]) ? 1 : 0;
    }
  }

  // エンゲージメント合成指標（0-1）：注視・感情の強さ・前のめり・眉上げの加重和
  if (m.faceDetected) {
    const lean = Math.max(0, Math.min(1, (m.leanIn ?? 0) * 5));
    m.engagement =
      0.45 * m.attention +
      0.2 * Math.min(1, Math.abs(m.valence) * 2) +
      0.2 * lean +
      0.15 * Math.min(1, m.browRaise * 2);
  }
  return m;
}

// 視線ゾーン名（参加者視点。x: left/center/right, y: up/middle/down）
function gazeZoneName(x, y) {
  const col = x < -0.33 ? "left" : x > 0.33 ? "right" : "center";
  const row = y < -0.3 ? "up" : y > 0.3 ? "down" : "middle";
  return `${col}-${row}`;
}

const ZONE_JA = { left: "左", center: "", right: "右", up: "上", middle: "", down: "下" };
function zoneLabelJa(zone) {
  if (!zone) return "–";
  const [col, row] = zone.split("-");
  return (ZONE_JA[col] ?? "") + (ZONE_JA[row] ?? "") || "中央";
}

// ---------- 基準値（キャリブレーション） ----------
function updateCalibration(m, now) {
  if (!calibrating) return;
  if (m.faceDetected) {
    calibBuf.push({ interocular: m.interocular, neckRatio: m.neckRatio, pitchProxy: m.pitchProxy });
  }
  if (now - recStartMs >= CALIB_MS) {
    calibrating = false;
    els.calibBadge.classList.add("hidden");
    const median = (arr) => {
      const v = arr.filter((x) => x != null).sort((a, b) => a - b);
      return v.length ? v[Math.floor(v.length / 2)] : null;
    };
    baseline = {
      interocular: median(calibBuf.map((s) => s.interocular)),
      neckRatio: median(calibBuf.map((s) => s.neckRatio)),
      pitchProxy: median(calibBuf.map((s) => s.pitchProxy)),
    };
    calibBuf = [];
  }
}

// ---------- UI 更新 ----------
const fmt = (v, d = 2) => (v == null ? "–" : v.toFixed(d));

function updateUI(m, now) {
  const smile = ema("smile", m.smile);
  const furrow = ema("furrow", m.furrow);
  const raise = ema("raise", m.browRaise);
  const eye = ema("eye", m.eyeOpen);
  const valence = ema("valence", m.valence);
  const engage = ema("engage", m.engagement);
  const lean = ema("lean", m.leanIn);

  setBar(els.barSmile, els.valSmile, smile);
  setBar(els.barFurrow, els.valFurrow, furrow);
  setBar(els.barRaise, els.valRaise, raise);
  setBar(els.barEye, els.valEye, eye);

  // 感情価バー（中央から左右に伸びる）
  if (valence == null) {
    els.valenceFill.style.width = "0";
    els.valValence.textContent = "–";
  } else {
    const pct = Math.min(50, Math.abs(valence) * 50);
    els.valenceFill.style.width = pct + "%";
    if (valence >= 0) {
      els.valenceFill.style.left = "50%";
      els.valenceFill.style.background = "var(--green)";
    } else {
      els.valenceFill.style.left = 50 - pct + "%";
      els.valenceFill.style.background = "var(--red)";
    }
    els.valValence.textContent = (valence >= 0 ? "+" : "") + valence.toFixed(2);
  }

  els.valEngage.textContent = engage == null ? "–" : Math.round(engage * 100);
  els.valAttention.textContent = m.faceDetected ? (m.attention ? "◎" : "✕") : "–";
  els.valAttention.style.color = m.attention ? "var(--green)" : "var(--muted)";
  els.valPosture.textContent = m.postureScore == null ? "–" : m.postureScore;
  els.valBlink.textContent = blinkTotal;
  els.valLean.textContent =
    lean == null ? "–" : (lean >= 0 ? "+" : "") + (lean * 100).toFixed(0) + "%";
  els.valGazeZone.textContent = m.faceDetected ? zoneLabelJa(m.gazeZone) : "–";

  // 詳細指標（表示中のみ更新して負荷を抑える）
  if (els.chkDetails.checked) {
    setBar(els.barDuchenne, els.valDuchenne, ema("duchenne", m.duchenne));
    setBar(els.barLipPress, els.valLipPress, ema("lipPress", m.lipPress));
    setBar(els.barSneer, els.valSneer, ema("sneer", m.noseSneer));
    setBar(els.barPucker, els.valPucker, ema("pucker", m.mouthPucker));
    const move = ema("move", m.headMove, 0.15);
    setBar(els.barMove, els.valMove, move == null ? null : Math.min(1, move / 0.15));
    els.valNod.textContent = nodTotal;
    els.valShake.textContent = shakeTotal;
    els.valTouch.textContent = m.faceTouch == null ? "–" : m.faceTouch ? "✋ 触れている" : "なし";
    updateGazeGrid(m.gazeZone);
  }

  // チャート
  chartHistory.push({ t: now, valence, engage });
  while (chartHistory.length && now - chartHistory[0].t > CHART_WINDOW_MS) {
    chartHistory.shift();
  }
  drawChart(now);
}

// 視線マップ：現在ゾーンを枠で、滞留割合を青の濃さで表す
function updateGazeGrid(zone) {
  const total = Object.values(gazeDwell).reduce((a, b) => a + b, 0);
  for (const cell of els.gazeGrid.children) {
    const z = cell.dataset.zone;
    cell.classList.toggle("active", z === zone);
    const share = total ? (gazeDwell[z] || 0) / total : 0;
    cell.style.background = share > 0 ? `rgba(79,140,255,${Math.min(0.9, share * 1.8).toFixed(3)})` : "";
  }
}

function setBar(barEl, valEl, v) {
  barEl.style.width = v == null ? "0%" : Math.min(100, v * 100) + "%";
  valEl.textContent = fmt(v);
}

function drawChart(now) {
  const c = els.chart;
  const ctx = c.getContext("2d");
  const w = (c.width = c.clientWidth * (window.devicePixelRatio || 1));
  const h = c.height;
  ctx.clearRect(0, 0, w, h);

  // ゼロライン
  ctx.strokeStyle = "rgba(147,161,189,.3)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, h / 2);
  ctx.lineTo(w, h / 2);
  ctx.stroke();

  const x = (t) => w - ((now - t) / CHART_WINDOW_MS) * w;
  const drawLine = (key, color, yFn) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    let started = false;
    for (const p of chartHistory) {
      if (p[key] == null) { started = false; continue; }
      const px = x(p.t), py = yFn(p[key]);
      if (!started) { ctx.moveTo(px, py); started = true; }
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  };
  // 感情価は -1..1 を上下に、エンゲージメントは 0..1 を全高に
  drawLine("valence", "#35c76f", (v) => h / 2 - v * (h / 2 - 4));
  drawLine("engage", "#4fb6ff", (v) => h - 4 - v * (h - 8));
}

function drawOverlay(faceResult, poseResult) {
  const ctx = els.overlay.getContext("2d");
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  if (!els.chkOverlay.checked) return;
  const W = els.overlay.width, H = els.overlay.height;

  const lm = faceResult?.faceLandmarks?.[0];
  if (lm) {
    ctx.fillStyle = "rgba(79,182,255,.55)";
    for (let i = 0; i < lm.length; i += 4) {
      ctx.fillRect(lm[i].x * W - 1, lm[i].y * H - 1, 2, 2);
    }
  }
  const plm = poseResult?.landmarks?.[0];
  if (plm && plm[11] && plm[12]) {
    ctx.strokeStyle = "rgba(53,199,111,.8)";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(plm[11].x * W, plm[11].y * H);
    ctx.lineTo(plm[12].x * W, plm[12].y * H);
    ctx.stroke();
    ctx.fillStyle = "rgba(53,199,111,.9)";
    ctx.beginPath();
    ctx.arc(plm[0].x * W, plm[0].y * H, 5, 0, Math.PI * 2);
    ctx.fill();
  }
}

// ---------- 記録 ----------
function startRecording() {
  samples = [];
  blinkTotal = 0;
  smileEvents = 0;
  blinkClosed = false;
  smileOn = false;
  nodTotal = 0;
  shakeTotal = 0;
  gazeDwell = {};
  prevNose = null;
  baseline = null;
  calibBuf = [];
  calibrating = true;
  recStartMs = performance.now();
  lastSampleAt = 0;
  recording = true;
  pendingMarker = "";
  markerSeq = 0;
  els.recBadge.classList.remove("hidden");
  els.calibBadge.classList.remove("hidden");
  els.btnRecord.textContent = "⏹ 記録停止";
  els.btnRecord.classList.add("recording");
  els.btnRecord.classList.remove("primary");
  els.btnMark.disabled = false;
  els.btnCsv.disabled = true;
  els.btnSummaryCsv.disabled = true;
  els.diagnosisPanel.classList.add("hidden");
  els.selfreportPanel.classList.add("hidden");
  setStatus("記録中：最初の2秒間は基準値の計測です。参加者は自然な姿勢でいてください。");
}

function stopRecording() {
  recording = false;
  calibrating = false;
  els.recBadge.classList.add("hidden");
  els.calibBadge.classList.add("hidden");
  els.btnRecord.textContent = "⏺ 記録開始";
  els.btnRecord.classList.remove("recording");
  els.btnRecord.classList.add("primary");
  els.btnMark.disabled = true;
  if (samples.length > 0) {
    els.btnCsv.disabled = false;
    els.btnSummaryCsv.disabled = false;
    showDiagnosis();
    lastSummary = summarize();
    if (lastSummary) els.selfreportPanel.classList.remove("hidden");
    setStatus(`記録停止（${samples.length} サンプル）。CSVをダウンロードできます。`);
  } else {
    setStatus("記録停止（サンプルなし）");
  }
}

function updateRecTimer(now) {
  const s = Math.floor((now - recStartMs) / 1000);
  els.recTime.textContent =
    String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
}

function pushSample(m, now) {
  samples.push({
    participant_id: els.inpParticipant.value || "",
    stimulus_label: els.inpStimulus.value || "",
    iso_time: new Date().toISOString(),
    elapsed_ms: Math.round(now - recStartMs),
    phase: calibrating ? "calibration" : "main",
    marker: pendingMarker,
    face_detected: m.faceDetected ? 1 : 0,
    smile: r3(m.smile),
    brow_furrow: r3(m.furrow),
    brow_raise: r3(m.browRaise),
    eye_open: r3(m.eyeOpen),
    jaw_open: r3(m.jawOpen),
    valence: r3(m.valence),
    yaw_proxy: r3(m.yawProxy),
    pitch_proxy: r3(m.pitchProxy),
    attention: m.attention,
    gaze_x: r3(m.gazeX),
    gaze_y: r3(m.gazeY),
    gaze_zone: m.gazeZone || "",
    blink_total: blinkTotal,
    smile_events: smileEvents,
    duchenne: r3(m.duchenne),
    cheek_squint: r3(m.cheekSquint),
    lip_press: r3(m.lipPress),
    nose_sneer: r3(m.noseSneer),
    mouth_pucker: r3(m.mouthPucker),
    head_move: r3(m.headMove),
    nod_total: nodTotal,
    shake_total: shakeTotal,
    lean_in: r3(m.leanIn),
    pose_detected: m.poseDetected ? 1 : 0,
    shoulder_tilt_deg: r3(m.shoulderTiltDeg),
    head_offset: r3(m.headOffset),
    neck_ratio: r3(m.neckRatio),
    posture_score: m.postureScore ?? "",
    face_touch: m.faceTouch ?? "",
    engagement: r3(m.engagement),
  });
  if (!calibrating && m.gazeZone) gazeDwell[m.gazeZone] = (gazeDwell[m.gazeZone] || 0) + 1;
  pendingMarker = "";
}

const r3 = (v) => (v == null ? "" : Math.round(v * 1000) / 1000);

function addMarker() {
  markerSeq++;
  const label = els.inpStimulus.value
    ? `${els.inpStimulus.value}_M${markerSeq}`
    : `M${markerSeq}`;
  pendingMarker = label;
  setStatus(`マーク「${label}」を記録しました`);
}

// ---------- CSV ----------
function downloadCsv(filename, rows) {
  if (!rows.length) return;
  const headers = [...new Set(rows.flatMap((r) => Object.keys(r)))];
  const esc = (v) => {
    const s = String(v ?? "");
    return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s;
  };
  const csv =
    "\uFEFF" + // BOM: Excel で文字化けさせないため
    headers.join(",") +
    "\n" +
    rows.map((r) => headers.map((h) => esc(r[h])).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  a.click();
  URL.revokeObjectURL(a.href);
}

function fileStem() {
  const p = els.inpParticipant.value || "anon";
  const s = els.inpStimulus.value || "stim";
  const t = new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19);
  return `reaction_${p}_${s}_${t}`;
}

// ---------- サマリー・診断 ----------
function summarize() {
  const main = samples.filter((s) => s.phase === "main" && s.face_detected === 1);
  const all = samples.filter((s) => s.phase === "main");
  if (!main.length) return null;

  const mean = (key) => {
    const v = main.map((s) => s[key]).filter((x) => typeof x === "number");
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
  };
  const max = (key) => {
    const v = main.map((s) => s[key]).filter((x) => typeof x === "number");
    return v.length ? Math.max(...v) : null;
  };
  const durationMin = (all.at(-1).elapsed_ms - all[0].elapsed_ms) / 60000 || 1 / 60;
  const last = samples.at(-1);

  // 視線ゾーンの最頻値と中央滞留率
  const zoneCounts = {};
  for (const s of main) if (s.gaze_zone) zoneCounts[s.gaze_zone] = (zoneCounts[s.gaze_zone] || 0) + 1;
  const zoneTotal = Object.values(zoneCounts).reduce((a, b) => a + b, 0);
  const gazeTop = Object.entries(zoneCounts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? "";

  // 感情価の変動（表情の起伏の大きさ）
  const vVals = main.map((s) => s.valence).filter((x) => typeof x === "number");
  const vMean = vVals.length ? vVals.reduce((a, b) => a + b, 0) / vVals.length : 0;
  const valenceSd = vVals.length
    ? Math.sqrt(vVals.reduce((a, b) => a + (b - vMean) ** 2, 0) / vVals.length)
    : null;

  return {
    participant_id: last.participant_id,
    stimulus_label: last.stimulus_label,
    duration_sec: Math.round((all.at(-1).elapsed_ms - all[0].elapsed_ms) / 1000),
    n_samples: main.length,
    face_detected_ratio: r3(main.length / all.length),
    valence_mean: r3(mean("valence")),
    valence_positive_ratio: r3(main.filter((s) => s.valence > 0.1).length / main.length),
    smile_mean: r3(mean("smile")),
    smile_peak: r3(max("smile")),
    smile_events: last.smile_events,
    brow_furrow_mean: r3(mean("brow_furrow")),
    attention_ratio: r3(mean("attention")),
    gaze_center_ratio: r3(zoneTotal ? (zoneCounts["center-middle"] || 0) / zoneTotal : null),
    gaze_zone_top: gazeTop,
    blink_per_min: r3(last.blink_total / durationMin),
    duchenne_mean: r3(mean("duchenne")),
    lip_press_mean: r3(mean("lip_press")),
    nose_sneer_mean: r3(mean("nose_sneer")),
    valence_sd: r3(valenceSd),
    nod_count: last.nod_total,
    shake_count: last.shake_total,
    face_touch_ratio: r3(mean("face_touch")),
    head_move_mean: r3(mean("head_move")),
    lean_in_mean: r3(mean("lean_in")),
    posture_score_mean: r3(mean("posture_score")),
    engagement_mean: r3(mean("engagement")),
  };
}

function showDiagnosis() {
  const s = summarize();
  if (!s) {
    els.diagnosisBody.innerHTML =
      "<p>顔が検出されたサンプルがなく、診断できませんでした。照明とカメラ位置を確認してください。</p>";
    els.diagnosisPanel.classList.remove("hidden");
    return;
  }

  const notes = [];
  // 感情価
  if (s.valence_mean >= 0.15) notes.push("😊 <strong>ポジティブ反応が優勢</strong>：笑顔が多く、好意的な反応の可能性があります。");
  else if (s.valence_mean <= -0.1) notes.push("😟 <strong>ネガティブ反応が優勢</strong>：眉間のしわが目立ちました。困惑・不快、または内容が難しかった可能性があります。");
  else notes.push("😐 表情反応は中立的でした。感情を強く動かす要素が少なかった可能性があります。");
  // 笑顔イベント
  if (s.smile_events >= 3) notes.push(`😄 明確な笑顔が ${s.smile_events} 回検出されました。どの時点かは時系列CSVの smile 列で確認できます。`);
  // 本物の笑顔（Duchenne）
  if (s.duchenne_mean >= 0.12) notes.push("😊 目元も動く<strong>「本物の笑顔」</strong>が見られました。表面的でない好意反応の可能性があります。");
  else if (s.smile_mean >= 0.25 && s.duchenne_mean != null && s.duchenne_mean < 0.05) notes.push("🙂 笑顔はありましたが目元の動きが小さく、<strong>社交的な笑顔</strong>（気遣い）の可能性があります。自己報告と照合してください。");
  // 嫌悪
  if (s.nose_sneer_mean >= 0.08) notes.push("🤢 <strong>嫌悪に関連する表情</strong>（鼻のしわ）が検出されました。デザイン・表現への生理的な忌避感の可能性があります。");
  // 注視
  if (s.attention_ratio >= 0.8) notes.push("👀 <strong>注視率が高い</strong>（" + Math.round(s.attention_ratio * 100) + "%）：画面/対象への関心が持続していました。");
  else if (s.attention_ratio < 0.5) notes.push("👀 <strong>注視率が低い</strong>（" + Math.round(s.attention_ratio * 100) + "%）：視線が外れがちでした。関心低下、または提示方法の問題の可能性があります。");
  // 視線
  if (s.gaze_center_ratio != null) {
    if (s.gaze_center_ratio >= 0.7) notes.push(`👁 <strong>視線が中央に集中</strong>（${Math.round(s.gaze_center_ratio * 100)}%）：対象をしっかり見ていました。`);
    else if (s.gaze_center_ratio < 0.4 && s.gaze_zone_top) notes.push(`👁 視線が中央から外れがちでした（最も長く見ていたのは「${zoneLabelJa(s.gaze_zone_top)}」）。注意が逸れたか、周辺の情報を見ていた可能性があります。`);
  }
  // うなずき・首振り
  if (s.nod_count >= 2) notes.push(`🙆 うなずきが ${s.nod_count} 回検出されました。同意・納得のサインの可能性があります。`);
  if (s.shake_count >= 2) notes.push(`🙅 首振りが ${s.shake_count} 回検出されました。否定・違和感のサインの可能性があります。`);
  // 顔タッチ・そわそわ
  if (s.face_touch_ratio > 0.15) notes.push(`✋ 顔に手を触れる時間が長めでした（${Math.round(s.face_touch_ratio * 100)}%）。思考・迷い・不安のサインとされます。`);
  if (s.head_move_mean > 0.08) notes.push("🌀 頭の動きが多く、<strong>落ち着きのなさ</strong>が見られました。退屈・違和感の可能性があります。");
  // 表情の起伏
  if (s.valence_sd >= 0.15) notes.push("🎢 表情の起伏が大きく、感情を動かす場面があったようです。時系列CSVの valence 列で山谷の時点を確認できます。");
  // まばたき
  if (s.blink_per_min > 30) notes.push(`👁 まばたきが多め（${Math.round(s.blink_per_min)}回/分）：緊張・認知負荷が高かった可能性があります。`);
  else if (s.blink_per_min < 10) notes.push(`👁 まばたきが少なめ（${Math.round(s.blink_per_min)}回/分）：対象への集中・没入を示す可能性があります。`);
  // 前のめり
  if (s.lean_in_mean > 0.05) notes.push("🪑 <strong>前のめり傾向</strong>：開始時より画面に近づいており、関心の高さを示す可能性があります。");
  else if (s.lean_in_mean < -0.05) notes.push("🪑 <strong>のけぞり傾向</strong>：開始時より画面から離れており、退屈・回避の可能性があります。");
  // 姿勢
  if (s.posture_score_mean != null) {
    if (s.posture_score_mean >= 75) notes.push(`🧍 姿勢は良好でした（平均スコア ${Math.round(s.posture_score_mean)}）。`);
    else if (s.posture_score_mean < 55) notes.push(`🧍 姿勢の崩れ（傾き・猫背傾向）が見られました（平均スコア ${Math.round(s.posture_score_mean)}）。疲労や集中低下のサインの可能性があります。`);
  }
  // データ品質
  if (s.face_detected_ratio < 0.7) notes.push(`⚠️ 顔検出率が ${Math.round(s.face_detected_ratio * 100)}% と低めです。照明・カメラ角度を改善すると精度が上がります。`);

  const rows = [
    ["記録時間", s.duration_sec + " 秒"],
    ["感情価（平均）", s.valence_mean],
    ["ポジティブ時間率", Math.round(s.valence_positive_ratio * 100) + "%"],
    ["注視率", Math.round(s.attention_ratio * 100) + "%"],
    ["視線中央率", s.gaze_center_ratio != null ? Math.round(s.gaze_center_ratio * 100) + "%" : "–"],
    ["うなずき/首振り", `${s.nod_count} / ${s.shake_count}`],
    ["まばたき/分", s.blink_per_min],
    ["エンゲージメント（平均）", s.engagement_mean != null ? Math.round(s.engagement_mean * 100) : "–"],
    ["姿勢スコア（平均）", s.posture_score_mean != null ? Math.round(s.posture_score_mean) : "–"],
    ["顔検出率", Math.round(s.face_detected_ratio * 100) + "%"],
  ];

  els.diagnosisBody.innerHTML =
    "<ul>" + notes.map((n) => `<li>${n}</li>`).join("") + "</ul>" +
    '<table class="diagnosis-summary-table"><tr>' +
    rows.map((r) => `<th>${r[0]}</th>`).join("") + "</tr><tr>" +
    rows.map((r) => `<td>${r[1]}</td>`).join("") + "</tr></table>";
  els.diagnosisPanel.classList.remove("hidden");
}

// ---------- セッション保存と自己報告尺度との関連 ----------
const SESSIONS_KEY = "reactionMeterSessions";
let sessions = [];
try {
  sessions = JSON.parse(localStorage.getItem(SESSIONS_KEY) || "[]");
} catch { sessions = []; }

// 相関のX軸に使える身体反応の要約指標
const CORR_X_OPTIONS = [
  ["engagement_mean", "エンゲージメント（平均）"],
  ["valence_mean", "感情価（平均）"],
  ["valence_sd", "表情の起伏（感情価SD）"],
  ["smile_mean", "笑顔（平均）"],
  ["duchenne_mean", "本物の笑顔（平均）"],
  ["smile_events", "笑顔回数"],
  ["brow_furrow_mean", "眉間のしわ（平均）"],
  ["nose_sneer_mean", "嫌悪表情（平均）"],
  ["attention_ratio", "注視率"],
  ["gaze_center_ratio", "視線中央率"],
  ["blink_per_min", "まばたき/分"],
  ["lean_in_mean", "前のめり度（平均）"],
  ["posture_score_mean", "姿勢スコア（平均）"],
  ["head_move_mean", "落ち着きのなさ（平均）"],
  ["nod_count", "うなずき回数"],
  ["shake_count", "首振り回数"],
  ["face_touch_ratio", "顔タッチ率"],
];
const CORR_Y_OPTIONS = [
  ["selfreport_automaticity", "自動性"],
  ["selfreport_activation", "活性"],
  ["selfreport_stability", "安定"],
];

function currentSelfreport() {
  const num = (el) => (el.value === "" ? null : Number(el.value));
  return {
    selfreport_automaticity: num(els.inpAuto),
    selfreport_activation: num(els.inpActive),
    selfreport_stability: num(els.inpStable),
  };
}

function saveSession() {
  if (!lastSummary) return;
  const sr = currentSelfreport();
  if (Object.values(sr).every((v) => v == null) &&
      !confirm("尺度得点が未入力です。保存後でもセッション一覧の表に直接入力できます。このまま保存しますか？")) {
    return;
  }
  sessions.push({ ...lastSummary, ...sr, saved_at: new Date().toISOString() });
  persistSessions();
  els.inpAuto.value = els.inpActive.value = els.inpStable.value = "";
  els.selfreportPanel.classList.add("hidden");
  lastSummary = null;
  setStatus(`セッションを保存しました（計 ${sessions.length} 件）`);
  renderSessions();
}

function persistSessions() {
  try {
    localStorage.setItem(SESSIONS_KEY, JSON.stringify(sessions));
  } catch (err) {
    setStatus(`セッションの保存に失敗しました: ${err.message}`, true);
  }
}

function deleteSession(i) {
  sessions.splice(i, 1);
  persistSessions();
  renderSessions();
}

function renderSessions() {
  els.sessionsPanel.classList.toggle("hidden", sessions.length === 0);
  els.sessionCount.textContent = sessions.length;
  if (!sessions.length) return;

  const fmtN = (v) => (typeof v === "number" ? v : "–");
  // 尺度得点は保存後でも直接編集できる入力欄にする
  const scoreCell = (i, key, v) =>
    `<td><input type="number" step="0.1" class="score-input" data-i="${i}" data-key="${key}"` +
    ` value="${typeof v === "number" ? v : ""}" placeholder="–"></td>`;
  els.sessionsTable.innerHTML =
    "<tr><th>参加者</th><th>刺激</th><th>日時</th><th>ｴﾝｹﾞｰｼﾞ</th><th>感情価</th><th>注視率</th><th>自動性</th><th>活性</th><th>安定</th><th></th></tr>" +
    sessions.map((s, i) =>
      `<tr><td>${escapeHtml(s.participant_id) || "–"}</td>` +
      `<td>${escapeHtml(s.stimulus_label) || "–"}</td>` +
      `<td>${(s.saved_at || "").slice(5, 16).replace("T", " ")}</td>` +
      `<td>${fmtN(s.engagement_mean)}</td><td>${fmtN(s.valence_mean)}</td><td>${fmtN(s.attention_ratio)}</td>` +
      scoreCell(i, "selfreport_automaticity", s.selfreport_automaticity) +
      scoreCell(i, "selfreport_activation", s.selfreport_activation) +
      scoreCell(i, "selfreport_stability", s.selfreport_stability) +
      `<td><button class="del-btn" data-i="${i}" title="このセッションを削除">✕</button></td></tr>`
    ).join("");
  renderCorrelation();
}

const escapeHtml = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function pearson(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let cov = 0, vx = 0, vy = 0;
  for (let i = 0; i < n; i++) {
    cov += (xs[i] - mx) * (ys[i] - my);
    vx += (xs[i] - mx) ** 2;
    vy += (ys[i] - my) ** 2;
  }
  return vx > 0 && vy > 0 ? cov / Math.sqrt(vx * vy) : null;
}

function renderCorrelation() {
  const xKey = els.selX.value, yKey = els.selY.value;
  const pairs = sessions
    .map((s) => [s[xKey], s[yKey]])
    .filter(([x, y]) => typeof x === "number" && typeof y === "number");
  drawScatter(pairs);

  if (pairs.length < 3) {
    els.corrResult.textContent = `有効なデータが ${pairs.length} 件です。相関を見るには自己報告入りのセッションが3件以上必要です。`;
    return;
  }
  const r = pearson(pairs.map((p) => p[0]), pairs.map((p) => p[1]));
  if (r == null) {
    els.corrResult.textContent = "値にばらつきがなく、相関を計算できません。";
    return;
  }
  const strength = Math.abs(r) >= 0.5 ? "強い" : Math.abs(r) >= 0.3 ? "中程度の" : Math.abs(r) >= 0.1 ? "弱い" : "ほぼ無";
  const dir = r > 0 ? "正の" : "負の";
  const caveat = pairs.length < 10 ? "（nが小さいため参考値）" : "";
  els.corrResult.innerHTML =
    `<strong>r = ${r.toFixed(2)}</strong>（n = ${pairs.length}）: ${strength}${Math.abs(r) >= 0.1 ? dir : ""}相関 ${caveat}`;
}

function drawScatter(pairs) {
  const c = els.scatter;
  const ctx = c.getContext("2d");
  const w = c.width, h = c.height;
  ctx.clearRect(0, 0, w, h);
  ctx.font = "11px sans-serif";
  ctx.fillStyle = "rgba(147,161,189,.9)";
  if (pairs.length < 2) {
    ctx.fillText("セッションが増えるとここに散布図が表示されます", 20, h / 2);
    return;
  }
  const pad = 36;
  const xs = pairs.map((p) => p[0]), ys = pairs.map((p) => p[1]);
  const pad10 = (min, max) => {
    const d = (max - min) || 1;
    return [min - d * 0.1, max + d * 0.1];
  };
  const [x0, x1] = pad10(Math.min(...xs), Math.max(...xs));
  const [y0, y1] = pad10(Math.min(...ys), Math.max(...ys));
  const px = (x) => pad + ((x - x0) / (x1 - x0)) * (w - pad * 2);
  const py = (y) => h - pad - ((y - y0) / (y1 - y0)) * (h - pad * 2);

  // 軸
  ctx.strokeStyle = "rgba(147,161,189,.4)";
  ctx.lineWidth = 1;
  ctx.strokeRect(pad, pad, w - pad * 2, h - pad * 2);
  ctx.fillText(x0.toFixed(2), pad, h - pad + 14);
  ctx.textAlign = "right";
  ctx.fillText(x1.toFixed(2), w - pad, h - pad + 14);
  ctx.fillText(y0.toFixed(1), pad - 4, h - pad);
  ctx.fillText(y1.toFixed(1), pad - 4, pad + 8);
  ctx.textAlign = "left";

  // 回帰直線（最小二乗）
  const r = pearson(xs, ys);
  if (r != null && pairs.length >= 3) {
    const mx = xs.reduce((a, b) => a + b, 0) / xs.length;
    const my = ys.reduce((a, b) => a + b, 0) / ys.length;
    const sx = Math.sqrt(xs.reduce((a, b) => a + (b - mx) ** 2, 0) / xs.length);
    const sy = Math.sqrt(ys.reduce((a, b) => a + (b - my) ** 2, 0) / ys.length);
    if (sx > 0) {
      const slope = (r * sy) / sx;
      ctx.strokeStyle = "rgba(53,199,111,.5)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px(x0), py(my + slope * (x0 - mx)));
      ctx.lineTo(px(x1), py(my + slope * (x1 - mx)));
      ctx.stroke();
    }
  }

  // 点
  ctx.fillStyle = "rgba(79,140,255,.85)";
  for (const [x, y] of pairs) {
    ctx.beginPath();
    ctx.arc(px(x), py(y), 5, 0, Math.PI * 2);
    ctx.fill();
  }
}

function initCorrControls() {
  els.selX.innerHTML = CORR_X_OPTIONS
    .map(([k, label]) => `<option value="${k}">${label}</option>`).join("");
  els.selY.innerHTML = CORR_Y_OPTIONS
    .map(([k, label]) => `<option value="${k}">${label}</option>`).join("");
}

// ---------- イベント ----------
els.btnCamera.addEventListener("click", () => (running ? stopCamera() : startCamera()));
els.chkDetails.addEventListener("change", () =>
  els.detailsSection.classList.toggle("hidden", !els.chkDetails.checked));
els.btnFacing.addEventListener("click", switchFacing);
els.chkMirror.addEventListener("change", applyMirror);
els.btnRecord.addEventListener("click", () => (recording ? stopRecording() : startRecording()));
els.btnMark.addEventListener("click", addMarker);
els.btnCsv.addEventListener("click", () => downloadCsv(fileStem() + "_timeseries.csv", samples));
els.btnSummaryCsv.addEventListener("click", () => {
  const s = summarize();
  if (s) downloadCsv(fileStem() + "_summary.csv", [{ ...s, ...currentSelfreport() }]);
});
els.btnSaveSession.addEventListener("click", saveSession);
els.sessionsTable.addEventListener("click", (e) => {
  const btn = e.target.closest(".del-btn");
  if (btn) deleteSession(Number(btn.dataset.i));
});
// 一覧の尺度得点セルを編集したらその場で保存して相関を更新
els.sessionsTable.addEventListener("change", (e) => {
  const inp = e.target.closest(".score-input");
  if (!inp || !sessions[Number(inp.dataset.i)]) return;
  sessions[Number(inp.dataset.i)][inp.dataset.key] =
    inp.value === "" ? null : Number(inp.value);
  persistSessions();
  renderCorrelation();
  setStatus("尺度得点を更新しました");
});
els.btnSessionsCsv.addEventListener("click", () =>
  downloadCsv(`reaction_sessions_${new Date().toISOString().slice(0, 10)}.csv`, sessions));
els.btnClearSessions.addEventListener("click", () => {
  if (confirm(`保存済みの ${sessions.length} セッションをすべて削除します。よろしいですか？`)) {
    sessions = [];
    persistSessions();
    renderSessions();
  }
});
els.selX.addEventListener("change", renderCorrelation);
els.selY.addEventListener("change", renderCorrelation);

initCorrControls();
renderSessions();
applyMirror();
if (!navigator.mediaDevices?.getUserMedia) {
  setStatus("このブラウザはカメラAPIに対応していません。HTTPS（または localhost）でアクセスしているか確認してください。", true);
}
