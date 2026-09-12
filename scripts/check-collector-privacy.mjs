#!/usr/bin/env node
// Source-level guard: first-run must not enable outbound send.
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const app = readFileSync(join(root, "app.js"), "utf8");
const html = readFileSync(join(root, "index.html"), "utf8");

const fail = (msg) => {
  console.error(`FAIL: ${msg}`);
  process.exitCode = 1;
};

const block = app.match(/const DEFAULT_COLLECTOR = \{[\s\S]*?\n\};/);
if (!block) {
  fail("DEFAULT_COLLECTOR block not found");
} else {
  const url = block[0].match(/url:\s*"([^"]*)"/);
  const autoSend = block[0].match(/autoSend:\s*(true|false)/);
  const sendTs = block[0].match(/sendTimeseries:\s*(true|false)/);
  if (!url || url[1] !== "") fail(`DEFAULT_COLLECTOR.url must be empty, got ${url && url[1]}`);
  if (!autoSend || autoSend[1] !== "false") fail(`DEFAULT_COLLECTOR.autoSend must be false`);
  if (!sendTs || sendTs[1] !== "false") fail(`DEFAULT_COLLECTOR.sendTimeseries must be false`);
}

if (/id="chkAutoSend"[^>]*\schecked/.test(html)) {
  fail("chkAutoSend must not be checked in HTML");
}
if (/id="chkSendTs"[^>]*\schecked/.test(html)) {
  fail("chkSendTs must not be checked in HTML");
}
if (!html.includes("データ回収は未設定です")) {
  fail("consent HTML must say collection is unset by default");
}
if (html.includes("データ回収が設定されている場合、上記の数値データのみが研究者へ送信されます")) {
  fail("consent HTML still uses the old optional/config-dependent wording");
}

if (!app.includes("RETIRED_DEFAULT_COLLECTOR_URL")) {
  fail("retired default URL wipe is missing");
}
if (!app.includes('autoSend: qs.get("auto") !== "0"')) {
  fail("settings-link auto-send default (auto!==0) is missing");
}
if (!app.includes("if (collectorCfg.url && collectorCfg.autoSend)")) {
  fail("session save must still require both destination and autoSend");
}
for (const phrase of [
  "データ回収は未設定です",
  "この画面ではデータ回収が有効です",
  "自動送信はオフです",
]) {
  if (!app.includes(phrase)) fail(`applyConsentCopy is missing: ${phrase}`);
}

if (process.exitCode) {
  console.error("collector privacy checks failed");
  process.exit(1);
}
console.log("collector privacy checks passed");
