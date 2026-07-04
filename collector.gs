// Reaction Meter データ回収スクリプト（Google Apps Script）
//
// 使い方:
// 1. Google スプレッドシートを新規作成
// 2. 拡張機能 → Apps Script を開き、このファイルの中身を貼り付けて保存
// 3. デプロイ → 新しいデプロイ → 種類「ウェブアプリ」
//    - 実行ユーザー: 自分 / アクセスできるユーザー: 全員
// 4. 発行された URL を Reaction Meter の「データ回収設定」に貼り付ける
//
// データはシート「summary」（1行=1セッション）と「timeseries」（200ms間隔の時系列）に
// 追記されます。列はアプリ側の更新に合わせて自動で右に増えます。

const CONFIG = {
  // 任意。値を設定すると、同じ collectorToken を含むリクエストだけを受け付ける。
  // 個人情報や本物のパスワードは入れないこと。
  COLLECTOR_TOKEN: "",
  MAX_ROWS: 20000,
  MAX_TEXT_LENGTH: 500,
};

function doGet() {
  return textResponse("Reaction Meter Collector is running.");
}

function doPost(e) {
  try {
    if (!e || !e.postData || !e.postData.contents) {
      return textResponse("missing body");
    }

    const payload = JSON.parse(e.postData.contents);
    if (CONFIG.COLLECTOR_TOKEN && payload.collectorToken !== CONFIG.COLLECTOR_TOKEN) {
      return textResponse("unauthorized");
    }

    const rows = Array.isArray(payload.rows) ? payload.rows.slice(0, CONFIG.MAX_ROWS) : [];
    if (rows.length === 0) {
      return textResponse("no rows");
    }

    const kind = payload.kind === "timeseries" ? "timeseries" : "summary";
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const sheet = ss.getSheetByName(kind) || ss.insertSheet(kind);

    // ヘッダー行を読み、未知の列は右に追加する（アプリ側の列追加に自動追従）
    let headers = [];
    if (sheet.getLastRow() > 0) {
      headers = sheet.getRange(1, 1, 1, sheet.getLastColumn()).getValues()[0].map(String);
    }
    if (headers.length === 0) {
      headers = ["received_at", "file_name"];
    }
    const seen = {};
    rows.forEach(function (row) {
      Object.keys(row).forEach(function (k) { seen[k] = true; });
    });
    Object.keys(seen).forEach(function (k) {
      if (headers.indexOf(k) === -1) headers.push(k);
    });
    sheet.getRange(1, 1, 1, headers.length).setValues([headers]);

    const receivedAt = new Date();
    const fileName = sanitizeText(payload.fileName || "");
    const values = rows.map(function (row) {
      return headers.map(function (h) {
        if (h === "received_at") return receivedAt;
        if (h === "file_name") return fileName;
        return sanitizeValue(row[h]);
      });
    });

    sheet.getRange(sheet.getLastRow() + 1, 1, values.length, headers.length).setValues(values);
    return textResponse("ok");
  } catch (error) {
    return textResponse("error");
  }
}

function sanitizeValue(value) {
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : "";
  }
  if (typeof value === "boolean") {
    return value;
  }
  return sanitizeText(value);
}

function sanitizeText(value) {
  const text = String(value == null ? "" : value)
    .replace(/[\u0000-\u001F\u007F]/g, " ")
    .slice(0, CONFIG.MAX_TEXT_LENGTH);

  // 数式インジェクション対策
  return /^[=+\-@]/.test(text) ? "'" + text : text;
}

function textResponse(text) {
  return ContentService.createTextOutput(text);
}
