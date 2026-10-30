// Halley's GUI: the pages, and how they call the API.
//
// The page reads its session token from the address it was opened at (#token=...), keeps it for
// the tab in sessionStorage and sends it with every API request. Every value it shows is set as
// text: nothing from the server, a model or a file is parsed as HTML.

import { el, formatValue, isNumber, lineChart } from "/chart.js";

const main = document.getElementById("main");
const jobChip = document.getElementById("job");
const tray = document.getElementById("tray");
const ended = document.getElementById("ended");
const chipList = document.getElementById("chips");
const themeButton = document.getElementById("theme");

// The tray (the running job, and the jobs that ended on a page no longer showing them) lies over
// the page's bottom corner. The page keeps room for it at its end, so its last lines can be
// scrolled out from under it, and a scroll to the focused control stops short of it, as it does
// of the top bar. The room grows with the tray while a page is shown and never shrinks under the
// user, which would move the page; the next page drawn takes it afresh.
let trayRoom = 0;
function fitTray(afresh = false) {
  const now = tray.offsetHeight ? tray.offsetHeight + 16 : 0;
  trayRoom = afresh ? now : Math.max(trayRoom, now);
  const root = document.documentElement.style;
  root.setProperty("--tray-room", `${trayRoom}px`);
  root.setProperty("--tray-now", `${now}px`);
}
// the tray's chips can be cleared without opening each one; Clear lies at the foot of the box,
// outside the list that scrolls, where it stays as chips come and go. Pressed from the keyboard,
// it gives the focus, which goes with the box, back to the control it came to the tray from;
// clicked or tapped, it leaves the focus to the browser, as any control that goes does
document.getElementById("chips-clear").addEventListener("click", (e) => {
  const had = ended.contains(document.activeElement);
  chipList.replaceChildren();
  ended.hidden = true;
  shadeEdges();
  if (had && e.detail === 0) focusBack();
});

// The control the focus came to the tray from (by Tab, say), while the focus is in the tray: a
// report opened from a chip, and closed, gives the focus back to the chip, and keeps it so, as
// does a switch to another window or tab and back, after which the focus comes back from nowhere
// to the control that had it.
let trayEntry = null;
let leftWindowFrom = null;
window.addEventListener("blur", () => {
  leftWindowFrom = tray.contains(document.activeElement) ? document.activeElement : null;
});
// back in the window: the focus comes back, if it does, as the window does; one that does not
// (a click elsewhere took it) leaves no control to come back to
window.addEventListener("focus", () => setTimeout(() => (leftWindowFrom = null), 0));
tray.addEventListener("focusin", (e) => {
  const from = e.relatedTarget;
  const back = !from && e.target === leftWindowFrom;
  leftWindowFrom = null;
  // whether the keyboard brought the focus, noted as it comes: a window away from the page
  // shows no focus at all
  keyedFocus = e.target.matches(":focus-visible") ? e.target : back ? keyedFocus : null;
  if (back || (from instanceof Element && (tray.contains(from) || from.closest("dialog")))) return;
  trayEntry = from instanceof Element ? from : null;
});
let keyedFocus = null;

// The focus back where it came to the tray from, if that control still takes it, or to the page.
function focusBack() {
  if (trayEntry?.isConnected) trayEntry.focus({ preventScroll: true });
  if (!trayEntry || document.activeElement !== trayEntry) focusQuietly(main);
}

// Give `node` the focus without scrolling to it. One that takes no focus of its own (the page, a
// box of text) takes it for that moment only: it gives it up at the next press of the pointer,
// so that a click on its text starts the next Tab from there, as on any text, not from the box.
function focusQuietly(node) {
  if (!node.matches("a[href], button, input, select, textarea, [tabindex]")) {
    node.setAttribute("tabindex", "-1");
    const drop = () => {
      node.removeAttribute("tabindex");
      node.removeEventListener("blur", drop);
      document.removeEventListener("pointerdown", drop, true);
    };
    node.addEventListener("blur", drop);
    document.addEventListener("pointerdown", drop, true);
  }
  node.focus({ preventScroll: true });
}

// Whether the pointer or the focus is in the tray: its chips do not move, nor go, meanwhile. The
// running job's chip goes, or shrinks, when its job ends, the one waiting starts or the next job's
// label is shorter, from under a pointer or a focus that was on it: the tray stays held for a
// moment after (the jobs poll sets heldUntil then, and only then), so the chip that job's end
// brings, which comes then, holds the chips too.
let heldUntil = 0;
let pointerAt = null; // where the pointer was last seen on the page
for (const type of ["pointermove", "pointerdown", "pointerover"]) {
  document.addEventListener(type, (e) => (pointerAt = { x: e.clientX, y: e.clientY }), { capture: true, passive: true });
}
// a pan (a touch the browser took to scroll), which leaves the hover where the last tap put it,
// and a zoom, which moves the page's pixels under a still pointer, leave its position unknown
document.addEventListener("pointercancel", () => (pointerAt = null), { capture: true, passive: true });
window.addEventListener("resize", () => (pointerAt = null));
function inTray() {
  return tray.matches(":hover") || tray.contains(document.activeElement) || Date.now() < heldUntil;
}


// The list's edges shade where more of it lies beyond them, over the chips, since a phone's
// scroll bar shows only while it scrolls; the list's own padding is not more of it.
function shadeEdges() {
  const style = getComputedStyle(chipList);
  const above = chipList.scrollTop;
  const below = chipList.scrollHeight - chipList.clientHeight - chipList.scrollTop;
  chipList.classList.toggle("more-above", above > parseFloat(style.paddingTop) + 0.5);
  chipList.classList.toggle("more-below", below > parseFloat(style.paddingBottom) + 0.5);
}
chipList.addEventListener("scroll", shadeEdges, { passive: true });
window.addEventListener("resize", shadeEdges);
if ("ResizeObserver" in window) {
  new ResizeObserver(() => fitTray()).observe(tray);
  const bar = document.querySelector(".topbar");
  new ResizeObserver(() => document.documentElement.style.setProperty("--bar-room", `${bar.offsetHeight}px`)).observe(bar);
}

// ------------------------------------------------------------------------------ the session

const TOKEN_KEY = "halley-gui-token";

function bootToken() {
  const match = location.hash.match(/^#token=([A-Za-z0-9_-]+)/);
  if (match) {
    try {
      sessionStorage.setItem(TOKEN_KEY, match[1]);
    } catch {
      /* storage refused: the token lives as long as this page */
    }
    history.replaceState(null, "", `${location.pathname}#/models`);
    return match[1];
  }
  try {
    return sessionStorage.getItem(TOKEN_KEY);
  } catch {
    return null;
  }
}

let TOKEN = bootToken();

class ApiFailure extends Error {
  constructor(status, body) {
    super(body?.error?.message || `the server answered ${status}`);
    this.status = status;
    this.detail = body?.error || {};
  }
}

class JobFailure extends Error {
  constructor(job) {
    super(job.error?.message || `the job was ${job.state}`);
    this.detail = job.error || { message: `the job was ${job.state}` };
    this.job = job;
  }
}

async function api(path, { params, body } = {}) {
  const url = new URL(path, location.origin);
  for (const [k, v] of Object.entries(params || {})) {
    if (v !== undefined && v !== null) url.searchParams.set(k, String(v));
  }
  const init = { method: body === undefined ? "GET" : "POST", headers: { "X-Halley-Token": TOKEN || "" } };
  if (body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  const response = await fetch(url, init);
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = { error: { message: text } };
  }
  if (!response.ok) throw new ApiFailure(response.status, data);
  return data;
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function waitJob(job, onTick) {
  let current = job;
  while (current.state === "queued" || current.state === "running") {
    onTick?.(current);
    await sleep(current.state === "queued" ? 250 : 400);
    current = await api("/api/job", { params: { id: job.id } });
  }
  onTick?.(current);
  if (current.state !== "done") throw new JobFailure(current);
  return current.result;
}

let sessionInfo = null;
function getSession() {
  // kept while it holds an answer; a refusal, or a server that was not there, is asked again
  if (!sessionInfo) {
    sessionInfo = api("/api/session").catch((err) => {
      sessionInfo = null;
      throw err;
    });
  }
  return sessionInfo;
}

// ------------------------------------------------------------------------------- helpers

function short(hash, n = 12) {
  return hash ? String(hash).slice(0, n) : "n/a";
}

function when(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? String(iso) : d.toLocaleString();
}

function plural(n, one, many) {
  return `${n} ${n === 1 ? one : many || `${one}s`}`;
}

function textOf(v) {
  if (v === null || v === undefined || v === "") return "n/a";
  if (typeof v === "number") return formatValue(v);
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

function cell(v) {
  return v === null || v === undefined || v === "" ? "" : textOf(v);
}

function chip(text, extra = "") {
  return el("span", { class: `chip ${extra}`.trim(), text });
}

function kv(pairs) {
  const list = el("dl", { class: "kv" });
  for (const [k, v] of pairs) {
    if (v === undefined) continue;
    list.append(el("dt", { text: k }), el("dd", {}, v instanceof Node ? v : textOf(v)));
  }
  return list;
}

function field(label, control, extra = "") {
  return el("label", { class: `field ${extra}`.trim() }, el("span", { text: label }), control);
}

function notice(text, kind = "notice") {
  return el("div", { class: kind, text });
}

function seconds(job) {
  return job.seconds === null || job.seconds === undefined ? "" : `${job.seconds.toFixed(1)} s`;
}

function progress(text) {
  return el("div", { class: "row muted" }, el("span", { class: "spinner", "aria-hidden": "true" }), el("span", { text }));
}

// Follow a job in `host`: one line whose spinner keeps turning while its text changes, with a
// Cancel button while the job waits behind another. A cancelled job ends in a plain notice.
async function watchJob(job, host, verb) {
  const text = el("span", { text: verb });
  const cancel = el("button", { type: "button", class: "button", text: "Cancel", hidden: true });
  cancel.addEventListener("click", async () => {
    cancel.disabled = true;
    try {
      await api("/api/job/cancel", { body: { id: job.id } });
    } catch {
      /* it may have started meanwhile; the next tick says so */
    }
  });
  host.replaceChildren(el("div", { class: "row muted" }, el("span", { class: "spinner", "aria-hidden": "true" }), text, cancel));
  kickJobs();
  return waitJob(job, (j) => {
    const waiting = j.state === "queued";
    text.textContent = waiting ? "Waiting for the job before it" : `${verb} ${seconds(j)}`;
    cancel.hidden = !waiting;
  });
}

function cancelled(err) {
  return err instanceof JobFailure && err.job.state === "cancelled";
}

// A job whose card is no longer shown when it ends (the user went elsewhere, or came back and the
// page was drawn again): the page is not changed under them; a chip in the tray says the job
// ended, the newest first. A chip for what the job made links to it. A chip for a failure, or for
// a result no page shows (a check of the inputs), opens a dialog with what the job said, and a
// link back to the page it was started on; it stays, read, so the report opens again however the
// dialog was closed. A chip goes when the chips are cleared, or once it has been read (a link's
// news is read when it comes, since the Models and Runs pages list what the job made) and two
// minutes have passed, and it has been on screen for half a minute since. A live region says
// that the job ended.
function finishedElsewhere(label, href, failed, detail) {
  const strip = chipList;
  // a chip goes; the last takes the box with it, and the focus, if the box had it, goes back
  // where it was, unless the chip goes to another page
  const leave = (chip, navigating = false) => {
    const had = ended.contains(document.activeElement);
    chip.remove();
    if (!strip.querySelector(".job-chip")) {
      ended.hidden = true;
      if (had && !navigating) focusBack();
    }
    shadeEdges();
  };
  let chip;
  if (detail) {
    const text = failed ? `Failed: ${label}. Why?` : `Done: ${label}. Show it`;
    chip = el("button", { type: "button", class: "job-chip", title: text, text });
    let again = null;
    chip.addEventListener("click", () => {
      if (again) again();
      else {
        chip.classList.add("read");
        // its name says it was opened; its title stays its visible text, which a screen reader
        // then does not say again as its description
        chip.setAttribute("aria-label", `${text} (opened)`);
        again = expireWhenSeen(chip, strip, leave);
      }
      showReport(label, href, detail, failed);
    });
  } else {
    const text = `Done: ${label}. Open it`;
    chip = el("a", { href, class: "job-chip", title: text, text });
    chip.addEventListener("click", (e) => {
      leave(chip, true);
      // a link to the page shown draws it again, as a link to another page draws that one
      if (location.hash === href) {
        e.preventDefault();
        route();
      }
    });
    expireWhenSeen(chip, strip, leave);
  }
  // the newest chip comes at the top of the list. Where the pointer or the focus is in the tray,
  // the chips stay where they are on screen, the new one coming above them, out of view once the
  // list scrolls; otherwise the list shows its top, and the new chip
  const held = inTray();
  ended.hidden = false;
  const hidden = strip.scrollHeight - strip.clientHeight;
  strip.prepend(chip);
  if (held) strip.scrollTop += strip.scrollHeight - strip.clientHeight - hidden;
  else strip.scrollTop = 0;
  shadeEdges();
  announce(failed ? `Failed: ${label}.` : `Done: ${label}.`);
}

// A chip whose news the user has had goes once two minutes have passed and it has been on
// screen, with nothing over it, for half a minute; never while a report is open, nor while the
// pointer or the focus is in the tray, where a chip going would move the others. What it gives
// back starts the two minutes and the half minute again (a report opened again).
function expireWhenSeen(chip, strip, leave) {
  let born = Date.now();
  let seen = 0;
  const tick = setInterval(() => {
    if (!chip.isConnected) {
      clearInterval(tick);
      return;
    }
    if (document.getElementById("report") || inTray()) return;
    if (document.visibilityState === "visible" && onScreen(chip, strip)) seen += 5000;
    if (seen >= 30000 && Date.now() - born >= 120000) {
      clearInterval(tick);
      leave(chip);
    }
  }, 5000);
  return () => {
    born = Date.now();
    seen = 0;
  };
}

// Whether a node can be seen: whole inside the box that scrolls it and what the screen shows of
// the page (the visual viewport, which a pinch zoom makes smaller than the window), and not under
// anything else (a dialog's backdrop, say).
function onScreen(node, box) {
  const r = node.getBoundingClientRect();
  const b = box.getBoundingClientRect();
  const v = window.visualViewport;
  const [top, left] = v ? [v.offsetTop, v.offsetLeft] : [0, 0];
  const [bottom, right] = v ? [top + v.height, left + v.width] : [innerHeight, innerWidth];
  if (r.height <= 0 || r.top < Math.max(b.top, top) || r.bottom > Math.min(b.bottom, bottom)) return false;
  if (r.left < left || r.right > right) return false;
  const hit = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
  return hit !== null && node.contains(hit);
}

// Say something in the live region: the open report's own while one is open, since a modal
// dialog leaves the rest of the page out of reach of a screen reader.
function announce(text) {
  const region = document.querySelector("#report .announce") || document.getElementById("announce");
  if (region) region.textContent = text;
}

// A dialog, over the page and the tray, with what a job (or a reload) said; `href`, when given,
// is the page it was started on, which it links back to. Close, Escape or a click on the
// backdrop closes it: a click that began there, not a double-click's second or a selection
// dragged out of the dialog. While one is open, another report waits in the tray as a chip.
function showReport(label, href, detail, failed = false) {
  if (document.getElementById("report")) {
    finishedElsewhere(label.charAt(0).toLowerCase() + label.slice(1), href, failed, detail);
    return;
  }
  const close = el("button", { type: "button", class: "button", text: "Close" });
  const back = href ? el("a", { href, text: "Go to the page it was started on" }) : null;
  const title = el("h2", { id: "report-title", tabindex: "-1", text: label.charAt(0).toUpperCase() + label.slice(1) });
  const region = el("span", { class: "announce visually-hidden", role: "status", "aria-live": "polite" });
  const dialog = el("dialog", { id: "report", class: "report", "aria-labelledby": "report-title" }, el("div", { class: "report-body stack" }, title, detail, el("div", { class: "row" }, back, close), region));
  const opened = Date.now();
  // a click that is not the second of a double-click whose first opened the report
  const meant = (e) => e.detail <= 1 && Date.now() - opened > 400;
  back?.addEventListener("click", (e) => {
    if (meant(e)) dialog.close();
    else e.preventDefault();
  });
  close.addEventListener("click", (e) => {
    if (meant(e)) dialog.close();
  });
  let pressedOnBackdrop = false;
  dialog.addEventListener("pointerdown", (e) => {
    pressedOnBackdrop = e.target === dialog; // the dialog's body fills the dialog: this is beside it
  });
  dialog.addEventListener("click", (e) => {
    if (e.target === dialog && pressedOnBackdrop && meant(e)) dialog.close();
  });
  dialog.addEventListener("close", () => dialog.remove());
  document.body.append(dialog);
  dialog.showModal();
  title.focus();
}

// The files a reload let go, being no longer where they were loaded from, as a notice; null for
// none.
function droppedNotice(dropped) {
  const gone = (dropped || []).map((path) => path.split(/[\\/]/).pop());
  if (!gone.length) return null;
  const one = gone.length === 1;
  return notice(`${gone.join(", ")} ${one ? "is" : "are"} no longer where ${one ? "it was" : "they were"} loaded from, so the GUI let ${one ? "it" : "them"} go with ${one ? "its" : "their"} models. Load ${one ? "it" : "them"} again from where ${one ? "it is" : "they are"} now.`);
}

function showDropped(dropped) {
  const gone = droppedNotice(dropped);
  if (gone) showReport("the files were reloaded", null, gone);
}

// After a reload under a report, asked for in a way whose answer does not say whether any file
// is still changed: if none is, the page's "changed" marks go, and its line, saying the reload
// ended, takes the place of the page's warning that the files changed.
async function settleWarning(drawnAt) {
  try {
    const listing = await api("/api/models");
    if (listing.changed) return;
    for (const m of document.querySelectorAll(".reload-mark")) m.remove();
    const warning = document.querySelector("main .files-changed");
    if (warning) warning.replaceChildren(endedLine(false, () => go(drawnAt), "The reload"));
  } catch {
    /* the next page drawn says it */
  }
}

// A load or a reload of the model files that ended once the page it was asked on was no longer
// shown, or was under an open report: the page the user is on stays as it is, but for the Models
// page's side lists when no report is open, and a chip says it ended, with what it let go, if
// anything, or a link to `href`; `status`, the load's own line while it is on the page, says it
// ended, with a link that shows what it would have shown (`show`); `what` names it there.
function filesEndedElsewhere(status, label, href, drawnAt, dropped, show, what) {
  if (!document.getElementById("report")) refreshSideLists();
  const gone = droppedNotice(dropped);
  if (gone) finishedElsewhere(label, drawnAt, false, gone);
  else finishedElsewhere(label, href, false, null);
  if (status.isConnected) status.replaceChildren(endedLine(false, show, what));
}

function filesFailedElsewhere(status, label, drawnAt, err, title, show, what) {
  if (err.detail?.dropped?.length && !document.getElementById("report")) refreshSideLists();
  const gone = () => droppedNotice((err.detail?.dropped || []).filter((f) => f !== err.detail?.file));
  finishedElsewhere(label, drawnAt, true, el("div", { class: "stack" }, errorBox(err, title), gone()));
  if (status.isConnected) status.replaceChildren(endedLine(true, show, what));
}

// The line a job's card, or a load's, shows in place of its progress for a job that ended while
// a report was open over it (the page beneath does not move) or while another page was coming:
// still true once the tray's chip has gone, with a link that shows what the card would have
// shown. What comes in the line's place (an error, a result) takes the focus its link had.
function endedLine(failed, show, what = "This") {
  const link = el("button", { type: "button", class: "link-button", text: failed ? "Show why" : "Show it" });
  const when = document.getElementById("report") ? "while a report was open" : "while another page was coming";
  const line = el("div", { class: "row muted" }, el("span", { text: `${what} ${failed ? "failed" : "ended"} ${when}.` }), link);
  link.addEventListener("click", () => {
    const host = line.parentElement;
    const had = document.activeElement === link;
    show();
    if (had && !link.isConnected && host?.isConnected && host.firstElementChild) focusQuietly(host.firstElementChild);
  });
  return line;
}

// Whether a job's card (or a load's line) is what the user sees now: on the page (see onPage),
// and not under an open report, beneath which the page does not move.
function seenNow(status, drawnBy) {
  return onPage(status, drawnBy) && !document.getElementById("report");
}

// A job's end its card cannot show as it would (the user went elsewhere, or a report is open
// over it): a chip in the tray, which the live region announces, with `detail()`, if given, for
// its report; and, while the card is still on the page, a line that says so, whose link shows
// the detail in the card, or does what the card would have done (`show`, by default opening
// `href`).
function endedAway(status, label, href, failed, detail, show = () => go(href)) {
  finishedElsewhere(label, href, failed, detail ? detail() : null);
  if (status.isConnected) status.replaceChildren(endedLine(failed, detail ? () => status.replaceChildren(detail()) : show));
}

function stamp() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
}

function slug(text) {
  return String(text)
    .replace(/([a-z0-9])([A-Z])/g, "$1-$2")
    .toLowerCase()
    .replace(/[^a-z0-9._-]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

function remember(key, value) {
  try {
    sessionStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage refused: the report is not shown after the page changes */
  }
}

function recall(key) {
  try {
    return JSON.parse(sessionStorage.getItem(key) || "null");
  } catch {
    return null;
  }
}

function go(hash) {
  if (location.hash === hash) route();
  else location.hash = hash;
}

function runHref(dir, tab = "overview", params) {
  const query = params ? `?${new URLSearchParams(params)}` : "";
  return `#/run/${encodeURIComponent(dir)}/${tab}${query}`;
}

function compareValues(a, b) {
  if (a === b) return 0;
  if (a === null || a === undefined) return 1;
  if (b === null || b === undefined) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  return String(a).localeCompare(String(b), undefined, { numeric: true });
}

// A table of rows: columns are {label, key, num, mono, render(row), sort(row) or false}. Headers
// sort the rows; a row with `onRow` opens on click or Enter.
function dataTable(columns, rows, { onRow, empty = "Nothing to show." } = {}) {
  const wrap = el("div", { class: "table-wrap" });
  const table = el("table", { class: "data" });
  const head = el("tr");
  const body = el("tbody");
  let sortBy = null;
  let direction = 1;
  for (const c of columns) {
    const sortable = c.sort !== false;
    const th = el("th", { scope: "col", class: `${c.num ? "num " : ""}${sortable ? "sortable" : ""}`, text: c.label });
    if (sortable) {
      th.tabIndex = 0;
      const toggle = () => {
        direction = sortBy === c ? -direction : 1;
        sortBy = c;
        for (const other of head.children) other.removeAttribute("aria-sort");
        th.setAttribute("aria-sort", direction > 0 ? "ascending" : "descending");
        fill();
      };
      th.addEventListener("click", toggle);
      th.addEventListener("keydown", (e) => {
        if (e.key === "Enter") toggle();
      });
    }
    head.append(th);
  }
  const value = (c, row) => (c.sort ? c.sort(row) : row[c.key]);
  function fill() {
    const list = rows.slice();
    if (sortBy) list.sort((a, b) => compareValues(value(sortBy, a), value(sortBy, b)) * direction);
    body.replaceChildren();
    for (const row of list) {
      const tr = el("tr", { class: onRow ? "clickable" : "" });
      for (const c of columns) {
        const content = c.render ? c.render(row) : cell(row[c.key]);
        tr.append(el("td", { class: `${c.num ? "num " : ""}${c.mono ? "code" : ""}`.trim() }, content));
      }
      if (onRow) {
        tr.tabIndex = 0;
        tr.addEventListener("click", (e) => {
          if (!e.target.closest("a, button, input, select")) onRow(row);
        });
        tr.addEventListener("keydown", (e) => {
          if (e.key === "Enter") onRow(row);
        });
      }
      body.append(tr);
    }
    if (!list.length) body.append(el("tr", {}, el("td", { colspan: String(columns.length), class: "muted", text: empty })));
  }
  fill();
  table.append(el("thead", {}, head), body);
  wrap.append(table);
  return wrap;
}

function snippetBlock(lines) {
  const pre = el("pre", { class: "code" });
  const width = String(lines[lines.length - 1].line).length;
  for (const line of lines) {
    // the line break stays outside the highlight, which is a block as wide as the code
    const text = `${String(line.line).padStart(width)}  ${line.text}`;
    pre.append(line.here ? el("span", { class: "hl", text }) : document.createTextNode(text), document.createTextNode("\n"));
  }
  return pre;
}

function where(loc) {
  if (!loc) return null;
  return el("code", { text: `${loc.function} (${loc.file}:${loc.line})` });
}

// An error from the API or a job: the message, the model source line it points at with the lines
// around it, and the traceback of an error that is not one of Halley's own.
function errorBox(err, title = "That did not work") {
  const d = err?.detail || {};
  const message = d.message || err?.message || String(err);
  const box = el("div", { class: "error", role: "alert" });
  const kinds = { unit: "Unit error", build: "Build error", load: "Input error", run: "Run error" };
  box.append(el("div", { class: "title", text: kinds[d.kind] ? `${title}: ${kinds[d.kind]}` : d.type ? `${title}: ${d.type}` : title }));
  box.append(el("div", { class: "message", text: message }));
  if (d.loc) box.append(el("div", { class: "where" }, "at ", where(d.loc)));
  if (d.snippet?.length) box.append(snippetBlock(d.snippet));
  for (const note of d.notes || []) box.append(el("div", { class: "message", text: note }));
  if (d.traceback) box.append(el("details", {}, el("summary", { text: "Traceback" }), el("pre", { class: "code", text: d.traceback })));
  return box;
}

function tabsBar(items, active, hrefOf) {
  const bar = el("nav", { class: "tabs", "aria-label": "Views" });
  for (const [key, label] of items) {
    bar.append(el("a", { href: hrefOf(key), class: key === active ? "on" : "", "aria-current": key === active ? "page" : null, text: label }));
  }
  return bar;
}

// Colour slots that follow an entity: a policy keeps its colour while others come and go.
class Slots {
  constructor() {
    this.map = new Map();
  }
  get(key) {
    if (!this.map.has(key)) {
      const used = new Set(this.map.values());
      let slot = 1;
      while (used.has(slot) && slot < 8) slot += 1;
      this.map.set(key, slot);
    }
    return this.map.get(key);
  }
  keep(keys) {
    for (const key of [...this.map.keys()]) if (!keys.includes(key)) this.map.delete(key);
  }
}

// --------------------------------------------------------------------------- jobs and theme

let jobTimer = null;
let afterJobChip = () => {}; // what the chip's change leaves to do once it is made
async function pollJobs() {
  jobTimer = null;
  try {
    const { jobs } = await api("/api/jobs");
    const active = jobs.filter((j) => j.state === "running" || j.state === "queued");
    // the chip may go, or shrink from an edge, from under the pointer or the focus: the tray
    // stays held if it went from under either, and a focus the keyboard gave its Cancel, gone
    // with it, goes back where it came to the tray from
    const hovered = jobChip.matches(":hover");
    const focused = jobChip.contains(document.activeElement) ? document.activeElement : null;
    const keyed = focused !== null && focused === keyedFocus;
    afterJobChip = () => {
      // still under the pointer: the chip, as the browser's own hit test finds it there now (a
      // pill's rounded ends leave the corners of its box to the page), or, where the page has
      // not seen the pointer yet, as long as the chip is there at all
      const at = pointerAt && document.elementFromPoint(pointerAt.x, pointerAt.y);
      const still = pointerAt ? !!at && jobChip.contains(at) : !jobChip.hidden;
      const gone = focused !== null && (!focused.isConnected || focused.closest("[hidden]"));
      if ((hovered && !still) || gone) heldUntil = Date.now() + 3000;
      if (keyed && gone) focusBack();
    };
    if (active.length) {
      const running = active.find((j) => j.state === "running") || active[0];
      const waiting = active.filter((j) => j.state === "queued");
      const text = `${running.label}${running.seconds !== null ? ` · ${Math.round(running.seconds)} s` : ""}`;
      if (!jobChip.firstChild) {
        jobChip.append(el("span", { class: "spinner", "aria-hidden": "true" }), el("span"), el("button", { type: "button", class: "legend-item", hidden: true }));
        jobChip.lastChild.addEventListener("click", async () => {
          const ids = JSON.parse(jobChip.lastChild.dataset.ids || "[]");
          for (const id of ids) {
            try {
              await api("/api/job/cancel", { body: { id } });
            } catch {
              /* started meanwhile */
            }
          }
          kickJobs();
        });
      }
      jobChip.children[1].textContent = text;
      jobChip.children[1].title = text;
      const cancel = jobChip.lastChild;
      cancel.hidden = waiting.length === 0;
      cancel.textContent = `Cancel ${waiting.length} waiting`;
      cancel.dataset.ids = JSON.stringify(waiting.map((j) => j.id));
      jobChip.hidden = false;
      jobTimer = setTimeout(pollJobs, 1000);
    } else {
      jobChip.hidden = true;
      jobChip.replaceChildren();
    }
  } catch {
    jobChip.hidden = true;
  }
  afterJobChip();
  afterJobChip = () => {};
}

function kickJobs() {
  if (jobTimer === null) pollJobs();
}

const THEMES = ["auto", "light", "dark"];
let theme = "auto";
try {
  theme = localStorage.getItem("halley-gui-theme") || "auto";
} catch {
  /* no storage: the theme follows the system */
}

function applyTheme() {
  if (theme === "auto") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
  themeButton.textContent = `Theme: ${theme}`;
}

themeButton.addEventListener("click", () => {
  theme = THEMES[(THEMES.indexOf(theme) + 1) % THEMES.length];
  try {
    localStorage.setItem("halley-gui-theme", theme);
  } catch {
    /* the choice lasts as long as the page */
  }
  applyTheme();
});
applyTheme();

// ------------------------------------------------------------------------------- models page

function welcome() {
  return el(
    "div",
    { class: "card stack" },
    el("h2", { text: "Start here" }),
    el("p", { text: "Pick a model on the left. A model is built first: its formulas are traced, their units, time references and calculation order are checked, and code is generated for them. Then it runs on a set of inputs, which writes a run directory with a manifest, a copy of every input and the outputs." }),
    el("p", { text: "To see it work, open TermAssurance, build it and run it on Reference case 1, the term assurance specification's worked example. Then explain a reserve: every value can be traced back to its formula and inputs." }),
    el("p", { class: "muted", text: "Your own models live in Python files. Load one with the field under the list; the GUI imports it as the module named after the file, and reloads it when you ask." }),
  );
}

// The Models page's lists: the shipped models, each loaded file's, and the reload of the files.
function drawSideLists(host, { models, files, changed }) {
  const selected = host.dataset.selected || null;
  const list = (items) => {
    const ul = el("ul", { class: "side-list" });
    for (const m of items) {
      const meta = [m.kind === "aggregation" ? "stage" : null, m.abstract ? "abstract" : null, m.builds.length ? `${plural(m.builds.length, "build")}` : null].filter(Boolean).join(" · ");
      ul.append(
        el(
          "li",
          {},
          el(
            "a",
            { href: `#/models/${encodeURIComponent(m.id)}`, class: m.id === selected ? "on" : "" },
            el("span", { class: "name", text: m.name }),
            el("span", { class: "what", text: meta || m.module }),
          ),
        ),
      );
    }
    return ul;
  };
  const shipped = models.filter((m) => m.origin === "shipped");
  host.replaceChildren(el("div", { class: "group-label", text: "Reference products" }), list(shipped.filter((m) => m.kind === "projection")));
  host.append(el("div", { class: "group-label", text: "IFRS 17 stages" }), list(shipped.filter((m) => m.kind === "aggregation")));
  for (const file of files) {
    host.append(el("div", { class: "group-label row" }, el("span", { text: file.path.split(/[\\/]/).pop() })), list(models.filter((m) => file.models.includes(m.id))));
  }
  if (files.length) {
    const reload = reloadControl(changed);
    host.append(el("div", { class: "stack" }, reload.row, reload.status));
  }
}

// The side lists of the Models page shown, drawn again from the server's list with the rest of
// the page left as it is: after a load or a reload that ended while the user was elsewhere, or
// one that let files go.
async function refreshSideLists() {
  const host = document.getElementById("side-lists");
  if (!host) return;
  const drawnBy = renderId;
  try {
    const listing = await api("/api/models");
    if (host.isConnected && drawnBy === renderId) drawSideLists(host, listing);
  } catch {
    /* the next page drawn lists them */
  }
}

async function modelsPage(ctx, selected, params) {
  const [listing, session] = await Promise.all([api("/api/models"), getSession()]);
  const lists = el("div", { id: "side-lists", "data-selected": selected || "" });
  drawSideLists(lists, listing);
  const side = el("aside", { class: "card", "aria-label": "Models" }, lists);
  const path = el("input", { type: "text", placeholder: "C:\\models\\term.py", "aria-label": "Model file" });
  const load = el("button", { type: "submit", class: "button", text: "Load" });
  const loadStatus = el("div");
  const form = el("form", { class: "stack" }, el("div", { class: "group-label", text: "Your models" }), field("A Python file of models", path), el("div", { class: "row" }, load), loadStatus);
  const formDrawnBy = renderId;
  const formDrawnAt = location.hash;
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    load.disabled = true;
    loadStatus.replaceChildren(progress("Importing"));
    const label = `loading ${path.value.split(/[\\/]/).pop() || "the file"}`;
    try {
      const answer = await api("/api/models/load", { body: { path: path.value } });
      const first = answer.models.find((m) => !m.abstract) || answer.models[0];
      const href = first ? `#/models/${encodeURIComponent(first.id)}` : "#/models";
      const show = () => {
        go(href);
        showDropped(answer.dropped);
      };
      // a file whose import takes a while: the user may have gone elsewhere meanwhile
      if (seenNow(loadStatus, formDrawnBy)) show();
      else {
        // a load of a loaded file reloads every file: one after which no file differs from what
        // was loaded takes the page's "changed" marks away, and its line takes the place of the
        // warning that the files changed, which no longer holds, as a reload's does
        const settled = answer.file && !answer.file.changed;
        const warning = settled ? document.querySelector("main .files-changed") : null;
        if (settled) for (const m of document.querySelectorAll(".reload-mark")) m.remove();
        if (warning) loadStatus.replaceChildren(); // its progress, said there instead
        filesEndedElsewhere(warning || loadStatus, label, href, formDrawnAt, answer.dropped, show, "The load");
      }
    } catch (err) {
      const show = () => {
        loadStatus.replaceChildren(errorBox(err, "The file did not load"));
        // a load of a loaded file reloads every file, and may let some go: the file asked for,
        // which the error names, and others
        if (err.detail?.dropped?.length) {
          refreshSideLists();
          document.getElementById("model-detail")?.redraw?.();
        }
        showDropped((err.detail?.dropped || []).filter((f) => f !== err.detail?.file));
      };
      if (seenNow(loadStatus, formDrawnBy)) show();
      else {
        filesFailedElsewhere(loadStatus, label, formDrawnAt, err, "The file did not load", show, "The load");
        // a load of a loaded file that is gone reloads every file first: when that leaves no file
        // differing from what was loaded, the page's "changed" marks and its warning go, as a
        // reload's do, its line saying the reload ended
        if (err.detail?.dropped) settleWarning(formDrawnAt);
      }
    } finally {
      load.disabled = false;
    }
  });
  side.append(form, el("p", { class: "hint", text: `Paths are on this computer; a relative one starts from ${session.cwd}. Loading a file runs its top-level code.` }));
  const detail = el("section", { id: "model-detail", "aria-live": "polite" });
  const drawDetail = async () => {
    if (!selected) return [welcome()];
    try {
      return [await modelDetail(ctx, selected, session, params)];
    } catch (err) {
      return [errorBox(err, "The model cannot be shown")];
    }
  };
  detail.append(...(await drawDetail()));
  // after a reload a load asked for, which ran every file again: the model shown is drawn again,
  // the Load form left as it is
  const drawnBy = renderId;
  detail.redraw = async () => {
    const shown = await drawDetail();
    if (detail.isConnected && drawnBy === renderId) detail.replaceChildren(...shown);
  };
  return el("div", { class: "split" }, side, detail);
}

async function modelDetail(ctx, id, session, params) {
  const model = await api("/api/model", { params: { id } });
  const root = el("div");
  const sub = el("div", { class: "sub" }, el("span", { class: "mono", text: model.full_name }));
  if (model.file) sub.append(el("span", { class: "muted", text: `  ${model.file}${model.line ? `:${model.line}` : ""}` }));
  root.append(
    el(
      "div",
      { class: "page-head" },
      el("div", {}, el("h1", { text: model.name }), sub),
      el("div", { class: "row" }, chip(model.kind === "aggregation" ? "aggregation stage" : "projection"), model.abstract ? chip("abstract") : null),
    ),
  );
  if (model.doc) root.append(el("p", { text: model.doc }));
  if (model.error) root.append(errorBox({ detail: model.error }, "The class's declarations cannot be read"));
  if (model.loaded_file?.changed) {
    const warning = el("div", { class: "warning stack files-changed" });
    const reload = reloadControl(true, warning);
    warning.append(el("div", { class: "row" }, el("span", { class: "grow", text: "A model file, or a module of yours one imports, has changed since it was loaded; reload the files to build what they say now." }), reload.row), reload.status);
    root.append(warning);
  }
  if (model.abstract) {
    root.append(notice("This class is abstract (it sets abstract = True itself), so it cannot be built: build one of its subclasses."));
    root.append(declaredView(model));
    return root;
  }
  const builds = model.builds;
  const chosen = builds.find((b) => b.id === params.get("build")) || builds.find((b) => !b.stale) || null;
  root.append(buildCard(model, session, builds, chosen));
  if (!chosen) {
    root.append(declaredView(model));
    return root;
  }
  const build = await api("/api/build", { params: { id: chosen.id } });
  root.append(runCard(model, build, session));
  const tab = params.get("tab") || "variables";
  const hrefOf = (key) => `#/models/${encodeURIComponent(model.id)}?${new URLSearchParams({ build: chosen.id, tab: key })}`;
  root.append(tabsBar([["variables", `Variables (${build.variables.length})`], ["inputs", "Inputs it reads"], ["invariants", "Invariants"], ["code", "Generated code"]], tab, hrefOf));
  if (tab === "inputs") root.append(declaredView(model, "inputs"));
  else if (tab === "invariants") root.append(declaredView(model, "invariants"));
  else if (tab === "code") root.append(await codeView(build));
  else root.append(variablesView(build, params.get("var")));
  return root;
}

// The one reload of the model files: every loaded file runs again, with every module of the
// user's own they import, so they keep one copy of each, as a new Python process would. Its row
// (the button) and its status (an error, which holds code lines) are placed apart, so an error
// never sets the width of the row it sits beside. A reload that ends under a report, after which
// no file differs from what was loaded, takes the page's "changed" marks away, and the page's
// warning that the files changed (the one it sits in, `box`, or the model page's, for the side
// list's), which no longer holds, says what happened instead.
function reloadControl(changed, box = null) {
  const drawnBy = renderId;
  const drawnAt = location.hash;
  const button = el("button", { type: "button", class: "button", text: "Reload the files" });
  const mark = changed ? chip("changed", "reload-mark") : null;
  const status = el("div", { class: "reload-status" });
  button.addEventListener("click", async () => {
    button.disabled = true;
    status.replaceChildren(progress("Reloading"));
    const label = "a reload of the model files";
    try {
      const answer = await api("/api/models/reload", { body: {} });
      const show = () => {
        go(drawnAt);
        showDropped(answer.dropped);
      };
      // a reload that takes a while: the user may have gone elsewhere meanwhile
      if (seenNow(status, drawnBy)) show();
      else {
        const still = answer.files.some((f) => f.changed); // a file saved while it ran
        const warning = still ? null : box || document.querySelector("main .files-changed");
        if (!still) for (const m of document.querySelectorAll(".reload-mark")) m.remove();
        if (warning && warning !== box) status.replaceChildren(); // its progress, said there instead
        filesEndedElsewhere(warning || status, label, drawnAt, drawnAt, answer.dropped, show, "The reload");
      }
    } catch (err) {
      const show = () => status.replaceChildren(errorBox(err, "The files did not load"));
      if (seenNow(status, drawnBy)) show();
      else filesFailedElsewhere(status, label, drawnAt, err, "The files did not load", show, "The reload");
    } finally {
      button.disabled = false;
    }
  });
  return { row: el("div", { class: "row" }, button, mark), status };
}

// A build's settings as the server writes them from the values themselves (1 and 1.0 apart, an
// enum's member by its name), never as JSON re-typed in the page.
function staticsLine(texts) {
  const entries = Object.entries(texts || {});
  return entries.length ? entries.map(([k, v]) => `${k} ${v}`).join(", ") : "no build-time settings";
}

// Whether a job's card is still the page the user is on: still in the page, and drawn by the
// route that is current. A route the user has asked for draws its page only when its data has
// come, and until then the old page, and its card, stay on screen.
function onPage(status, drawnBy) {
  return status.isConnected && drawnBy === renderId;
}

function buildCard(model, session, builds, chosen) {
  const drawnBy = renderId;
  const drawnAt = location.hash;
  const card = el("section", { class: "card stack" }, el("h2", { text: "Build" }));
  const backend = el("select", { "aria-label": "Backend" });
  for (const b of session.backends) {
    const label = b.name === "numpy" ? "NumPy, the reference" : b.name === "numba" ? `Numba${b.available ? "" : " (not installed)"}` : b.name;
    backend.append(el("option", { value: b.name, disabled: !b.available, text: label }));
  }
  if (chosen) backend.value = chosen.backend;
  const fields = el("div", { class: "fields" }, field("Backend", backend));
  const statics = [];
  for (const p of model.statics) {
    // the field shows the shown build's own setting when there is one, the class's default
    // otherwise, as the server writes it. What the field holds once set is its baseline: a
    // field left as it is sends nothing, so the server keeps that build's value or the default
    const shown = chosen?.form && p.name in chosen.form ? chosen.form[p.name] : p.text;
    const label = `${p.name} (Static[${p.kind}]${p.input === "json" ? ", as JSON" : ""})`;
    let input;
    if (p.input === "bool") {
      input = el("input", { type: "checkbox" });
      input.checked = shown === "true";
      fields.append(el("label", { class: "check" }, input, el("span", {}, el("code", { text: p.name }), " (Static[bool])")));
    } else if (p.input === "choice") {
      input = el("select", {});
      // the members in the enum's own order, with the shown or the default text before them
      // when it is not a member; "(none)" only for a default of None, or before any choice
      const extra = [shown, p.text].filter((c, i, all) => !p.choices.includes(c) && all.indexOf(c) === i && (c !== "" || shown === "" || p.has_default));
      for (const c of [...extra, ...p.choices]) input.append(el("option", { value: c, text: c || "(none)" }));
      input.value = shown;
      fields.append(field(label, input));
    } else if (p.input === "flags") {
      // a flag's members, any of them joined by |, typed with the names suggested
      const listId = `flags-${p.name}-${Math.random().toString(36).slice(2)}`;
      input = el("input", { type: "text", list: listId, spellcheck: "false", autocomplete: "off", placeholder: p.choices.join("|") });
      input.value = shown;
      fields.append(field(`${p.name} (Static[${p.kind}], members joined by |)`, el("div", { class: "stack" }, input, el("datalist", { id: listId }, ...[...new Set([shown, p.text, ...p.choices])].filter(Boolean).map((c) => el("option", { value: c }))))));
    } else if (shown.includes("\n") || p.text.includes("\n")) {
      // a text a one-line field would lose its line breaks from, shown or the default's
      const lines = Math.max(shown.split("\n").length, p.text.split("\n").length);
      input = el("textarea", { rows: String(Math.min(8, lines)), spellcheck: "false" });
      input.value = shown;
      fields.append(field(label, input, "wide"));
    } else {
      // a date field holds a date only; a date and time a Static[date] was given, or has as its
      // default, stays text
      const plainDate = (text) => text === "" || /^\d{4}-\d{2}-\d{2}$/.test(text);
      const dateField = p.input === "date" && plainDate(shown) && plainDate(p.text);
      const inputmode = p.input === "int" ? "numeric" : p.input === "float" ? "decimal" : null;
      input = el("input", { type: dateField ? "date" : "text", inputmode, spellcheck: "false", autocomplete: "off" });
      input.value = shown;
      fields.append(field(label, input));
    }
    // a check box for a parameter with no default and no value shown always sends its state:
    // left clear, it means False, which a build must be given
    const unset = p.input === "bool" && !p.has_default && shown === "";
    statics.push([p, input, p.input === "bool" ? input.checked : input.value, unset]);
  }
  const button = el("button", { type: "button", class: "button primary", text: chosen ? "Build again" : "Build" });
  const hint = el("span", { class: "hint" });
  const sayHint = () => {
    hint.textContent = backend.value === "numba" ? "A first Numba build compiles every kernel, which can take minutes; later builds load them from Numba's cache." : "Traces the formulas, checks their units, time references and order, and generates the code that runs them.";
  };
  backend.addEventListener("change", sayHint);
  sayHint();
  const status = el("div", { role: "status" });
  button.addEventListener("click", async () => {
    // only the fields changed from what the form showed are sent, as typed: the server starts
    // from the shown build's own settings (or the class's defaults) and reads each changed one by
    // its declared type, so an untouched form builds the same model again, IR hash and all
    const values = {};
    for (const [p, input, baseline, unset] of statics) {
      const now = p.input === "bool" ? input.checked : input.value;
      if (now !== baseline || unset) values[p.name] = now;
    }
    const from = drawnAt; // the page the button was drawn on, not where the user is now
    button.disabled = true;
    try {
      const { job } = await api("/api/build", { body: { model: model.id, backend: backend.value, statics: values, from_build: chosen ? chosen.id : null } });
      const result = await watchJob(job, status, "Building");
      const href = `#/models/${encodeURIComponent(model.id)}?${new URLSearchParams({ build: result.build })}`;
      if (seenNow(status, drawnBy)) go(href);
      else endedAway(status, job.label, href, false);
    } catch (err) {
      if (cancelled(err)) status.replaceChildren(notice("The build was cancelled before it started."));
      else if (seenNow(status, drawnBy)) status.replaceChildren(errorBox(err, "The build failed"));
      else endedAway(status, `a build of ${model.name}`, from, true, () => errorBox(err, "The build failed"));
    } finally {
      button.disabled = false;
    }
  });
  card.append(fields, el("div", { class: "row" }, button, hint), status);
  if (chosen) {
    const facts = [`${chosen.backend}`, `${chosen.seconds} s`, `IR hash ${short(chosen.ir_sha256)}`, staticsLine(chosen.statics_text)];
    const line = el("div", { class: "row" }, el("span", { class: "status good", text: `Built: ${facts.join(" · ")}` }));
    if (chosen.stale) line.append(chip("from the file as it was"));
    card.append(line);
    for (const w of chosen.warnings) card.append(notice(`Build warning: ${w}`, "warning"));
  }
  if (builds.length > 1) {
    const others = el("div", { class: "row hint" }, el("span", { text: "Builds this session:" }));
    for (const b of builds) {
      const label = `${b.id} (${b.backend}; ${staticsLine(b.statics_text)}${b.stale ? "; earlier file" : ""})`;
      others.append(b.id === chosen?.id ? el("strong", { text: label }) : el("a", { href: `#/models/${encodeURIComponent(model.id)}?build=${b.id}`, text: label }));
    }
    card.append(others);
  }
  return card;
}

function runCard(model, build, session) {
  const drawnBy = renderId;
  const drawnAt = location.hash;
  const card = el("section", { class: "card stack" }, el("h2", { text: "Run" }));
  if (build.stage) {
    card.append(notice("An aggregation stage runs on projection run directories, through halley.aggregation.run_aggregation, not on model points. Its runs are on the Runs page: open one to read it, replay it from its own directory and explain its values."));
    return card;
  }
  const choice = el("div", { class: "stack", role: "radiogroup", "aria-label": "Inputs" });
  const radios = [];
  const usable = model.examples.filter((e) => !e.missing);
  for (const ex of model.examples) {
    const radio = el("input", { type: "radio", name: "inputs", value: ex.id, disabled: Boolean(ex.missing) });
    radios.push([radio, ex]);
    choice.append(
      el(
        "label",
        { class: "check" },
        radio,
        el("span", {}, el("strong", { text: ex.label }), el("span", { class: "muted", text: ` ${ex.missing ? ex.missing : ex.description}` })),
      ),
    );
  }
  const filesRadio = el("input", { type: "radio", name: "inputs", value: "" });
  radios.push([filesRadio, null]);
  choice.append(el("label", { class: "check" }, filesRadio, el("span", {}, el("strong", { text: "Files" }), el("span", { class: "muted", text: " Parquet or CSV files on this computer, one per input the model declares." }))));
  (usable.length ? radios.find(([, ex]) => ex === usable[0])[0] : filesRadio).checked = true;

  const files = el("div", { class: "fields" });
  const paths = { model_points: el("input", { type: "text", placeholder: "model_points.parquet" }), tables: {}, scenarios: {} };
  files.append(field("Model points", paths.model_points, "wide"));
  for (const name of build.tables) {
    paths.tables[name] = el("input", { type: "text", placeholder: `${name}.parquet` });
    files.append(field(`Table ${name}`, paths.tables[name]));
  }
  for (const name of build.scenarios) {
    paths.scenarios[name] = el("input", { type: "text", placeholder: `${name}.parquet` });
    files.append(field(`Scenario input ${name}`, paths.scenarios[name]));
  }
  const filesBox = el("div", { class: "stack" }, files, el("p", { class: "hint", text: `A relative path starts from ${session.cwd}. A CSV file is read with pyarrow's CSV reader, as the tutorial reads one.` }));

  const date = el("input", { type: "date", required: true });
  const horizon = el("input", { type: "number", min: "1", step: "1", placeholder: build.horizon_field ? `from ${build.horizon_field}` : "required" });
  const batch = el("input", { type: "number", min: "1", step: "1", value: "10000" });
  const workers = el("input", { type: "number", min: "1", step: "1", value: "1" });
  const label = el("input", { type: "text", placeholder: "optional" });
  const name = el("input", { type: "text", value: `${slug(model.name)}-${stamp()}` });
  const outputs = el("select", { multiple: true, "aria-label": "Outputs" });
  for (const v of build.needed) {
    const option = el("option", { value: v, text: v });
    option.selected = build.outputs ? build.outputs.includes(v) : false;
    outputs.append(option);
  }
  const outputsNote = el("p", { class: "hint", text: build.outputs ? `By default a run stores the model's ${build.outputs.length} outputs; choose others with Ctrl or Shift.` : "By default a run stores every variable it computes; choose fewer with Ctrl or Shift." });
  const settings = el(
    "div",
    { class: "fields" },
    field("Valuation date", date),
    field("Horizon in months", horizon),
    field("Batch size", batch),
    field("Worker processes", workers),
    field("Label", label),
    field("Run directory name", name),
  );
  const outputsBox = el("details", {}, el("summary", { text: "Outputs to store" }), outputs, outputsNote);

  const example = () => radios.find(([r]) => r.checked)?.[1] || null;
  const sync = () => {
    const ex = example();
    filesBox.hidden = Boolean(ex);
    date.value = ex ? ex.valuation_date : date.value || model.valuation_date || "";
    batch.value = ex ? String(ex.batch_size) : batch.value;
  };
  for (const [radio] of radios) radio.addEventListener("change", sync);
  sync();

  const request = () => {
    const ex = example();
    const inputs = ex
      ? { example: ex.id }
      : {
          model_points: paths.model_points.value,
          tables: Object.fromEntries(Object.entries(paths.tables).map(([k, input]) => [k, input.value])),
          scenarios: Object.fromEntries(Object.entries(paths.scenarios).map(([k, input]) => [k, input.value])),
        };
    const chosen = [...outputs.selectedOptions].map((o) => o.value);
    const defaults = build.outputs || [];
    const same = chosen.length === defaults.length && chosen.every((v) => defaults.includes(v));
    return {
      build: build.id,
      inputs,
      name: name.value.trim(),
      settings: {
        valuation_date: date.value,
        horizon: horizon.value === "" ? null : Number(horizon.value),
        batch_size: batch.value === "" ? null : Number(batch.value),
        workers: workers.value === "" ? null : Number(workers.value),
        label: label.value,
        outputs: same || !chosen.length ? null : chosen,
      },
    };
  };
  const check = el("button", { type: "button", class: "button", text: "Check the inputs" });
  const go_ = el("button", { type: "button", class: "button primary", text: "Run" });
  const status = el("div", { role: "status" });
  check.addEventListener("click", async () => {
    const from = drawnAt; // the page the button was drawn on, not where the user is now
    const what = `a check of the inputs of ${model.name}`;
    check.disabled = true;
    try {
      const { job } = await api("/api/inputs/check", { body: request() });
      const r = await watchJob(job, status, "Loading and checking the inputs");
      const passed = () => el("div", { class: "status good", text: `The inputs pass every check: ${plural(r.policies, "policy", "policies")}, ${plural(r.scenarios, "scenario")}, a horizon of ${r.horizon} months, ${plural(r.batches, "batch", "batches")} on ${plural(r.processes, "process", "processes")}.` });
      if (seenNow(status, drawnBy)) status.replaceChildren(passed());
      else endedAway(status, what, from, false, passed);
    } catch (err) {
      if (cancelled(err)) status.replaceChildren(notice("The check was cancelled before it started."));
      else if (seenNow(status, drawnBy)) status.replaceChildren(errorBox(err, "The inputs do not pass"));
      else endedAway(status, what, from, true, () => errorBox(err, "The inputs do not pass"));
    } finally {
      check.disabled = false;
    }
  });
  go_.addEventListener("click", async () => {
    const from = drawnAt; // the page the button was drawn on, not where the user is now
    go_.disabled = true;
    let asked = null;
    try {
      asked = request();
      const { job } = await api("/api/run", { body: asked });
      const r = await watchJob(job, status, "Running");
      if (seenNow(status, drawnBy)) go(runHref(r.run));
      else {
        endedAway(status, job.label, runHref(r.run), false);
        // the card stays, and the run took its name: the next run gets another, unless one was
        // typed meanwhile
        if (name.value.trim() === asked.name) name.value = `${slug(model.name)}-${stamp()}`;
      }
    } catch (err) {
      if (cancelled(err)) status.replaceChildren(notice("The run was cancelled before it started."));
      else if (seenNow(status, drawnBy)) status.replaceChildren(errorBox(err, "The run failed"));
      else endedAway(status, `a run of ${model.name}`, from, true, () => errorBox(err, "The run failed"));
      // only a name that is taken needs another; one the user typed stays for the next try
      if (err instanceof ApiFailure && err.status === 409 && /exists already/.test(err.message)) name.value = `${slug(model.name)}-${stamp()}`;
    } finally {
      go_.disabled = false;
    }
  });
  card.append(choice, filesBox, settings, outputsBox, el("div", { class: "row" }, go_, check), status);
  return card;
}

function variablesView(build, focus) {
  const box = el("div", { class: "stack" });
  const filter = el("input", { type: "text", placeholder: "Filter by name, unit, role or formula", "aria-label": "Filter the variables" });
  const detail = el("div");
  const tableHost = el("div");
  const byName = new Map(build.variables.map((v) => [v.name, v]));
  const show = (name) => {
    const v = byName.get(name);
    if (!v) return;
    detail.replaceChildren(variableCard(v, show));
    detail.scrollIntoView({ block: "nearest", behavior: "smooth" });
  };
  const columns = [
    { label: "Variable", key: "name", mono: true },
    { label: "Unit", key: "unit" },
    { label: "Role", key: "role", render: (v) => v.role || "" },
    { label: "Timing", key: "timing", render: (v) => v.timing || "" },
    { label: "Runs", key: "direction", render: (v) => (v.direction === "none" ? (v.time ? "whole" : "once") : v.direction) },
    { label: "From t", key: "start", num: true, render: (v) => (v.time ? String(v.start) : "") },
    { label: "Stored", key: "stored", render: (v) => (v.stored ? "yes" : "") },
    { label: "Formula", key: "formula", mono: true, render: (v) => el("span", { title: v.formula, text: v.formula.length > 90 ? `${v.formula.slice(0, 88)}...` : v.formula }) },
  ];
  const fill = () => {
    const q = filter.value.trim().toLowerCase();
    const rows = build.variables.filter((v) => !q || [v.name, v.unit, v.role, v.formula, v.doc].some((s) => s && String(s).toLowerCase().includes(q)));
    tableHost.replaceChildren(dataTable(columns, rows, { onRow: (v) => show(v.name), empty: "No variable matches." }));
  };
  filter.addEventListener("input", fill);
  fill();
  const unused = build.variables.filter((v) => !v.needed).map((v) => v.name);
  box.append(
    el("p", { class: "hint", text: `${plural(build.variables.length, "variable")} in ${plural(build.components.length, "component")}, evaluated in dependency order; a recursive component steps forward from t = 1 or backward from the horizon.${unused.length ? ` Not computed (nothing needs them): ${unused.join(", ")}.` : ""}` }),
    filter,
    tableHost,
    detail,
  );
  if (focus) show(focus);
  return box;
}

function variableCard(v, show) {
  const card = el("div", { class: "card stack" });
  card.append(el("div", { class: "row" }, el("h2", { text: v.name }), chip(v.unit), v.role ? chip(v.role) : null, v.timing ? chip(`paid at the ${v.timing}`) : null, chip(v.direction === "none" ? "not recursive" : `${v.direction} recursion`)));
  if (v.doc) card.append(el("p", { text: v.doc }));
  if (v.loc) card.append(el("div", { class: "muted defined" }, "defined at ", where(v.loc)));
  if (v.source) card.append(el("pre", { class: "code", text: v.source }));
  const facts = [["Formula", el("code", { text: v.formula })]];
  if (v.init) facts.push(["Value at t = 0 (init)", el("code", { text: v.init })]);
  if (v.terminal) facts.push(["Value at the horizon (terminal)", el("code", { text: v.terminal })]);
  if (v.formula_range) facts.push(["Formula applies", `from t = ${v.formula_range[0]} to t = ${v.formula_range[1]}`]);
  const links = (names) => {
    if (!names.length) return "nothing";
    const span = el("span", { class: "row" });
    for (const item of names) {
      const target = typeof item === "string" ? item : item.var;
      const text = typeof item === "string" ? item : item.label;
      const a = el("a", { href: "#", class: "mono", text });
      a.addEventListener("click", (e) => {
        e.preventDefault();
        show(target);
      });
      span.append(a);
    }
    return span;
  };
  facts.push(["Reads", links(v.reads)], ["Read by", links(v.read_by)]);
  if (v.inputs.length) facts.push(["Inputs", el("span", { class: "mono", text: v.inputs.join(", ") })]);
  card.append(kv(facts));
  return card;
}

function declaredView(model, which = "inputs") {
  const box = el("div", { class: "stack" });
  if (which === "invariants") {
    box.append(el("p", { class: "hint", text: "Checked on every batch of every run, after the built-in ones that the variables' roles call for: in force never negative or rising, decrements within what is in force, probabilities in [0, 1], and reserves reaching their terminal values. A failure stops the run with an error that names the policy and the month." }));
    box.append(
      dataTable(
        [
          { label: "Invariant", key: "name" },
          { label: "Kind", key: "kind" },
          { label: "Tolerance", key: "tol", num: true },
          { label: "Months", key: "months", render: (i) => (i.months ? i.months.join(" to ") : "every month") },
          { label: "Defined at", key: "loc", sort: false, render: (i) => where(i.loc) || "" },
          { label: "About", key: "doc" },
        ],
        model.invariants,
        { empty: "The model declares no invariants of its own." },
      ),
    );
    return box;
  }
  const schema = model.schema;
  if (schema) {
    box.append(el("h3", { text: "Model points" }));
    box.append(el("p", { class: "hint", text: `Keyed on ${schema.id_field}${schema.horizon ? `; the largest ${schema.horizon} sets the horizon` : ""}${schema.valuation_date_rule ? "; the model also has a rule on the valuation date" : ""}.` }));
    const bounds = (f) => [f.gt !== null ? `> ${f.gt}` : null, f.ge !== null ? `>= ${f.ge}` : null, f.lt !== null ? `< ${f.lt}` : null, f.le !== null ? `<= ${f.le}` : null].filter(Boolean).join(", ");
    box.append(
      dataTable(
        [
          { label: "Field", key: "name", mono: true },
          { label: "Kind", key: "kind" },
          { label: "Unit or values", key: "unit", render: (f) => (f.enum ? f.enum.join(", ") : f.unit || "") },
          { label: "Bounds", key: "bounds", sort: false, render: bounds },
          { label: "About", key: "doc" },
        ],
        schema.fields,
      ),
    );
    if (schema.derived.length) {
      box.append(el("h3", { text: "Derived at load" }));
      box.append(dataTable([{ label: "Field", key: "name", mono: true }, { label: "Kind", key: "kind" }, { label: "Unit", key: "unit" }, { label: "About", key: "doc" }], schema.derived));
    }
    if (schema.rules.length) {
      box.append(el("h3", { text: "Rules on the model points" }));
      box.append(dataTable([{ label: "Rule", key: "name" }, { label: "About", key: "doc" }], schema.rules));
    }
  }
  if (model.tables.length) {
    box.append(el("h3", { text: "Assumption tables" }));
    for (const t of model.tables) {
      const keys = t.keys.map((k) => `${k.name} ${k.enum ? `in {${k.enum.join(", ")}}` : `${k.lo}..${k.hi}`}`).join("; ");
      const columns = t.columns.map((c) => `${c.name} [${c.unit}]`).join(", ");
      box.append(
        el(
          "div",
          { class: "card" },
          el("div", { class: "row" }, el("strong", { class: "mono", text: t.name }), t.doc ? el("span", { class: "muted", text: t.doc }) : null),
          kv([
            ["Keys", keys || "none (one row)"],
            ["Columns", columns],
            ["Metadata it must carry", t.metadata.length ? t.metadata.map((m) => `halley.${m}`).join(", ") : undefined],
            ["Rules", t.rules.length ? t.rules.map((r) => r.name).join("; ") : undefined],
          ]),
        ),
      );
    }
  }
  if (model.scenarios.length) {
    box.append(el("h3", { text: "Scenario inputs" }));
    box.append(
      dataTable(
        [
          { label: "Input", key: "name", mono: true },
          { label: "Unit", key: "unit" },
          { label: "Column", key: "column", mono: true },
          { label: "Must cover", key: "covers", render: (s) => (s.covers ? s.covers.join(" to ") : "the months read") },
          { label: "About", key: "doc" },
        ],
        model.scenarios,
      ),
    );
    box.append(el("p", { class: "hint", text: "A scenario file has a scenario column (0, 1, ...), a month column t or u, and the value column." }));
  }
  if (model.input_rules.length) {
    box.append(el("h3", { text: "Rules across the inputs" }));
    box.append(dataTable([{ label: "Rule", key: "name" }, { label: "About", key: "doc" }], model.input_rules));
  }
  if (model.kind === "aggregation") {
    box.append(el("h3", { text: "Bases and series" }));
    box.append(el("pre", { class: "code", text: JSON.stringify({ bases: model.bases, series: model.series, member_rules: model.member_rules }, null, 2) }));
  }
  return box;
}

async function codeView(build) {
  const { source, artefact } = await api("/api/build/source", { params: { id: build.id } });
  return el("div", { class: "stack" }, el("p", { class: "hint", text: `The code the ${build.backend} backend generated from the checked formulas; a run writes it to generated/${artefact}. It is never edited by hand, and every error points at the model's own lines instead.` }), el("pre", { class: "code tall", text: source }));
}

// --------------------------------------------------------------------------------- runs page

async function runsPage() {
  const { root, runs } = await api("/api/runs");
  const page = el("div");
  const refresh = el("button", { type: "button", class: "button", text: "Refresh" });
  refresh.addEventListener("click", () => route());
  page.append(el("div", { class: "page-head" }, el("div", {}, el("h1", { text: "Runs" }), el("div", { class: "sub", text: `Run directories in ${root}` })), refresh));
  const picked = new Set();
  const compare = el("button", { type: "button", class: "button primary", text: "Compare the two selected", disabled: true });
  compare.addEventListener("click", () => {
    const [a, b] = [...picked];
    go(`#/compare?${new URLSearchParams({ a, b })}`);
  });
  const pick = (run) => {
    const box = el("input", { type: "checkbox", "aria-label": `Select ${run.name}` });
    box.checked = picked.has(run.dir);
    box.addEventListener("change", () => {
      if (box.checked) picked.add(run.dir);
      else picked.delete(run.dir);
      compare.disabled = picked.size !== 2;
    });
    return box;
  };
  const checks = (r) => {
    if (r.error) return el("span", { class: "status bad", text: "unreadable" });
    const n = r.invariants?.checked ?? 0;
    return el("span", { class: "status good", text: `${n} passed` });
  };
  const columns = [
    { label: "", key: "pick", sort: false, render: pick },
    { label: "Created", key: "created_at", render: (r) => when(r.created_at) },
    { label: "Name", key: "name", render: (r) => el("a", { href: runHref(r.dir), text: r.name }) },
    { label: "Model", key: "model", render: (r) => (r.model ? String(r.model).split(".").pop() : r.error || "") },
    { label: "Label", key: "label" },
    { label: "Run id", key: "run_id", mono: true, render: (r) => short(r.run_id, 16) },
    { label: "Lanes", key: "policies", num: true, render: (r) => (r.policies === undefined ? "" : `${r.policies} x ${r.scenarios} x ${r.horizon + 1} months`) },
    { label: "Backend", key: "backend" },
    { label: "Checks", key: "invariants", sort: false, render: checks },
    { label: "", key: "in_session", sort: false, render: (r) => (r.in_session ? chip("explainable") : r.rerun ? chip("rerun") : "") },
  ];
  page.append(dataTable(columns, runs, { onRow: (r) => go(runHref(r.dir)), empty: "No runs yet. Build a model and run it, or open a run directory below." }));
  const path = el("input", { type: "text", placeholder: "C:\\work\\runs\\first-run", "aria-label": "Run directory" });
  const open = el("button", { type: "submit", class: "button", text: "Open" });
  const status = el("div");
  const form = el("form", { class: "card stack" }, el("h2", { text: "Open a run directory" }), el("p", { class: "hint", text: "Any directory a run wrote, wherever it is: it holds manifest.json." }), el("div", { class: "row" }, el("div", { class: "grow" }, path), open), status);
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      const r = await api("/api/runs/open", { body: { path: path.value } });
      go(runHref(r.dir));
    } catch (err) {
      status.replaceChildren(errorBox(err, "That is not a run directory"));
    }
  });
  page.append(el("div", { class: "row" }, compare), form);
  return page;
}

// ---------------------------------------------------------------------------------- run page

const runState = new Map();

function stateOf(dir) {
  if (!runState.has(dir)) runState.set(dir, { slots: new Slots(), policies: null, variable: null, scenario: 0, explain: null, trail: [] });
  return runState.get(dir);
}

async function runPage(ctx, dir, tab, params) {
  const verify = params.get("verify") === "1";
  const run = await api("/api/run", { params: { dir, verify: verify ? "1" : null } });
  const state = stateOf(dir);
  const page = el("div");
  const checks = run.invariants.length;
  const head = el("div", { class: "row" }, chip(run.kind === "aggregation" ? "aggregation run" : "projection run"), chip(`run id ${run.run_id}`, "mono"), el("span", { class: "status good", text: `${checks} checks passed` }));
  if (run.explainable) head.append(chip("explainable in this session"));
  if (verify) head.append(el("span", { class: "status good", text: "every recorded hash verified" }));
  page.append(
    el(
      "div",
      { class: "page-head" },
      el("div", {}, el("h1", { text: run.name }), el("div", { class: "sub", text: `${run.model.name} · ${when(run.created_at)}${run.settings.label ? ` · ${run.settings.label}` : ""} · ${plural(run.n_policies, run.lane === "group" ? "group" : "policy", run.lane === "group" ? "groups" : "policies")} x ${plural(run.n_scenarios, "scenario")} x ${run.H + 1} months` })),
      head,
    ),
  );
  const tabs = [
    ["overview", "Overview"],
    ["charts", "Charts"],
    ["policies", run.lane === "group" ? "Groups" : "Policies"],
    ["explain", "Explain"],
    ["invariants", "Checks"],
    ["inputs", "Inputs"],
    ["files", "Files"],
  ];
  page.append(tabsBar(tabs, tab, (key) => runHref(dir, key)));
  const body = el("div");
  page.append(body);
  if (tab === "charts") body.append(chartsView(run, state, params));
  else if (tab === "policies") body.append(policiesView(run, state));
  else if (tab === "explain") body.append(explainView(run, state, params));
  else if (tab === "invariants") body.append(invariantsView(run));
  else if (tab === "inputs") body.append(inputsView(run, params));
  else if (tab === "files") body.append(filesView(run));
  else body.append(overviewView(run, verify));
  return page;
}

// `drawnBy` is the route that drew the page the button is on: a card drawn after its page's
// own request has answered (Explain's, when explain is refused) is still that route's.
function rerunButton(run, then, drawnBy = renderId, drawnAt = location.hash) {
  const replay = replays(run);
  const word = replay ? "replay" : "rerun";
  const button = el("button", { type: "button", class: "button primary", text: replay ? "Replay it from its directory" : "Rerun it from its own inputs" });
  const status = el("div", { role: "status" });
  button.addEventListener("click", async () => {
    const from = drawnAt; // the page the button was drawn on, not where the user is now
    button.disabled = true;
    try {
      const { job } = await api("/api/run/rerun", { body: { dir: run.dir } });
      const r = await watchJob(job, status, `Rebuilding the model and ${replay ? "replaying" : "rerunning"}`);
      remember(`halley-rerun:${r.run}`, r);
      if (seenNow(status, drawnBy)) then(r);
      else endedAway(status, job.label, runHref(r.run), false, null, () => then(r));
    } catch (err) {
      const title = `The ${word} failed`;
      if (cancelled(err)) status.replaceChildren(notice(`The ${word} was cancelled before it started.`));
      else if (seenNow(status, drawnBy)) status.replaceChildren(errorBox(err, title));
      else endedAway(status, `a ${word} of ${run.name}`, from, true, () => errorBox(err, title));
    } finally {
      button.disabled = false;
    }
  });
  return el("div", { class: "stack" }, el("div", { class: "row" }, button), status);
}

// A stage run that run_aggregation made keeps its groups and membership and is replayed; any
// other run, a stage's made by run() too, is run again: as the server decides.
function replays(run) {
  const kinds = new Set(run.inputs.map((i) => i.kind));
  return kinds.has("groups") && kinds.has("membership");
}

function rerunReport(r) {
  const same = r.same_run_id && r.outputs?.bit_for_bit;
  const lines = [same ? "Reproduced: the same run id, and every stored value the same bit for bit." : r.same_run_id ? "The same run id, but some values differ." : "Not the same run id."];
  if (r.outputs && !r.outputs.bit_for_bit) lines.push(`Variables that differ: ${r.outputs.differ.join(", ")}.`);
  if (r.outputs && r.outputs.bit_for_bit && !r.outputs.files) lines.push("The output files' bytes differ (another pyarrow writes them), though the values are the same.");
  for (const n of r.notes || []) lines.push(n);
  return el("div", { class: same ? "notice" : "warning", text: lines.join(" ") });
}

function overviewView(run, verify) {
  const box = el("div", { class: "stack" });
  if (!run.explainable) {
    box.append(
      el(
        "div",
        { class: "card stack" },
        el("h2", { text: "Reproduce this run" }),
        el("p", { text: replays(run) ? "Replays the stage from this directory alone: the copies of its groups, membership, series, tables and scenarios, verified against their hashes, with the recorded settings. The replay goes to the reruns folder, is compared with this run bit for bit, and holds what explain needs." : "Builds the model the manifest names, with its recorded build-time settings, and runs it again on this directory's own copies of its inputs, each verified against its hash, with the recorded settings. The rerun goes to the reruns folder, is compared with this run bit for bit, and holds what explain needs." }),
        rerunButton(run, (r) => go(runHref(r.run))),
      ),
    );
  }
  const report = recall(`halley-rerun:${run.dir}`);
  if (report) box.prepend(rerunReport(report));
  const m = run.model;
  const s = run.settings;
  const d = run.dimensions;
  box.append(
    el(
      "div",
      { class: "card" },
      el("h2", { text: "Model" }),
      kv([
        ["Model", el("span", { class: "mono", text: m.name })],
        ["Source", m.source_file],
        ["Source sha256", el("span", { class: "mono", text: short(m.source_sha256, 16) })],
        ["Git commit", m.git_commit || "not in a repository"],
        ["Build-time settings", run.statics_text && run.statics_text !== "{}" ? run.statics_text : "none"],
        ["IR hash", el("span", { class: "mono", text: run.ir_sha256 })],
        ["Halley", run.halley],
      ]),
    ),
    el(
      "div",
      { class: "card" },
      el("h2", { text: "Settings" }),
      kv([
        ["Valuation date", s.valuation_date],
        ["Horizon", s.horizon === null ? `${d.horizon} months, from the model points` : `${s.horizon} months`],
        ["Batch size", s.batch_size],
        ["Outputs stored", s.outputs === null ? "the model's outputs" : s.outputs.join(", ")],
        ["Label", s.label || "none"],
        ["Worker processes", s.workers],
        ["Run id", el("span", { class: "mono", text: run.run_id })],
        ["Manifest sha256", el("span", { class: "mono", text: run.manifest_sha256 || "" })],
      ]),
      el("p", { class: "hint", text: "The run id covers the model's IR, every input's hash, the settings that change the numbers (all but the label and the workers) and the backend with its versions: two runs with the same id declare the same computation." }),
    ),
    el(
      "div",
      { class: "card" },
      el("h2", { text: "Backend" }),
      kv(Object.entries(run.backend).map(([k, v]) => [k, k === "cpu_features" ? `${String(v).slice(0, 80)}...` : v])),
    ),
  );
  if (Object.keys(run.stage).length) {
    box.append(el("div", { class: "card" }, el("h2", { text: "Stage records" }), el("pre", { class: "code tall", text: JSON.stringify(run.stage, null, 2) })));
  }
  if (!verify) {
    const check = el("a", { href: `${runHref(run.dir)}?verify=1`, text: "Verify every file against the hash the manifest records" });
    box.append(el("p", {}, check));
  }
  return box;
}

function variablesOf(run) {
  const monthly = run.variables.filter((v) => v.time);
  const single = run.variables.filter((v) => !v.time);
  return { monthly, single };
}

function variableSelect(run, current, { allowSingle = true } = {}) {
  const select = el("select", { "aria-label": "Variable" });
  const { monthly, single } = variablesOf(run);
  const group = (label, items) => {
    if (!items.length) return;
    const g = el("optgroup", { label });
    for (const v of items) g.append(el("option", { value: v.name, text: `${v.name}${v.unit ? ` [${v.unit}]` : ""}` }));
    select.append(g);
  };
  group("Monthly", monthly);
  if (allowSingle) group("One value per policy", single);
  if (current) select.value = current;
  return select;
}

function defaultVariable(run) {
  const reserve = run.variables.find((v) => v.role === "reserve" && v.time);
  return (reserve || run.variables.find((v) => v.time) || run.variables[0])?.name || null;
}

// Find a policy by its id: among the ids the run's page holds, or by asking the server, for a run
// with more policies than that. Gives its index, or null.
async function findPolicy(run, text) {
  const wanted = text.trim();
  if (!wanted) return null;
  const local = run.ids.findIndex((id) => String(id) === wanted);
  if (local >= 0) return local;
  if (!run.ids_truncated) return null;
  const found = await api("/api/run/find", { params: { dir: run.dir, id: wanted } });
  return found.p.length ? found.p[0] : null;
}

function policyLabel(run, p) {
  return p < run.ids.length ? String(run.ids[p]) : `#${p + 1}`;
}

function policyPicker(run, chosen, onChange, max = 8) {
  const box = el("div", { class: "row" });
  const listId = `ids-${Math.random().toString(36).slice(2)}`;
  const list = el("datalist", { id: listId });
  run.ids.slice(0, 5000).forEach((id) => list.append(el("option", { value: String(id) })));
  const input = el("input", { type: "text", list: listId, placeholder: `Add a ${run.lane} by id`, "aria-label": `Add a ${run.lane}` });
  const message = el("span", { class: "hint" });
  const add = async () => {
    message.textContent = "";
    const index = await findPolicy(run, input.value);
    if (index === null) {
      message.textContent = `No ${run.lane} has the id ${input.value.trim()}.`;
      return;
    }
    if (chosen.includes(index) || chosen.length >= max) return;
    onChange([...chosen, index]);
  };
  input.addEventListener("change", add);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") add();
  });
  for (const p of chosen) {
    const remove = el("button", { type: "button", class: "legend-item", title: "Remove", "aria-label": `Remove ${policyLabel(run, p)}` }, el("span", { text: `${policyLabel(run, p)} \u00d7` }));
    remove.addEventListener("click", () => onChange(chosen.filter((q) => q !== p)));
    box.append(remove);
  }
  box.append(input, list, message);
  if (chosen.length >= max) box.append(el("span", { class: "hint", text: `A chart shows at most ${max}.` }));
  if (run.ids_truncated) box.append(el("span", { class: "hint", text: `The list suggests the first 5000 ids of ${run.n_policies}; any id can be typed.` }));
  return box;
}

function chartsView(run, state, params) {
  const box = el("div", { class: "stack" });
  if (!run.variables.length) {
    box.append(notice("The run stores no variables."));
    return box;
  }
  state.variable = params.get("var") || state.variable || defaultVariable(run);
  if (params.get("p") !== null) {
    const p = Number(params.get("p"));
    state.policies = [p, ...(state.policies || []).filter((q) => q !== p)].slice(0, 8);
  }
  state.policies = state.policies || [0];
  const select = variableSelect(run, state.variable);
  const scenario = el("select", { "aria-label": "Scenario" });
  for (let s = 0; s < run.n_scenarios; s++) scenario.append(el("option", { value: String(s), text: `scenario ${s}` }));
  if (run.n_scenarios > 1) scenario.append(el("option", { value: "all", text: "every scenario, first policy" }));
  scenario.value = String(state.scenario);
  const pickerHost = el("div");
  const charts = el("div");
  const controls = el("div", { class: "fields" }, field("Variable", select), run.n_scenarios > 1 ? field("Scenario", scenario) : null);
  box.append(controls, pickerHost, charts);
  let seq = 0;
  const draw = async () => {
    const mine = ++seq;
    state.variable = select.value;
    state.scenario = scenario.value === "all" ? "all" : Number(scenario.value);
    pickerHost.replaceChildren(
      el("div", { class: "field" }, el("span", { text: run.lane === "group" ? "Groups" : "Policies" }), policyPicker(run, state.policies, (next) => {
        state.policies = next.length ? next : [0];
        draw();
      })),
    );
    const info = run.variables.find((v) => v.name === state.variable);
    const everyScenario = state.scenario === "all";
    const ps = everyScenario ? [state.policies[0]] : state.policies;
    charts.replaceChildren(progress("Reading the values"));
    try {
      const [lanes, totals] = await Promise.all([
        api("/api/run/lanes", { params: { dir: run.dir, var: state.variable, p: ps.join(","), s: everyScenario ? "all" : String(state.scenario) } }),
        api("/api/run/totals", { params: { dir: run.dir, var: state.variable, s: everyScenario ? "all" : String(state.scenario) } }),
      ]);
      if (mine !== seq) return;
      const out = [];
      if (lanes.time) {
        state.slots.keep(lanes.series.map((s) => (everyScenario ? `s${s.scenario}` : s.p)));
        out.push(
          lineChart({
            title: `${state.variable} ${everyScenario ? `of ${run.lane} ${lanes.series[0]?.id} in every scenario` : `by ${run.lane}`}`,
            unit: info?.unit || "",
            x: lanes.t,
            series: lanes.series.map((s) => ({ label: everyScenario ? `scenario ${s.scenario}` : `${s.id}`, values: s.values, slot: state.slots.get(everyScenario ? `s${s.scenario}` : s.p) })),
          }),
        );
        if (lanes.asked > lanes.series.length) out.push(notice(`Showing ${lanes.series.length} of the ${lanes.asked} series asked for: a chart draws at most ${lanes.series.length}.`));
        out.push(
          lineChart({
            title: `${state.variable} summed over every ${run.lane}${totals.series.length > 1 ? ", by scenario" : ""}`,
            unit: info?.unit || "",
            x: totals.t,
            series: totals.series.map((s) => ({ label: `scenario ${s.scenario}`, values: s.values, slot: s.scenario % 8 + 1 })),
          }),
        );
        if (totals.asked > totals.series.length) out.push(notice(`Showing the totals of ${totals.series.length} of ${totals.asked} scenarios.`));
      } else {
        out.push(
          el("div", { class: "card stack" }, el("h2", { text: `${state.variable}${info?.unit ? ` [${info.unit}]` : ""}` }), el("p", { class: "hint", text: "One value per policy and scenario, without a month." }), dataTable([{ label: run.lane === "group" ? "Group" : "Policy", key: "id" }, { label: "Scenario", key: "scenario", num: true }, { label: "Value", key: "value", num: true, render: (r) => formatValue(r.value) }], lanes.series.map((s) => ({ id: s.id, scenario: s.scenario, value: s.values[0] })))),
          el("div", { class: "card" }, kv([[`Summed over every ${run.lane}`, totals.series.map((s) => `${formatValue(s.values[0])} (scenario ${s.scenario})`).join("; ")]])),
        );
      }
      charts.replaceChildren(...out);
    } catch (err) {
      if (mine === seq) charts.replaceChildren(errorBox(err));
    }
  };
  select.addEventListener("change", draw);
  scenario.addEventListener("change", draw);
  draw();
  return box;
}

function policiesView(run, state) {
  const box = el("div", { class: "stack" });
  const select = variableSelect(run, state.variable || defaultVariable(run));
  const month = el("input", { type: "number", min: "0", max: String(run.H), step: "1", value: "0" });
  const scenario = el("select", { "aria-label": "Scenario" });
  for (let s = 0; s < run.n_scenarios; s++) scenario.append(el("option", { value: String(s), text: `scenario ${s}` }));
  if (state.scenario !== "all") scenario.value = String(state.scenario || 0);
  const sort = el("select", { "aria-label": "Order" }, el("option", { value: "value:desc", text: "largest value first" }), el("option", { value: "value:asc", text: "smallest value first" }), el("option", { value: "id:asc", text: "by id" }));
  const host = el("div");
  const pager = el("div", { class: "row" });
  let offset = 0;
  let seq = 0;
  const limit = 100;
  const controls = el("div", { class: "fields" }, field("Variable", select), field("Month t", month), run.n_scenarios > 1 ? field("Scenario", scenario) : null, field("Order", sort));
  box.append(controls, host, pager);
  const load = async () => {
    const mine = ++seq;
    const [by, order] = sort.value.split(":");
    const info = run.variables.find((v) => v.name === select.value);
    month.disabled = !info?.time;
    host.replaceChildren(progress("Reading"));
    try {
      const r = await api("/api/run/policies", { params: { dir: run.dir, var: select.value, t: month.value || 0, s: scenario.value || 0, sort: by, order, offset, limit } });
      if (mine !== seq) return;
      const columns = [{ label: run.lane === "group" ? "Group" : "Policy", key: "id", sort: false, render: (row) => String(row.id) }];
      columns.push({ label: `${r.variable}${r.t !== null ? ` at t = ${r.t}` : ""}`, key: "value", num: true, sort: false, render: (row) => formatValue(row.value) });
      r.fields.forEach((f, i) => columns.push({ label: f, key: f, sort: false, num: typeof r.rows[0]?.fields[i] === "number", render: (row) => cell(row.fields[i]) }));
      host.replaceChildren(
        dataTable(columns, r.rows, {
          onRow: (row) => {
            state.variable = select.value;
            state.scenario = Number(scenario.value || 0);
            const params = { p: row.p, var: select.value, s: state.scenario };
            if (run.explainable) go(runHref(run.dir, "explain", { ...params, t: info?.time ? month.value || 0 : "" }));
            else go(runHref(run.dir, "charts", params));
          },
        }),
      );
      pager.replaceChildren(el("span", { class: "hint", text: `${r.offset + 1} to ${Math.min(r.offset + r.limit, r.total)} of ${r.total}. ${run.explainable ? "Open a row to explain its value." : "Open a row to chart it."}` }));
      if (offset > 0) {
        const prev = el("button", { type: "button", class: "button", text: "Previous" });
        prev.addEventListener("click", () => {
          offset = Math.max(0, offset - limit);
          load();
        });
        pager.append(prev);
      }
      if (offset + limit < r.total) {
        const next = el("button", { type: "button", class: "button", text: "Next" });
        next.addEventListener("click", () => {
          offset += limit;
          load();
        });
        pager.append(next);
      }
    } catch (err) {
      if (mine === seq) host.replaceChildren(errorBox(err));
    }
  };
  for (const control of [select, month, scenario, sort]) {
    control.addEventListener("change", () => {
      offset = 0;
      load();
    });
  }
  load();
  return box;
}

function explainView(run, state, params) {
  const drawnBy = renderId;
  const drawnAt = location.hash;
  const box = el("div", { class: "stack" });
  if (!run.explainable) {
    box.append(
      el(
        "div",
        { class: "card stack" },
        el("h2", { text: "Explain needs the model and the inputs" }),
        el("p", { text: `A run read back from its directory holds its outputs, not the model or the loaded inputs that explain evaluates formulas with. ${replays(run) ? "Replay it here from its directory: the replay" : "Rerun it here from its own inputs: the rerun"} is checked against this run bit for bit, and its values can then be explained.` }),
        rerunButton(run, (r) => go(runHref(r.run, "explain", Object.fromEntries(params)))),
      ),
    );
    return box;
  }
  const fromParams = params.get("var") ? { p: Number(params.get("p") || 0), var: params.get("var"), t: params.get("t") === null || params.get("t") === "" ? null : Number(params.get("t")), s: Number(params.get("s") || 0) } : null;
  if (fromParams) state.trail = [fromParams];
  if (!state.trail.length) {
    const v = defaultVariable(run);
    const info = run.variables.find((x) => x.name === v);
    state.trail = [{ p: 0, var: v, t: info?.time ? 0 : null, s: 0 }];
  }
  const current = () => state.trail[state.trail.length - 1];
  // the policy is typed by its id, with the first ids suggested: a run may have more policies
  // than a list can hold, and the field reaches every one of them
  const listId = `explain-ids-${Math.random().toString(36).slice(2)}`;
  const ids = el("datalist", { id: listId });
  run.ids.slice(0, 5000).forEach((id) => ids.append(el("option", { value: String(id) })));
  const policy = el("input", { type: "text", list: listId, "aria-label": run.lane });
  const policyNote = el("span", { class: "hint" });
  const select = variableSelect(run, current().var);
  const month = el("input", { type: "number", min: "0", max: String(run.H), step: "1" });
  const scenario = el("select", { "aria-label": "Scenario" });
  for (let s = 0; s < run.n_scenarios; s++) scenario.append(el("option", { value: String(s), text: `scenario ${s}` }));
  const crumbs = el("nav", { class: "crumbs", "aria-label": "What you have opened" });
  const result = el("div");
  box.append(el("div", { class: "fields" }, field(run.lane === "group" ? "Group" : "Policy", el("div", { class: "stack" }, policy, ids, policyNote)), field("Variable", select), field("Month t", month), run.n_scenarios > 1 ? field("Scenario", scenario) : null), crumbs, result);
  let seq = 0;
  const sync = () => {
    const c = current();
    policy.value = policyLabel(run, c.p);
    select.value = c.var;
    const info = run.variables.find((x) => x.name === c.var);
    month.disabled = !info?.time;
    month.value = c.t === null || c.t === undefined ? "" : String(c.t);
    scenario.value = String(c.s);
  };
  const show = async () => {
    const mine = ++seq;
    sync();
    crumbs.replaceChildren();
    state.trail.forEach((step, i) => {
      if (i) crumbs.append(el("span", { class: "sep", text: "\u203a" }));
      const text = `${step.var}${step.t !== null && step.t !== undefined ? `(${step.t})` : ""}`;
      if (i === state.trail.length - 1) crumbs.append(el("strong", { class: "mono", text }));
      else {
        const a = el("a", { href: "#", class: "mono", text });
        a.addEventListener("click", (e) => {
          e.preventDefault();
          state.trail = state.trail.slice(0, i + 1);
          show();
        });
        crumbs.append(a);
      }
    });
    const c = current();
    result.replaceChildren(progress("Explaining"));
    try {
      const e = await api("/api/run/explain", { params: { dir: run.dir, p: c.p, var: c.var, t: c.t === null || c.t === undefined ? "" : c.t, s: c.s } });
      if (mine !== seq) return;
      result.replaceChildren(
        explanationCard(e, run, (target) => {
          if (target.kind === "variable") {
            const info = run.variables.find((x) => x.name === target.var);
            const t = info?.time ? (target.t ?? c.t ?? 0) : null;
            state.trail.push({ p: c.p, var: target.var, t, s: c.s });
            show();
          } else if (target.kind === "lane") {
            state.scenario = c.s;
            go(runHref(run.dir, "charts", { p: c.p, var: target.var }));
          } else if (target.kind === "table" || target.kind === "scenario" || target.kind === "series") {
            go(runHref(run.dir, "inputs", { kind: target.kind, name: target.name }));
          }
        }),
      );
    } catch (err) {
      if (mine !== seq) return;
      result.replaceChildren(errorBox(err, "Explain did not work"));
      // a run this session made and has since let go of to make room: rerun it to explain it
      if (err instanceof ApiFailure && err.status === 409 && err.detail?.rerun) {
        result.append(rerunButton(run, (r) => go(runHref(r.run, "explain", { p: c.p, var: c.var, t: c.t ?? "", s: c.s })), drawnBy, drawnAt));
      }
    }
  };
  // a change of one control keeps the others: the month, variable or scenario of the policy
  // shown, or another policy at the same month
  const restart = async (changed) => {
    const c = current();
    let p = c.p;
    if (changed === policy) {
      policyNote.textContent = "";
      const found = await findPolicy(run, policy.value);
      if (found === null) {
        policyNote.textContent = `No ${run.lane} has the id ${policy.value.trim()}.`;
        return;
      }
      p = found;
    }
    const info = run.variables.find((x) => x.name === select.value);
    state.trail = [{ p, var: select.value, t: info?.time ? Number(month.value || 0) : null, s: Number(scenario.value || 0) }];
    show();
  };
  for (const control of [policy, select, month, scenario]) control.addEventListener("change", () => restart(control));
  show();
  return box;
}

function explanationCard(e, run, open) {
  const card = el("div", { class: "card stack" });
  const scenarioText = run.n_scenarios > 1 ? `, scenario ${e.scenario}` : "";
  card.append(el("div", { class: "muted", text: `${e.variable}${e.t !== null ? ` at t = ${e.t}` : ""} for ${e.lane} ${e.policy}${scenarioText}` }));
  card.append(el("div", { class: "explain-value" }, el("span", { text: formatValue(e.value) }), " ", el("span", { class: "chart-unit", text: e.unit })));
  const facts = el("div", { class: "row" });
  if (e.role) facts.append(chip(e.role));
  if (e.timing) facts.append(chip(`paid at the ${e.timing} of the month`));
  facts.append(chip(e.direction === "none" ? "not recursive" : `${e.direction} recursion`));
  if (e.formula_range) facts.append(chip(`formula from t = ${e.formula_range[0]} to ${e.formula_range[1]}`));
  card.append(facts);
  if (e.loc) card.append(el("div", { class: "muted defined" }, "defined at ", where(e.loc)));
  if (e.boundary) card.append(notice(`${e.boundary.note}${e.boundary.expr ? ` Its ${e.boundary.kind} value is ${e.boundary.expr}.` : ""}`));
  if (e.doc) card.append(el("p", { text: e.doc }));
  if (e.source) card.append(el("pre", { class: "code", text: e.source }));
  const actions = { variable: "Explain", lane: "Chart", table: "Show the table", scenario: "Show the file", series: "Show the series" };
  const outside = (target) => target.kind === "variable" && target.t !== null && target.t !== undefined && (target.t < 0 || target.t > run.H);
  if (e.inputs.length) {
    card.append(el("h3", { text: e.boundary ? "What the formula would read (not used at this month)" : "Inputs" }));
    card.append(
      dataTable(
        [
          { label: "Input", key: "text", mono: true },
          { label: "Value", key: "value", num: true, sort: false, render: (i) => (i.error ? el("span", { class: "status bad", text: i.value }) : el("span", { class: i.unused ? "muted" : "", text: formatValue(i.value) })) },
          {
            label: "",
            key: "target",
            sort: false,
            render: (i) => {
              const action = i.target ? actions[i.target.kind] : null;
              if (!action) return "";
              if (outside(i.target)) {
                return el("button", { type: "button", class: "button", disabled: true, title: `Month ${i.target.t} is outside the run's months 0 to ${run.H}`, text: action });
              }
              const button = el("button", { type: "button", class: "button", text: action });
              button.addEventListener("click", () => open(i.target));
              return button;
            },
          },
        ],
        e.inputs,
      ),
    );
  } else {
    card.append(el("p", { class: "muted", text: "The formula reads nothing: it is a constant." }));
  }
  const r = e.recomputed;
  if (r.error) card.append(el("div", { class: "status bad", text: `The interpreter could not recompute it: ${r.error}` }));
  else if (r.same) card.append(el("div", { class: "status good", text: "Recomputed from the stored inputs by Halley's interpreter, independently of the generated code: the same value, bit for bit." }));
  else if (r.backend !== "numpy" && isNumber(r.relative) && r.relative <= 1e-9) {
    card.append(el("div", { class: "status good", text: `Recomputed by Halley's interpreter, which follows the NumPy backend: ${formatValue(r.value)}, within ${r.relative.toExponential(1)} relative of the ${r.backend} backend's value, inside the 1e-9 the backends are held to.` }));
  } else {
    card.append(el("div", { class: "status bad", text: `Recomputed by Halley's interpreter: ${formatValue(r.value)}, not the stored ${formatValue(e.value)}${isNumber(r.relative) ? ` (${r.relative.toExponential(1)} relative)` : ""}. In two documented cases (a power whose exponent is 0.5, 2 or -1 read from a model point field or a scalar, and a month-column exponent over a one-month window) the two can differ in the last place.` }));
  }
  const copy = el("button", { type: "button", class: "button", text: "Copy as text" });
  copy.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(e.text);
      copy.textContent = "Copied";
    } catch {
      copy.textContent = "The browser refused";
    }
  });
  card.append(el("div", { class: "row" }, copy));
  return card;
}

function invariantsView(run) {
  const box = el("div", { class: "stack" });
  box.append(el("p", { class: "hint", text: "The checks the run made on its own results, built-in first, then the model's: every one passed, since a failing check stops a run before it writes anything. The worst residual is how close it came to its tolerance." }));
  box.append(
    dataTable(
      [
        { label: "Check", key: "name" },
        { label: "Result", key: "passed", render: (r) => el("span", { class: `status ${r.passed ? "good" : "bad"}`, text: r.passed ? "passed" : "failed" }) },
        { label: "Worst residual", key: "worst", num: true, render: (r) => formatValue(r.worst) },
        { label: "Detail", key: "detail" },
      ],
      run.invariants,
      { empty: "The run recorded no checks." },
    ),
  );
  return box;
}

function inputsView(run, params) {
  const box = el("div", { class: "stack" });
  const preview = el("div");
  let seq = 0;
  const show = async (row) => {
    const mine = ++seq;
    preview.replaceChildren(progress("Reading the copy and checking its hash"));
    try {
      const kind = row.kind;
      const r = await api("/api/run/input", { params: { dir: run.dir, kind, name: kind === "model_points" || kind === "groups" || kind === "derived_groups" || kind === "membership" ? "" : row.name } });
      if (mine !== seq) return;
      const columns = r.columns.map((c, i) => ({ label: `${c.name} (${c.type})`, key: String(i), sort: false, num: /int|double|float|decimal/.test(c.type), render: (cells) => cell(cells[i]) }));
      preview.replaceChildren(
        el(
          "div",
          { class: "card stack" },
          el("div", { class: "row" }, el("h2", { text: `${row.kind === "model_points" ? "Model points" : row.name}` }), chip(`sha256 ${short(r.sha256, 16)}`, "mono"), chip("hash verified")),
          Object.keys(r.metadata).length ? kv(Object.entries(r.metadata).map(([k, v]) => [`halley.${k}`, v])) : null,
          dataTable(columns, r.rows),
          r.total > r.rows.length ? el("p", { class: "hint", text: `The first ${r.rows.length} of ${r.total} rows.` }) : null,
        ),
      );
    } catch (err) {
      if (mine === seq) preview.replaceChildren(errorBox(err, "The copy could not be read"));
    }
  };
  box.append(el("p", { class: "hint", text: "A run keeps a copy of every input it read, the bytes its sha256 hashes, so it can be repeated from its own directory." }));
  box.append(
    dataTable(
      [
        { label: "Kind", key: "kind", render: (r) => r.kind.replace("_", " ") },
        { label: "Name", key: "name", mono: true },
        { label: "Rows", key: "rows", num: true },
        { label: "Version", key: "version" },
        { label: "sha256", key: "sha256", mono: true, render: (r) => short(r.sha256, 16) },
        { label: "Read from", key: "path", render: (r) => r.path || "memory" },
      ],
      run.inputs,
      { onRow: show },
    ),
    preview,
  );
  const kind = params.get("kind");
  if (kind) {
    const row = run.inputs.find((r) => r.kind === kind && (r.name === params.get("name") || kind === "model_points"));
    if (row) show(row);
  }
  return box;
}

function filesView(run) {
  const box = el("div", { class: "stack" });
  const host = el("div");
  let seq = 0;
  const open = async (name) => {
    const mine = ++seq;
    host.replaceChildren(progress("Reading"));
    try {
      const r = await api("/api/run/file", { params: { dir: run.dir, name } });
      if (mine === seq) host.replaceChildren(el("p", { class: "hint", text: r.path }), el("pre", { class: "code tall", text: r.text }));
    } catch (err) {
      if (mine === seq) host.replaceChildren(errorBox(err));
    }
  };
  const row = el("div", { class: "row" });
  for (const [name, label] of [["manifest", "manifest.json"], ["invariants", "invariants.json"], ["generated", "Generated code"]]) {
    const button = el("button", { type: "button", class: "button", text: label });
    button.addEventListener("click", () => open(name));
    row.append(button);
  }
  box.append(row, host);
  open("manifest");
  return box;
}

// ------------------------------------------------------------------------------ compare page

async function comparePage(ctx, params) {
  const { runs } = await api("/api/runs");
  const page = el("div");
  page.append(el("div", { class: "page-head" }, el("div", {}, el("h1", { text: "Compare two runs" }), el("div", { class: "sub", text: "What each run's id covers, side by side, and the largest difference in every variable both hold." }))));
  const readable = runs.filter((r) => !r.error);
  const select = (value, label) => {
    const s = el("select", { "aria-label": label });
    s.append(el("option", { value: "", text: "Choose a run" }));
    for (const r of readable) s.append(el("option", { value: r.dir, text: `${r.name} · ${String(r.model).split(".").pop()} · ${when(r.created_at)}` }));
    s.value = value || "";
    return s;
  };
  const a = select(params.get("a"), "Run A");
  const b = select(params.get("b"), "Run B");
  const button = el("button", { type: "button", class: "button primary", text: "Compare" });
  button.addEventListener("click", () => {
    if (a.value && b.value) go(`#/compare?${new URLSearchParams({ a: a.value, b: b.value })}`);
  });
  page.append(el("div", { class: "fields" }, field("Run A", a), field("Run B", b)), el("div", { class: "row" }, button));
  if (!params.get("a") || !params.get("b")) {
    if (readable.length < 2) page.append(notice("Two runs are needed; make another on the Models page, for instance on another basis."));
    return page;
  }
  const c = await api("/api/compare", { params: { a: params.get("a"), b: params.get("b") } });
  page.append(
    el("div", { class: c.same_run_id ? "notice" : "warning", text: c.same_run_id ? `The runs have the same run id, ${c.a.run_id}: they declare the same computation.` : `The run ids differ (${c.a.run_id} and ${c.b.run_id}); the table says in what.` }),
  );
  page.append(el("h3", { text: "What the run ids cover" }));
  page.append(
    dataTable(
      [
        { label: "", key: "what" },
        { label: `A: ${c.a.name}`, key: "a", mono: true, render: (r) => textOf(r.a) },
        { label: `B: ${c.b.name}`, key: "b", mono: true, render: (r) => textOf(r.b) },
        { label: "", key: "same", render: (r) => el("span", { class: `status ${r.same ? "good" : "bad"}`, text: r.same ? "same" : "differs" }) },
      ],
      c.identity,
    ),
  );
  if (!c.comparable) {
    page.append(notice(`The values cannot be compared: ${c.reason}.`, "warning"));
    return page;
  }
  const hide = el("input", { type: "checkbox" });
  hide.checked = true;
  const tableHost = el("div");
  const chartHost = el("div");
  const fill = () => {
    const rows = c.variables.filter((v) => !hide.checked || !(v.max_abs === 0));
    tableHost.replaceChildren(
      dataTable(
        [
          { label: "Variable", key: "variable", mono: true },
          { label: "Largest |A - B|", key: "max_abs", num: true, render: (r) => formatValue(r.max_abs) },
          { label: "Relative", key: "relative", num: true, render: (r) => (isNumber(r.relative) ? r.relative.toExponential(2) : textOf(r.relative)) },
          { label: "Policy", key: "policy", render: (r) => String(r.policy) },
          { label: "t", key: "t", num: true },
          { label: "Scenario", key: "scenario", num: true },
          { label: "A", key: "a", num: true, render: (r) => formatValue(r.a) },
          { label: "B", key: "b", num: true, render: (r) => formatValue(r.b) },
        ],
        rows,
        { onRow: (r) => overlay(r), empty: "Every variable both runs hold agrees exactly." },
      ),
    );
  };
  let seq = 0;
  const overlay = async (r) => {
    const mine = ++seq;
    chartHost.replaceChildren(progress("Reading both lanes"));
    try {
      const [la, lb] = await Promise.all([
        api("/api/run/lanes", { params: { dir: c.a.dir, var: r.variable, p: r.p, s: r.scenario } }),
        api("/api/run/lanes", { params: { dir: c.b.dir, var: r.variable, p: r.p, s: r.scenario } }),
      ]);
      if (mine !== seq) return;
      if (!la.time) {
        chartHost.replaceChildren(el("div", { class: "card" }, kv([[`A: ${c.a.name}`, formatValue(la.series[0].values[0])], [`B: ${c.b.name}`, formatValue(lb.series[0].values[0])]])));
        return;
      }
      chartHost.replaceChildren(
        lineChart({
          title: `${r.variable} of ${r.policy}${c.scenarios > 1 ? `, scenario ${r.scenario}` : ""}`,
          unit: r.unit || "",
          x: la.t,
          series: [
            { label: `A: ${c.a.name}`, values: la.series[0].values, slot: 1 },
            { label: `B: ${c.b.name}`, values: lb.series[0].values, slot: 2 },
          ],
        }),
      );
      chartHost.scrollIntoView({ block: "nearest", behavior: "smooth" });
    } catch (err) {
      if (mine === seq) chartHost.replaceChildren(errorBox(err));
    }
  };
  hide.addEventListener("change", fill);
  page.append(el("h3", { text: "Values" }), el("label", { class: "check" }, hide, el("span", { text: "Hide the variables that agree exactly" })), tableHost, chartHost);
  if (c.only_a.length || c.only_b.length) page.append(el("p", { class: "hint", text: `Held by one run only: ${[...c.only_a.map((n) => `${n} (A)`), ...c.only_b.map((n) => `${n} (B)`)].join(", ")}.` }));
  fill();
  return page;
}

// ---------------------------------------------------------------------------------- routing

let renderId = 0;

async function route() {
  const id = ++renderId;
  if (!TOKEN) {
    main.replaceChildren(
      el("div", { class: "card stack" }, el("h1", { text: "Open the address the server printed" }), el("p", { text: "This page needs the session's token, which comes in the address python -m halley.gui prints when it starts (it ends in #token=...). Open that address, or start the GUI again." })),
    );
    return;
  }
  const hash = location.hash || "#/models";
  const [path, query = ""] = hash.slice(1).split("?");
  const params = new URLSearchParams(query);
  const parts = path.split("/").filter(Boolean);
  const section = parts[0] || "models";
  for (const a of document.querySelectorAll(".nav a")) a.classList.toggle("on", a.dataset.section === section || (section === "run" && a.dataset.section === "runs"));
  const ctx = { alive: () => id === renderId };
  let page;
  try {
    if (section === "models") page = await modelsPage(ctx, parts[1] ? decodeURIComponent(parts[1]) : null, params);
    else if (section === "runs") page = await runsPage(ctx);
    else if (section === "run") page = await runPage(ctx, decodeURIComponent(parts[1] || ""), parts[2] || "overview", params);
    else if (section === "compare") page = await comparePage(ctx, params);
    else page = notice(`There is no page ${path}.`);
  } catch (err) {
    if (err instanceof ApiFailure && err.status === 403) {
      page = el("div", { class: "card stack" }, el("h1", { text: "The server refused this page" }), el("p", { text: err.message }));
    } else {
      page = errorBox(err);
    }
  }
  if (id === renderId) {
    main.replaceChildren(page);
    fitTray(true);
    document.title = `Halley · ${section[0].toUpperCase()}${section.slice(1)}`;
  }
}

window.addEventListener("hashchange", () => {
  // the address a server printed, opened in a tab that holds the page of an earlier session: the
  // page does not load again, so the new token is read here
  if (location.hash.startsWith("#token=")) {
    TOKEN = bootToken();
    sessionInfo = null; // another server's session, or the same server's answered for another token
  }
  route();
});
route();
kickJobs();
