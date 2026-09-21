// Drive a dashboard panel against scripted responses and report what it built.
//
// The render harness next door checks one function. This runs a whole panel:
// it loads the real script, hands the panel a scripted sequence of HTTP
// responses, replays operator actions against the real event handlers, and
// prints the transcript, the composer state and every request the panel made.
//
// That last part is the point. Whether a conversation waits for a terminal
// answer, and whether an answer lands under the message that asked for it,
// are properties of the request sequence -- invisible to anything that only
// inspects the page source.
//
//   node dashboard_panel_harness.js <script.js> <scenario.json>
//
// scenario: {panel, session?, responses[], actions[], scroll?, maxRequests?}
// responses: [{status, ctype, body}]  consumed in order; the last repeats
// actions:   [{type:"type"|"key"|"click", text?, key?, shift?}]
"use strict";
const fs = require("fs");
const vm = require("vm");

const src = fs.readFileSync(process.argv[2], "utf8");
const scn = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const hostTimeout = setTimeout;

const byId = new Map();
function element(tag) {
  const n = {
    tag: tag, className: "", title: "", value: "", placeholder: "",
    disabled: false, hidden: false, _text: "", children: [], style: {},
    scrollTop: 0, scrollHeight: 0, clientHeight: 0,
    onclick: null, onkeydown: null,
    classList: {toggle() {}, add() {}, remove() {}},
    focus() {},
    appendChild(c) { this.children.push(c); return c; },
    append() { for (const c of arguments) this.children.push(c); },
    removeChild(c) {
      const i = this.children.indexOf(c);
      if (i >= 0) this.children.splice(i, 1);
      return c;
    },
    insertBefore(c, ref) {
      const i = this.children.indexOf(ref);
      this.children.splice(i < 0 ? this.children.length : i, 0, c);
      return c;
    },
  };
  Object.defineProperty(n, "textContent", {
    get() { return this._text; },
    set(v) { this._text = String(v); this.children.length = 0; },
  });
  Object.defineProperty(n, "innerHTML", {
    get() { return ""; },
    set(_v) { this.children.length = 0; },
  });
  return n;
}

// -- scripted transport ------------------------------------------------------
let phase = "boot";
const requests = [];
const errors = [];
let inflight = 0;

function respond(i) {
  const r = scn.responses[Math.min(i, scn.responses.length - 1)] || {};
  const ctype = r.ctype === undefined ? "application/json" : r.ctype;
  return {
    status: r.status === undefined ? 200 : r.status,
    headers: {get: (h) => (String(h).toLowerCase() === "content-type"
                           ? ctype : null)},
    text: async () => (r.body === undefined ? "" : r.body),
  };
}

function fetchShim(_url, opts) {
  // The script boots itself on load. Nothing during boot should consume a
  // scripted response, so it hangs harmlessly until the panel under test
  // takes over.
  if (phase === "boot") return new Promise(() => {});
  const body = JSON.parse(opts.body);
  requests.push({method: body.method, params: body.params,
                 token: (opts.headers || {})["X-Amoeba-Operator"]});
  if (scn.maxRequests && requests.length > scn.maxRequests) {
    return new Promise(() => {});   // freeze: the panel is still waiting
  }
  if (scn.responses.length === 0) return Promise.reject(new Error("offline"));
  inflight += 1;
  const i = requests.length - 1;
  return new Promise((resolve) => hostTimeout(() => {
    inflight -= 1;
    resolve(respond(i));
  }, 0));
}

const ctx = {
  document: {
    createElement: element,
    getElementById(id) {
      if (!byId.has(id)) byId.set(id, element("div"));
      return byId.get(id);
    },
    querySelectorAll: () => [],
  },
  localStorage: {getItem: () => scn.session || "", setItem: () => {}},
  fetch: fetchShim,
  // Real ordering, no real waiting: the poll interval is not what is under
  // test and a test should not spend it.
  setTimeout: (fn) => hostTimeout(fn, 0),
  setInterval: () => 0,
  alert: () => {},
  console: {log() {}, error() { errors.push([...arguments].map(String).join(" ")); }},
};

vm.createContext(ctx);
// `render` is a top-level `const`, which in a vm script is a lexical
// binding rather than a property of the global object. One appended
// line publishes it; the script itself is not otherwise touched.
vm.runInContext(src + "\n;globalThis.render = render;\n", ctx,
                {filename: "dashboard.js"});

// -- replay ------------------------------------------------------------------
function findByClass(node, cls, out) {
  out = out || [];
  if (String(node.className).split(" ").indexOf(cls) >= 0) out.push(node);
  for (const k of node.children) findByClass(k, cls, out);
  return out;
}

function transcript(main) {
  return findByClass(main, "turn").map((t) => ({
    side: t.className.replace("turn", "").trim(),
    who: t.children[0] ? t.children[0].textContent : "",
    cls: t.children[1] ? t.children[1].className : "",
    text: t.children[1] ? t.children[1].textContent : "",
  }));
}

async function idle() {
  for (let i = 0; i < 600; i++) {
    await new Promise((r) => hostTimeout(r, 1));
    const frozen = scn.maxRequests && requests.length > scn.maxRequests;
    if ((inflight === 0 && i > 4) || frozen) {
      await new Promise((r) => hostTimeout(r, 3));
      if (inflight === 0 || frozen) return;
    }
  }
}

(async () => {
  const main = element("div");
  phase = "test";
  const panel = ctx.render[scn.panel];
  if (typeof panel !== "function") {
    console.error("no such panel: " + scn.panel);
    process.exit(2);
  }
  await panel(main);

  const stream = findByClass(main, "stream")[0];
  if (stream && scn.scroll) Object.assign(stream, scn.scroll);

  const box = findByClass(main, "composer")[0].children[0];
  const send = findByClass(main, "composer")[0].children[1];

  for (const a of scn.actions || []) {
    if (a.type === "type") box.value = a.text;
    else if (a.type === "click") send.onclick();
    else if (a.type === "key") {
      box.onkeydown({key: a.key, shiftKey: !!a.shift, isComposing: !!a.composing,
                     ctrlKey: false, altKey: false, metaKey: false,
                     preventDefault() { this.defaulted = true; }});
    }
    await idle();
  }

  const notice = findByClass(main, "notice")[0] || {};
  process.stdout.write(JSON.stringify({
    transcript: transcript(main),
    composer: box.value,
    sendDisabled: send.disabled,
    notice: notice.textContent || "",
    noticeHidden: notice.hidden !== false,
    tokenPrompt: findByClass(main, "row").length > 0,
    pills: findByClass(main, "pill").map((p) => p.textContent),
    requests: requests,
    scrollTop: stream ? stream.scrollTop : null,
    consoleErrors: errors,
  }));
  process.exit(0);
})();
