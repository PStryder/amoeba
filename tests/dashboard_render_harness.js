// Run the dashboard's own kv() against a payload and report what it built.
//
// The dashboard's rendering was only ever checked by reading the Python
// string that contains it, which is how a structured metric came to be
// stringified into a column sized for a number without any test noticing.
// This loads the real script in a V8 context with just enough DOM to run,
// calls the real renderer, and prints the resulting tree as JSON so a test
// can assert on what an operator would actually see.
//
//   node dashboard_render_harness.js <script.js> <payload.json>
"use strict";
const fs = require("fs");
const vm = require("vm");

const src = fs.readFileSync(process.argv[2], "utf8");
const payload = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));

function element(tag) {
  const n = {
    tag: tag, className: "", title: "", _text: "", children: [], style: {},
    classList: {toggle() {}, add() {}, remove() {}},
    appendChild(c) { this.children.push(c); return c; },
    append() { for (const c of arguments) this.children.push(c); },
  };
  Object.defineProperty(n, "textContent", {
    get() { return this._text; },
    // Matching the DOM: setting text discards children.
    set(v) { this._text = String(v); this.children.length = 0; },
  });
  Object.defineProperty(n, "innerHTML", {
    get() { return ""; },
    set(_v) { this.children.length = 0; },
  });
  return n;
}

const ctx = {
  document: {
    createElement: element,
    getElementById: () => element("div"),
    querySelectorAll: () => [],
  },
  localStorage: {getItem: () => null, setItem: () => {}},
  // Never settles, so the script's boot call hangs harmlessly instead of
  // rejecting. A rejection here would fail the harness for a reason that has
  // nothing to do with rendering.
  fetch: () => new Promise(() => {}),
  setInterval: () => 0,
  setTimeout: () => 0,
  alert: () => {},
  console: console,
};

vm.createContext(ctx);
vm.runInContext(src, ctx, {filename: "dashboard.js"});

if (typeof ctx.kv !== "function") {
  console.error("the dashboard script exposes no kv()");
  process.exit(2);
}

function serialize(n) {
  return {
    tag: n.tag, cls: n.className, text: n.textContent, title: n.title,
    kids: (n.children || []).map(serialize),
  };
}
process.stdout.write(JSON.stringify(serialize(ctx.kv(payload))));
