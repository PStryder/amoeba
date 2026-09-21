"""The operator console: a cockpit over Harness operations.

Every panel is a call to `/operator/rpc`, which is a call into the Harness,
which validates and receipts it. The page holds no authority of its own and has
no path to the database or the filesystem -- running on loopback does not make
JavaScript privileged.

Kept as a single self-contained document on purpose: a build step between the
operator and their console is a way for the console to be unavailable exactly
when something is wrong.
"""

DASHBOARD_HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Amoeba — operator console</title>
<style>
  :root {
    --bg:#0f1115; --panel:#171a21; --line:#252a34; --ink:#e6e8ec;
    --dim:#8b93a3; --accent:#7aa2f7; --warn:#e0af68; --bad:#f7768e;
    --good:#9ece6a;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.5 ui-sans-serif,system-ui,"Segoe UI",sans-serif; }
  header { display:flex; align-items:baseline; gap:1rem; padding:.75rem 1rem;
           border-bottom:1px solid var(--line); background:var(--panel); }
  header h1 { font-size:15px; margin:0; letter-spacing:.02em; }
  header .meta { color:var(--dim); font-size:12px; }
  nav { display:flex; gap:.25rem; padding:.5rem 1rem; border-bottom:1px solid var(--line);
        flex-wrap:wrap; }
  nav button { background:transparent; border:1px solid transparent; color:var(--dim);
               padding:.35rem .7rem; border-radius:6px; cursor:pointer; font:inherit; }
  nav button:hover { color:var(--ink); }
  nav button.on { background:var(--bg); border-color:var(--line); color:var(--accent); }
  main { padding:1rem; display:grid; gap:1rem;
         grid-template-columns:repeat(auto-fit,minmax(340px,1fr)); }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px;
          padding:.75rem .9rem; overflow:hidden; }
  .card h2 { margin:0 0 .5rem; font-size:12px; text-transform:uppercase;
             letter-spacing:.08em; color:var(--dim); font-weight:600; }
  .kv { display:grid; grid-template-columns:auto 1fr; gap:.15rem .75rem; font-size:13px; }
  .kv dt { color:var(--dim); } .kv dd { margin:0; font-variant-numeric:tabular-nums; }
  pre { margin:0; white-space:pre-wrap; word-break:break-word; font-size:12px;
        color:var(--dim); max-height:22rem; overflow:auto; }
  .wide { grid-column:1/-1; }
  .row { display:flex; gap:.5rem; align-items:center; margin-bottom:.5rem; }
  input, textarea, select { background:var(--bg); border:1px solid var(--line);
    color:var(--ink); border-radius:6px; padding:.4rem .5rem; font:inherit; flex:1; }
  textarea { min-height:5rem; resize:vertical; }
  button.go { background:var(--accent); border:none; color:#0b0d12; font-weight:600;
              padding:.4rem .9rem; border-radius:6px; cursor:pointer; }
  .pill { display:inline-block; padding:.05rem .45rem; border-radius:999px;
          font-size:11px; border:1px solid var(--line); color:var(--dim); }
  .good { color:var(--good); } .warn { color:var(--warn); } .bad { color:var(--bad); }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th, td { text-align:left; padding:.25rem .4rem; border-bottom:1px solid var(--line); }
  th { color:var(--dim); font-weight:500; }
</style>
</head>
<body>
<header>
  <h1>Amoeba</h1>
  <span class="meta" id="meta">connecting…</span>
</header>
<nav id="nav"></nav>
<main id="main"></main>

<script>
const PANELS = ["overview","work","blackboard","memory","artifacts","prompts",
                "environment","turns",
                "health","provenance","converse","consult id","backchannel"];
let session = localStorage.getItem("amoeba_operator") || "";
let current = "overview";

async function rpc(method, params={}) {
  const r = await fetch("/operator/rpc", {
    method:"POST",
    headers:{"Content-Type":"application/json","X-Amoeba-Operator":session},
    body: JSON.stringify({jsonrpc:"2.0", id:Date.now(), method, params})
  });
  const j = await r.json();
  if (j.error) throw new Error(j.error.message + " (" + j.error.code + ")");
  return j.result;
}
const el = (t,c,x) => { const n=document.createElement(t); if(c)n.className=c;
                        if(x!==undefined)n.textContent=x; return n; };
function card(title, wide) {
  const c = el("div","card"+(wide?" wide":"")); c.appendChild(el("h2",null,title)); return c;
}
function kv(obj) {
  const d = el("dl","kv");
  for (const [k,v] of Object.entries(obj)) {
    d.appendChild(el("dt",null,k));
    d.appendChild(el("dd",null, typeof v==="object"&&v!==null?JSON.stringify(v):String(v)));
  }
  return d;
}
function dump(o){ const p=el("pre"); p.textContent=JSON.stringify(o,null,2); return p; }
function table(rows, cols) {
  const t = el("table"), h = el("tr");
  cols.forEach(c=>h.appendChild(el("th",null,c)));
  t.appendChild(h);
  rows.forEach(r=>{ const tr=el("tr");
    cols.forEach(c=>tr.appendChild(el("td",null, r[c]===undefined?"":String(r[c]))));
    t.appendChild(tr); });
  return t;
}

const render = {
  async overview(main) {
    const o = await rpc("operator_overview");
    document.getElementById("meta").textContent =
      "state v" + o.state_version + " · up " + Math.round(o.harness.uptime_seconds) + "s";
    const a = card("organism");
    a.appendChild(kv({"state version":o.state_version,
      "uptime s":Math.round(o.harness.uptime_seconds),
      "supervision passes":o.harness.supervision_passes,
      "schema":o.harness.schema_version}));
    main.appendChild(a);
    const r = card("roles");
    for (const [role,f] of Object.entries(o.roles)) {
      r.appendChild(el("div",null,role+": "+(f.reachable?"reachable":"UNREACHABLE")
        +" · inc "+f.incarnation+" · ctx "+(f.context_tokens??"?")
        +"/"+(f.max_context_tokens??"?")));
    }
    main.appendChild(r);
    const s = card("scheduler"); s.appendChild(kv(o.scheduler)); main.appendChild(s);
    const w = card("work"); w.appendChild(kv(o.work.by_status)); main.appendChild(w);
    const f = card("failures (5m / 1h)");
    f.appendChild(kv({"5m":o.failures.last_5m_total,"1h":o.failures.last_1h_total,
                      "detail":o.failures.last_1h})); main.appendChild(f);
    const p = card("pending decisions"); p.appendChild(kv(o.pending_decisions));
    main.appendChild(p);
    const st = card("storage"); st.appendChild(kv(o.storage)); main.appendChild(st);
    const res = card("resource versions", true);
    const rows = Object.entries(o.resources.configured).map(([k,v])=>
      ({resource:k, version:v.short}));
    res.appendChild(table(rows,["resource","version"]));
    res.appendChild(el("div","pill","embodied ego prompt: "
      +(o.resources.embodied["prompt.ego"].matches_configured?"matches config":"DIFFERS from config")));
    main.appendChild(res);
  },
  async work(main) {
    const v = await rpc("ego_work_view", {limit:60});
    const c = card("work items", true);
    c.appendChild(kv(v.by_status));
    c.appendChild(table(v.items, ["work_id","work_class","status","origin_actor",
                                  "board_access","attempt"]));
    main.appendChild(c);
  },
  async blackboard(main) {
    const b = await rpc("board_read", {reader:"operator", limit:40, record:false});
    const c = card("blackboard", true);
    c.appendChild(table(b.posts, ["post_id","post_type","author","body"]));
    main.appendChild(c);
  },
  async memory(main) {
    const m = await rpc("recall", {limit:40});
    const c = card("maintained memory", true);
    c.appendChild(table(m, ["memory_id","kind","confidence","status","claim"]));
    main.appendChild(c);
  },
  async artifacts(main) {
    const a = await rpc("artifact_list", {limit:50});
    const c = card("artifacts", true);
    c.appendChild(table(a, ["artifact_id","status","path","proposed_by"]));
    const row = el("div","row");
    const id = el("input"); id.placeholder="artifact_id";
    const ok = el("button","go","promote"), no = el("button","go","reject");
    ok.onclick = async()=>{ try { await rpc("artifact_promote",
      {artifact_id:id.value, decided_by:"operator"}); go(current); }
      catch(e){ alert(e.message); } };
    no.onclick = async()=>{ try { await rpc("artifact_reject",
      {artifact_id:id.value, reason:"operator rejected"}); go(current); }
      catch(e){ alert(e.message); } };
    row.append(id, ok, no); c.appendChild(row);
    main.appendChild(c);
  },
  async prompts(main) {
    // The family tree: what is selected, what is pending, and the lineage
    // every node resolves to. Every action here is a call into the Harness;
    // this panel decides nothing.
    const t = await rpc("prompt_tree", {});
    const c = card("prompt library \u2014 cognitive family tree", true);
    c.appendChild(table(t.nodes.map(n => ({
      namespace: n.namespace,
      selected: n.selected ? n.selected.profile_ref : "\u2014",
      versions: n.version_count,
      pending: n.pending.length,
      experimental: n.experimental || "\u2014",
    })), ["namespace","selected","versions","pending","experimental"]));
    c.appendChild(el("div","pill", t.note));

    // Explain what a namespace actually resolves to, level by level.
    const erow = el("div","row");
    const ens = el("input"); ens.placeholder = "namespace or namespace@3.7.5";
    const ego = el("button","go","explain");
    ego.onclick = async()=>{ try {
      const v = ens.value.trim();
      const args = v.includes("@") ? {profile_ref:v} : {namespace:v};
      const x = await rpc("explain_profile", args);
      c.appendChild(el("h2",null,"resolved " + x.profile_ref));
      c.appendChild(table(x.lineage, ["namespace","local_version","prompt_mode",
        "changed_prompt","resulting_prompt_chars","model_vars_set",
        "model_vars_shadowed_by"]));
      c.appendChild(dump({model_var_source:x.model_var_source,
                          unset:x.unset_model_vars,
                          prompt_sha256:x.prompt_sha256}));
    } catch(e){ alert(e.message); } };
    erow.append(ens, ego); c.appendChild(erow);

    // Pending candidates across the whole tree, with the governance actions.
    const pending = [];
    for (const n of t.nodes) for (const v of n.pending)
      pending.push({namespace:n.namespace, ...v});
    c.appendChild(el("h2",null,"pending candidates"));
    c.appendChild(table(pending, ["namespace","local_version","version_id",
                                  "state","origin","created_by"]));
    const grow = el("div","row");
    const vid = el("input"); vid.placeholder = "version_id";
    const st = el("input"); st.placeholder = "state (validated|proposed|production_approved|rejected)";
    const setst = el("button","go","set state");
    const sel = el("button","go","select");
    setst.onclick = async()=>{ try{ await rpc("operator_prompt_state",
      {version_id:vid.value, state:st.value}); go(current);}catch(e){alert(e.message);} };
    sel.onclick = async()=>{ try{
      const row = pending.concat(t.nodes.map(n=>n.selected).filter(Boolean));
      const ns = (pending.find(p=>p.version_id===vid.value)||{}).namespace;
      if (!ns) { alert("that version_id is not a pending candidate here"); return; }
      await rpc("operator_prompt_select", {namespace:ns, version_id:vid.value});
      go(current);}catch(e){alert(e.message);} };
    grow.append(vid, st, setst, sel); c.appendChild(grow);
    c.appendChild(el("div","pill",
      "approving makes a version selectable; selecting changes what is born "
      "next. A running Ego, Id or neuocyte keeps the profile it was bound to."));

    // Cascade, previewed before it is run.
    const crow = el("div","row");
    const cns = el("input"); cns.placeholder = "namespace";
    const cv = el("input"); cv.placeholder = "local_version";
    const cm = el("input"); cm.placeholder = "none|queue|approve"; cm.value = "queue";
    const plan = el("button","go","preview"), run = el("button","go","cascade");
    plan.onclick = async()=>{ try{
      const p = await rpc("operator_prompt_cascade_plan",
        {namespace:cns.value, local_version:Number(cv.value), mode:cm.value});
      c.appendChild(el("h2",null,"cascade plan (" + p.mode + ")"));
      c.appendChild(table(p.steps, ["namespace","from_version","new_local_version",
                                    "new_parent_namespace","new_parent_version"]));
      c.appendChild(table(p.skipped, ["namespace","reason"]));
    }catch(e){alert(e.message);} };
    run.onclick = async()=>{ try{ await rpc("operator_prompt_cascade",
      {namespace:cns.value, local_version:Number(cv.value), mode:cm.value});
      go(current);}catch(e){alert(e.message);} };
    crow.append(cns, cv, cm, plan, run); c.appendChild(crow);
    main.appendChild(c);

    // What the shipped files did at the last start.
    const b = await rpc("operator_prompt_bootstrap_report", {});
    const bc = card("bootstrap files", true);
    bc.appendChild(el("div","pill", b.note));
    bc.appendChild(table(b.pending_file_deltas,
                         ["namespace","local_version","version_id","source"]));
    main.appendChild(bc);

    // What was actually born with what.
    const inc = await rpc("prompt_incarnations", {limit:25});
    const ic = card("incarnations", true);
    ic.appendChild(table(inc.bindings, ["actor_id","actor_kind","incarnation",
                                        "profile_ref","prompt_sha256","work_id"]));
    main.appendChild(ic);

    // The older proposal log, kept because it is history.
    const p = await rpc("operator_prompt_library", {limit:15});
    const oc = card("root suggestions (legacy proposal log)", false);
    oc.appendChild(table(p.pending, ["proposal_id","role","candidate_sha256","rationale"]));
    oc.appendChild(el("div","pill",
      "ego and id are bootstrap-only roots: these are suggested wordings, not "
      "library candidates. Adopting one means editing the shipped prompt file."));
    main.appendChild(oc);
  },
  async turns(main) {
    // What each persistent role is doing, what is waiting for it, and what it
    // has thought recently. A cockpit over Harness state: the scheduler is
    // deterministic substrate, and nothing here decides anything.
    const mb = await rpc("role_mailbox", {});
    for (const role of ["ego","id"]) {
      const r = mb[role];
      const c = card(role + " \u2014 " + r.state, true);
      c.appendChild(dump({
        state: r.state,
        queued: r.queued,
        queued_kinds: r.queued_kinds,
        current_turn: r.current_turn ? r.current_turn.turn_id : null,
        current_profile: r.current_turn ? r.current_turn.profile_ref : null,
        last_stop_reason: r.last_turn ? r.last_turn.stop_reason : null,
        next_heartbeat_in: r.next_heartbeat
          ? Math.max(0, Math.round(r.next_heartbeat - Date.now()/1000)) + "s"
          : null,
      }));
      if (r.next_triggers.length) {
        c.appendChild(el("h2",null,"waiting for the next turn"));
        c.appendChild(table(r.next_triggers,
                            ["kind","source","summary","trigger_id"]));
      }
      main.appendChild(c);
    }

    const t = await rpc("role_turns", {limit:30});
    const c = card("recent bounded turns", true);
    c.appendChild(table(t.turns.map(x => ({
      turn_id: x.turn_id, role: x.role, why: (x.trigger_kinds||[]).join(","),
      n: x.trigger_count, stop_reason: x.stop_reason, status: x.status,
      tools: x.tool_call_count, continues: x.parent_turn || "",
    })), ["turn_id","role","why","n","stop_reason","status","tools","continues"]));
    c.appendChild(el("div","pill",
      "a continuation is a new bounded turn whose parent is recorded, not an "
      "invisible extension of the previous one"));

    // Drill into one turn: the exact bundle and environment it was given.
    const row = el("div","row");
    const id = el("input"); id.placeholder = "turn_id";
    const go2 = el("button","go","inspect");
    go2.onclick = async()=>{ try {
      const d = await rpc("role_turn", {turn_id:id.value.trim()});
      c.appendChild(el("h2",null,"turn " + d.turn_id));
      c.appendChild(dump({profile_ref:d.profile_ref,
                          environment_sha256:d.environment_sha256,
                          bundle_sha256:d.bundle_sha256,
                          stop_reason:d.stop_reason,
                          parent_turn:d.parent_turn}));
      c.appendChild(table(d.triggers,
                          ["kind","source","summary","status","deliveries"]));
      if (d.bundle) c.appendChild(dump(d.bundle.text));
    } catch(e){ alert(e.message); } };
    row.append(id, go2); c.appendChild(row);

    // Put something in a role's mailbox. Input, not authority.
    const mrow = el("div","row");
    const who = el("input"); who.placeholder = "ego|id"; who.value = "ego";
    const msg = el("input"); msg.placeholder = "message";
    const send = el("button","go","queue message");
    send.onclick = async()=>{ try{
      await rpc("operator_message_role", {role:who.value.trim(), message:msg.value});
      go(current);}catch(e){alert(e.message);} };
    mrow.append(who, msg, send); c.appendChild(mrow);
    c.appendChild(el("div","pill",
      "a message wakes the role and is attributable; it carries no authority, "
      "and the role acts only through its own effectors"));
    main.appendChild(c);
  },
  async environment(main) {
    // What each role is told it can do, right now. The same manifest the
    // Harness hands the role at the start of a turn -- built from the live
    // dispatch table, not a description of it.
    for (const role of ["ego","id"]) {
      const e = await rpc("role_environment", {role});
      const m = e.manifest;
      const c = card(role + " environment \u2014 " + e.environment_sha256.slice(0,12), true);
      c.appendChild(dump({
        profile: m.bound_profile.profile_ref,
        model_generation: m.resources.model_generation,
        capabilities: m.capabilities.length,
        available_profiles: m.available_profiles.length,
        blob: e.environment_blob,
      }));
      c.appendChild(el("h2",null,"available cognitive profiles"));
      c.appendChild(table(m.available_profiles,
                          ["namespace","profile_ref","prompt_mode","state"]));
      c.appendChild(el("h2",null,"capabilities this role may invoke"));
      c.appendChild(table(m.capabilities.map(x => ({
        verb: x.verb, summary: x.summary,
        arguments: (x.arguments||[]).map(a => a.name + (a.required ? "" : "?")).join(", "),
      })), ["verb","arguments","summary"]));
      c.appendChild(el("div","pill", m.contract));
      main.appendChild(c);
    }
  },
  async health(main) {
    const p = await rpc("system_pulse", {max_age_seconds:0});
    const c = card("id telemetry (system_pulse)", true);
    c.appendChild(dump(p)); main.appendChild(c);
  },
  async provenance(main) {
    const h = await rpc("history", {limit:60});
    const c = card("recent events", true);
    c.appendChild(table(h, ["seq","kind","actor_id"])); main.appendChild(c);
  },
  async converse(main) {
    const c = card("chat with ego", true);
    const t = el("textarea"); t.placeholder = "message to Ego…";
    const b = el("button","go","send"); const out = el("pre");
    b.onclick = async()=>{ out.textContent="thinking…";
      try { out.textContent = JSON.stringify(
        await rpc("ego_converse",{message:t.value}), null, 2); }
      catch(e){ out.textContent = e.message; } };
    const row = el("div","row"); row.append(b);
    c.append(t, row, out); main.appendChild(c);
  },
  async ["consult id"](main) {
    const c = card("consult id", true);
    const t = el("textarea"); t.placeholder = "question for Id…";
    const b = el("button","go","ask"); const out = el("pre");
    b.onclick = async()=>{ out.textContent="thinking…";
      try { out.textContent = JSON.stringify(
        await rpc("operator_consult_id",{question:t.value}), null, 2); }
      catch(e){ out.textContent = e.message; } };
    const row = el("div","row"); row.append(b);
    c.append(t, row, out,
      el("div","pill","an input into Id's reasoning; it carries no capability"));
    main.appendChild(c);
  },
  async backchannel(main) {
    const b = await rpc("operator_backchannel", {limit:40});
    const c = card("ego ↔ id backchannel", true);
    c.appendChild(table(b.transcript, ["seq","kind","actor","from_role","to_role"]));
    const row = el("div","row");
    const who = el("select");
    ["ego","id"].forEach(r=>{const o=el("option",null,r); o.value=r; who.appendChild(o);});
    const m = el("input"); m.placeholder="message…";
    const go2 = el("button","go","send");
    go2.onclick = async()=>{ try{ await rpc("operator_backchannel",
      {to_role:who.value, message:m.value}); go(current);}catch(e){alert(e.message);} };
    row.append(who, m, go2); c.appendChild(row);
    c.appendChild(el("div","pill", b.note));
    main.appendChild(c);
  },
};

async function go(panel) {
  current = panel;
  [...document.querySelectorAll("nav button")]
    .forEach(b => b.classList.toggle("on", b.textContent === panel));
  const main = document.getElementById("main");
  main.innerHTML = "";
  try { await (render[panel] || render.overview)(main); }
  catch (e) {
    const c = card("error", true);
    c.appendChild(el("pre",null,e.message));
    if (String(e.message).includes("-32000")) {
      const row = el("div","row");
      const i = el("input"); i.placeholder = "operator session token";
      const b = el("button","go","use");
      b.onclick = ()=>{ session = i.value.trim();
        localStorage.setItem("amoeba_operator", session); go(current); };
      row.append(i,b); c.appendChild(row);
      c.appendChild(el("div","pill",
        "the supervisor prints this token at startup"));
    }
    main.appendChild(c);
  }
}

const nav = document.getElementById("nav");
PANELS.forEach(p => { const b = el("button",null,p); b.onclick = ()=>go(p); nav.appendChild(b); });
go("overview");
setInterval(()=>{ if (current==="overview") go("overview"); }, 5000);
</script>
</body>
</html>
"""
