# Interfaces: who is talking, and what that lets them do

```
                          OPERATOR
                             |
                      governance UI
                             |
                         HARNESS
                    /              \
             internal mind       semantic I/O
          Ego / Id / neuocytes     /       \
                                 MCP       API
```

Three kinds of caller, and they are **not** equivalent authority classes.

| Caller | Reaches | Credential |
|---|---|---|
| **Operator** | the governance surface | control token + session header |
| **MCP client** | input and output | external API key → adapter's `external_io` scope |
| **API client** | input and output | external API key → adapter's `external_io` scope |

## The distinction everything rests on

> **external input changing what Amoeba thinks about**
> is not
> **external control mutating Amoeba's protected state**

A client saying *"investigate this codebase and tell me whether the bug is a
race condition"* may cause Ego to request workers, neuocytes to run tools, the
blackboard to fill with findings, artifacts to be proposed, Id to notice
something, and maintained cognition to change. That is Amoeba processing input
using **its own** authority.

The caller never receives the verbs that did it. `"stop investigating and
answer with what you have"` is valid input; `work.cancel(work_id)` is not a
thing an external client can call.

## External I/O surface

Eight verbs, and that is the whole of it:

| Verb | What it does |
|---|---|
| `io_capabilities` | describe this surface |
| `io_submit` | hand Amoeba something to think about |
| `io_await` | block until it settles |
| `io_status` | progress on **your own** interaction |
| `io_output` | the answer, plus results surfaced to you |
| `io_list` | your own interactions |
| `io_attach_input` | admit bytes as input |
| `io_result` | fetch a result deliberately surfaced to you |

**Identity is the credential.** `client_id` comes from the authenticated API
key. Fields named `client_id`, `role`, `actor`, `caller`, `scope`, `from_role`
or `origin_actor` in a request are *discarded* before dispatch. "My
interactions" is a fact about who asked.

**Input bytes are input.** Attachments are content-addressed with exact-byte
provenance, exactly as an operator-attached file is. An attachment name is a
label, not a path: no host path is accepted, no filespace root is touched, and
materialising anything into a compute sandbox stays a Harness act.

**Identifiers are not authority.** An artifact is fetchable only once it has
been deliberately surfaced as a result of *that client's* interaction. Knowing
an artifact id or a blob digest buys nothing.

## Protocol

JSON-RPC 2.0 over HTTP. Verb-oriented, because Amoeba's interface is
interaction-shaped rather than resource-shaped, and transport-independent
enough that another adapter can be added.

```
POST /rpc                 external I/O          Authorization: Bearer <api key>
GET  /rpc                 discovery (this surface only)
GET  /events              SSE, your interactions only
POST /operator/rpc        governance             X-Amoeba-Operator: <session>
GET  /                    operator console
GET  /health              liveness
```

Request and response are ordinary JSON-RPC:

```json
{"jsonrpc":"2.0","id":1,"method":"io_submit","params":{"text":"..."}}
{"jsonrpc":"2.0","id":1,"result":{"interaction_id":"ixn_...","status":"accepted"}}
```

An unreachable verb returns `-32601 unknown method` — never an authorization
decision, because the operation does not exist on that surface.

**Streaming.** `GET /events` is SSE carrying progress on the subscriber's own
interactions. It is deliberately *not* a window onto the event log: publishing
the organism's history because a stream exists would turn a convenience into a
disclosure channel.

**Discovery** advertises the external surface and nothing else.

## Authentication and identity

**Localhost is not authentication.** Binding to `127.0.0.1` keeps other
machines out; it says nothing about other local processes, or a web page in the
user's browser POSTing to loopback. So:

* every request carries a credential, loopback or not;
* cross-origin browser requests are refused before dispatch;
* the operator session travels in `X-Amoeba-Operator`, a header — never a
  cookie alone, because a cookie is exactly what a hostile page can make the
  browser send for you.

Credentials live beside the other secrets: `api_clients.json` maps keys to
client identities, `operator.session` holds the console credential.

## Why MCP/API clients have no route to protected state

Not documentation. Three independent structural facts:

1. **The adapter dispatches nothing outside its own list.** A method not in
   `io_api.EXTERNAL_VERBS` is never looked up.
2. **The credential cannot name anything else.** The external adapter connects
   with the `external_io` scope, and the supervisor put only I/O verbs in that
   table. Scope comes from the secret presented; there is no role field in the
   handshake to forge.
3. **There is no generic dispatcher behind either.** The MCP facade holds the
   same narrow credential, and its tools map one-to-one onto I/O verbs.

Either of the first two alone would suffice. Both are in place so one mistake
does not open the door, and a test pins them to agree so they cannot drift.

**What changed.** The MCP facade previously held `cfg.token_path` — the control
token, which resolves to the full method table — and exposed 23 tools including
`mind_file_write`, `mind_file_delete`, `mind_artifact_promote`,
`id_maintenance`, `board_post` and `mind_cancel`. Every MCP client was
effectively the operator. It now holds `external_io` and exposes eight
`amoeba_*` tools.

## Operator surface

The console is a **cockpit, not an authority**. Every panel is a call to
`/operator/rpc`, which is a call into the Harness, which validates and
receipts. Dashboard code opens no database and touches no filesystem — running
on loopback does not make JavaScript privileged.

Views: organism state, Ego/Id incarnations, neuocytes and work, blackboard,
maintained memory, artifacts and proposals, provenance, prompt library and
lineage, Id telemetry, security and sandbox posture, resource versions,
backchannel transcript, Ego conversation, Id consultation.

**Operator authority does not leak into cognition.** Talking to Ego and
consulting Id are *inputs* — auditable, attributed, carrying no capability. Ego
does not gain the power to promote a prompt because the Operator asked it a
question. Communication carries information, not capability.

**Accepting a prompt candidate records a decision.** It does not hot-swap
anything: a running role keeps the prompt it started with until it is reborn.
Changing how the organism thinks from a dashboard click, with no moment at
which anyone chose to, is precisely the failure that separation exists to
prevent.

## Residual limits

The credentials are files in the state directory, readable by the Amoeba
account — the same boundary documented in `SANDBOX.md`. This separates
authority between *callers*, not against a determined process already running
as the user.

The external surface currently runs cognition synchronously on a worker thread
per interaction. There is no per-client rate limit or concurrency cap yet; a
client can queue as many interactions as it likes. That is a resource-policy
gap, not an authority gap, and it belongs with the Arbiter.
