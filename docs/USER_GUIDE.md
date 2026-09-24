# Amoeba User Guide

**Download, run, and operate your own Amoeba on Windows.**

This guide describes the controls and commands in the revision it ships with.
It is kept current with the code: when behaviour changes, this file changes in
the same commit. Amoeba is under active development, and the short
[Current operating limits](#16-current-operating-limits) section records
behaviour to account for in this version.

## Contents

1. [What you will be running](#1-what-you-will-be-running)
2. [What you need](#2-what-you-need)
3. [Download and install Amoeba](#3-download-and-install-amoeba)
4. [Try the console without a model](#4-try-the-console-without-a-model)
5. [Set up the real local model](#5-set-up-the-real-local-model)
6. [Start Amoeba and sign in](#6-start-amoeba-and-sign-in)
7. [Your first conversation](#7-your-first-conversation)
8. [Find your way around the console](#8-find-your-way-around-the-console)
9. [Work with files and artifacts](#9-work-with-files-and-artifacts)
10. [Review and change prompts](#10-review-and-change-prompts)
11. [Connect an MCP application](#11-connect-an-mcp-application)
12. [Operator commands for less common tasks](#12-operator-commands-for-less-common-tasks)
13. [Stop, back up, reset, and update](#13-stop-back-up-reset-and-update)
14. [Adjust resource use](#14-adjust-resource-use)
15. [Troubleshooting](#15-troubleshooting)
16. [Current operating limits](#16-current-operating-limits)
17. [Daily reference](#17-daily-reference)

## 1. What you will be running

Amoeba runs a local language model together with a persistent record of its
work. You can talk to it in a browser, ask it to investigate a question,
inspect its activity, and decide which proposed files and prompt changes to
accept.

Four names appear throughout the console:

| Name | What it means when you are using Amoeba |
|---|---|
| **Ego** | The part you talk to: it handles conversation and outward work. |
| **Id** | The part you consult about Amoeba itself: its health, evidence, disagreements, and use of resources. |
| **Neuocyte** | A temporary worker assigned a bounded piece of work. Workers can finish and disappear without erasing their recorded results. |
| **Harness / supervisor** | The machinery that starts processes, enforces limits, records state, and carries out permitted actions. You start it once and leave it running. |

Your browser is the **operator console**. An MCP application is an **external
client**. The console gives you oversight and decision controls; MCP lets
another application submit questions and collect answers.

Closing a browser tab or disconnecting an MCP client does not shut Amoeba
down. Use the shutdown command when you want it to stop.

## 2. What you need

The documented operating environment is native Windows. You do not need
Docker, WSL, PyTorch, or a hosted-model API key for this setup.

Install:

- **64-bit Python 3.11**, including the Windows Python launcher. The
  repository's measured installation uses 3.11.8; its package permits 3.12
  (`requires-python = ">=3.11,<3.13"`), but this guide stays with the
  documented 3.11 path. [Python downloads](https://www.python.org/downloads/windows/)
- **Git for Windows**, if you want the download and update commands below.
  [Git download](https://git-scm.com/downloads/win)
- A browser and PowerShell.

For actual model-generated answers, you also need a compatible NVIDIA
GPU/driver, the specific llama.cpp runtime, and the model file described in
section 5. The repository's measured configuration uses an **RTX 4080 with
16 GB of VRAM** and about **64 GB of system RAM**. These are reference-machine
specifications, not established minimum requirements. Its configured model and
context pool use roughly 10 GB of VRAM before allowing for the desktop and
other use.

You can try the interface without a GPU or model using the simulated setup in
section 4. Its answers are placeholders, so it cannot demonstrate reasoning
quality.

## 3. Download and install Amoeba

### Choose a home folder

Open PowerShell. The following creates an `Amoeba` folder under your Windows
user profile:

```powershell
$amoebaHome = Join-Path $env:USERPROFILE 'Amoeba'
New-Item -ItemType Directory -Force -Path $amoebaHome | Out-Null
git clone https://github.com/PStryder/amoeba.git "$amoebaHome\source"
Set-Location "$amoebaHome\source"
```

Keep these folders separate:

| Folder | What belongs there |
|---|---|
| `Amoeba\source` | The downloaded application. |
| `Amoeba\runtime` | The llama.cpp runtime and its accompanying DLLs. |
| `Amoeba\models` | The downloaded GGUF model. |
| `Amoeba\state` | Your real Amoeba's accumulated state, records, credentials, and accepted artifacts. Back this up. |
| `Amoeba\state-demo` | Disposable practice state for the simulated setup. |
| `Amoeba\settings` | Your own configuration files, outside the application checkout. |

If you prefer a ZIP download, open the
[Amoeba repository](https://github.com/PStryder/amoeba), choose
**Code → Download ZIP**, and extract it so that `pyproject.toml` is directly
inside your chosen `source` folder. The Git update instructions later apply
only to a Git clone.

### Install the Python dependencies

In the same PowerShell window:

```powershell
py -3.11 --version
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
New-Item -ItemType Directory -Force -Path "$amoebaHome\settings", "$amoebaHome\runtime", "$amoebaHome\models" | Out-Null
```

The editable installation lets this environment use the code in `source`.
There is no need to activate the environment: every command below names its
Python executable explicitly.

**Do not start with the repository's configuration files unchanged.** They
contain paths for the author's `F:` drive. The next sections create settings
for your own account instead.

## 4. Try the console without a model

This optional trial checks installation, startup, sign-in, and basic
interaction. It uses separate state and ports, disables sandbox compute, and
does not download a model.

From `Amoeba\source`, create the demo configuration:

```powershell
$amoebaHome = Join-Path $env:USERPROFILE 'Amoeba'
$amoebaPath = $amoebaHome.Replace('\', '/')
$amoebaDemoConfig = @"
state_dir = "$amoebaPath/state-demo"
runtime_dir = "$amoebaPath/runtime"
models_dir = "$amoebaPath/models"
supervisor_host = "127.0.0.1"
supervisor_port = 8721
inference_port = 8722
ego_port = 8723
id_port = 8724
api_host = "127.0.0.1"
api_port = 8725
api_enabled = true
log_level = "INFO"

[backend]
kind = "deterministic"
n_ctx = 32768
n_seq_max = 8

[arbiter]
max_neuocytes = 2

[sandbox]
enabled = false
"@
[IO.File]::WriteAllText("$amoebaHome\settings\demo.toml", $amoebaDemoConfig, [Text.UTF8Encoding]::new($false))
```

The command saves TOML as UTF-8 without a byte-order mark. If you edit the
file later, preserve that encoding and use forward slashes in paths.

Check and start the demo:

```powershell
.\.venv\Scripts\python.exe -m amoeba doctor --config ..\settings\demo.toml
.\.venv\Scripts\python.exe -m amoeba supervise --config ..\settings\demo.toml
```

Leave that window open. In a **second PowerShell window**, copy the sign-in
token:

```powershell
Get-Content "$env:USERPROFILE\Amoeba\state-demo\operator.session" -Raw | Set-Clipboard
```

Open [http://127.0.0.1:8725](http://127.0.0.1:8725). Paste into **operator
session token** and click **use**. Open **converse** and send a short message.
Expect simulated output rather than an intelligent answer.

When finished, run this in the second window:

```powershell
Set-Location "$env:USERPROFILE\Amoeba\source"
.\.venv\Scripts\python.exe -m amoeba shutdown --config ..\settings\demo.toml
```

The demo does not prove that GPU inference or sandbox computation works. Use
the real setup for those. Keep `state-demo` separate from `state` so
placeholder results do not become part of the record you intend to keep.

## 5. Set up the real local model

### Download the matching runtime

This version of Amoeba expects **llama.cpp b11057**. Its binding checks that
build's interface; substituting the newest llama.cpp release can prevent
startup.

Download these two files from the official
[b11057 release](https://github.com/ggml-org/llama.cpp/releases/tag/b11057):

- [llama-b11057-bin-win-cuda-12.4-x64.zip](https://github.com/ggml-org/llama.cpp/releases/download/b11057/llama-b11057-bin-win-cuda-12.4-x64.zip)
- [cudart-llama-bin-win-cuda-12.4-x64.zip](https://github.com/ggml-org/llama.cpp/releases/download/b11057/cudart-llama-bin-win-cuda-12.4-x64.zip)

Extract both into `Amoeba\runtime\llama.cpp-b11057-win-cuda12.4`, keeping the
runtime's accompanying DLLs together. Locate `llama.dll`: the configuration
below assumes it is directly inside that folder. If extraction produces an
extra subfolder, use the actual folder containing `llama.dll` for both runtime
paths in the configuration.

The repository uses these prebuilt binaries. It does not require you to
compile llama.cpp or install PyTorch.

### Download the model

The repository's reference model is **Qwen3-4B-Instruct-2507**, quantized as
**Q5_K_M**, in GGUF format. Save the file as:

```text
Amoeba\models\Qwen3-4B-Instruct-2507-Q5_K_M.gguf
```

One available source of that filename is the
[bartowski GGUF distribution](https://huggingface.co/bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF).
In **Files and versions**, choose `Qwen3-4B-Instruct-2507-Q5_K_M.gguf`; the
download is approximately 2.9 GB. The Amoeba repository does not record a
publisher or checksum for its original model download, so this link identifies
a matching model/quantization, not a verified byte-for-byte copy of the
author's file.

Download the actual `.gguf` file, not a model-page HTML file or a small Git
pointer. The model and runtime are separate downloads; cloning Amoeba does not
install either.

### Create your real configuration

This example follows the repository's 16 GB GPU configuration. In PowerShell,
from `Amoeba\source`:

```powershell
$amoebaHome = Join-Path $env:USERPROFILE 'Amoeba'
$amoebaPath = $amoebaHome.Replace('\', '/')
$amoebaLocalConfig = @"
state_dir = "$amoebaPath/state"
runtime_dir = "$amoebaPath/runtime/llama.cpp-b11057-win-cuda12.4"
models_dir = "$amoebaPath/models"
supervisor_host = "127.0.0.1"
supervisor_port = 8711
inference_port = 8712
ego_port = 8713
id_port = 8714
api_host = "127.0.0.1"
api_port = 8715
api_enabled = true
log_level = "INFO"

[backend]
kind = "llama_cpp"
lib_path = "$amoebaPath/runtime/llama.cpp-b11057-win-cuda12.4/llama.dll"
model_path = "$amoebaPath/models/Qwen3-4B-Instruct-2507-Q5_K_M.gguf"
n_gpu_layers = -1
n_ctx = 49152
n_seq_max = 8
n_batch = 1024
n_ubatch = 512
n_threads = 8
flash_attn = true
type_k = "f16"
type_v = "f16"

[arbiter]
max_neuocytes = 3
max_outstanding_work = 64
max_prompt_tokens = 6144
max_completion_tokens = 3072
neuocyte_wall_seconds = 180.0
neuocyte_token_budget = 512
lease_seconds = 90.0
user_reserved_slots = 1
maintenance_reserved_slots = 1

[ego]
max_context_tokens = 16384

[id]
max_context_tokens = 8192
"@
[IO.File]::WriteAllText("$amoebaHome\settings\local.toml", $amoebaLocalConfig, [Text.UTF8Encoding]::new($false))
```

You can edit the resulting file later with:

```powershell
notepad ..\settings\local.toml
```

The two file paths under `[backend]` must name files that exist. The
`runtime_dir` and `models_dir` settings do not automatically find or download
those files.

## 6. Start Amoeba and sign in

### Check the installation

With the supervisor stopped, run:

```powershell
Set-Location "$env:USERPROFILE\Amoeba\source"
.\.venv\Scripts\python.exe -m amoeba doctor --config ..\settings\local.toml
```

The report uses `ok: true` and `ok: false`. Check the individual entries,
including:

- `mcp_sdk` and `numpy`: dependencies import successfully.
- `llama_dll` and `model_file`: the runtime and model are present.
- `llama_abi`: the runtime matches the expected interface.
- `cuda_device` and `gpu_offload_supported`: GPU support was detected.
- `hash_chain` and `content_addressable_store`: the stored record passes its
  checks.

`supervisor_running: false` is expected before the first start. The overall
`ok` field covers fatal setup checks; it does not mean every individual check
passed. This command also prepares/checks state, so it is not a guarantee that
a running model will fit in GPU memory.

### Start the supervisor

```powershell
.\.venv\Scripts\python.exe -m amoeba supervise --config ..\settings\local.toml
```

Leave this window open. Startup loads the model and starts Ego and Id. Allow
time for model loading; the console can become reachable before both roles are
ready.

In another PowerShell window:

```powershell
Set-Location "$env:USERPROFILE\Amoeba\source"
.\.venv\Scripts\python.exe -m amoeba status --config ..\settings\local.toml
Get-Content "$env:USERPROFILE\Amoeba\state\operator.session" -Raw | Set-Clipboard
```

Open [http://127.0.0.1:8715](http://127.0.0.1:8715). Paste the token into
**operator session token**, then click **use**.

The token is an operator credential, not a password you choose. It is
regenerated whenever the HTTP console starts. After a supervisor restart, copy
the new value and sign in again. The browser remembers the previous value,
which may produce an invalid-session message until you replace it.

On **overview**, wait for Ego and Id to become reachable. A quiet, idle Ego is
normal: it wakes for input and relevant events. Id also reviews the organism
periodically and when certain problems are detected.

## 7. Your first conversation

1. Open **converse**.
2. Enter a short, specific request.
3. Click **send**, or press **Enter**. Use **Shift+Enter** for a new line.
4. Wait while **Ego is thinking…** is shown.

For a first real-model request, try:

> Compare keeping a research journal in one Markdown file with keeping one
> file per topic. Give me three tradeoffs and a recommendation for a solo
> project.

For work that benefits from checking:

> Investigate this claim using only the information below. Separate what the
> evidence establishes from what remains uncertain. If you delegate checks,
> tell me what each check contributed.

Then provide the relevant information. Asking Amoeba to investigate does not
give it internet access or automatically make a host folder readable.

Requests can take several bounded thinking turns. If the console says **this
answer was cut off before it was finished**, treat it as partial. Ask for a
narrower follow-up or continuation, and inspect **turns** if it happens
repeatedly.

Keep the conversation panel open while waiting. The current panel builds its
visible transcript in the browser; leaving or reloading it does not restore
that scrollback. The underlying requests and answers have durable records, but
the console is not a full conversation-history browser. Copy answers you want
to reuse before navigating away.

### Talk to Id when the question is about Amoeba

Use **consult id**, for example:

> Check whether any work has failed repeatedly or any recorded conclusions
> have unresolved disagreements. Tell me which items need my attention and
> why.

This asks Id to examine the organism. It does not itself approve changes or
cancel work.

### Use the backchannel for shared context

Open **backchannel** to watch short messages and send a note to both Ego and
Id:

> For this session, prioritize finishing the current investigation before
> starting unrelated work.

The note is queued for the roles; it is not an immediate command that
overrides their limits. The displayed room is a temporary, bounded live view
and disappears when the supervisor restarts. Use **converse** when you need a
direct answer to a question.

## 8. Find your way around the console

| Tab | Use it to… |
|---|---|
| **overview** | Check reachability, whether each role is thinking, repeated failures, work counts, resource use, and pending decisions. Refreshes automatically about every five seconds. |
| **work** | See work IDs, origins, queue states, and attempts. Copy an ID when you need to inspect or cancel a specific item. |
| **blackboard** | Read workers' shared posts. A post is a finding or message, not automatically an accepted belief. |
| **memory** | Read maintained claims, their confidence, and their status. Treat confidence as part of the record, not a guarantee of correctness. |
| **artifacts** | Find proposed files and promote or reject them. |
| **prompts** | Inspect selected profiles, pending candidates, and which profiles running roles received. Approve/select changes deliberately. |
| **environment** | See the capabilities and profiles currently described to each role. Useful when a role says it cannot do something. |
| **turns** | See what is queued, what is running, why a turn stopped, and whether it continued. Enter a `turn_id` and click **inspect** for more detail. |
| **health** | Inspect a detailed snapshot of measured health and resource pressure. |
| **provenance** | See recent recorded events and the actors responsible for them. |
| **converse** | Talk to Ego. |
| **consult id** | Ask Id about the organism. |
| **backchannel** | Watch the live room and send context to both roles. |

Most inspection panels load a snapshot when opened. Reopen the tab to refresh
it. The live backchannel and the overview update themselves.

In **work**, `queued` means waiting, `leased` means assigned to a worker, and
`blocked` means not currently runnable. Completed, failed, and cancelled items
remain part of the record. An increasing attempt count or repeated failures
deserves investigation.

In **turns**, a turn is one bounded piece of thinking. A completed turn is not
necessarily a completed answer: the thought may have earned a continuation.
For external submissions, collect the interaction's output to see whether the
whole answer is complete.

On **overview**, a role can be reachable and still not thinking. Reachability
means its process answers; **NOT THINKING** means its recent turns have failed
in a row. The Harness records that, attempts a bounded number of repairs, and
then leaves the role visibly unwell rather than restarting it endlessly.

## 9. Work with files and artifacts

### Understand the three user-facing stages

1. **Input:** content you provide for Amoeba to use.
2. **Proposal:** a file produced by a worker and offered for review.
3. **Promotion:** your decision to save the proposed bytes as accepted output.

Workers compute in temporary sandboxes. Sandbox files are not a permanent
output folder. Proposed artifacts are preserved separately, so a proposal can
still be promoted after its worker and sandbox have gone away.

### Review and accept a proposed file

1. Open **artifacts** and locate a row with status `proposed`.
2. Copy its `artifact_id`.
3. Review its content. The current panel lists metadata rather than a file
   preview; section 12 gives a command to retrieve a text preview and its
   recorded digest.
4. Paste the ID into the panel's `artifact_id` field.
5. Click **promote** to accept it, or **reject** to decline it.

The panel's **promote** button saves into the internal artifact store at
`Amoeba\state\artifacts`, under a name the Harness picks. Open that folder in
File Explorer to use the accepted file. The table's `path` can describe the
original proposal path; it is not necessarily the final host destination.

Promotion saves the file; it does not run it. A proposed script remains a
script you must review before choosing to execute.

### Allow an explicit output folder

By default there are no configured host-file roots. To allow promotion into a
folder you choose, create it first:

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\Amoeba\output" | Out-Null
```

Append this to `settings\local.toml`, replacing `YOUR_NAME` with your actual
Windows profile folder:

```toml
[[filespace.roots]]
name = "out"
path = "C:/Users/YOUR_NAME/Amoeba/output"
mode = "read_write"
description = "Accepted output from Amoeba"
```

Restart Amoeba after changing the configuration. The custom-destination
command is in section 12. The console's simple **promote** button continues to
use the internal artifact folder.

A source folder can be added as another root using `mode = "read_only"`.
Naming a root permits the relevant Harness operations; it does not
automatically attach its files to a conversation or let workers browse your
computer. The folder must already exist when Amoeba starts.

### Attach a document to a question

The browser conversation panel currently has no upload control. The external
HTTP interface can upload bytes, obtain an input ID, and submit a request with
that ID. Section 12 provides a complete example.

Both `converse` and `investigate` submissions now carry their interaction
context, so an investigation can read the files its own request arrived with
and can return one as a surfaced result. The MCP attachment tool still admits
bytes while its ask/submit tools take no attachment IDs, so use the HTTP
example for an end-to-end attachment workflow in this revision.

Sandbox computation uses a restricted, standard-library Python environment. It
has no network access and does not inherit packages such as NumPy from your
main installation. An attached PDF or image is not automatically parsed or
understood just because its upload succeeded; supply text or a format the
available tools can handle.

## 10. Review and change prompts

You do not need to customize prompts to start using Amoeba. The shipped
profiles establish a baseline on first startup.

When you do want a change, the important distinction is:

- **Candidate:** a proposed version exists, but is not in use.
- **Approved:** the version is eligible for selection.
- **Selected:** new instances of that role or worker profile will use it.
- **Running:** an existing role keeps the profile it received when it started.

Open **prompts** to see the family tree and pending candidates. Enter a
namespace such as `ego` or `id`, then click **explain** to inspect its selected
profile's lineage and settings. The **incarnations** table shows what was
actually assigned to running/recent instances.

For an existing candidate, copy its `version_id` and review the actual
proposed text before approving it; section 12 shows how to retrieve all
versions for a namespace. The state progression is governed by the
application: `candidate → validated → proposed → production_approved`, with
`evaluated` available between `validated` and `proposed`, and `rejected`
reachable from each. Use **set state** for a permitted transition, then select
the approved version. Approval alone changes nothing running.

The command examples in section 12 provide a reliable way to approve and
select by ID, including versions no longer shown as pending in the panel. To
apply a new Ego or Id selection to the running organism, perform a clean
shutdown and restart after current work settles.

Changing a parent profile does not automatically rewrite its descendants. The
cascade controls let you **preview** a propagation plan. `none` leaves
descendants unchanged, `queue` creates candidates for review, and `approve`
propagates with approval. Read the preview and skipped entries before using
**cascade**.

To propose a change to the baseline Ego or Id wording, first save a copy of
the corresponding file under `source\src\amoeba\promptlib\prompts\ego.md` or
`id.md`. Edit its prompt text while preserving the settings header, then
restart. The change appears as a governed candidate; it does not silently
replace an existing selected profile. Review, approve, and select it as
described above, then restart again when ready to use it. Keep your
prompt-file edits under version control so an application update does not lose
them. Do not add a `system_prompt` setting to `[ego]` or `[id]`: the current
configuration loader refuses that retired setting.

## 11. Connect an MCP application

Start the supervisor first. Then configure your MCP-compatible application to
launch this command as a **local stdio MCP server**:

```text
Executable: C:\Users\YOUR_NAME\Amoeba\source\.venv\Scripts\python.exe
Arguments:  -m amoeba mcp --config C:\Users\YOUR_NAME\Amoeba\settings\local.toml --transport stdio
```

Use your actual account path. The executable and arguments must be separate
fields if your application asks for them separately. For clients that use the
common `mcpServers` JSON format:

```json
{
  "mcpServers": {
    "amoeba": {
      "command": "C:\\Users\\YOUR_NAME\\Amoeba\\source\\.venv\\Scripts\\python.exe",
      "args": [
        "-m", "amoeba", "mcp",
        "--config", "C:\\Users\\YOUR_NAME\\Amoeba\\settings\\local.toml",
        "--transport", "stdio"
      ]
    }
  }
}
```

The exact location and outer format of the MCP configuration depend on your
client. The local facade reads its own credential from the configured state
directory; do not paste the operator session token into this configuration.

The current tools are:

| Tool | Purpose |
|---|---|
| `amoeba_capabilities` | Check what the external interface offers. |
| `amoeba_ask` | Submit a question and wait for up to the requested interval. |
| `amoeba_submit` | Submit without waiting. |
| `amoeba_status` | Check an interaction ID. |
| `amoeba_output` | Collect its answer and surfaced-result list. |
| `amoeba_list` | Find recent interactions belonging to this client identity. |
| `amoeba_attach` | Store input bytes; see the current attachment limitation in section 16. |
| `amoeba_result` | Fetch the bytes of an explicitly surfaced result. |

Ask your client to call `amoeba_capabilities` first, then submit a short
question. Keep the returned `interaction_id` if the answer is still running.
Use `amoeba_status` and `amoeba_output` rather than repeatedly resubmitting the
question.

MCP results are wrapped: inspect the interaction state inside `result`, not
just the outer envelope's `status`. A waiting timeout ends the client's wait;
it does not necessarily end the work.

MCP has no direct artifact-promotion, prompt-approval, host-file-management, or
work-cancellation tools. Use the operator controls for those decisions.
Disconnecting the client does not cancel work.

All current MCP facades for the same state directory use the identity `mcp`.
Treat one Amoeba as one shared cognitive trust domain, not as isolated private
minds for unrelated users. Separate HTTP client keys restrict direct result
retrieval, but do not create separate model contexts.

## 12. Operator commands for less common tasks

These PowerShell recipes use the implemented HTTP interface for actions that do
not have a complete console workflow. They are optional; ordinary conversation
needs none of them.

### Prepare an operator session

Run this in a second PowerShell window while the real supervisor is running:

```powershell
$amoebaBase = 'http://127.0.0.1:8715'
$amoebaState = Join-Path $env:USERPROFILE 'Amoeba\state'
$amoebaOperatorHeaders = @{
    'X-Amoeba-Operator' = (Get-Content "$amoebaState\operator.session" -Raw).Trim()
}

function Invoke-AmoebaOperator {
    param([string]$Method, [hashtable]$Parameters = @{})
    $body = @{ jsonrpc = '2.0'; id = 1; method = $Method; params = $Parameters } | ConvertTo-Json -Depth 20
    $reply = Invoke-RestMethod -Uri "$amoebaBase/operator/rpc" -Method Post -Headers $amoebaOperatorHeaders -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
    if ($reply.error) { throw ($reply.error | ConvertTo-Json -Depth 10) }
    $reply.result
}
```

Refresh the token variables after a supervisor restart. Replace example IDs
below with IDs copied from your console or a preceding result.

**Inspect an artifact before deciding:**

```powershell
Invoke-AmoebaOperator 'ego_artifact_evidence' @{ artifact_id = 'YOUR_ARTIFACT_ID' } | ConvertTo-Json -Depth 10
```

This returns metadata and, for small UTF-8 content, a text preview. Large or
non-text files may have no inline preview. Do not interpret a missing preview
as an empty file.

**Promote an artifact to the configured `out` root:**

```powershell
Invoke-AmoebaOperator 'artifact_promote' @{
    artifact_id = 'YOUR_ARTIFACT_ID'
    decided_by = 'operator'
    root = 'out'
    path = 'report.md'
} | ConvertTo-Json -Depth 10
```

The destination is relative to that root. An existing destination is versioned
before replacement under the default settings.

**Inspect, then cancel a specific work item:**

```powershell
Invoke-AmoebaOperator 'get_work' @{ work_id = 'YOUR_WORK_ID' } | ConvertTo-Json -Depth 12
Invoke-AmoebaOperator 'cancel_work' @{
    work_id = 'YOUR_WORK_ID'
    actor = 'operator'
    reason = 'No longer needed'
}
```

This targets that work item. It is not a global pause button or a promise to
cancel every activity related to a conversation.

**Review prompt versions, then approve and select one:**

```powershell
Invoke-AmoebaOperator 'prompt_versions' @{ namespace = 'ego' } | ConvertTo-Json -Depth 20
```

After reviewing the candidate and choosing its ID, advance it from its current
state through the required sequence. For a fresh `candidate`:

```powershell
$amoebaCandidate = 'YOUR_VERSION_ID'
Invoke-AmoebaOperator 'operator_prompt_state' @{ version_id = $amoebaCandidate; state = 'validated' }
Invoke-AmoebaOperator 'operator_prompt_state' @{ version_id = $amoebaCandidate; state = 'proposed' }
Invoke-AmoebaOperator 'operator_prompt_state' @{ version_id = $amoebaCandidate; state = 'production_approved'; rationale = 'Reviewed for this deployment' }
Invoke-AmoebaOperator 'operator_prompt_select' @{ namespace = 'ego'; version_id = $amoebaCandidate }
```

If the version is already partway through this sequence, start at its next
permitted transition. Restart the roles through a clean supervisor restart to
use the new selection. An older approved version can be selected the same way
for a prompt rollback.

### Attach a text file through the external interface

This uses the separate external API key, keeping the submission in the external
interaction workflow. In PowerShell:

```powershell
$amoebaBase = 'http://127.0.0.1:8715'
$amoebaState = Join-Path $env:USERPROFILE 'Amoeba\state'
$amoebaKeys = Get-Content "$amoebaState\api_clients.json" -Raw | ConvertFrom-Json
$amoebaKey = ($amoebaKeys.PSObject.Properties | Select-Object -First 1).Name
$amoebaExternalHeaders = @{ Authorization = "Bearer $amoebaKey" }

function Invoke-AmoebaExternal {
    param([string]$Method, [hashtable]$Parameters = @{})
    $body = @{ jsonrpc = '2.0'; id = 1; method = $Method; params = $Parameters } | ConvertTo-Json -Depth 20
    $reply = Invoke-RestMethod -Uri "$amoebaBase/rpc" -Method Post -Headers $amoebaExternalHeaders -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
    if ($reply.error) { throw ($reply.error | ConvertTo-Json -Depth 10) }
    $reply.result
}

$amoebaInputFile = 'C:\path\to\notes.txt'
$amoebaAttachment = Invoke-AmoebaExternal 'io_attach_input' @{
    filename = [IO.Path]::GetFileName($amoebaInputFile)
    media_type = 'text/plain'
    content_base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes($amoebaInputFile))
}
$amoebaRequest = Invoke-AmoebaExternal 'io_submit' @{
    text = 'Read the attached notes and summarize the unresolved questions.'
    kind = 'converse'
    input_ids = @($amoebaAttachment.input_id)
}
$amoebaRequest
```

Use a UTF-8 text file and replace the example path. Uploads are limited to
8 MiB per attachment and eight attachment IDs per submission. Admission
preserves bytes; it does not promise that an entire large document will fit
into a model's reading budget.

Collect the result later:

```powershell
$amoebaAnswer = Invoke-AmoebaExternal 'io_output' @{ interaction_id = $amoebaRequest.interaction_id }
$amoebaAnswer | ConvertTo-Json -Depth 20
```

If its status is `accepted` or `running`, wait and repeat the output call.
`complete` is a finished answer, `incomplete` is a partial answer, and `failed`
carries an error. When `results` lists a file, fetch it using the returned
result ID:

```powershell
$amoebaFile = Invoke-AmoebaExternal 'io_result' @{ result_id = 'YOUR_RESULT_ID' }
$amoebaDownload = Join-Path $env:USERPROFILE 'Amoeba\downloaded-result.bin'
[IO.File]::WriteAllBytes($amoebaDownload, [Convert]::FromBase64String($amoebaFile.content_base64))
```

Choose the destination name and extension yourself from the returned metadata.
Use a new destination if you need to preserve an existing local file. A
surfaced result and an operator-promoted artifact are different ways to receive
output; one does not imply that the other action occurred.

## 13. Stop, back up, reset, and update

### Stop cleanly

Let important answers settle and save their output first. From a second
PowerShell window:

```powershell
Set-Location "$env:USERPROFILE\Amoeba\source"
.\.venv\Scripts\python.exe -m amoeba shutdown --config ..\settings\local.toml
```

Wait for the supervisor window to return to its prompt. Shutdown stops child
processes and destroys temporary sandboxes. Durable state and promoted
artifacts remain.

Restart with the same `supervise` command and the same configuration. **The
state directory is what identifies the Amoeba you are continuing.** Pointing to
a new empty directory creates a fresh record; it does not move the old one.

There is no automatic Windows service or startup installation in this guide.
Start the supervisor again after a machine reboot when you want Amoeba running.

### Back up the whole state directory

After clean shutdown:

```powershell
$amoebaHome = Join-Path $env:USERPROFILE 'Amoeba'
$amoebaBackup = Join-Path $amoebaHome ('backups\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
New-Item -ItemType Directory -Force -Path $amoebaBackup | Out-Null
Copy-Item -LiteralPath "$amoebaHome\state" -Destination "$amoebaBackup\state" -Recurse
Copy-Item -LiteralPath "$amoebaHome\settings" -Destination "$amoebaBackup\settings" -Recurse
```

Back up any configured external output folders separately. Keep a note of the
code revision, runtime build, and model filename as well:

```powershell
Set-Location "$env:USERPROFILE\Amoeba\source"
git rev-parse HEAD
```

Do not back up only `mind.sqlite3`: the `blobs` folder contains referenced
content, and the rest of the state tree contains other necessary records and
files. Stopping first avoids taking an inconsistent copy while work is
changing.

Backups include local credentials. Store them as private application data.

### Start a fresh organism

`amoeba reset` moves the current organism aside and lets the next start build a
new one. With the supervisor stopped:

```powershell
Set-Location "$env:USERPROFILE\Amoeba\source"
.\.venv\Scripts\python.exe -m amoeba reset --config ..\settings\local.toml --dry-run
.\.venv\Scripts\python.exe -m amoeba reset --config ..\settings\local.toml
```

The first command lists what would move and moves nothing. The second archives
the database, blobs, artifacts, logs and runtime markers into a timestamped
folder beside the state directory — `Amoeba\state.reset-20260924-113000`, for
example — and prints where the old organism is readable.

- It **refuses while a supervisor owns the state directory**. Stop it first.
- **Credentials are kept**: `control.token`, `scope.*.token`,
  `operator.session` and `api_clients.json` stay, so existing clients keep
  working. Add `--rotate-credentials` to reissue them too.
- `--delete` removes the old state instead of archiving it, and refuses
  without `--yes`.
- It touches nothing outside the state directory; configured file roots and
  their contents are untouched.

The next `supervise` starts a new organism at incarnation 1 and rebootstraps
the prompt library from the shipped files. Prompt candidates you approved into
the old database do not carry over; keep such edits in the prompt files under
version control (section 10) if you want them in a fresh organism.

### Restore without overwriting your current record

With Amoeba stopped, copy the backed-up `state` folder into a **new empty
recovery location**. Point a copy of your configuration at that restored
directory, use the recorded application/runtime version where possible, and run
`doctor` before starting. Keep the original state folder until you have checked
the restored instance.

Start only one supervisor for a given state directory. When moving state to a
different machine, recheck runtime/model paths and folder permissions; those
paths are machine-specific.

### Update the application

1. Finish or save important work, then shut down.
2. Back up state and settings.
3. In `Amoeba\source`, run:

```powershell
git status --short
git pull --ff-only
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m amoeba doctor --config ..\settings\local.toml
```

If `git status` shows your own changes, preserve them before updating. If Git
refuses the update, resolve that situation rather than forcing it and
discarding files.

Review the new version's runtime/configuration requirements, then start the
supervisor and sign in with its new token. Your configuration lives outside the
checkout, so a normal source update does not replace it. Shipped prompt changes
may appear as pending candidates; updating the code does not automatically
adopt them into an existing mind.

Database downgrade compatibility is not established. If you need to undo an
application update, use the matching pre-update state backup rather than
assuming older code can read state already changed by newer code.

## 14. Adjust resource use

Make configuration changes while stopped, then restart. Begin with the
reference settings and change one thing at a time.

| Setting | What it controls |
|---|---|
| `[backend] n_ctx` | The total shared model-context pool. Larger values consume more memory; this is not a separate allowance for every worker. |
| `[backend] n_seq_max` | The inference sequence capacity shared by roles, workers, and other held contexts. |
| `[arbiter] max_neuocytes` | The maximum number of temporary workers. Reducing it limits concurrent worker demand. |
| `[ego] max_context_tokens` / `[id] max_context_tokens` | Each persistent role's logical context ceiling within the shared pool. |
| `[arbiter] max_completion_tokens` | A hard platform cap on generated output. Individual profile ceilings must fit under it. |
| `[scheduler] max_continuations` | How many follow-on turns an interrupted thought can receive. Default: 3. |
| `[scheduler] id_heartbeat_seconds` | The starting periodic review interval for Id. Default: 300 seconds; setting 0 disables periodic heartbeats. |

To add scheduler settings, append one `[scheduler]` section to your
configuration. Do not duplicate a section that already exists. For example:

```toml
[scheduler]
id_heartbeat_seconds = 600.0
max_continuations = 3
```

Turning off periodic heartbeats does not disable Id: it can still receive input
and event/condition wakes. In this revision, quiet heartbeat intervals back off
to at most one hour, while a burst of failures, critical pool pressure, or a
peer role that has stopped thinking wakes Id sooner.

Smaller GPUs may need a smaller shared pool and fewer workers. Reducing the
pool alone can leave configured role allowances competing for insufficient
capacity. Check **overview**, **health**, and the startup logs after
adjustments; the repository does not provide a validated preset for every GPU
size.

Automatic context maintenance can rebuild a role's working context. Persistent
records are separate from the model's current context, so this is not the same
as deleting the organism's recorded memory.

## 15. Troubleshooting

| Symptom | What to do |
|---|---|
| `py -3.11` cannot find Python | Install 64-bit Python 3.11 with the launcher, or use the full path to that installation's `python.exe` to create `.venv`. |
| `No module named amoeba` | Use `source\.venv\Scripts\python.exe` and rerun `pip install -e .` from the source folder. |
| Startup mentions an `F:` drive | You are using the author's sample paths or omitted your `--config` argument. Use `settings\local.toml`. |
| TOML will not parse | Check quotes, duplicate section names, file encoding, and Windows backslashes. Forward-slash paths avoid TOML escape problems. |
| `llama.dll` exists but fails to load | Check that it is the b11057 CUDA 12.4 build and that both runtime archives' DLLs were extracted together. Inspect `doctor` and the inference log. |
| `llama_abi` fails | Restore the expected runtime build. Do not bypass the compatibility check. |
| No CUDA device or GPU offload detected | Check the NVIDIA driver and accompanying runtime DLLs. A present model file does not establish GPU acceleration. |
| GPU out-of-memory or constant resource deferrals | Close other GPU-heavy workloads, inspect pool pressure, and reduce worker/pool demand consistently. |
| Browser cannot connect | Check that the supervisor is running, `api_enabled` is true, and the URL uses the configured `api_port`: 8715 for this real setup, 8725 for the demo. |
| Console says the session is invalid | Copy the current `operator.session` value for that instance and click **use** again. |
| Port already in use | Check for another Amoeba instance or another application. If running multiple instances, give each separate state and all five separate ports. |
| Another supervisor owns the state directory | Check for a still-running supervisor using that state. Do not delete its lock file to force a second one to start. |
| `amoeba reset` refuses | It names the process that owns the state directory. Shut that supervisor down first. |
| A role is reachable but **NOT THINKING** | Reachability and successful thinking are different. Inspect **turns**, **health**, and role logs for repeated errors; the Harness attempts bounded repairs and then leaves the role visibly unwell. |
| An answer is cut off | Read it as partial, inspect its stop reason in **turns**, and ask a narrower follow-up. Review limits if this becomes routine. |
| A file root is missing | Create the directory before startup and verify its absolute path and mode in your configuration. |
| A prompt edit had no effect | Check pending candidates, approval, selection, and the profile bound to the running role. Restarting alone does not approve a candidate. |
| MCP has no tools or cannot connect | Check the executable/config paths, start the supervisor, and restart the client connection. Stdio is the supported MCP transport. |
| An external interaction still reports `running` | Answers are delivered from the record on the next supervision pass, including after a restart, so a persistent `running` means the thought itself has not answered yet. Inspect **turns** before resubmitting; avoid duplicating work. |
| An external interaction reports `failed` after a long wait | The submission's patience ran out. The underlying thought can still be queued or running; check **turns** rather than assuming the work stopped. |

### Read the logs

Logs are in `Amoeba\state\logs`. Start with the supervisor and inference logs:

```powershell
Get-Content "$env:USERPROFILE\Amoeba\state\logs\supervisor.log" -Tail 80
Get-Content "$env:USERPROFILE\Amoeba\state\logs\inference.log" -Tail 80
```

Other components have their own logs in the same folder. For a useful problem
report, record the source revision, relevant configuration values, the affected
interaction/work/turn ID, and the corresponding log excerpt. Remove credentials
and private input before sharing.

## 16. Current operating limits

These are practical limits of this revision, not requirements for future
features:

- **A long message is stored whole and read in part.** The external interface
  accepts up to 32,000 characters and the whole request is now preserved, but
  a role reads a bounded rendering of about 8,000 characters, which states how
  much it withheld and which stored copy holds the rest. Put what matters
  early, or split very long material.
- **An external submission can time out while its thought continues.** If the
  wait exceeds the submission's patience, the interaction is recorded as
  `failed` even though the queued thought may still be running. Restarts no
  longer strand answers — those are delivered from the record — but a
  `failed` interaction is terminal and will not collect a later answer.
- **MCP attachment submission is incomplete.** `amoeba_attach` stores bytes,
  but `amoeba_ask` and `amoeba_submit` have no attachment-ID argument. Use the
  HTTP recipe in section 12 for file-based questions.
- **Poll for external completion.** The event stream publishes acceptance
  only, not the rest of the lifecycle; use `io_status`/`io_output` or their MCP
  counterparts.
- **Treat one instance as one cognitive trust domain.** Separate interaction
  IDs and client keys do not isolate everything a persistent model has already
  seen.
- **Allow disk space for the record.** Content blobs are not
  garbage-collected. Do not manually delete them to reduce storage use;
  recorded evidence and artifacts may depend on them.
- **Evaluate the answers.** The repository establishes many operational checks
  but does not establish model quality for every investigation or audit.
  Simulated output is never evidence of real model capability.

## 17. Daily reference

Run commands from `Amoeba\source` with its virtual-environment Python:

| Task | Command or location |
|---|---|
| Start | `.\.venv\Scripts\python.exe -m amoeba supervise --config ..\settings\local.toml` |
| Check running state | `.\.venv\Scripts\python.exe -m amoeba status --config ..\settings\local.toml` |
| Open console | [http://127.0.0.1:8715](http://127.0.0.1:8715) |
| Sign-in token | `Amoeba\state\operator.session` |
| Talk | **converse** |
| Ask about health/evidence | **consult id** |
| Inspect progress | **work**, **turns**, **health** |
| Review proposed files | **artifacts** |
| Accepted files from the panel | `Amoeba\state\artifacts` |
| Logs | `Amoeba\state\logs` |
| Stop | `.\.venv\Scripts\python.exe -m amoeba shutdown --config ..\settings\local.toml` |
| Start a fresh organism | `.\.venv\Scripts\python.exe -m amoeba reset --config ..\settings\local.toml` (stopped; archives the old one) |
| Preserve the organism | Stop, then back up the whole `state` folder and your settings. |

---

### Guide basis and upkeep

The operating steps were checked against the repository's command-line entry
point, configuration loader, browser console, MCP tools, and operator/external
interfaces at the revision this file ships in — ports, console panels, verb and
tool names, doctor checks, prompt state transitions, attachment limits, and the
reference runtime build and model were each read from the code rather than
carried over from earlier prose.

This guide is part of the code's contract with its operator: a change that
alters what an operator does, sees, or must account for updates this file in
the same commit as the change. The
[implementation notes](IMPLEMENTATION.md),
[runtime findings](RUNTIME.md) and
[interface description](INTERFACES.md) provide further background, and
[ARCHITECTURE.md](ARCHITECTURE.md) states the invariants behind the behaviour
described here. Where older prose and current controls differ, this guide
follows the current controls.
