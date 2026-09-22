"""SQLite (WAL) schema and connection helper.

A single state-writer process owns writes; every other component reads. WAL mode
lets readers proceed during a write transaction.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

SCHEMA_VERSION = 1

SCHEMA_SQL = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

-- ------------------------------------------------------------------
-- Append-only raw history. Never updated, never deleted.
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS events (
  seq               INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id          TEXT NOT NULL UNIQUE,
  ts                REAL NOT NULL,
  run_id            TEXT NOT NULL,
  actor_id          TEXT NOT NULL,
  actor_incarnation INTEGER NOT NULL DEFAULT 0,
  operation_id      TEXT,
  causation_id      TEXT,
  correlation_id    TEXT,
  kind              TEXT NOT NULL,
  payload_sha256    TEXT,
  payload_inline    TEXT,
  prev_hash         TEXT NOT NULL,
  event_hash        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_events_op   ON events(operation_id);
CREATE INDEX IF NOT EXISTS ix_events_kind ON events(kind, seq);
CREATE INDEX IF NOT EXISTS ix_events_corr ON events(correlation_id);

CREATE TABLE IF NOT EXISTS blobs (
  sha256     TEXT PRIMARY KEY,
  size       INTEGER NOT NULL,
  encoding   TEXT NOT NULL,
  schema     TEXT,
  created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS receipts (
  receipt_id     TEXT PRIMARY KEY,
  mutation_id    TEXT NOT NULL UNIQUE,
  operation_id   TEXT,
  prior_version  INTEGER NOT NULL,
  result_version INTEGER NOT NULL,
  event_seq_from INTEGER NOT NULL,
  event_seq_to   INTEGER NOT NULL,
  outcome        TEXT NOT NULL,
  detail         TEXT,
  ts             REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS state_version (
  id      INTEGER PRIMARY KEY CHECK (id = 1),
  version INTEGER NOT NULL
);

-- ------------------------------------------------------------------
-- Maintained memory: interpretations, not raw history.
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory_items (
  memory_id     TEXT PRIMARY KEY,
  kind          TEXT NOT NULL,
  claim         TEXT NOT NULL,
  confidence    REAL NOT NULL,
  status        TEXT NOT NULL,
  version       INTEGER NOT NULL,
  supersedes    TEXT,
  tags          TEXT,
  created_by    TEXT NOT NULL,
  created_at    REAL NOT NULL,
  updated_at    REAL NOT NULL,
  state_version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_memory_status ON memory_items(status, kind);

CREATE TABLE IF NOT EXISTS memory_evidence (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  memory_id   TEXT NOT NULL REFERENCES memory_items(memory_id),
  stance      TEXT NOT NULL,
  event_seq   INTEGER,
  event_id    TEXT,
  blob_sha256 TEXT,
  note        TEXT
);
CREATE INDEX IF NOT EXISTS ix_mem_ev ON memory_evidence(memory_id);

CREATE TABLE IF NOT EXISTS conclusions (
  conclusion_id  TEXT PRIMARY KEY,
  claim          TEXT NOT NULL,
  uncertainty    REAL,
  alternatives   TEXT,
  operation_id   TEXT,
  produced_by    TEXT NOT NULL,
  review_status  TEXT NOT NULL DEFAULT 'unreviewed',
  -- Whether the claim is still being made. Deliberately not `review_status`,
  -- which says what an audit found: a claim may be audited `contested` and
  -- still stand, and may be withdrawn with no audit having happened.
  standing       TEXT NOT NULL DEFAULT 'active',   -- active|retracted|superseded
  superseded_by  TEXT,
  withdrawn_at   REAL,
  withdrawn_by   TEXT,
  withdrawn_reason TEXT,
  model_identity TEXT,
  snapshot_id    TEXT,
  created_at     REAL NOT NULL,
  state_version  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_concl_op ON conclusions(operation_id);

CREATE TABLE IF NOT EXISTS conclusion_evidence (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  conclusion_id TEXT NOT NULL REFERENCES conclusions(conclusion_id),
  event_seq     INTEGER,
  event_id      TEXT,
  blob_sha256   TEXT,
  memory_id     TEXT,
  note          TEXT
);
CREATE INDEX IF NOT EXISTS ix_concl_ev ON conclusion_evidence(conclusion_id);

-- ------------------------------------------------------------------
-- Work, operations, agents.
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS operations (
  operation_id    TEXT PRIMARY KEY,
  kind            TEXT NOT NULL,
  actor           TEXT NOT NULL,
  status          TEXT NOT NULL,
  idempotency_key TEXT UNIQUE,
  request_blob    TEXT,
  result_blob     TEXT,
  work_id         TEXT,
  receipt_id      TEXT,
  limitations     TEXT,
  created_at      REAL NOT NULL,
  updated_at      REAL NOT NULL,
  state_version   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_ops_status ON operations(status);

CREATE TABLE IF NOT EXISTS work_items (
  work_id           TEXT PRIMARY KEY,
  objective         TEXT NOT NULL,
  work_class        TEXT NOT NULL,
  origin_actor      TEXT NOT NULL,
  operation_id      TEXT,
  priority          INTEGER NOT NULL DEFAULT 0,
  depends_on        TEXT,
  snapshot_id       TEXT,
  model_generation  TEXT,
  pinned_state_ver  INTEGER,
  status            TEXT NOT NULL,
  lease_owner       TEXT,
  lease_expires     REAL,
  attempt           INTEGER NOT NULL DEFAULT 0,
  fencing_token     INTEGER NOT NULL DEFAULT 0,
  budget_tokens     INTEGER,
  deadline          REAL,
  maintenance_depth INTEGER NOT NULL DEFAULT 0,
  -- none | read | read_write. 'none' makes the neuocyte board-naive by
  -- construction, which is what turns agreement between two neuocytes into
  -- evidence of independent replication rather than an echo.
  board_access      TEXT NOT NULL DEFAULT 'read_write',
  sandbox_allowed   INTEGER NOT NULL DEFAULT 0,
  result_blob       TEXT,
  failure           TEXT,
  created_at        REAL NOT NULL,
  -- The neuocyte specialisation Ego asked for, as a leaf under its
  -- role's namespace: 'research' means `ego.neuocyte.research`. A
  -- request, not an instruction -- the library governs whether such a
  -- profile exists and is approved, and `profile_fallback` records the
  -- answer so a specialisation that quietly stopped applying is
  -- visible rather than merely absent.
  specialisation    TEXT,
  profile_fallback  TEXT,
  updated_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_work_status ON work_items(status, work_class, priority, created_at);

CREATE TABLE IF NOT EXISTS agents (
  agent_id         TEXT PRIMARY KEY,
  role             TEXT NOT NULL,
  incarnation      INTEGER NOT NULL DEFAULT 0,
  status           TEXT NOT NULL,
  pid              INTEGER,
  session_handle   TEXT,
  snapshot_id      TEXT,
  model_generation TEXT,
  work_id          TEXT,
  started_at       REAL,
  heartbeat_at     REAL,
  retired_at       REAL,
  detail           TEXT
);

-- ------------------------------------------------------------------
-- Ego snapshots (UKV): immutable published prefixes of the Ego context.
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS snapshots (
  snapshot_id      TEXT PRIMARY KEY,
  version          INTEGER NOT NULL,
  actor            TEXT NOT NULL,
  model_generation TEXT NOT NULL,
  token_count      INTEGER NOT NULL,
  tokens_blob      TEXT NOT NULL,
  text_blob        TEXT,
  kv_mode          TEXT NOT NULL,
  backend_handle   TEXT,
  refcount         INTEGER NOT NULL DEFAULT 0,
  status           TEXT NOT NULL,
  created_at       REAL NOT NULL,
  released_at      REAL,
  state_version    INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_snap_ver ON snapshots(actor, version);

CREATE TABLE IF NOT EXISTS snapshot_refs (
  ref_id      TEXT PRIMARY KEY,
  snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
  holder      TEXT NOT NULL,
  acquired_at REAL NOT NULL,
  released_at REAL
);
CREATE INDEX IF NOT EXISTS ix_snapref ON snapshot_refs(snapshot_id, released_at);

-- ------------------------------------------------------------------
-- Id outputs: audits and disagreements.
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audits (
  audit_id      TEXT PRIMARY KEY,
  target_kind   TEXT NOT NULL,
  target_id     TEXT NOT NULL,
  focus         TEXT,
  verdict       TEXT NOT NULL,
  findings      TEXT,
  unresolved    TEXT,
  evidence      TEXT,
  operation_id  TEXT,
  created_at    REAL NOT NULL,
  state_version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_audit_target ON audits(target_kind, target_id);

CREATE TABLE IF NOT EXISTS disagreements (
  disagreement_id TEXT PRIMARY KEY,
  subject_kind    TEXT NOT NULL,
  subject_id      TEXT NOT NULL,
  claim_a         TEXT NOT NULL,
  actor_a         TEXT NOT NULL,
  claim_b         TEXT NOT NULL,
  actor_b         TEXT NOT NULL,
  evidence_a      TEXT,
  evidence_b      TEXT,
  status          TEXT NOT NULL,   -- open|superseded|retracted|
                                   -- resolved_supported|closed_by_operator
  -- What the opening audit was performed against. A later audit may close
  -- this dispute only if its own basis differs: the same adjudicator
  -- changing its mind about the same evidence is not a resolution.
  evidence_basis_digest TEXT,
  resolution      TEXT,
  resolved_at     REAL,
  resolved_by     TEXT,
  -- How many times the same contradiction has been reached again. A repeat
  -- attaches here instead of opening a rival row.
  recurrences     INTEGER NOT NULL DEFAULT 0,
  created_at      REAL NOT NULL,
  state_version   INTEGER NOT NULL
);
-- One live dispute per disputed thing. Without this, auditing one conclusion
-- twenty times turns one unresolved issue into twenty rows, and the count
-- stops describing anything.
CREATE UNIQUE INDEX IF NOT EXISTS ux_one_open_disagreement_per_subject
  ON disagreements(subject_kind, subject_id) WHERE status = 'open';

-- ------------------------------------------------------------------
-- Cognitive blackboard: neuocyte-to-neuocyte communication.
--
-- This is NOT authoritative Mind State. A post is something a neuocyte said,
-- not something the organism believes. Promotion into memory_items is a
-- separate, receipted act.
--
-- board_reads exists for one reason: to tell independent replication apart
-- from socially propagated agreement. Two neuocytes reaching the same finding
-- means something very different depending on whether the second had read the
-- first, so every read is recorded with a timestamp and every post snapshots
-- what its author had already seen.
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS board_posts (
  post_id          TEXT PRIMARY KEY,
  -- Wall-clock is NOT a usable cursor here: time.time() on this platform has
  -- ~0.5ms granularity and returns identical values for consecutive calls, so
  -- a "since <timestamp>" poll silently drops posts written in the same tick.
  -- seq is assigned under the writer's transaction lock and is strictly
  -- increasing, so it is the cursor neuocytes should page on.
  seq              INTEGER NOT NULL DEFAULT 0,
  thread_id        TEXT NOT NULL,
  author           TEXT NOT NULL,
  author_kind      TEXT NOT NULL,          -- ego | id | neuocyte | operator
  author_incarnation INTEGER,
  -- Which attempt at the work wrote this. A work item can fail one attempt
  -- and succeed on the next; a post from the fenced attempt must not be
  -- laundered through the later success, so the fate that matters is the
  -- author's lease, identified by the token that was live when it posted.
  -- Read from the work row by the Harness, never supplied by the author.
  author_fencing_token INTEGER,
  author_attempt   INTEGER,
  work_id          TEXT,
  operation_id     TEXT,
  post_type        TEXT NOT NULL,          -- finding|question|hypothesis|challenge|request|answer|note|retraction
  title            TEXT,
  body             TEXT NOT NULL,
  confidence       REAL,
  snapshot_id      TEXT,
  model_generation TEXT,
  status           TEXT NOT NULL DEFAULT 'open',   -- open|resolved|retracted|superseded
  supersedes       TEXT,
  -- independence bookkeeping, written at post time and never edited
  informed_by      TEXT NOT NULL DEFAULT '[]',     -- post_ids this author had read BEFORE posting
  read_count_before INTEGER NOT NULL DEFAULT 0,
  board_naive      INTEGER NOT NULL DEFAULT 1,     -- 1 = author had read nothing at all
  created_at       REAL NOT NULL,
  state_version    INTEGER NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ix_board_seq ON board_posts(seq);
CREATE INDEX IF NOT EXISTS ix_board_thread ON board_posts(thread_id, seq);
CREATE INDEX IF NOT EXISTS ix_board_type   ON board_posts(post_type, created_at);
CREATE INDEX IF NOT EXISTS ix_board_author ON board_posts(author, created_at);
CREATE INDEX IF NOT EXISTS ix_board_work   ON board_posts(work_id);

CREATE TABLE IF NOT EXISTS board_evidence (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  post_id     TEXT NOT NULL REFERENCES board_posts(post_id),
  event_id    TEXT,
  event_seq   INTEGER,
  blob_sha256 TEXT,
  memory_id   TEXT,
  artifact_id TEXT,
  note        TEXT
);
CREATE INDEX IF NOT EXISTS ix_board_ev ON board_evidence(post_id);

CREATE TABLE IF NOT EXISTS board_relations (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  from_post TEXT NOT NULL REFERENCES board_posts(post_id),
  to_post   TEXT NOT NULL REFERENCES board_posts(post_id),
  relation  TEXT NOT NULL,   -- reply_to|challenges|supports|refines|duplicates|answers
  created_at REAL NOT NULL,
  UNIQUE(from_post, to_post, relation)
);
CREATE INDEX IF NOT EXISTS ix_board_rel_from ON board_relations(from_post);
CREATE INDEX IF NOT EXISTS ix_board_rel_to   ON board_relations(to_post);

CREATE TABLE IF NOT EXISTS board_reads (
  id        INTEGER PRIMARY KEY AUTOINCREMENT,
  post_id   TEXT NOT NULL REFERENCES board_posts(post_id),
  reader    TEXT NOT NULL,
  work_id   TEXT,
  read_at   REAL NOT NULL,
  query     TEXT,
  -- The fate as it was rendered to this reader. A fate changes after the
  -- read, and `informed_by` is frozen for exactly this reason: what a reader
  -- was shown stops being answerable once the world moves on. Without this,
  -- a reader influenced by `attempt: running` is indistinguishable later from
  -- one influenced by `attempt: fenced`.
  attempt_fate_at_read TEXT,
  work_status_at_read  TEXT,
  state_version_at_read INTEGER
);
CREATE INDEX IF NOT EXISTS ix_board_reads_reader ON board_reads(reader, read_at);
CREATE INDEX IF NOT EXISTS ix_board_reads_post   ON board_reads(post_id);

-- ------------------------------------------------------------------
-- Compute sandboxes: ephemeral execution scratch and promotion proposals.
-- ------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sandboxes (
  sandbox_id    TEXT PRIMARY KEY,
  owner         TEXT NOT NULL,
  work_id       TEXT,
  container_sid TEXT,
  root          TEXT NOT NULL,
  limits        TEXT,
  status        TEXT NOT NULL,          -- active | destroyed
  created_at    REAL NOT NULL,
  destroyed_at  REAL,
  state_version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
  artifact_id   TEXT PRIMARY KEY,
  sandbox_id    TEXT,
  proposed_by   TEXT NOT NULL,
  work_id       TEXT,
  path          TEXT NOT NULL,          -- path within the compute sandbox
  sha256        TEXT NOT NULL,
  bytes         INTEGER NOT NULL,
  media_type    TEXT,
  rationale     TEXT,
  status        TEXT NOT NULL,          -- proposed | promoted | rejected
  decided_by    TEXT,
  decided_at    REAL,
  reason        TEXT,
  created_at    REAL NOT NULL,
  state_version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_artifact_status ON artifacts(status, created_at);

-- Targeted messages to a *work item*, not to a worker process. A running
-- neuocyte is not addressable: the Harness records a message here and the
-- neuocyte collects it at a turn boundary, so mid-flight communication cannot
-- become a side channel into a live sandbox.
--
-- `consumed_at` is what makes influence visible: a finding produced after a
-- message was collected is a finding the message may have shaped, and the work
-- record says so rather than leaving a reader to guess.
CREATE TABLE IF NOT EXISTS work_messages (
  message_id    TEXT PRIMARY KEY,
  work_id       TEXT NOT NULL,
  from_role     TEXT NOT NULL,          -- authenticated scope, never claimed
  body          TEXT NOT NULL,
  kind          TEXT NOT NULL,          -- clarification | constraint | context
  created_at    REAL NOT NULL,
  consumed_at   REAL,
  consumed_by   TEXT,
  state_version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_work_message_work ON work_messages(work_id, created_at);

-- External interactions: one row per unit of input an outside client sent.
--
-- This is the whole external surface. A client owns its interactions and sees
-- nothing else: `client_id` is the authenticated credential's identity, never
-- a field the caller supplies, so scoping a read to "mine" is a fact rather
-- than a filter someone could ask to have widened.
--
-- Input bytes are content-addressed like any other admitted input, so what the
-- client actually sent is recoverable by digest and cannot be confused with
-- what Amoeba decided to do about it.
CREATE TABLE IF NOT EXISTS interactions (
  interaction_id  TEXT PRIMARY KEY,
  client_id       TEXT NOT NULL,          -- authenticated identity, never claimed
  surface         TEXT NOT NULL,          -- mcp | api
  kind            TEXT NOT NULL,          -- converse | investigate
  conversation_id TEXT,
  input_sha256    TEXT NOT NULL,
  input_preview   TEXT,
  status          TEXT NOT NULL,          -- accepted | running | complete
                                          --   | incomplete | failed
  operation_id    TEXT,
  output_sha256   TEXT,
  output_preview  TEXT,
  error           TEXT,
  created_at      REAL NOT NULL,
  completed_at    REAL,
  state_version   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_interaction_client
  ON interactions(client_id, created_at);

-- Files an external client attached to an interaction. Admitted input, nothing
-- more: exact bytes, a digest, and a record of which client sent them. Never a
-- host path, and never a Filespace write.
CREATE TABLE IF NOT EXISTS interaction_inputs (
  input_id       TEXT PRIMARY KEY,
  interaction_id TEXT,
  client_id      TEXT NOT NULL,
  filename       TEXT NOT NULL,
  sha256         TEXT NOT NULL,
  bytes          INTEGER NOT NULL,
  media_type     TEXT,
  created_at     REAL NOT NULL,
  state_version  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_interaction_input
  ON interaction_inputs(interaction_id);

-- Results deliberately surfaced outward. An artifact is readable by a client
-- only if it appears here: knowing an artifact id is not authority to fetch it,
-- and there is no route from an identifier to the blob store.
CREATE TABLE IF NOT EXISTS interaction_results (
  result_id      TEXT PRIMARY KEY,
  interaction_id TEXT NOT NULL,
  client_id      TEXT NOT NULL,
  artifact_id    TEXT,
  sha256         TEXT NOT NULL,
  filename       TEXT,
  media_type     TEXT,
  bytes          INTEGER NOT NULL,
  surfaced_by    TEXT NOT NULL,
  created_at     REAL NOT NULL,
  state_version  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_interaction_result
  ON interaction_results(interaction_id);

-- The Prompt Library: Amoeba's versioned cognitive family tree.
--
-- Every row is an immutable node version that pins the EXACT parent version it
-- inherits from. That pinning is the architecture: promoting a new parent
-- cannot silently change what an existing child means, because the child names
-- the version it was built against rather than "whatever the parent is now".
--
-- A new local version is created when local properties change OR when the node
-- is rebased onto a different parent version. Two consecutive versions may
-- therefore have identical local definitions and still be legitimately
-- different profiles, because their inherited environment differs.
CREATE TABLE IF NOT EXISTS prompt_versions (
  version_id       TEXT PRIMARY KEY,
  namespace        TEXT NOT NULL,
  local_version    INTEGER NOT NULL,
  parent_namespace TEXT,                  -- NULL only for a root
  parent_version   INTEGER,               -- the pinned parent local version
  prompt_mode      TEXT NOT NULL,         -- inherit|append|prepend|replace
  prompt_text      TEXT NOT NULL DEFAULT '',
  model_vars       TEXT NOT NULL DEFAULT '{}',   -- canonical JSON, local only
  local_sha256     TEXT NOT NULL,         -- digest of the LOCAL definition
  state            TEXT NOT NULL,         -- see promptlib.store.STATES
  origin           TEXT NOT NULL,         -- bootstrap|id|operator|cascade
  created_by       TEXT NOT NULL,
  created_at       REAL NOT NULL,
  rationale        TEXT,
  state_version    INTEGER NOT NULL,
  UNIQUE(namespace, local_version)
);
CREATE INDEX IF NOT EXISTS ix_prompt_ns ON prompt_versions(namespace, local_version);
CREATE INDEX IF NOT EXISTS ix_prompt_state ON prompt_versions(state, created_at);

-- Which version is currently selected for a purpose. Selection is separate
-- from approval: a version can be approved and not selected, and history keeps
-- every previously selected version usable.
CREATE TABLE IF NOT EXISTS prompt_selections (
  namespace     TEXT NOT NULL,
  purpose       TEXT NOT NULL,            -- production|experimental
  version_id    TEXT NOT NULL,
  selected_by   TEXT NOT NULL,
  selected_at   REAL NOT NULL,
  state_version INTEGER NOT NULL,
  PRIMARY KEY (namespace, purpose)
);

-- Id's evaluation of a candidate. Advisory: Id evaluates, the Operator decides.
CREATE TABLE IF NOT EXISTS prompt_evaluations (
  evaluation_id TEXT PRIMARY KEY,
  version_id    TEXT NOT NULL,
  evaluator     TEXT NOT NULL,
  verdict       TEXT NOT NULL,            -- endorse|concern|oppose
  notes         TEXT,
  evidence      TEXT,
  created_at    REAL NOT NULL,
  state_version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_prompt_eval ON prompt_evaluations(version_id);

-- The Operator's decision, recorded like any other consequential act.
CREATE TABLE IF NOT EXISTS prompt_decisions (
  decision_id   TEXT PRIMARY KEY,
  version_id    TEXT NOT NULL,
  decision      TEXT NOT NULL,            -- production|experimental|rejected|retired
  decided_by    TEXT NOT NULL,
  rationale     TEXT,
  propagation   TEXT,                     -- cascade_approve|cascade_queue|none
  created_at    REAL NOT NULL,
  state_version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_prompt_decision ON prompt_decisions(version_id);

-- What a cognition-producing incarnation actually received, frozen at birth.
-- Historical cognition must never depend on re-resolving against a future
-- library, so the resolved text and digests are stored rather than recomputed.
CREATE TABLE IF NOT EXISTS incarnation_profiles (
  binding_id        TEXT PRIMARY KEY,
  actor_id          TEXT NOT NULL,
  actor_kind        TEXT NOT NULL,        -- ego|id|neuocyte
  incarnation       INTEGER,
  work_id           TEXT,
  namespace         TEXT NOT NULL,
  profile_ref       TEXT NOT NULL,        -- ego.neuocyte.research@4.8.5
  lineage           TEXT NOT NULL,        -- canonical JSON, leaf->root
  prompt_sha256     TEXT NOT NULL,
  config_sha256     TEXT NOT NULL,
  profile_sha256    TEXT NOT NULL,
  model_generation  TEXT,
  effective_settings TEXT NOT NULL,       -- after Harness constraints
  harness_constraints TEXT NOT NULL DEFAULT '{}',
  created_at        REAL NOT NULL,
  state_version     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_incarnation_actor
  ON incarnation_profiles(actor_id, created_at);

-- ===================================================================
-- Persistent roles: the mailbox, the bundle, and the turn
-- ===================================================================
-- Ego and Id are persistent identities whose cognition happens in bounded
-- turns. The Harness owns when a turn begins, what woke it, what inputs it is
-- given, and what happens when it ends; the model owns only the reasoning
-- inside it. These three tables are where that ownership lives.

-- One thing that happened which a role may need to think about. Being queued
-- is NOT being seen: a trigger becomes a cognitive input only when the Harness
-- puts it in a specific turn's bundle.
CREATE TABLE IF NOT EXISTS role_triggers (
  trigger_id    TEXT PRIMARY KEY,
  target_role   TEXT NOT NULL,           -- ego | id
  kind          TEXT NOT NULL,           -- see mailbox.TRIGGER_KINDS
  source        TEXT NOT NULL,           -- who or what produced it
  source_ref    TEXT,                    -- work_id, interaction_id, post_id...
  summary       TEXT NOT NULL DEFAULT '',-- bounded, rendered to the model
  payload_sha256 TEXT,                   -- immutable full payload, if any
  correlation_id TEXT,
  operation_id  TEXT,                    -- the externally visible unit
                                         -- that produced this trigger
  causal_parent TEXT,                    -- the turn that caused this one
  status        TEXT NOT NULL,           -- queued|claimed|consumed|expired
  bundle_id     TEXT,                    -- set when claimed
  turn_id       TEXT,                    -- the turn that consumed it
  created_at    REAL NOT NULL,
  claimed_at    REAL,
  consumed_at   REAL,
  deliveries    INTEGER NOT NULL DEFAULT 0,
  -- Someone is waiting on this one. A request expects an answer; an event
  -- that merely wakes a role does not, and conflating them is how a work
  -- completion ended up "answering" a user's question.
  expects_answer INTEGER NOT NULL DEFAULT 0,
  -- Whose information this is. Derived rather than required: an interaction
  -- id, a conversation, or the operation that started the lineage. NULL means
  -- unowned, which is not the same as ambient.
  lineage        TEXT,
  -- Every turn may see it: a resource change, an approved profile, an
  -- operator announcement. Explicit, because "unrelated" must never become
  -- the default merely because nothing is owed a reply.
  ambient        INTEGER NOT NULL DEFAULT 0,
  answer_sha256  TEXT,             -- the answer to THIS request
  answer_status  TEXT,             -- NULL | answered | incomplete
                                   --   | unanswerable
  answered_at    REAL,
  answered_by_turn TEXT,           -- which turn finally produced it
  state_version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_trigger_queue
  ON role_triggers(target_role, status, created_at);
CREATE INDEX IF NOT EXISTS ix_trigger_turn ON role_triggers(turn_id);
CREATE INDEX IF NOT EXISTS ix_trigger_lineage
  ON role_triggers(target_role, lineage, status);
CREATE INDEX IF NOT EXISTS ix_trigger_awaiting
  ON role_triggers(target_role, expects_answer, answer_status);

-- One bounded turn of a persistent role. Rows are append-only: a turn is
-- opened, then closed with its stop reason. The environment and profile are
-- recorded as digests plus an immutable blob so the exact cognition can be
-- reconstructed after the library and the world have moved on.
CREATE TABLE IF NOT EXISTS role_turns (
  turn_id          TEXT PRIMARY KEY,
  role             TEXT NOT NULL,
  incarnation      INTEGER,
  profile_ref      TEXT,
  profile_sha256   TEXT,
  environment_sha256 TEXT,
  environment_blob TEXT,
  bundle_id        TEXT,
  bundle_sha256    TEXT,                 -- digest of the exact rendered bundle
  bundle_blob      TEXT,                 -- the bundle bytes, recoverable
  trigger_kinds    TEXT NOT NULL DEFAULT '[]',
  trigger_count    INTEGER NOT NULL DEFAULT 0,
  started_at       REAL NOT NULL,
  finished_at      REAL,
  status           TEXT NOT NULL,        -- running|completed|failed|abandoned
  stop_reason      TEXT,                 -- see mailbox.STOP_REASONS
  model_generation TEXT,
  tool_call_count  INTEGER NOT NULL DEFAULT 0,
  result_sha256    TEXT,
  -- Which tokens of the session this turn occupies. Exact, not estimated:
  -- a role runs one turn at a time, so everything appended between these two
  -- lengths belongs to this turn. Offsets are meaningless across a
  -- rejuvenation, hence the handle they were measured in.
  session_handle   TEXT,
  token_start      INTEGER,
  token_end        INTEGER,
  parent_turn      TEXT,                 -- the turn this continues
  lineage          TEXT,                 -- whose interaction this turn serves
  operation_id     TEXT,
  state_version    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_role_turn ON role_turns(role, started_at);
CREATE INDEX IF NOT EXISTS ix_role_turn_status ON role_turns(role, status);

-- Exactly one turn may be open per persistent role. A UNIQUE index over
-- (role) filtered to open turns makes a second concurrent turn a constraint
-- violation in the database rather than a convention somebody has to keep.
CREATE UNIQUE INDEX IF NOT EXISTS ux_one_open_turn_per_role
  ON role_turns(role) WHERE status = 'running';

CREATE TABLE IF NOT EXISTS conversations (
  conversation_id TEXT PRIMARY KEY,
  created_at      REAL NOT NULL,
  updated_at      REAL NOT NULL,
  turn_count      INTEGER NOT NULL DEFAULT 0
);
"""


def connect(path: Path, *, read_only: bool = False, timeout: float = 30.0) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if read_only and path.exists():
        uri = f"file:{path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=timeout, check_same_thread=False)
    else:
        conn = sqlite3.connect(str(path), timeout=timeout, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    if not read_only:
        conn.execute("PRAGMA synchronous=FULL")
    return conn


class Database:
    """Owns one SQLite connection. Writers use a single instance per process.

    ``tx_lock`` serialises *transactions*, not individual statements. Python's
    sqlite3 is built in serialized mode, so a single ``execute`` from two
    threads is safe -- but a connection has exactly one transaction. Without
    this lock, two threads inside ``StateWriter.apply`` would interleave: the
    second ``BEGIN IMMEDIATE`` fails or silently joins the first transaction,
    and one thread's ``commit`` publishes the other's half-finished work. The
    observable symptom is a broken event hash chain, because two writers
    computed ``prev_hash`` from the same tip.
    """

    def __init__(self, path: Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self.tx_lock = threading.RLock()
        self.conn = connect(self.path, read_only=read_only)
        if not read_only:
            self.initialize()

    def initialize(self) -> None:
        # Before the schema script, not after: the script creates indexes that
        # name the ownership columns, so against a database written by the
        # previous release it would abort at `CREATE INDEX ... (lineage)`
        # having already run half of itself. On a fresh database the tables do
        # not exist yet and this is a no-op.
        self._migrate_turn_ownership()
        self.conn.executescript(SCHEMA_SQL)
        cur = self.conn.execute("SELECT version FROM state_version WHERE id = 1")
        if cur.fetchone() is None:
            self.conn.execute("INSERT INTO state_version(id, version) VALUES (1, 0)")
        self.conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._migrate_vocabulary()
        self.conn.commit()

    def _migrate_vocabulary(self) -> None:
        """Normalise the pre-rename vocabulary in an existing database.

        Disposable cognitive workers are called neuocytes. A database written
        before the rename stores the role as 'worker', and leaving it would
        make live_agents(role=...) quietly miss them. Rewriting these two
        columns is safe: they are enumerations, not evidence, and the
        append-only event payloads that recorded the old word are deliberately
        left alone.
        """
        for table, column in (("agents", "role"), ("board_posts", "author_kind")):
            try:
                cur = self.conn.execute(
                    f"UPDATE {table} SET {column} = 'neuocyte' WHERE {column} = 'worker'")
                if cur.rowcount:
                    self.conn.execute(
                        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                        (f"migrated_{table}_{column}", str(cur.rowcount)))
            except sqlite3.OperationalError:
                pass

    # The columns that make an answer belong to a request rather than to a
    # turn, and information belong to a lineage rather than to whoever is
    # nearby. Table, column, and the DDL fragment to add it with.
    _OWNERSHIP_COLUMNS = (
        ("role_triggers", "expects_answer", "INTEGER NOT NULL DEFAULT 0"),
        ("role_triggers", "answer_sha256", "TEXT"),
        ("role_triggers", "answer_status", "TEXT"),
        ("role_triggers", "answered_at", "REAL"),
        ("role_triggers", "answered_by_turn", "TEXT"),
        ("role_triggers", "lineage", "TEXT"),
        ("role_triggers", "ambient", "INTEGER NOT NULL DEFAULT 0"),
        ("role_turns", "lineage", "TEXT"),
        ("work_items", "specialisation", "TEXT"),
        ("work_items", "profile_fallback", "TEXT"),
        ("role_turns", "session_handle", "TEXT"),
        ("role_turns", "token_start", "INTEGER"),
        ("role_turns", "token_end", "INTEGER"),
        ("conclusions", "standing", "TEXT NOT NULL DEFAULT 'active'"),
        ("conclusions", "superseded_by", "TEXT"),
        ("conclusions", "withdrawn_at", "REAL"),
        ("conclusions", "withdrawn_by", "TEXT"),
        ("conclusions", "withdrawn_reason", "TEXT"),
        ("disagreements", "evidence_basis_digest", "TEXT"),
        ("disagreements", "resolution", "TEXT"),
        ("disagreements", "resolved_at", "REAL"),
        ("disagreements", "resolved_by", "TEXT"),
        ("disagreements", "recurrences", "INTEGER NOT NULL DEFAULT 0"),
        ("board_posts", "author_fencing_token", "INTEGER"),
        ("board_posts", "author_attempt", "INTEGER"),
        ("board_reads", "attempt_fate_at_read", "TEXT"),
        ("board_reads", "work_status_at_read", "TEXT"),
        ("board_reads", "state_version_at_read", "INTEGER"),
    )

    def _migrate_turn_ownership(self) -> None:
        """Add the request-ownership columns to a database that predates them.

        `CREATE TABLE IF NOT EXISTS` does nothing to a table that already
        exists, so a database written by the previous release keeps the old
        `role_triggers` and every mailbox read fails on a missing column. The
        live table is inspected rather than a version number trusted: adding a
        column that is already there is then simply a no-op, and a database
        that was interrupted half-way through an upgrade still converges.

        Every added column is nullable or carries a default, so existing rows
        stay valid. An old trigger gets `expects_answer = 0` and no lineage,
        which reads as "nothing is owed a reply, and this belongs to nobody" --
        the conservative interpretation, and the one that cannot invent an
        answer for a request that never asked for one.
        """
        for table, column, ddl in self._OWNERSHIP_COLUMNS:
            try:
                cur = self.conn.execute(f"PRAGMA table_info({table})")
                existing = {row[1] for row in cur.fetchall()}
            except sqlite3.OperationalError:
                continue
            if not existing or column in existing:
                continue
            try:
                self.conn.execute(
                    f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
                self.conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
                    (f"migrated_{table}_{column}", "added"))
            except sqlite3.OperationalError:
                # Lost a race with another process applying the same schema.
                pass

    def close(self) -> None:
        try:
            self.conn.close()
        except sqlite3.Error:
            pass

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
