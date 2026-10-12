-- One row identifying this file. id must match the directory name.
CREATE TABLE tenant (
  id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  llm_api_key_env TEXT NOT NULL DEFAULT 'AGENT_LLM_API_KEY',
  singleton INTEGER NOT NULL DEFAULT 1 CHECK (singleton = 1) UNIQUE
);

-- One coding task. fencing_token is the commit capability. workspace_root is relative to the tenant directory.
CREATE TABLE tasks (
  tenant_id TEXT NOT NULL,
  id TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed', 'rejected')),
  fencing_token INTEGER NOT NULL DEFAULT 0,
  lease_owner TEXT,
  lease_expires_at REAL,
  workspace_root TEXT NOT NULL,
  PRIMARY KEY (tenant_id, id)
);

-- Durable tool intent. Status moves from intent_logged to committed, failed, or rolled_back.
CREATE TABLE step_intents (
  tenant_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  step_index INTEGER NOT NULL,
  idempotency_key TEXT NOT NULL,
  tool_name TEXT NOT NULL,
  args_json TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('intent_logged', 'committed', 'failed', 'rolled_back')),
  result_json TEXT,
  PRIMARY KEY (tenant_id, task_id, step_index)
);

-- Result cache keyed by the SHA256 idempotency key inside this tenant file.
CREATE TABLE idempotency_cache (
  tenant_id TEXT NOT NULL,
  idempotency_key TEXT NOT NULL,
  result_json TEXT NOT NULL,
  result_sha256 TEXT NOT NULL,
  PRIMARY KEY (tenant_id, idempotency_key)
);

-- Chat transcript continued by a successor worker. seq is the order inside one task.
CREATE TABLE messages (
  tenant_id TEXT NOT NULL,
  task_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  role TEXT NOT NULL,
  body_json TEXT NOT NULL,
  PRIMARY KEY (tenant_id, task_id, seq)
);

-- Abort a task row whose tenant_id disagrees with the one-row tenant table.
CREATE TRIGGER tasks_tenant_stamp
BEFORE INSERT ON tasks
WHEN NEW.tenant_id != (SELECT id FROM tenant)
BEGIN
  SELECT RAISE(ABORT, 'tenant mismatch');
END;

-- Abort a step row whose tenant_id disagrees with the one-row tenant table.
CREATE TRIGGER step_intents_tenant_stamp
BEFORE INSERT ON step_intents
WHEN NEW.tenant_id != (SELECT id FROM tenant)
BEGIN
  SELECT RAISE(ABORT, 'tenant mismatch');
END;

-- Abort a cache row whose tenant_id disagrees with the one-row tenant table.
CREATE TRIGGER idempotency_cache_tenant_stamp
BEFORE INSERT ON idempotency_cache
WHEN NEW.tenant_id != (SELECT id FROM tenant)
BEGIN
  SELECT RAISE(ABORT, 'tenant mismatch');
END;

-- Abort a message row whose tenant_id disagrees with the one-row tenant table.
CREATE TRIGGER messages_tenant_stamp
BEFORE INSERT ON messages
WHEN NEW.tenant_id != (SELECT id FROM tenant)
BEGIN
  SELECT RAISE(ABORT, 'tenant mismatch');
END;
