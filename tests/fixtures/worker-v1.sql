-- Frozen pre-refactor schema and completed job; do not regenerate with Worker.
CREATE TABLE worker_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE docker_jobs (
    job_id TEXT PRIMARY KEY, job_key TEXT UNIQUE NOT NULL, spec TEXT NOT NULL,
    policy_hash TEXT NOT NULL, status TEXT NOT NULL, container TEXT NOT NULL UNIQUE,
    created REAL NOT NULL, deadline REAL, lease_owner TEXT, lease_until REAL,
    stop_reason TEXT, exit_code INTEGER, reason TEXT, receipt TEXT);
CREATE TABLE docker_stop_intents (
    job_id TEXT PRIMARY KEY REFERENCES docker_jobs(job_id),
    reason TEXT NOT NULL CHECK(reason IN ('timed_out','cancelled')),
    requested REAL NOT NULL, last_checked REAL,
    container_state TEXT, confirmed REAL, detail TEXT);
INSERT INTO worker_meta VALUES ('owner', '0123456789abcdef0123456789abcdef');
INSERT INTO docker_jobs (
    job_id, job_key, spec, policy_hash, status, container, created, deadline,
    exit_code, receipt
) VALUES (
    'job-07b7991def2e396f507936e2a635b5de78a3f915',
    '07b7991def2e396f507936e2a635b5de78a3f9150f264f4c287e567f1b281a83',
    '{"config":"default","dataset_id":"tiny","limits":{"cpus":2.0,"memory_mb":2048,"pids":128,"seconds":600},"model_job_id":null,"phase":"train","seed":42,"source_id":"baseline"}',
    'e504bd5bf9323834ccb8393ec62900e267049f5744e2b6fa2fe859d81ade3fe1',
    'completed', 'reclab-07b7991def2e396f507936e2a635b5de78a3f915', 1000.0, 1600.0,
    0,
    '{"bytes":3,"files":{"model.json":{"bytes":3,"sha256":"ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356"}},"sha256":"44f2014af64f4344819e2da45413274dae5e685a0ced3f7102e3985bca936ac4"}'
);
-- Stop requests can be retained after a terminal job to supervise late restarts.
INSERT INTO docker_stop_intents VALUES (
    'job-07b7991def2e396f507936e2a635b5de78a3f915', 'cancelled', 1700.0,
    1701.0, 'exited', 1701.0, 'legacy stop observation'
);
