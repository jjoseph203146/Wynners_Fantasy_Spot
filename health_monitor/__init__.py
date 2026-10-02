"""
NFL APP Health Monitor.

V1:
    isolated observation-only production monitor

V2:
    persistent incident state
    alert deduplication/recovery semantics
    repair authorization/circuit breaker
    guarded allowlisted repair runner
    dry-run-by-default controller
    durable provider-neutral notification outbox

Scheduling and external notification delivery remain separate.
"""
