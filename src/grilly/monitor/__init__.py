"""
Grilly Reasoning Monitor — lightweight HTML dashboard for live training metrics.

Replaces ImGui with a zero-dependency web UI that works on headless servers
via SSH tunnel (ssh -L 8080:localhost:8080 user@a40-host).

Usage:
    from grilly.monitor import ReasoningMonitor

    monitor = ReasoningMonitor(port=8080)
    monitor.start()

    # In training loop:
    monitor.record(surprise=0.42, stress=0.15)

    # With full cache stats (from VSACache.stats()):
    monitor.record(surprise=0.42, stress=0.15, cache_stats=cache_stats_dict)

    monitor.stop()
"""

from grilly.monitor.server import ReasoningMonitor

__all__ = ["ReasoningMonitor"]
