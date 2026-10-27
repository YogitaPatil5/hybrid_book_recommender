"""
Gunicorn production config.

    gunicorn -c gunicorn.conf.py "bookmind.main:create_app()"

Every value can be overridden with the matching BOOKMIND_* env var.
"""

import os

bind = f"{os.getenv('BOOKMIND_HOST', '0.0.0.0')}:{os.getenv('BOOKMIND_PORT', '8000')}"
workers = int(os.getenv("BOOKMIND_WORKERS", "2"))
worker_class = "uvicorn_worker.UvicornWorker"

# Each worker loads its own copy of the model + index (~150 MB for 7k books
# with MiniLM). Size `workers` to available RAM, not just CPU count.
timeout = int(os.getenv("BOOKMIND_TIMEOUT", "60"))
graceful_timeout = 30
keepalive = 5

# Recycle workers periodically to cap memory growth from fragmentation.
max_requests = int(os.getenv("BOOKMIND_MAX_REQUESTS", "2000"))
max_requests_jitter = 200

# Our middleware writes structured access logs; gunicorn logs only errors.
accesslog = None
errorlog = "-"
loglevel = os.getenv("BOOKMIND_LOG_LEVEL", "info").lower()

# Trust X-Forwarded-* only from these proxy IPs (set to your LB / ingress).
forwarded_allow_ips = os.getenv("FORWARDED_ALLOW_IPS", "127.0.0.1")
