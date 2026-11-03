#!/usr/bin/env bash
# Create broken (and one healthy) containers for triage practice and evals.
set -euo pipefail

docker rm -f payments worker api web >/dev/null 2>&1 || true

# 1. Crash loop: dependency unreachable. Exits 1, Docker restarts it up to 5 times.
docker run -d --name payments --restart on-failure:5 alpine:3.20 \
  sh -c 'echo "payments-svc starting"; echo "connecting to postgres at 10.0.3.14:5432"; sleep 1;
         echo "Error: connect ECONNREFUSED 10.0.3.14:5432" >&2; exit 1'

# 2. OOM kill: 32 MB memory limit (and no swap), app tries to allocate ~200 MB.
docker run -d --name worker -m 32m --memory-swap 32m python:3.12-alpine \
  python -c 'print("loading batch into memory", flush=True); data = b"x" * (200 * 1024 * 1024); print("done")'

# 3. Config error: PORT must be an integer. DB_PASSWORD is here to prove secrets are hidden.
docker run -d --name api -e PORT=abc -e DB_PASSWORD=s3cret python:3.12-alpine \
  python -c 'import os; print("api starting", flush=True); port = int(os.environ["PORT"])'

# 4. Healthy control: a good agent should NOT report a problem here.
docker run -d --name web -p 8080:80 nginx:alpine

sleep 8
docker ps -a --format 'table {{.Names}}\t{{.Status}}'