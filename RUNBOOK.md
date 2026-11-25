# Runbook: ECG API on EC2

How to redeploy the service and how to roll back if a deployment breaks.
First-time setup is in [DEPLOYMENT.md](DEPLOYMENT.md).

**Service:** container `ecg-api` on the EC2 instance, port 80 → 8000.
**Images:** `ghcr.io/jawad1569/ecganomaly:<tag>`, built by CI on every push to `main`.
Tags: `<short commit SHA>` (7 characters, one per commit), `latest`, and `v*` release tags.
**Deploy tool:** `~/deploy.sh` on the server ([deploy/deploy.sh](deploy/deploy.sh)). It records the
running image in `~/.ecg-api/current`, the one before it in `~/.ecg-api/previous`, and every deploy in
`~/.ecg-api/history.log`.

Connect to the server (from the repository root on your PC):

```powershell
$ip = (Get-Content .ec2-instance.json | ConvertFrom-Json).public_ip
ssh -i "$env:USERPROFILE\.ssh\ecg-key.pem" ubuntu@$ip
```

## Check the service

```bash
~/deploy.sh --status            # current + previous image, docker ps (look for "healthy"), /health, /stats
docker logs --tail 50 ecg-api   # recent requests and errors
cat ~/.ecg-api/history.log      # what was deployed when
```

From anywhere: `curl http://<ip>/health` and `curl http://<ip>/stats`.

## Redeploy

1. Push to `main` and wait for CI (lint, test, build) to go green. The build job pushes the image.
2. Find the tag: the short SHA of the commit (`git rev-parse --short=7 HEAD`), also listed on GitHub under
   **Packages → ecganomaly**.
3. On the server:

   ```bash
   ~/deploy.sh <short-sha>       # e.g. ~/deploy.sh 3f9c2ab
   ```

4. The script pulls the image, replaces the container and waits up to 120 s for Docker to report it
   **healthy** (`HEALTHCHECK` on `/health`).
   - healthy → `OK: ... is healthy`, done. The image that was running before becomes "previous".
   - not healthy → it prints the container's last logs, **automatically starts the previous image again**
     and exits with an error. Fix the problem and redeploy.
5. Check `curl http://<ip>/health` and `curl http://<ip>/stats` from your PC.

To update `deploy.sh` itself first: `scp -i ... deploy/deploy.sh ubuntu@<ip>:~/`.
Without GHCR (image built on your PC): `powershell -ExecutionPolicy Bypass -File deploy\ship_image.ps1`.

## Roll back

Use this when the new version is healthy but wrong (bad predictions, errors in `/stats`, slow responses).

**To the previous image** (one step back):

```bash
~/deploy.sh --rollback
```

**To any earlier version**, by tag:

```bash
cat ~/.ecg-api/history.log         # find a tag that was OK
docker images ghcr.io/jawad1569/ecganomaly   # images still on the server
~/deploy.sh <older-short-sha>      # or a release tag, e.g. ~/deploy.sh v1.0-milestone1
```

A rollback is just a deploy of an older tag, so it gets the same health check and automatic fallback.
`--rollback` swaps current and previous, so running it twice returns to where you started.

**Manual fallback** (if `deploy.sh` is not available):

```bash
docker rm -f ecg-api
docker run -d --name ecg-api --restart unless-stopped -p 80:8000 ghcr.io/jawad1569/ecganomaly:<good-tag>
docker ps     # wait for (healthy)
```

## Common problems

| Symptom | Likely cause | Fix |
|---|---|---|
| `curl` times out | instance stopped, or security group lost port 80 | `aws ec2 start-instances ...`; check inbound rule 80 from 0.0.0.0/0 |
| `ssh` times out | your IP changed (SSH only allowed from one IP) | rerun `deploy\aws_create_ec2.ps1` (adds your new IP) |
| `docker pull` denied | GHCR package still private | make it public, or `docker login ghcr.io` on the server |
| container restarts / unhealthy | crash at startup (bad model file, missing dependency) | `docker logs ecg-api`; `~/deploy.sh --rollback` |
| `no space left on device` | old images | `docker image prune -a` (keeps the running one) |

Notes:
- `/stats` counters live in the container, so every redeploy or rollback starts them from zero.
- Run multi-line commands after `ssh`-ing in (or through Git Bash). Multi-line strings sent from Windows
  PowerShell arrive with CRLF line endings and the server's shell ignores them.

## Rehearsal log

| Date | Where | What | Result |
|---|---|---|---|
| 2026-11-25 | local Docker Desktop | deploy v1 → deploy v2 → deploy broken image → `--rollback` | broken image rolled back to v2 automatically (exit 1); `--rollback` → v1 healthy |
| 2026-11-25 | EC2 `i-0e013a660f56a20d8` (ap-southeast-2) | deploy `ecg-api:rehearsal-v2` → deploy a broken image (crashes on start) → `--rollback` | crash detected after 3 s, automatic rollback to `rehearsal-v2` healthy after 7 s (exit 1); `--rollback` → `ecg-api:284f588-dirty` healthy. Full trail in `~/.ecg-api/history.log` |
