# Deploying the ECG API to AWS EC2

One Ubuntu `t3.micro` runs the Docker container on port 80. Everything is scripted in [`deploy/`](deploy):

| Script | Runs on | Does |
|---|---|---|
| `deploy/aws_create_ec2.ps1` | your PC | creates key pair, security group, instance, Elastic IP |
| `deploy/ec2_setup.sh` | server (automatic on first boot) | installs Docker, 2 GB swap, log rotation |
| `deploy/deploy.sh` | server | deploys an image tag, health-checks it, rolls back if unhealthy |
| `deploy/ship_image.ps1` | your PC | builds the image locally, copies it to the server, deploys it (no GHCR needed) |
| `deploy/aws_destroy_ec2.ps1` | your PC | deletes everything again (after grading) |

Redeploy and rollback procedures are in [RUNBOOK.md](RUNBOOK.md).

## 0. Prerequisites (once)

- AWS CLI v2 signed in: `aws login --profile ecg-project` (browser sign-in, no access keys), and `aws sts get-caller-identity --profile ecg-project` works.
- **Use your project's Region.** Our AWS project uses the *new AWS experience*, where every project is locked to one selected Region (**`ap-southeast-2`**, Sydney; see AWS Settings → project → Additional info). Any other Region fails with *"explicit deny in a service control policy"*. Set it once: `aws configure set region ap-southeast-2 --profile ecg-project`.
- Docker Desktop running (only needed for step 3B).

All commands below run in PowerShell from the repository root.

## 1. Create the server

```powershell
powershell -ExecutionPolicy Bypass -File deploy\aws_create_ec2.ps1 -Profile ecg-project
```

It prints the instance ID and the **public IP**, saves the SSH key to `%USERPROFILE%\.ssh\ecg-key.pem` and writes `.ec2-instance.json` (git-ignored). SSH is only open to the IP you ran it from; if your IP changes, run the script again to add the new one.

## 2. Wait for Docker on the server (~2 minutes)

```powershell
$ip = (Get-Content .ec2-instance.json | ConvertFrom-Json).public_ip
ssh -i "$env:USERPROFILE\.ssh\ecg-key.pem" ubuntu@$ip "test -f /var/log/ecg-setup-done && docker --version"
```

Repeat until it prints the Docker version (answer `yes` to the host-key question the first time).

## 3. Deploy the container

Pick one.

### 3A. From GHCR (after CI has pushed the image)

CI pushes `ghcr.io/jawad1569/ecganomaly:<short-sha>` and `:latest` on every push to `main`.
GHCR packages are private at first; either make it public once (GitHub → your profile → **Packages → ecganomaly → Package settings → Change visibility → Public**), or log in on the server with a token that has `read:packages`.

```powershell
scp -i "$env:USERPROFILE\.ssh\ecg-key.pem" deploy/deploy.sh ubuntu@${ip}:~/
ssh -i "$env:USERPROFILE\.ssh\ecg-key.pem" ubuntu@$ip "chmod +x ~/deploy.sh && ~/deploy.sh latest"
```

Use a commit SHA tag (`~/deploy.sh a1b2c3d`) instead of `latest` so rollbacks have an exact version to return to.

### 3B. Straight from your PC (works before anything is pushed)

```powershell
powershell -ExecutionPolicy Bypass -File deploy\ship_image.ps1
```

Builds `ecg-api:<git sha>` locally, copies it over SSH (~600 MB, compressed in transit), loads it on the server and runs `deploy.sh`.

Either way `deploy.sh` waits until Docker reports the container **healthy**; if it does not, it restarts the previous image and exits with an error.

## 4. Check the live service

```powershell
curl.exe http://$ip/health
curl.exe -X POST http://$ip/predict -H "Content-Type: application/json" -d "@samples/request_example.json"
curl.exe http://$ip/stats
```

Interactive docs: `http://<ip>/docs`. Put the IP in the README's *Deployment* section.

On the server:

```bash
~/deploy.sh --status          # current/previous image, docker ps, /health, /stats
docker logs --tail 20 ecg-api # one log line per request
```

## 5. After grading: stop paying

```powershell
# pause (keeps disk and Elastic IP, URL unchanged; the Elastic IP is billed while the instance is stopped)
aws ec2 stop-instances --profile ecg-project --instance-ids (Get-Content .ec2-instance.json | ConvertFrom-Json).instance_id

# delete everything
powershell -ExecutionPolicy Bypass -File deploy\aws_destroy_ec2.ps1
```

Then update the README to say the service was shut down after grading.

## Cost

In `ap-southeast-2` a `t3.micro` is about USD 0.0132/h, plus USD 0.005/h for the public IPv4 address and about USD 1.50/month for the 16 GB gp3 disk: roughly USD 15/month in total, covered by the project's AWS credits.
