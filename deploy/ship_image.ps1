<#
Copies a locally built image to the EC2 server and deploys it, without GHCR.
Useful before the code is pushed to GitHub, or if GHCR is not reachable.
Run from the repository root:

    powershell -ExecutionPolicy Bypass -File deploy\ship_image.ps1                 # builds ecg-api:<git sha>
    powershell -ExecutionPolicy Bypass -File deploy\ship_image.ps1 -NoBuild -Tag latest

Reads the server address and key from .ec2-instance.json (written by aws_create_ec2.ps1),
or pass -HostIp and -Key.
#>
param(
    [string]$Tag = "",
    [string]$HostIp = "",
    [string]$Key = "",
    [string]$User = "ubuntu",
    [switch]$NoBuild
)

$ErrorActionPreference = "Continue"
if (Test-Path .ec2-instance.json) {
    $info = Get-Content .ec2-instance.json -Raw | ConvertFrom-Json
    if (-not $HostIp) { $HostIp = $info.public_ip }
    if (-not $Key) { $Key = $info.key }
}
if (-not $HostIp -or -not $Key) { throw "Pass -HostIp and -Key, or run deploy\aws_create_ec2.ps1 first" }
if (-not $Tag) {
    $Tag = (git rev-parse --short=7 HEAD).Trim()
    if (git status --porcelain) { $Tag = "$Tag-dirty" }   # uncommitted changes in this build
}
$image = "ecg-api:$Tag"
$ssh = @("-i", $Key, "-o", "StrictHostKeyChecking=accept-new")

if (-not $NoBuild) {
    Write-Host "==> Building $image"
    docker build -t $image .
    if ($LASTEXITCODE -ne 0) { throw "docker build failed" }
}

$tar = Join-Path $env:TEMP "ecg-api-$Tag.tar"
Write-Host "==> Saving $image to $tar"
docker save -o $tar $image
if ($LASTEXITCODE -ne 0) { throw "docker save failed" }

Write-Host "==> Copying image and deploy.sh to $User@$HostIp (compressed)"
scp @ssh -C $tar "deploy/deploy.sh" "${User}@${HostIp}:~/"
if ($LASTEXITCODE -ne 0) { throw "scp failed" }
Remove-Item $tar

Write-Host "==> Loading and deploying on the server"
$name = Split-Path $tar -Leaf
ssh @ssh "${User}@${HostIp}" "sed -i 's/\r$//' ~/deploy.sh && chmod +x ~/deploy.sh && docker load -i ~/$name && rm ~/$name && ~/deploy.sh $image"
if ($LASTEXITCODE -ne 0) { throw "deploy on the server failed (see output above)" }

Write-Host ""
Write-Host "Live: http://$HostIp/docs   health: http://$HostIp/health   stats: http://$HostIp/stats"
