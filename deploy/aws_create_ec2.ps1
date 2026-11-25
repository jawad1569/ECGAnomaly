<#
Creates everything the ECG API needs on AWS EC2, using the AWS CLI profile you set up
with `aws configure`. Run from the repository root:

    powershell -ExecutionPolicy Bypass -File deploy\aws_create_ec2.ps1

What it creates (all tagged Project=ecg-api, removed again by deploy\aws_destroy_ec2.ps1):
  - key pair "ecg-key"            -> private key saved to %USERPROFILE%\.ssh\ecg-key.pem
  - security group "ecg-api-sg"   -> SSH (22) from your current IP only, HTTP (80) from anywhere
  - one Ubuntu 24.04 t3.micro     -> 16 GB disk, IMDSv2, runs deploy\ec2_setup.sh on first boot
  - an Elastic IP                 -> fixed public address for the README (use -NoElasticIp to skip)

The instance details are written to .ec2-instance.json (git-ignored).
Cost: t3.micro is free-tier eligible (750 h/month for 12 months); outside the free tier
about USD 0.0104/h plus public IPv4 USD 0.005/h.
#>
param(
    [string]$Profile = $env:AWS_PROFILE,   # e.g. -Profile ecg-project (aws login); empty = default profile
    [string]$Region = "",
    [string]$InstanceType = "t3.micro",
    [string]$KeyName = "ecg-key",
    [string]$GroupName = "ecg-api-sg",
    [string]$Name = "ecg-api",
    [int]$DiskGb = 16,
    [switch]$NoElasticIp
)

# "Continue": in Windows PowerShell 5.1, "Stop" turns any stderr line from aws.exe into a
# fatal error, even for expected "not found" answers. Failures are checked via $LASTEXITCODE.
$ErrorActionPreference = "Continue"
$aws = (Get-Command aws -ErrorAction SilentlyContinue).Source
if (-not $aws) { $aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe" }
if (-not (Test-Path $aws)) { throw "AWS CLI not found. Install it: winget install --id Amazon.AWSCLI -e" }

function Invoke-Aws {
    $out = & $aws @args --output json
    if ($LASTEXITCODE -ne 0) { throw "aws $($args -join ' ') failed" }
    if ($out) { return ($out | Out-String | ConvertFrom-Json) }
}

if ($Profile) { $env:AWS_PROFILE = $Profile }
if (-not $Region) { $Region = (& $aws configure get region) }
if (-not $Region) { throw "No region set. Run: aws configure" }
# New-AWS-experience projects can only create resources in their one selected Region
# (AWS Settings > project > Additional info > Region); other Regions fail with an SCP deny.
$env:AWS_DEFAULT_REGION = $Region
$tags = "{Key=Project,Value=ecg-api},{Key=Name,Value=$Name}"
Write-Host "Region: $Region  Profile: $(if ($env:AWS_PROFILE) { $env:AWS_PROFILE } else { 'default' })"

# 1. key pair
$pem = Join-Path $env:USERPROFILE ".ssh\$KeyName.pem"
$existingKey = & $aws ec2 describe-key-pairs --key-names $KeyName --query "KeyPairs[0].KeyName" --output text 2>$null
if ($LASTEXITCODE -ne 0 -or -not $existingKey -or $existingKey -eq "None") {
    Write-Host "Creating key pair $KeyName -> $pem"
    New-Item -ItemType Directory -Force (Split-Path $pem) | Out-Null
    $material = & $aws ec2 create-key-pair --key-name $KeyName --key-type ed25519 `
        --tag-specifications "ResourceType=key-pair,Tags=[$tags]" --query KeyMaterial --output text
    if ($LASTEXITCODE -ne 0) { throw "create-key-pair failed" }
    [IO.File]::WriteAllText($pem, (($material -join "`n") + "`n"))
    # OpenSSH refuses keys other users can read
    icacls $pem /inheritance:r /grant:r "$($env:USERNAME):R" | Out-Null
} else {
    Write-Host "Key pair $KeyName already exists"
    if (-not (Test-Path $pem)) { Write-Warning "$pem is missing; delete the key pair in AWS or pass -KeyName with a new name" }
}

# 2. security group in the default VPC
$vpc = (& $aws ec2 describe-vpcs --filters Name=is-default,Values=true --query "Vpcs[0].VpcId" --output text)
if (-not $vpc -or $vpc -eq "None") { throw "No default VPC in $Region" }
$myIp = (Invoke-RestMethod -Uri "https://checkip.amazonaws.com").Trim()
$sg = (& $aws ec2 describe-security-groups --filters Name=group-name,Values=$GroupName Name=vpc-id,Values=$vpc --query "SecurityGroups[0].GroupId" --output text)
if (-not $sg -or $sg -eq "None") {
    Write-Host "Creating security group $GroupName (SSH from $myIp, HTTP from anywhere)"
    $sg = (Invoke-Aws ec2 create-security-group --group-name $GroupName --vpc-id $vpc `
        --description "ECG API: HTTP from anywhere, SSH from admin IP" `
        --tag-specifications "ResourceType=security-group,Tags=[$tags]").GroupId
    & $aws ec2 authorize-security-group-ingress --group-id $sg --protocol tcp --port 80 --cidr 0.0.0.0/0 | Out-Null
    & $aws ec2 authorize-security-group-ingress --group-id $sg --protocol tcp --port 22 --cidr "$myIp/32" | Out-Null
} else {
    Write-Host "Security group $GroupName exists ($sg); allowing SSH from $myIp"
    & $aws ec2 authorize-security-group-ingress --group-id $sg --protocol tcp --port 22 --cidr "$myIp/32" 2>$null | Out-Null
}

# 3. latest Ubuntu 24.04 LTS AMI from Canonical
$ami = (& $aws ec2 describe-images --owners 099720109477 `
    --filters "Name=name,Values=ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*" "Name=state,Values=available" `
    --query "sort_by(Images, &CreationDate)[-1].ImageId" --output text)
if (-not $ami -or $ami -eq "None") { throw "Could not find an Ubuntu 24.04 AMI" }
Write-Host "AMI: $ami"

# 4. instance (ec2_setup.sh runs as user-data on first boot)
$existing = (& $aws ec2 describe-instances --filters "Name=tag:Project,Values=ecg-api" "Name=instance-state-name,Values=pending,running,stopped" `
    --query "Reservations[0].Instances[0].InstanceId" --output text)
if ($existing -and $existing -ne "None") {
    Write-Host "Instance already exists: $existing"
    $instanceId = $existing
} else {
    $userData = (Resolve-Path "deploy\ec2_setup.sh").Path
    Write-Host "Launching $InstanceType ..."
    $instanceId = (Invoke-Aws ec2 run-instances --image-id $ami --instance-type $InstanceType `
        --key-name $KeyName --security-group-ids $sg --count 1 `
        --user-data "file://$userData" `
        --metadata-options "HttpTokens=required,HttpEndpoint=enabled" `
        --block-device-mappings "DeviceName=/dev/sda1,Ebs={VolumeSize=$DiskGb,VolumeType=gp3,DeleteOnTermination=true}" `
        --tag-specifications "ResourceType=instance,Tags=[$tags]" "ResourceType=volume,Tags=[$tags]").Instances[0].InstanceId
}
if (-not $instanceId) { throw "Instance was not created" }
Write-Host "Waiting for $instanceId to be running ..."
& $aws ec2 wait instance-running --instance-ids $instanceId

# 5. Elastic IP (fixed address across stop/start)
$ip = $null
if (-not $NoElasticIp) {
    $alloc = (& $aws ec2 describe-addresses --filters "Name=tag:Project,Values=ecg-api" --query "Addresses[0].AllocationId" --output text)
    if (-not $alloc -or $alloc -eq "None") {
        $alloc = (Invoke-Aws ec2 allocate-address --domain vpc --tag-specifications "ResourceType=elastic-ip,Tags=[$tags]").AllocationId
    }
    & $aws ec2 associate-address --instance-id $instanceId --allocation-id $alloc | Out-Null
    $ip = (& $aws ec2 describe-addresses --allocation-ids $alloc --query "Addresses[0].PublicIp" --output text)
} else {
    $ip = (& $aws ec2 describe-instances --instance-ids $instanceId --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
}

$info = [ordered]@{ region = $Region; profile = $env:AWS_PROFILE; instance_id = $instanceId; public_ip = $ip; key = $pem; security_group = $sg; user = "ubuntu" }
$info | ConvertTo-Json | Set-Content -Encoding utf8 .ec2-instance.json

Write-Host ""
Write-Host "Instance ready: $instanceId  public IP: $ip"
Write-Host "Docker is being installed by ec2_setup.sh (about 2 minutes). Then:"
Write-Host "  ssh -i `"$pem`" ubuntu@$ip `"test -f /var/log/ecg-setup-done && docker --version`""
Write-Host "Next step: see DEPLOYMENT.md, section 3."
