<#
Removes everything deploy\aws_create_ec2.ps1 created (resources tagged Project=ecg-api),
so nothing keeps costing money after grading. Run from the repository root:

    powershell -ExecutionPolicy Bypass -File deploy\aws_destroy_ec2.ps1          # asks first
    powershell -ExecutionPolicy Bypass -File deploy\aws_destroy_ec2.ps1 -Yes     # no prompt
    powershell -ExecutionPolicy Bypass -File deploy\aws_destroy_ec2.ps1 -KeepKey # keep the key pair

To only pause (keeps the disk and the Elastic IP, so the URL stays the same):
    aws ec2 stop-instances --instance-ids <id>      # start again with start-instances
#>
param(
    [string]$Profile = $env:AWS_PROFILE,
    [string]$Region = "",
    [string]$KeyName = "ecg-key",
    [string]$GroupName = "ecg-api-sg",
    [switch]$KeepKey,
    [switch]$Yes
)

$ErrorActionPreference = "Continue"   # see aws_create_ec2.ps1
$aws = (Get-Command aws -ErrorAction SilentlyContinue).Source
if (-not $aws) { $aws = "C:\Program Files\Amazon\AWSCLIV2\aws.exe" }
if (-not $Profile -and (Test-Path .ec2-instance.json)) { $Profile = (Get-Content .ec2-instance.json -Raw | ConvertFrom-Json).profile }
if ($Profile) { $env:AWS_PROFILE = $Profile }
if (-not $Region -and (Test-Path .ec2-instance.json)) { $Region = (Get-Content .ec2-instance.json -Raw | ConvertFrom-Json).region }
if (-not $Region) { $Region = (& $aws configure get region) }
$env:AWS_DEFAULT_REGION = $Region

$instances = @(& $aws ec2 describe-instances --filters "Name=tag:Project,Values=ecg-api" "Name=instance-state-name,Values=pending,running,stopping,stopped" `
    --query "Reservations[].Instances[].InstanceId" --output text) -split "\s+" | Where-Object { $_ }
$allocs = @(& $aws ec2 describe-addresses --filters "Name=tag:Project,Values=ecg-api" --query "Addresses[].AllocationId" --output text) -split "\s+" | Where-Object { $_ }
$sg = (& $aws ec2 describe-security-groups --filters Name=group-name,Values=$GroupName --query "SecurityGroups[0].GroupId" --output text)

Write-Host "Region $Region will lose:"
Write-Host "  instances:    $($instances -join ', ')  (and their disks)"
Write-Host "  elastic IPs:  $($allocs -join ', ')"
Write-Host "  security grp: $sg"
if (-not $KeepKey) { Write-Host "  key pair:     $KeyName" }
if (-not $Yes) {
    $answer = Read-Host "Type 'delete' to continue"
    if ($answer -ne "delete") { Write-Host "Cancelled."; exit 1 }
}

foreach ($a in $allocs) {
    $assoc = (& $aws ec2 describe-addresses --allocation-ids $a --query "Addresses[0].AssociationId" --output text)
    if ($assoc -and $assoc -ne "None") { & $aws ec2 disassociate-address --association-id $assoc | Out-Null }
    & $aws ec2 release-address --allocation-id $a | Out-Null
    Write-Host "Released Elastic IP $a"
}
if ($instances) {
    & $aws ec2 terminate-instances --instance-ids @instances | Out-Null
    Write-Host "Terminating $($instances -join ', ') ..."
    & $aws ec2 wait instance-terminated --instance-ids @instances
}
if ($sg -and $sg -ne "None") { & $aws ec2 delete-security-group --group-id $sg | Out-Null; Write-Host "Deleted security group $sg" }
if (-not $KeepKey) { & $aws ec2 delete-key-pair --key-name $KeyName | Out-Null; Write-Host "Deleted key pair $KeyName (local .pem left in place)" }
if (Test-Path .ec2-instance.json) { Remove-Item .ec2-instance.json }
Write-Host "Done."
