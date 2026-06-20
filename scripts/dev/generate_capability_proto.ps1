param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$ProtoFile = Join-Path $RepoRoot "proto\hey_robot\capability\v1\capability.proto"
$ProtoRoot = Join-Path $RepoRoot "proto"
$SrcRoot = Join-Path $RepoRoot "src"
$GeneratedRoot = Join-Path $SrcRoot "hey_robot\capability"
$ContractRoot = Join-Path $GeneratedRoot "contract\v1"
$LegacyRoot = Join-Path $GeneratedRoot "v1"

if (-not (Test-Path $Python)) {
    throw "Python executable not found: $Python"
}

if (-not (Test-Path $ProtoFile)) {
    throw "Proto file not found: $ProtoFile"
}

& $Python -m grpc_tools.protoc `
    -I $ProtoRoot `
    --python_out=$SrcRoot `
    --pyi_out=$SrcRoot `
    --grpc_python_out=$SrcRoot `
    $ProtoFile

if (-not (Test-Path $LegacyRoot)) {
    throw "Expected generated directory missing: $LegacyRoot"
}

New-Item -ItemType Directory -Force -Path $ContractRoot | Out-Null

Move-Item -Force `
    -LiteralPath (Join-Path $LegacyRoot "capability_pb2.py") `
    -Destination (Join-Path $ContractRoot "capability_pb2.py")

Move-Item -Force `
    -LiteralPath (Join-Path $LegacyRoot "capability_pb2.pyi") `
    -Destination (Join-Path $ContractRoot "capability_pb2.pyi")

Move-Item -Force `
    -LiteralPath (Join-Path $LegacyRoot "capability_pb2_grpc.py") `
    -Destination (Join-Path $ContractRoot "capability_pb2_grpc.py")

$GrpcFile = Join-Path $ContractRoot "capability_pb2_grpc.py"
$GrpcContent = Get-Content $GrpcFile -Raw
$GrpcContent = $GrpcContent.Replace(
    "from hey_robot.capability.v1 import capability_pb2 as hey__robot_dot_capability_dot_v1_dot_capability__pb2",
    "from hey_robot.capability.contract.v1 import capability_pb2 as hey__robot_dot_capability_dot_v1_dot_capability__pb2"
)
$GrpcContent = $GrpcContent.Replace(
    "hey_robot/capability/v1/capability_pb2_grpc.py",
    "hey_robot/capability/contract/v1/capability_pb2_grpc.py"
)
Set-Content -Path $GrpcFile -Value $GrpcContent -Encoding utf8

if (Test-Path $LegacyRoot) {
    Remove-Item -Recurse -Force $LegacyRoot
}

Write-Host "Generated capability proto contract into $ContractRoot"
