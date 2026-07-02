param(
    [string]$Python = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$ProtoFile = Join-Path $RepoRoot "proto\hey_robot\model_service\v1\model_service.proto"
$ProtoRoot = Join-Path $RepoRoot "proto"
$SrcRoot = Join-Path $RepoRoot "src"
$GeneratedRoot = Join-Path $SrcRoot "hey_robot\model_service"
$GeneratedV1Root = Join-Path $GeneratedRoot "v1"
$ContractRoot = Join-Path $SrcRoot "hey_robot\foundation\contract\v1"

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

if (-not (Test-Path $GeneratedV1Root)) {
    throw "Expected generated directory missing: $GeneratedV1Root"
}

New-Item -ItemType Directory -Force -Path $ContractRoot | Out-Null

Move-Item -Force `
    -LiteralPath (Join-Path $GeneratedV1Root "model_service_pb2.py") `
    -Destination (Join-Path $ContractRoot "model_service_pb2.py")

Move-Item -Force `
    -LiteralPath (Join-Path $GeneratedV1Root "model_service_pb2.pyi") `
    -Destination (Join-Path $ContractRoot "model_service_pb2.pyi")

Move-Item -Force `
    -LiteralPath (Join-Path $GeneratedV1Root "model_service_pb2_grpc.py") `
    -Destination (Join-Path $ContractRoot "model_service_pb2_grpc.py")

$GrpcFile = Join-Path $ContractRoot "model_service_pb2_grpc.py"
$GrpcContent = Get-Content $GrpcFile -Raw
$GrpcContent = $GrpcContent.Replace(
    "from hey_robot.model_service.v1 import model_service_pb2 as hey__robot_dot_model__service_dot_v1_dot_model__service__pb2",
    "from hey_robot.foundation.contract.v1 import model_service_pb2 as hey__robot_dot_model__service_dot_v1_dot_model__service__pb2"
)
$GrpcContent = $GrpcContent.Replace(
    "hey_robot/model_service/v1/model_service_pb2_grpc.py",
    "hey_robot/foundation/contract/v1/model_service_pb2_grpc.py"
)
Set-Content -Path $GrpcFile -Value $GrpcContent -Encoding utf8

foreach ($GeneratedFile in @(
    (Join-Path $ContractRoot "model_service_pb2.py"),
    (Join-Path $ContractRoot "model_service_pb2_grpc.py")
)) {
    $Content = Get-Content $GeneratedFile -Raw
    if (-not $Content.StartsWith("# ruff: noqa")) {
        Set-Content -Path $GeneratedFile -Value ("# ruff: noqa`n" + $Content) -Encoding utf8
    }
}

if (Test-Path $GeneratedRoot) {
    Remove-Item -Recurse -Force $GeneratedRoot
}

Write-Host "Generated model service proto contract into $ContractRoot"
