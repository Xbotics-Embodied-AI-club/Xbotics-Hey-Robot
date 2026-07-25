# ModelService Proto 与 Codegen 规范

本文描述当前 ModelService gRPC contract。旧的 `CapabilityService`、
`ExecuteCapability` 和 `CancelCapability` 已从代码中移除；不要再把
capability-rpc 作为系统的一等架构概念。

## 1. 边界目标

ModelService RPC 用于把 VLA、VLN 等 Foundation Model 从主系统进程中独立部署。
它不直接暴露 RobotDriver primitive，也不替代主Harness的本地Skill执行链。

在快慢双系统视角中，ModelService 属于下层快系统的学习型决策组件。这里的“快”指
短时域具身决策层级，不保证每次模型推理具有更低的绝对延迟。

当前主Skill链使用进程内端口：

```text
Agent -> SkillClient -> SkillWorker -> Skill handler -> LocalRobotClient
```

模型执行边界使用gRPC：

```text
Skill handler / option runner -> ModelRouter -> ModelService -> inference result
```

SkillWorker/SkillRunner负责资源锁、超时和结果归一化；VLA/VLN option负责有界控制循环；
Robot Runtime仍是唯一硬件执行边界。NATS承载会话、状态和事件投影，不承载生产
ModelService调用。

## 2. Source of truth

Proto source：

- [proto/hey_robot/model_service/v1/model_service.proto](../../proto/hey_robot/model_service/v1/model_service.proto)

Generated Python contract：

- [src/hey_robot/foundation/contract/v1/model_service_pb2.py](../../src/hey_robot/foundation/contract/v1/model_service_pb2.py)
- [src/hey_robot/foundation/contract/v1/model_service_pb2.pyi](../../src/hey_robot/foundation/contract/v1/model_service_pb2.pyi)
- [src/hey_robot/foundation/contract/v1/model_service_pb2_grpc.py](../../src/hey_robot/foundation/contract/v1/model_service_pb2_grpc.py)

生成入口：

- [scripts/dev/generate_model_service_proto.ps1](../../scripts/dev/generate_model_service_proto.ps1)
- `uv run --group dev poe proto`

## 3. Namespace 与目录

Proto tree：

```text
proto/
  hey_robot/
    model_service/
      v1/
        model_service.proto
```

Python implementation tree：

```text
src/hey_robot/foundation/
  backends/
  catalog/
  clients/
  contract/
    v1/
  transport/
    grpc/
```

两棵树不需要镜像：

- `proto/` 按 wire contract namespace 和版本组织；
- `foundation/` 按 Python 模块职责组织。

当前 protobuf package：

```proto
package hey_robot.model_service.v1;
```

规则：

- package 使用 `<project>.<domain>.<version>`；
- 修改已有 package name 属于 breaking change；
- wire-compatible 的新增字段继续放在 `v1`；
- 删除字段、改变字段语义或 RPC 语义时新建 `v2`。

## 4. 当前 RPC

```proto
service ModelService {
  rpc GetHealth(GetHealthRequest) returns (GetHealthResponse);
  rpc ExecuteSkill(ExecuteSkillRequest) returns (ExecuteSkillResponse);
  rpc CancelSkill(CancelSkillRequest) returns (CancelSkillResponse);
}
```

### GetHealth

返回：

- service 和 robot identity；
- online / loaded / busy；
- current skill；
- error code/message；
- metrics 和 contract version。

ModelService client提供health接口；具体Skill执行通过ModelRouter调用服务。部署和运维应
在放开模型驱动Skill前检查online、loaded和busy状态，不能假设本地SkillWorker已经替代
所有启动health gate。

### ExecuteSkill

请求包含：

- `service_id`
- `trace_id`
- `episode_id`
- `skill_id`
- `skill_name`
- `robot_id`
- `objective`
- `arguments`
- `timeout_sec`
- `metadata`

响应包含：

- success / status / summary；
- failure mode；
- error code/message；
- metrics。

`arguments` 和 `metrics` 使用 `google.protobuf.Struct`，使不同 Foundation backend
能够携带模型特有数据。稳定的业务语义应优先提升为明确字段，不应无限堆入 Struct。

### CancelSkill

传入 `service_id` 和 `skill_id`，返回 cancel 是否被接受。取消是协作式语义：
executor 必须自行检查或响应取消信号，gRPC 返回 accepted 不代表模型线程已经立即退出。

## 5. 路由语义

Deployment 通过 `provides` 声明服务可处理的 Skill 名：

```yaml
model_services:
  vln_nav:
    type: vln_planner
    robot_id: sim_robot
    target: grpc://127.0.0.1:9091
    provides:
      - navigate_to
      - approach_object
```

`ModelServiceRegistry.service_for(skill_name, robot_id)` 只有在以下条件全部满足时才返回服务：

1. service enabled；
2. `robot_id` 匹配；
3. `skill_name` 出现在 `provides`；
4. 对应 client 已创建。

因此 `SkillSpec.required_model_service`、执行时传给 `ctx.model_services.call()` 的名称和
deployment `provides` 必须一致。部署校验当前能验证 required service 是否存在，但不能
证明 Skill 实现运行时使用了同一个名称，新增 Skill 时必须添加端到端路由测试。

## 6. Client 与 Server

Client：

```python
from hey_robot.foundation.transport.grpc.client import GrpcModelServiceClient
```

Server：

```python
from hey_robot.foundation.transport.grpc.server import ModelServiceServicer
```

配置中的 target 可写为：

```text
grpc://127.0.0.1:9091
```

项目 client 会去掉 `grpc://` 后再传给 gRPC。直接使用 grpcurl 等工具时应使用
`127.0.0.1:9091`。

当前 client/server 使用 insecure channel。生产网络需要在 transport 层增加 TLS、
认证、服务身份和访问策略，不能把开发配置直接暴露到不可信网络。

## 7. Codegen

统一命令：

```bash
uv run --group dev poe proto
```

脚本负责：

1. 从 `proto/` 读取 source；
2. 调用 `grpc_tools.protoc`；
3. 生成 `--python_out`、`--pyi_out` 和 `--grpc_python_out`；
4. 将文件放入 `src/hey_robot/foundation/contract/v1/`；
5. 修正 generated gRPC import 为项目 package path。

Generated 文件不承载手写业务逻辑。`.pyi` 用于严格 mypy 和 IDE typing，必须和
`.py`、`_grpc.py` 一起提交。

允许的 import：

```python
from hey_robot.foundation.contract.v1 import model_service_pb2
from hey_robot.foundation.contract.v1 import model_service_pb2_grpc
```

不要：

- 从临时 codegen 目录 import；
- 手改 generated 文件加入业务逻辑；
- 重新创建旧的 `src/hey_robot/capability/` contract tree；
- 在 Skill 或 Agent 中直接依赖 protobuf message。

上层应通过 `foundation.clients` 的 request/result model 使用 ModelService。

## 8. Architecture guard

对应测试：

- [tests/architecture/test_model_service_proto_contract.py](../../tests/architecture/test_model_service_proto_contract.py)
- [tests/integration/test_model_service_grpc_flow.py](../../tests/integration/test_model_service_grpc_flow.py)

测试固定以下约束：

- proto source 和 generated artifact 存在；
- package、service 和 RPC 使用当前 ModelService 命名；
- generated import path 正确；
- 旧 `ExecuteCapability` 不重新出现；
- gRPC client/server 能完成 deployment-style ExecuteSkill flow。

## 9. 当前限制

- 一个 servicer 同时只执行一个请求，busy 时直接拒绝；
- executor 通过 `asyncio.to_thread` 调用同步模型代码；
- cancel 是否及时生效取决于 backend；
- 没有服务端队列、优先级或多租户隔离；
- 没有默认 TLS/auth；
- metrics 使用 Struct，schema 约束有限；
- gRPC 合同测试不代表真实模型 checkpoint、GPU 和第三方仓库已经可用。

这些限制不影响 contract 分层，但必须纳入生产部署设计。
