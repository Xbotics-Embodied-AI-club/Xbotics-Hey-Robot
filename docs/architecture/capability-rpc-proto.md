# Capability Proto 与 Codegen 规范

## 1. 目标

这份文档定义 capability RPC contract 的生产级规范，重点回答四个问题：

- `proto source` 放在哪里
- `generated contract` 放在哪里
- `codegen flow` 怎么跑
- 后续扩展新的 RPC contract 时，目录和命名怎么保持一致

Capability proto 面向 capability service 的执行请求，而不是直接暴露 robot driver primitive。当前 VLA 路径统一使用 `vla_manipulation` 作为 skill/capability 名称；普通 XLeRobot 原子能力（例如 `move_base`、`turn_base`、`set_gripper`）走 RobotRuntime，不需要 capability service。当前 XLeRobot real/sim 配置默认不把 `vla_manipulation` 加入 `skills.enabled`，因此 Agent 默认不能直接调用它。

当前 source of truth：

- [proto/hey_robot/capability/v1/capability.proto](../../proto/hey_robot/capability/v1/capability.proto)

当前 generated Python contract：

- [src/hey_robot/capability/contract/v1/__init__.py](../../src/hey_robot/capability/contract/v1/__init__.py)
- [src/hey_robot/capability/contract/v1/capability_pb2.py](../../src/hey_robot/capability/contract/v1/capability_pb2.py)
- [src/hey_robot/capability/contract/v1/capability_pb2.pyi](../../src/hey_robot/capability/contract/v1/capability_pb2.pyi)
- [src/hey_robot/capability/contract/v1/capability_pb2_grpc.py](../../src/hey_robot/capability/contract/v1/capability_pb2_grpc.py)

## 2. 为什么 `proto/` 目录是这样组织的

当前目录：

```text
proto/
  hey_robot/
    capability/
      v1/
        capability.proto
```

这个结构表达的是 `protocol namespace`，不是 Python runtime 的实现分层。

三层含义分别是：

1. `hey_robot`
   项目级 namespace，避免和外部 protobuf package 冲突。
2. `capability`
   domain boundary，表示这是 capability 子系统的 contract。
3. `v1`
   wire-contract version，给未来的 breaking change 预留清晰演进路径。

所以 `proto/` 的组织原则是：

- 面向 `wire contract`
- 面向 `multi-language codegen`
- 面向 `versioned namespace`

不是面向 Python 包内部怎么分模块。

## 3. 为什么 `proto tree` 和 `src tree` 不完全一样

因为两棵树服务的是不同职责。

### `proto/...`

这是 `contract source tree`，关心的是：

- package naming
- RPC surface
- field compatibility
- versioning

### `src/hey_robot/capability/...`

这是 `runtime implementation tree`，关心的是：

- import boundary
- transport isolation
- runtime responsibility
- maintainability

当前 Python 结构：

```text
src/hey_robot/capability/
  catalog/
  contract/
    v1/
  runtime/
  sensors/
  transport/
    grpc/
```

结论很直接：

- `proto/` 按 `contract namespace` 组织
- `src/...` 按 `implementation responsibility` 组织

两者相关，但不应强行做成镜像目录。

## 4. Naming 规范

### 4.1 Proto package

当前 package：

```proto
package hey_robot.capability.v1;
```

规则：

- package 必须遵守 `<project>.<domain>.<version>`
- 不要随意修改已有 package name
- 修改 package name 属于 `contract-level breaking change`

### 4.2 Proto file name

当前 file：

```text
capability.proto
```

规则：

- 一个稳定的 capability domain，在早期可以只有一个主 proto 文件
- message 明显变复杂以后，再考虑拆出 `common.proto`
- 不做过早拆分

### 4.3 Service name

当前 service：

```proto
service CapabilityService
```

规则：

- service name 要稳定，面向 domain，而不是面向某次实现
- 不要因为 server/client 内部实现变化去 rename service
- service identity 面向 capability domain；具体执行目标用 implementation capability name 表达，不用 Agent-visible semantic skill name 表达

## 5. Versioning 规范

### 5.1 什么情况继续留在 `v1`

只要 `wire compatibility` 还成立，就继续留在 `v1`。

比如：

- 新增 optional field
- 扩展 `Struct` 中的 metadata 或 metrics
- 增加不破坏兼容性的 health 信息

### 5.2 什么情况要开 `v2`

以下情况应新建 `v2`：

- 删除 existing field
- 改变 field 语义且不兼容
- 改变 RPC 的 contract semantics
- 改变 protobuf package namespace

目标结构会变成：

```text
proto/
  hey_robot/
    capability/
      v1/
        capability.proto
      v2/
        capability.proto
```

## 6. Generated contract 规范

`proto source` 固定放在：

```text
proto/hey_robot/capability/v1/capability.proto
```

`generated Python artifacts` 固定落在：

```text
src/hey_robot/capability/contract/v1/
  capability_pb2.py
  capability_pb2.pyi
  capability_pb2_grpc.py
```

其中：

- `capability_pb2.py` 是 protobuf runtime code
- `capability_pb2.pyi` 是 typing stub
- `capability_pb2_grpc.py` 是 gRPC stub / servicer glue

这里的关键点是：

- `.py` 负责 runtime
- `.pyi` 负责 `mypy` 和 IDE typing

没有 `.pyi`，generated protobuf module 很容易在严格 `mypy` 下丢失 attribute typing。

## 7. Codegen flow

统一入口：

- `uv run --group dev poe proto`

实际执行脚本：

- [scripts/dev/generate_capability_proto.ps1](../../scripts/dev/generate_capability_proto.ps1)

脚本职责是固定的：

1. 从 `proto/` 读取 source proto
2. 调用 `grpc_tools.protoc`
3. 同时生成：
   - `--python_out`
   - `--pyi_out`
   - `--grpc_python_out`
4. 将 generated file 移动到 `src/hey_robot/capability/contract/v1/`
5. 修正 `capability_pb2_grpc.py` 中的 import path，使其指向 `contract.v1`
6. 删除临时 legacy generated path

## 8. Import 规范

业务代码允许的 import pattern：

```python
from hey_robot.capability.contract.v1 import capability_pb2, capability_pb2_grpc
```

或者：

```python
from hey_robot.capability.contract.v1 import ExecuteCapabilityRequest
```

不允许：

- 直接从临时 generated path 引用
- 在 runtime code 中手工拼接 proto import path
- 重新引入 `src/hey_robot/capability/v1/` 这种 legacy generated 目录

## 9. Generated boundary 的工程约束

`generated files` 和 `hand-written source` 应该被视为两种不同资产。

### 对 generated files 的要求

- 必须可重复生成
- 不手工维护业务逻辑
- 不作为人工风格优化对象

### 对 tooling 的要求

- `mypy` 需要消费 `.pyi`，从而让上层手写代码获得正确 typing
- `ruff`/`mypy` 不应该把 generated files 当成 hand-written business module 去要求完全一致的风格

这不是“放过错误”，而是把 `tooling boundary` 和 `ownership boundary` 对齐。

## 10. Architecture guard

当前仓库已经通过 architecture test 固化以下约束：

- proto source 必须存在
- `contract/v1` 必须存在
- `capability_pb2.py`、`capability_pb2.pyi`、`capability_pb2_grpc.py` 必须存在
- `src/hey_robot/capability/v1/` 不允许重新出现
- generated gRPC import 必须指向 `hey_robot.capability.contract.v1`

对应测试：

- [tests/architecture/test_capability_proto_contract.py](../../tests/architecture/test_capability_proto_contract.py)

## 11. 后续扩展建议

如果未来再增加新的 RPC contract，建议保持同一套模式：

### 新增 capability contract version

```text
proto/hey_robot/capability/v2/
src/hey_robot/capability/contract/v2/
```

### 新增新的 RPC domain

如果未来 capability contract 明显分化，可以扩展成：

```text
proto/
  hey_robot/
    capability/
      manipulation/v1/
      navigation/v1/
```

但前提是 wire contract 真正已经分域，不做过度设计。

## 12. 当前 boundary 结论

当前 capability proto 只保留三个 RPC：

- `GetHealth`
- `ExecuteCapability`
- `CancelCapability`

这说明本次重构控制得比较克制：

- 只把 `capability boundary` 提升为 RPC contract
- 主 agent-skill event chain 仍然保持 event-driven

也就是说，当前系统不是“全系统 RPC 化”，而是把最有必要、最稳定、最适合远程边界的那一层做成 RPC。
