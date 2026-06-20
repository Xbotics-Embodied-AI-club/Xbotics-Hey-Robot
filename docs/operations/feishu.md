# Feishu 接入说明

本文面向使用 `hey-robot` 项目的用户，说明如何把机器人接到飞书。

这套项目里的飞书通道已经内置，不需要另外开发。你主要要做三件事：

1. 在飞书开放平台创建应用
2. 配好本机环境变量
3. 在 deployment 配置里启用 `feishu` channel

飞书开放平台入口：

- https://open.feishu.cn/?lang=zh-CN

按你当前看到的飞书页面，相关配置主要在这两个位置：

- `基础信息`
- `事件与回调`

## 适用范围

当前仓库里的飞书通道：

- 代码位置：`src/hey_robot/channels/feishu/`
- 部署配置：`configs/xlerobot.real.windows.yaml`
- 接入方式：长连接事件通道

对使用者来说，最重要的一点是：

- 不需要自己写飞书机器人逻辑
- 不需要自己再接一层 webhook 服务
- 正常启动 `hey-robot run` 后，项目会自己连接飞书

## 先准备什么

你需要准备下面几项：

- 一个飞书开放平台应用
- 应用的 `App ID`
- 应用的 `App Secret`
- `Encrypt Key`
- `Verification Token`

项目默认从环境变量读取这些值：

- `FEISHU_APP_ID`
- `FEISHU_APP_SECRET`
- `FEISHU_ENCRYPT_KEY`
- `FEISHU_VERIFICATION_TOKEN`

## 飞书平台怎么配

建议按这个顺序做：

### 1. 创建应用

在飞书开放平台创建一个机器人应用。

### 2. 开启机器人相关能力

至少要保证应用能：

- 接收用户消息
- 给用户发消息
- 回复消息

如果你希望机器人能收发图片、文件，也要把对应能力打开。

### 3. 配置事件与回调

项目当前使用的是飞书消息事件接入。你需要在飞书开放平台的：

```text
事件与回调
```

页面里完成相关配置。

常见做法：

- 开启应用的消息接收能力
- 订阅机器人接收消息事件

你现在在页面里能看到两项关键值：

- `Encrypt Key`
- `Verification Token`

它们分别对应：

```dotenv
FEISHU_ENCRYPT_KEY=
FEISHU_VERIFICATION_TOKEN=
```

注意：

- 现在平台里显示的是“事件与回调”
- 不是旧说法里的“事件订阅”
- 文档里如果看到“事件订阅”，可以直接理解成你现在页面里的“事件与回调”配置区

## 本机怎么配

这个项目会自动加载根目录的 `.env` 文件。

所以最简单的做法是直接编辑项目根目录的：

```text
.env
```

加入下面 4 个变量：

```dotenv
FEISHU_APP_ID=你的 App ID
FEISHU_APP_SECRET=你的 App Secret
FEISHU_ENCRYPT_KEY=你的 Encrypt Key
FEISHU_VERIFICATION_TOKEN=你的 Verification Token
```

如果你的项目里还没有 `.env`，可以先复制：

```text
.env.example
```

再把里面的飞书配置补全。

## 项目配置怎么改

打开：

```text
configs/xlerobot.real.windows.yaml
```

找到 `channels.feishu`，把 `enabled` 改成 `true`：

```yaml
channels:
  feishu:
    type: feishu
    enabled: true
    account_id: xlerobot-feishu
    app_id_env: FEISHU_APP_ID
    app_secret_env: FEISHU_APP_SECRET
    encrypt_key_env: FEISHU_ENCRYPT_KEY
    verification_token_env: FEISHU_VERIFICATION_TOKEN
    group_policy: mention
    reply_to_message: true
    domain: feishu
    allow_from:
      - "*"
    media_root: runtime/xlerobot.real.windows/media/feishu
```

下面几个字段最常用：

- `enabled`: 是否启用飞书通道
- `group_policy`: 群聊策略
  - `mention`: 只有在群里 @ 机器人时才处理
  - `open`: 群里所有消息都处理
- `reply_to_message`: 是否直接回复原消息
- `allow_from`: 允许哪些用户发消息给机器人
  - `["*"]` 表示不限制
- `domain`: 国内飞书一般用 `feishu`
  - 如果你接的是国际版 Lark，再改成 `lark`
- `media_root`: 飞书图片和文件落地目录

## 怎么启动

先确认依赖已经安装：

```powershell
uv sync --dev
```

然后启动系统：

```powershell
uv run hey-robot run --config configs\xlerobot.real.windows.yaml
```

## 怎么验证是否接通

建议按这个顺序验证：

### 1. 看启动日志

正常情况下，日志里应该出现类似信息：

```text
gateway channel [feishu] feishu 就绪
```

如果这里没有出现，通常说明：

- `enabled` 还没有打开
- `.env` 没配好
- 飞书应用配置不完整

### 2. 先做单聊测试

先在飞书里和机器人单聊，发一句简单的话，比如：

```text
你好
```

### 3. 再做群聊测试

如果 `group_policy: mention`，那你需要在群里：

```text
@机器人 你好
```

如果没有 `@`，项目会忽略这条群消息，这是正常行为。

## 常见问题

### 1. 为什么机器人没有回复

先检查这几项：

- 飞书应用是否真的开了消息能力
- `事件与回调` 里的相关配置是否已经完成
- `.env` 里的飞书变量是否正确
- `channels.feishu.enabled` 是否已经设成 `true`
- 群聊里是否真的 `@` 了机器人

### 2. 为什么群里发消息没反应

大多数情况下是这个原因：

- 你当前配置的是 `group_policy: mention`

这表示只有 `@机器人` 才会进入 agent。

### 3. 图片和文件会保存到哪里

默认会保存到：

```text
runtime/xlerobot.real.windows/media/feishu
```

### 4. 一定要配 `Encrypt Key` 和 `Verification Token` 吗

按当前项目实现，建议配置。

代码在初始化飞书通道时会读取它们，因此最稳妥的方式就是四个值都配齐。

## 最小可用步骤

如果你只想尽快打通，按这个最小流程做：

1. 创建飞书应用
2. 开启机器人消息能力
3. 在 `.env` 里配好四个飞书变量
4. 把 `channels.feishu.enabled` 改成 `true`
5. 运行 `uv run hey-robot run --config configs\xlerobot.real.windows.yaml`
6. 在飞书里给机器人发一条消息

做到这里，通常就已经接通了。

## 这 4 个值在哪里找

### 1. `FEISHU_APP_ID`

在飞书开放平台应用的：

```text
基础信息
```

里找 `App ID`。

### 2. `FEISHU_APP_SECRET`

通常也在：

```text
基础信息
```

或应用凭证相关区域里。

### 3. `FEISHU_ENCRYPT_KEY`

在：

```text
事件与回调
```

页面里找 `Encrypt Key`。

### 4. `FEISHU_VERIFICATION_TOKEN`

也在：

```text
事件与回调
```

页面里找 `Verification Token`。
