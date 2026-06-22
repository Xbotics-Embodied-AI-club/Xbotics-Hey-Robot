# 飞书接入

项目已内置飞书通道（`src/hey_robot/channels/feishu/`），不需要自己写机器人逻辑或接 webhook 服务。启动 `hey-robot run` 后自动连接飞书。

飞书开放平台入口：https://open.feishu.cn/?lang=zh-CN

## 准备工作

在飞书开放平台创建机器人应用，准备好四项凭证：

| 凭证 | 环境变量 | 在飞书后台的位置 |
|---|---|---|
| App ID | `FEISHU_APP_ID` | 基础信息 |
| App Secret | `FEISHU_APP_SECRET` | 基础信息 |
| Encrypt Key | `FEISHU_ENCRYPT_KEY` | 事件与回调 |
| Verification Token | `FEISHU_VERIFICATION_TOKEN` | 事件与回调 |

编辑项目根目录 `.env`：

```dotenv
FEISHU_APP_ID=你的 App ID
FEISHU_APP_SECRET=你的 App Secret
FEISHU_ENCRYPT_KEY=你的 Encrypt Key
FEISHU_VERIFICATION_TOKEN=你的 Verification Token
```

## 飞书平台配置

1. 创建应用，开启机器人相关能力（接收消息、发送消息、回复消息）
2. 在 **事件与回调** 页面订阅 `im.message.receive_v1` 事件
3. 应用的可用范围需包含你自己

## 项目配置

在 deployment 配置文件中找到 `channels.feishu`，确认 `enabled: true`：

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
    media_root: runtime/xlerobot.real.ubuntu/media/feishu
```

常用配置项：

| 字段 | 说明 |
|---|---|
| `enabled` | 是否启用飞书通道 |
| `group_policy` | `mention`：仅 @机器人 时处理；`open`：所有群消息都处理 |
| `reply_to_message` | 是否直接回复原消息 |
| `allow_from` | `["*"]` 不限制发送者；可限定 open_id 列表 |
| `domain` | 国内飞书用 `feishu`，国际版 Lark 用 `lark` |
| `media_root` | 飞书图片和文件落地目录 |

## 验证是否接通

启动系统后，日志应出现：

```
gateway channel [feishu] feishu 就绪
```

然后给机器人发一条私聊消息，观察日志是否有 trace 进入。如果没有，检查：

1. 飞书应用是否发布了（或把你加到测试用户列表）
2. `事件与回调` 是否订阅了 `im.message.receive_v1`
3. `.env` 四个变量是否正确
4. `channels.feishu.enabled` 是否为 `true`
5. 群聊是否 `@` 了机器人（`group_policy: mention` 时）
6. `allow_from` 是否包含你的 open_id 或 `"*"`

## 图片和文件

收到图片/文件会自动保存到 `media_root` 指定目录。
