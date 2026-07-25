# NATS ACL 状态说明

`deploy/nats/autonomy-acl.conf`来自旧event-driven Skill控制面，目前不能直接视为当前生产
ACL模板。

## 为什么不能直接使用

旧ACL假设：

- `autonomy_supervisor`是唯一允许发布`skill.intent`的角色；
- Agent角色名为`agent`；
- `skill_controller`是独立控制服务。

当前代码已经移除Supervisor/SkillController主控制链：

- Agent到Skill通过进程内`SkillClient`；
- Skill由同进程`SkillWorker`执行；
- Agent创建NATS client时请求的credential role是`robot-agent`；
- `skill_controller`role只用于Skill event projector；
- `skill.intent`不再是生产提交topic。

因此旧ACL中的`agent`credential与当前`robot-agent`查找键不一致；启用
`deployment.bus.options.credentials`时，如果没有`robot-agent`条目，Agent会在构造阶段报
`NATS credentials missing role: robot-agent`。

## 当前需要保护的NATS边界

当前NATS主要承载：

- Gateway发布`conversation.turn`和control消息；
- Agent订阅conversation/control并发布conversation result；
- Robot发布status、observation、runtime event和camera frame；
- Skill projector发布`skill.event`；
- Human Follow订阅camera frame并发布受限velocity stream；
- Gateway订阅面向UI和Channel的结果/投影。

`robot.action`仍是RobotService兼容入口。如果在部署中不需要外部动作生产者，应默认拒绝
所有非授权角色发布该topic。

## 生产化要求

在重新启用本文件前，应先：

1. 以当前源码中的role名称建立credential map；
2. 按每个service实际publish/subscribe topic生成最小权限，而不是`allow: [">" ]`；
3. 删除对`autonomy_supervisor`和`skill.intent`唯一发布者的旧假设；
4. 明确是否允许任何外部角色发布`robot.action`；
5. 添加启动测试，使用真实NATS配置验证Gateway、Agent、Robot、Skill projector和
   Human Follow均能连接；
6. 为生产环境配置TLS，并根据重放需求决定是否启用JetStream。

在完成上述工作之前，开发环境应使用无role credential配置的本地NATS，或使用
`deployment.bus.type: in_memory`。不要把当前`autonomy-acl.conf`直接部署到真机网络。
