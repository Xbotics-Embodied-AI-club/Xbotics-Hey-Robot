# NATS Autonomy ACL

Start NATS with `deploy/nats/autonomy-acl.conf` and set all six
`HEY_ROBOT_NATS_*_PASSWORD` environment variables. Configure
`deployment.bus.options.credentials` with the matching role names and
`password_env` values. `autonomy_supervisor` is the only user permitted to
publish `skill.intent`; all other roles have an explicit deny rule.
