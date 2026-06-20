from __future__ import annotations

from enum import StrEnum


class Severity(StrEnum):
    DEBUG = "debug"
    INFO = "info"
    WARN = "warn"
    ERROR = "error"


class EventKind(StrEnum):
    GATEWAY_START = "gateway.start"
    GATEWAY_READY = "gateway.ready"
    GATEWAY_SHUTDOWN = "gateway.shutdown"
    CHANNEL_STARTED = "channel.lifecycle.started"
    CHANNEL_STOPPED = "channel.lifecycle.stopped"
    CHANNEL_INBOUND = "channel.message.inbound"
    CHANNEL_OUTBOUND_SENT = "channel.message.outbound_sent"
    CHANNEL_OUTBOUND_FAILED = "channel.message.outbound_failed"
    EPISODE_ALLOCATED = "episode.allocated"
    AGENT_TURN_START = "agent.turn.start"
    AGENT_TURN_END = "agent.turn.end"
    AGENT_SKILL_SUBMITTED = "agent.skill.submitted"
    ROBOT_STARTED = "robot.started"
    ROBOT_STATUS = "robot.status"
    ROBOT_SKILL_RECEIVED = "robot.skill.received"
    POLICY_STARTED = "policy.started"
    POLICY_ACTION = "policy.action"
    BUS_PUBLISH_FAILED = "bus.publish.failed"
