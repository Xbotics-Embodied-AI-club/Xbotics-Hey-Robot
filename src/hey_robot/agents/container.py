from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path

from hey_robot.agents.busy_turn import BusyTurnHandler
from hey_robot.agents.context import RobotContextBuilder
from hey_robot.agents.core import RobotAgentCore
from hey_robot.agents.loop import RobotAgentLoop
from hey_robot.agents.notification_runtime import AgentNotificationRuntime
from hey_robot.agents.progress import RobotAgentProgress
from hey_robot.agents.scene_runtime import SceneRuntime
from hey_robot.agents.session import AgentTurnSessions
from hey_robot.agents.task_runtime import RobotStateCache, TaskRunManager
from hey_robot.capability.catalog.loader import CapabilityLoader
from hey_robot.config import DeploymentConfig
from hey_robot.episode import JsonlEpisodeStore, RobotEpisodeStateStore
from hey_robot.gateway.identity import IdentityResolver
from hey_robot.media import LocalMediaStore, MediaResolver
from hey_robot.memory import SceneSummarizer
from hey_robot.notifications import NotificationPolicy, NotificationService
from hey_robot.perception.scene import build_scene_captioner
from hey_robot.protocol import AgentReply, SkillEvent, SkillIntent

ReplyPublisher = Callable[[AgentReply], Awaitable[None]]
ProgressPublisher = Callable[[RobotAgentProgress], Awaitable[None]]
SkillEventPublisher = Callable[[SkillEvent], Awaitable[None]]
SkillIntentPublisher = Callable[[SkillIntent], Awaitable[None]]


@dataclass
class RobotAgentRuntimeContainer:
    episodes: JsonlEpisodeStore
    robot_states: RobotEpisodeStateStore
    robot_cache: RobotStateCache
    turn_sessions: AgentTurnSessions
    task_runtime: TaskRunManager
    notification_runtime: AgentNotificationRuntime
    media_resolver: MediaResolver
    scene_captioner: object
    scene_runtime: SceneRuntime
    busy_turns: BusyTurnHandler
    core: RobotAgentCore
    loop: RobotAgentLoop
    capabilities: CapabilityLoader
    turn_timeout_sec: float
    skill_lease_timeout_sec: float

    @classmethod
    def build(
        cls,
        *,
        config: DeploymentConfig,
        agent_id: str,
        episode_dir: str | Path | None,
        io,
        publish_reply: ReplyPublisher,
        publish_progress: ProgressPublisher,
        publish_skill_event: SkillEventPublisher,
        publish_skill_intent: SkillIntentPublisher,
    ) -> RobotAgentRuntimeContainer:
        spec = config.agents.get(agent_id)
        if spec is None:
            raise KeyError(f"unknown agent: {agent_id}")

        episode_root = episode_dir or config.resources.episodes_root
        episodes = JsonlEpisodeStore(episode_root)
        robot_states = RobotEpisodeStateStore(episode_root)
        robot_cache = RobotStateCache()
        turn_sessions = AgentTurnSessions()
        task_runtime = TaskRunManager(
            episode_root=episode_root,
            runtime_dir=config.resources.runtime_dir,
            events_max_items=config.resources.events_max_items,
            robot_states=robot_states,
        )
        identity = IdentityResolver(
            config.identity,
            state_path=Path(config.resources.runtime_dir)
            / "identity"
            / "bindings.json",
        )
        notification_runtime = AgentNotificationRuntime(
            agent_id=agent_id,
            default_robot=config.default_robot_id(agent_id),
            service=NotificationService(
                episodes,
                publish_reply,
                policy=NotificationPolicy(config.notifications),
                linked_target_provider=identity.linked_channel_targets,
            ),
        )
        media_resolver = MediaResolver(LocalMediaStore(config.resources.media_root))
        scene_captioner = build_scene_captioner(
            config, agent_id, image_resolver=media_resolver
        )
        scene_runtime = SceneRuntime(
            agent_id=agent_id,
            robot_cache=robot_cache,
            task_runtime=task_runtime,
            captioner=scene_captioner,
            max_memory_tasks=int(spec.settings.get("scene_memory_max_tasks", 2)),
            summarizer=SceneSummarizer(),
        )
        busy_turns = BusyTurnHandler(
            agent_id=agent_id,
            default_robot=config.default_robot_id(agent_id),
            robot_cache=robot_cache,
            task_runtime=task_runtime,
            turn_sessions=turn_sessions,
            publish_reply=publish_reply,
            publish_progress=publish_progress,
            publish_skill_event=publish_skill_event,
            publish_skill_intent=publish_skill_intent,
        )
        runtime_spec = replace(
            spec,
            settings={
                **spec.settings,
                "_deployment_config": config,
                "_scene_memory": task_runtime.scene_memory,
            },
        )
        core = RobotAgentCore(
            agent_id=agent_id,
            spec=runtime_spec,
            io=io,
            media_resolver=media_resolver,
        )
        loop = RobotAgentLoop(
            core,
            task_runtime=task_runtime,
            context_builder=RobotContextBuilder(
                capability_manifest_provider=core.capability_manifest,
            ),
        )
        capabilities = CapabilityLoader(
            tools=core.runtime.tools,
            robot_skills=core.capabilities.robot_skills,
        )
        return cls(
            episodes=episodes,
            robot_states=robot_states,
            robot_cache=robot_cache,
            turn_sessions=turn_sessions,
            task_runtime=task_runtime,
            notification_runtime=notification_runtime,
            media_resolver=media_resolver,
            scene_captioner=scene_captioner,
            scene_runtime=scene_runtime,
            busy_turns=busy_turns,
            core=core,
            loop=loop,
            capabilities=capabilities,
            turn_timeout_sec=float(spec.settings.get("turn_timeout_sec", 120.0)),
            skill_lease_timeout_sec=float(
                spec.settings.get("skill_lease_timeout_sec", 300.0)
            ),
        )
