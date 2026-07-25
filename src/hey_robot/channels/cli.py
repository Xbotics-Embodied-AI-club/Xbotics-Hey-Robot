from __future__ import annotations

import asyncio
import sys
import time

from hey_robot.channels.base import ChannelContext, InboundHandler
from hey_robot.events import RuntimeEvent
from hey_robot.logging import HeyRobotLogger
from hey_robot.protocol import AgentReply, Envelope, UserTurn

logger = HeyRobotLogger(name="cli")


class CLIChannel:
    def __init__(self, context: ChannelContext) -> None:
        self.context = context
        self.name = context.name
        self._task: asyncio.Task | None = None
        self._stopped = asyncio.Event()
        self._streamed_replies: dict[str, str] = {}

    async def start(self, handler: InboundHandler) -> None:
        if not self._supports_interactive_input():
            logger.info(
                f"cli channel [{self.name}] started without interactive stdin; input loop disabled"
            )
            return
        self._task = asyncio.create_task(self._input_loop(handler))

    async def send(self, reply: AgentReply) -> None:
        prefix = self.context.spec.settings.get("reply_prefix", "assistant")
        stream_key = str(
            reply.metadata.get("interaction_id") or reply.envelope.trace_id
        )
        if not reply.final:
            if stream_key not in self._streamed_replies:
                sys.stdout.write(f"{prefix}> ")
                self._streamed_replies[stream_key] = ""
            sys.stdout.write(reply.text)
            sys.stdout.flush()
            self._streamed_replies[stream_key] += reply.text
            return
        streamed = self._streamed_replies.pop(stream_key, None)
        if streamed is not None:
            if reply.text.startswith(streamed):
                sys.stdout.write(f"{reply.text[len(streamed) :]}\n")
            elif reply.text != streamed:
                sys.stdout.write(f"\n{prefix}> {reply.text}\n")
            else:
                sys.stdout.write("\n")
            sys.stdout.flush()
            return
        sys.stdout.write(f"{prefix}> {reply.text}\n")

    async def on_event(self, _event: RuntimeEvent) -> None:
        return None

    async def stop(self) -> None:
        self._stopped.set()
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _input_loop(self, handler: InboundHandler) -> None:
        prompt = self.context.spec.settings.get("prompt", "user> ")
        sender_id = self.context.spec.settings.get("sender_id", "local")
        chat_id = self.context.spec.settings.get("chat_id", "local")
        while not self._stopped.is_set():
            try:
                text = (await asyncio.to_thread(input, prompt)).strip()
            except (EOFError, KeyboardInterrupt):
                self._stopped.set()
                break
            if not text:
                continue
            envelope = Envelope(
                channel=self.name,
                account_id=self.context.spec.account_id or self.name,
                user_id=(
                    str(self.context.spec.settings.get("user_id")).strip() or None
                    if self.context.spec.settings.get("user_id") is not None
                    else None
                ),
                chat_id=chat_id,
                chat_type="direct",
                sender_id=sender_id,
                deployment_id=self.context.deployment_id,
                timestamp=time.time(),
            )
            await handler(UserTurn(envelope=envelope, text=text))

    @staticmethod
    def _supports_interactive_input() -> bool:
        stdin = sys.stdin
        return bool(stdin and hasattr(stdin, "isatty") and stdin.isatty())
