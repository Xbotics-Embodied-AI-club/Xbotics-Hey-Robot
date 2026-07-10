#!/usr/bin/env python3
"""VLN navigation diagnostics — save frames, test movement, monitor VLN behavior.

Usage:
    python scripts/dev/diag_vln_navigation.py
    python scripts/dev/diag_vln_navigation.py --task "导航去厨房" --save-frames
    python scripts/dev/diag_vln_navigation.py --test-movement
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import aiohttp
import nats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hey_robot.protocol import Topics

# ── Frame saver ───────────────────────────────────────────────────────────


class FrameSaver:
    """Save camera frames received over NATS for visual inspection."""

    def __init__(self, output_dir: str = "diag_frames"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frames: list[dict] = []
        self._nc: nats.NATS | None = None

    async def start(self, nats_url: str = "nats://127.0.0.1:4222"):
        self._nc = await nats.connect(nats_url)
        # Subscribe to raw camera frame bytes
        await self._nc.subscribe("robot.camera.frame.>", cb=self._on_frame)

    async def stop(self):
        if self._nc:
            await self._nc.close()

    async def _on_frame(self, msg):
        try:
            # Decode the frame packet (same format as frame_stream.py)
            import struct

            header_size = struct.unpack("!I", msg.data[:4])[0]
            metadata = json.loads(msg.data[4 : 4 + header_size].decode("utf-8"))
            image_bytes = msg.data[4 + header_size :]
        except Exception:
            return

        frame_id = metadata.get("frame_id", 0)
        camera = metadata.get("camera", "unknown")
        robot_id = metadata.get("robot_id", "unknown")

        # Save to disk
        fname = f"{robot_id}_{camera}_f{frame_id:06d}.jpg"
        fpath = self.output_dir / fname
        fpath.write_bytes(image_bytes)

        # Compute hash for dedup check
        img_hash = hashlib.sha256(image_bytes).hexdigest()[:8]

        self.frames.append(
            {
                "frame_id": frame_id,
                "camera": camera,
                "robot_id": robot_id,
                "hash": img_hash,
                "file": str(fpath),
                "size": len(image_bytes),
            }
        )
        print(f"  [FRAME] {fname}  hash={img_hash}  size={len(image_bytes)}")


# ── Web client ────────────────────────────────────────────────────────────


class WebClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url

    async def send_turn(self, text: str, chat_id: str = "diag-test") -> dict:
        async with (
            aiohttp.ClientSession() as s,
            s.post(
                f"{self.base_url}/turn",
                json={"text": text, "chat_id": chat_id, "sender_id": "diag-user"},
            ) as resp,
        ):
            return await resp.json()


# ── NATS monitor ──────────────────────────────────────────────────────────


class NATSMonitor:
    """Collect all VLN-related events."""

    def __init__(self, nats_url: str = "nats://127.0.0.1:4222"):
        self.nats_url = nats_url
        self.nc: nats.NATS | None = None
        self.topics = Topics()
        self.skill_events: list[dict] = []
        self.skill_results: list[dict] = []
        self.agent_replies: list[dict] = []
        self.robot_statuses: list[dict] = []

    async def __aenter__(self):
        self.nc = await nats.connect(self.nats_url)
        await self.nc.subscribe(self.topics.skill_event, cb=self._on_skill_event)
        await self.nc.subscribe(self.topics.skill_result, cb=self._on_skill_result)
        await self.nc.subscribe(self.topics.agent_reply, cb=self._on_agent_reply)
        await self.nc.subscribe(self.topics.robot_status, cb=self._on_robot_status)
        return self

    async def __aexit__(self, *args: object) -> None:
        if self.nc:
            await self.nc.close()

    async def _on_skill_event(self, msg):
        e = json.loads(msg.data.decode())
        self.skill_events.append(e)
        name = e.get("name", "?")
        phase = e.get("phase", "?")
        step = e.get("step", "")
        summary = (e.get("summary", "") or "")[:80]
        meta = e.get("metadata", {}) or {}
        ux = meta.get("ux", {})
        planner = ux.get("planner", {})
        if planner:
            print(f"  [VLN] name={name} phase={phase} step={step}")
            print(
                f"        mode={planner.get('mode')} heading={planner.get('heading_deg')} "
                f"pixel={planner.get('pixel_goal')} stop={planner.get('stop')}"
            )
            cmd = ux.get("primitive")
            if cmd:
                print(f"        primitive={cmd} args={ux.get('arguments')}")
        else:
            print(f"  [EVENT] name={name} phase={phase} summary={summary[:60]}")

    async def _on_skill_result(self, msg):
        r = json.loads(msg.data.decode())
        self.skill_results.append(r)
        status = r.get("status", "?")
        success = r.get("success")
        summary = (r.get("summary", "") or "")[:100]
        fm = r.get("failure_mode", "")
        print(
            f"  [RESULT] status={status} success={success} failure={fm} summary={summary}"
        )

    async def _on_agent_reply(self, msg):
        r = json.loads(msg.data.decode())
        self.agent_replies.append(r)
        text = (r.get("text", "") or "")[:150]
        print(f"  [AGENT] {text}")

    async def _on_robot_status(self, msg):
        s = json.loads(msg.data.decode())
        self.robot_statuses.append(s)


# ── Frame uniqueness check ────────────────────────────────────────────────


async def check_frame_uniqueness(
    nats_url: str = "nats://127.0.0.1:4222",
    duration: float = 10.0,
) -> dict:
    """Subscribe to camera frames and check if they're actually changing."""
    nc = await nats.connect(nats_url)
    frames: list[tuple[int, str]] = []  # (frame_id, hash)

    async def on_frame(msg):
        try:
            import struct

            header_size = struct.unpack("!I", msg.data[:4])[0]
            image_bytes = msg.data[4 + header_size :]
            metadata = json.loads(msg.data[4 : 4 + header_size].decode("utf-8"))
            img_hash = hashlib.sha256(image_bytes).hexdigest()[:8]
            frames.append((metadata.get("frame_id", 0), img_hash))
        except Exception as exc:
            print(f"[diag] frame parse warning: {exc}")

    await nc.subscribe("robot.camera.frame.>", cb=on_frame)
    print(f"\n  Collecting camera frames for {duration}s...")
    await asyncio.sleep(duration)
    await nc.close()

    if not frames:
        return {"error": "no frames received", "count": 0}

    hashes = [h for _, h in frames]
    unique = len(set(hashes))
    return {
        "count": len(frames),
        "unique": unique,
        "all_same": unique == 1,
        "frame_ids": [fid for fid, _ in frames],
        "hashes": hashes,
    }


# ── Main ──────────────────────────────────────────────────────────────────


async def run_diag(
    task: str,
    save_frames: bool,
    timeout_sec: float,
    web_url: str,
    nats_url: str,
) -> dict:
    web = WebClient(web_url)

    # Phase 0: Check frame uniqueness
    print("=" * 60)
    print("PHASE 0: Frame uniqueness check")
    print("=" * 60)
    frame_check = await check_frame_uniqueness(nats_url, duration=5.0)
    print(f"  Frames received: {frame_check.get('count', 0)}")
    print(f"  Unique frames: {frame_check.get('unique', 0)}")
    if frame_check.get("all_same"):
        print("  ⚠ ALL FRAMES ARE IDENTICAL — camera may be frozen!")

    # Phase 1: Send task and monitor
    print(f"\n{'=' * 60}")
    print("PHASE 1: Send task via Web API")
    print(f"  Task: {task}")
    print(f"{'=' * 60}")

    if save_frames:
        saver = FrameSaver("diag_frames")
        await saver.start(nats_url)

    async with NATSMonitor(nats_url) as monitor:
        resp = await web.send_turn(task)
        trace_id = resp.get("trace_id", "")
        print(f"  trace_id={trace_id}")

        # Wait for agent reply
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if monitor.agent_replies:
                break
            await asyncio.sleep(0.5)

        if time.time() >= deadline:
            print(f"  TIMEOUT after {timeout_sec}s")

    if save_frames:
        await saver.stop()
        print(f"\n  Saved {len(saver.frames)} frames to {saver.output_dir}/")

        # Check frame uniqueness in saved frames
        if saver.frames:
            hashes = [f["hash"] for f in saver.frames]
            unique = set(hashes)
            print(f"  Unique frames: {len(unique)}/{len(hashes)}")
            if len(unique) == 1:
                print("  ⚠ All saved frames are IDENTICAL!")

    # Phase 2: Analyze VLN events
    print(f"\n{'=' * 60}")
    print("PHASE 2: VLN Analysis")
    print(f"{'=' * 60}")

    vln_events = [
        e
        for e in monitor.skill_events
        if (e.get("metadata", {}) or {}).get("ux", {}).get("planner")
    ]
    print(f"  VLN planner events: {len(vln_events)}")

    modes = []
    headings = []
    for e in vln_events:
        planner = (e.get("metadata", {}) or {}).get("ux", {}).get("planner", {})
        mode = planner.get("mode")
        if mode:
            modes.append(mode)
        h = planner.get("heading_deg")
        if h is not None:
            headings.append(h)

    if modes:
        from collections import Counter

        print(f"  VLN modes: {dict(Counter(modes))}")
    if headings:
        print(f"  VLN headings: {[f'{h:.0f}°' for h in headings]}")

    # Check for diversity
    unique_modes = set(modes)
    unique_headings = {round(h) for h in headings}

    issues = []
    if (
        len(vln_events) > 1
        and len(unique_modes) == 1
        and "heading" in unique_modes
        and len(unique_headings) == 1
    ):
        issues.append(
            f"Model ONLY outputs {next(iter(unique_modes))} mode with "
            f"heading={next(iter(unique_headings))}° — possible frozen camera or model issue"
        )

    # Check skill results
    last_result = monitor.skill_results[-1] if monitor.skill_results else {}
    print(
        f"\n  Final skill result: status={last_result.get('status')} "
        f"success={last_result.get('success')}"
    )

    # Check agent replies
    if monitor.agent_replies:
        print(f"  Agent reply: {monitor.agent_replies[0].get('text', '')[:200]}")

    return {
        "task": task,
        "trace_id": trace_id,
        "frame_check": frame_check,
        "vln_events": len(vln_events),
        "vln_modes": modes,
        "vln_headings": headings,
        "issues": issues,
        "final_status": last_result.get("status"),
        "final_success": last_result.get("success"),
    }


async def test_movement(nats_url: str) -> dict:
    """Test basic move/turn primitives by sending direct NATS messages."""
    nc = await nats.connect(nats_url)
    topics = Topics()

    async def collect_statuses(duration: float) -> list[dict]:
        statuses = []

        async def on_status(msg):
            statuses.append(json.loads(msg.data.decode()))

        await nc.subscribe(topics.robot_status, cb=on_status)
        await asyncio.sleep(duration)
        return statuses

    # Test turn right
    print("\n  Testing turn_base right 30°...")
    tid = f"diag_turn_{uuid.uuid4().hex[:8]}"
    turn_msg = {
        "envelope": {
            "trace_id": tid,
            "episode_id": f"ep_{tid}",
            "agent_id": "diag",
            "robot_id": "sim_robot",
            "timestamp": time.time(),
        },
        "skill": "turn_base",
        "arguments": {"direction": "right", "angle_deg": 30},
    }
    await nc.publish(topics.skill_intent, json.dumps(turn_msg).encode())
    await asyncio.sleep(3.0)

    # Test move forward
    print("  Testing move_base forward 30cm...")
    tid = f"diag_move_{uuid.uuid4().hex[:8]}"
    move_msg = {
        "envelope": {
            "trace_id": tid,
            "episode_id": f"ep_{tid}",
            "agent_id": "diag",
            "robot_id": "sim_robot",
            "timestamp": time.time(),
        },
        "skill": "move_base",
        "arguments": {"direction": "forward", "distance_cm": 30},
    }
    await nc.publish(topics.skill_intent, json.dumps(move_msg).encode())
    await asyncio.sleep(3.0)

    await nc.close()
    return {"tested": ["turn_base", "move_base"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLN Navigation Diagnostics")
    parser.add_argument("--task", default="导航去厨房")
    parser.add_argument(
        "--save-frames", action="store_true", help="Save camera frames to disk"
    )
    parser.add_argument(
        "--test-movement", action="store_true", help="Test basic movement primitives"
    )
    parser.add_argument("--web", default="http://127.0.0.1:8080")
    parser.add_argument("--nats", default="nats://127.0.0.1:4222")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()

    if args.test_movement:
        results = asyncio.run(test_movement(args.nats))
        print(f"\nResults: {json.dumps(results, indent=2, ensure_ascii=False)}")
    else:
        results = asyncio.run(
            run_diag(args.task, args.save_frames, args.timeout, args.web, args.nats)
        )
        print(f"\n{'=' * 60}")
        print("DIAGNOSIS")
        print(f"{'=' * 60}")
        for issue in results.get("issues", []):
            print(f"  ⚠ {issue}")
        if not results.get("issues"):
            print("  No obvious issues detected from event stream.")
        print(
            f"\nFull result: {json.dumps(results, indent=2, ensure_ascii=False, default=str)}"
        )
