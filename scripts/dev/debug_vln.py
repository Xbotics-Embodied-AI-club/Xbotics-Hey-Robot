#!/usr/bin/env python3
"""Step-by-step VLN navigation debugging.

Phase 1: Test basic movement primitives via NATS
Phase 2: Test VLN with mock mode (no GPU needed)
Phase 3: Test frame uniqueness over time
Phase 4: Full VLN test with frame saving

Usage:
    python scripts/dev/debug_vln.py phase1    # Test movement
    python scripts/dev/debug_vln.py phase2    # Test mock VLN
    python scripts/dev/debug_vln.py phase3    # Check frames
    python scripts/dev/debug_vln.py phase4    # Full VLN test with frames
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import struct
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import nats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from hey_robot.protocol import Topics

# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Test basic movement primitives
# ═══════════════════════════════════════════════════════════════════════════


async def phase1_test_movement(nats_url: str):
    """Send move_base and turn_base primitives directly and monitor results."""
    nc = await nats.connect(nats_url)
    topics = Topics()

    skill_results: list[dict] = []
    robot_statuses: list[dict] = []

    async def on_skill_result(msg):
        skill_results.append(json.loads(msg.data.decode()))

    async def on_robot_status(msg):
        robot_statuses.append(json.loads(msg.data.decode()))

    await nc.subscribe(topics.skill_result, cb=on_skill_result)
    await nc.subscribe(topics.robot_status, cb=on_robot_status)

    tests = [
        (
            "move_base forward 20cm",
            {
                "name": "move_base",
                "arguments": {"direction": "forward", "distance_cm": 20},
            },
        ),
        (
            "turn_base right 30°",
            {
                "name": "turn_base",
                "arguments": {"direction": "right", "angle_deg": 30},
            },
        ),
        (
            "turn_base left 30°",
            {
                "name": "turn_base",
                "arguments": {"direction": "left", "angle_deg": 30},
            },
        ),
        (
            "move_base backward 10cm",
            {
                "name": "move_base",
                "arguments": {"direction": "backward", "distance_cm": 10},
            },
        ),
    ]

    for label, cmd in tests:
        print(f"\n{'─' * 50}")
        print(f"  Testing: {label}")
        print(f"{'─' * 50}")

        skill_results.clear()
        robot_statuses.clear()

        tid = f"diag_{uuid.uuid4().hex[:8]}"
        msg = {
            "envelope": {
                "trace_id": tid,
                "episode_id": f"ep_{tid}",
                "agent_id": "diag",
                "robot_id": "sim_robot",
                "timestamp": time.time(),
            },
            **cmd,
        }
        await nc.publish(topics.skill_intent, json.dumps(msg).encode())

        # Wait for result
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if any(r.get("envelope", {}).get("trace_id") == tid for r in skill_results):
                break
            await asyncio.sleep(0.2)

        matching = [
            r
            for r in skill_results
            if (r.get("envelope", {}) or {}).get("trace_id") == tid
        ]
        if matching:
            r = matching[0]
            print(f"  Result: success={r.get('success')} status={r.get('status')}")
            print(f"  Summary: {r.get('summary', 'N/A')}")
        else:
            print("  TIMEOUT — no result received")

        # Check robot status for position changes
        pos_updates = [s for s in robot_statuses if s.get("position")]
        if pos_updates:
            last = pos_updates[-1]
            pos = last.get("position", {})
            print(
                f"  Robot position: x={pos.get('x', '?'):.2f}, "
                f"y={pos.get('y', '?'):.2f}, yaw={pos.get('yaw', '?'):.2f}"
            )

        await asyncio.sleep(1.0)

    await nc.close()
    print(f"\n{'=' * 50}")
    print("Phase 1 complete: movement primitives tested")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Mock VLN test
# ═══════════════════════════════════════════════════════════════════════════


async def phase2_mock_vln(web_url: str, nats_url: str, timeout_sec: float = 60.0):
    """Send a web turn and monitor that navigate_to uses mock VLN output."""
    import aiohttp

    # Collect events
    skill_events: list[dict] = []
    skill_results: list[dict] = []
    agent_replies: list[dict] = []

    nc = await nats.connect(nats_url)
    topics = Topics()

    async def on_event(msg):
        skill_events.append(json.loads(msg.data.decode()))

    async def on_result(msg):
        skill_results.append(json.loads(msg.data.decode()))

    async def on_reply(msg):
        agent_replies.append(json.loads(msg.data.decode()))

    await nc.subscribe(topics.skill_event, cb=on_event)
    await nc.subscribe(topics.skill_result, cb=on_result)
    await nc.subscribe(topics.agent_reply, cb=on_reply)

    # Send web turn
    async with (
        aiohttp.ClientSession() as s,
        s.post(
            f"{web_url}/turn",
            json={"text": "导航去厨房", "chat_id": "mock-test", "sender_id": "diag"},
        ) as resp,
    ):
        turn_resp = await resp.json()
    trace_id = turn_resp.get("trace_id", "")
    print(f"  Web turn accepted, trace_id={trace_id}")

    # Wait for agent reply
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if agent_replies:
            break
        await asyncio.sleep(0.5)

    # Analysis
    print(f"\n  Events received: {len(skill_events)}")
    print(f"  Results received: {len(skill_results)}")
    print(f"  Agent replies: {len(agent_replies)}")

    # Check VLN events
    vln_events = []
    for e in skill_events:
        meta = e.get("metadata", {}) or {}
        ux = meta.get("ux", {})
        planner = ux.get("planner", {})
        if planner:
            vln_events.append(
                {
                    "name": e.get("name"),
                    "phase": e.get("phase"),
                    "step": e.get("step"),
                    "mode": planner.get("mode"),
                    "heading": planner.get("heading_deg"),
                    "pixel": planner.get("pixel_goal"),
                    "stop": planner.get("stop"),
                    "primitive": ux.get("primitive"),
                    "args": ux.get("arguments"),
                }
            )

    if vln_events:
        print(f"\n  VLN Events ({len(vln_events)}):")
        for ve in vln_events:
            print(f"    name={ve['name']} phase={ve['phase']} step={ve['step']}")
            print(
                f"    mode={ve['mode']} heading={ve['heading']} pixel={ve['pixel']} "
                f"stop={ve['stop']}"
            )
            print(f"    primitive={ve['primitive']} args={ve['args']}")

    # Final result
    if skill_results:
        last = skill_results[-1]
        print(
            f"\n  Final result: success={last.get('success')} "
            f"status={last.get('status')}"
        )
        print(f"  Summary: {(last.get('summary') or '')[:200]}")
        if last.get("failure_mode"):
            print(f"  Failure: {last.get('failure_mode')}")

    if agent_replies:
        print(f"\n  Agent: {(agent_replies[0].get('text') or '')[:300]}")

    await nc.close()
    return {
        "trace_id": trace_id,
        "vln_events": len(vln_events),
        "agent_replies": [r.get("text", "")[:100] for r in agent_replies],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3: Frame uniqueness check
# ═══════════════════════════════════════════════════════════════════════════


async def phase3_check_frames(nats_url: str, duration: float = 15.0):
    """Monitor camera frames and check for uniqueness over time."""
    nc = await nats.connect(nats_url)
    frames: list[dict] = []

    async def on_frame(msg):
        try:
            header_size = struct.unpack("!I", msg.data[:4])[0]
            image_bytes = msg.data[4 + header_size :]
            metadata = json.loads(msg.data[4 : 4 + header_size].decode("utf-8"))
            frames.append(
                {
                    "frame_id": metadata.get("frame_id", 0),
                    "camera": metadata.get("camera", "unknown"),
                    "robot_id": metadata.get("robot_id", "unknown"),
                    "hash": hashlib.sha256(image_bytes).hexdigest()[:8],
                    "size": len(image_bytes),
                    "ts": time.time(),
                }
            )
        except Exception as exc:
            print(f"[phase3] frame parse warning: {exc}")

    await nc.subscribe("robot.camera.frame.>", cb=on_frame)

    print(f"  Collecting frames for {duration}s...")
    t0 = time.time()
    await asyncio.sleep(duration)
    elapsed = time.time() - t0
    await nc.close()

    if not frames:
        print("  ⚠ NO FRAMES RECEIVED!")
        return {"error": "no frames", "count": 0}

    print(
        f"  Received {len(frames)} frames in {elapsed:.1f}s "
        f"({len(frames) / elapsed:.1f} Hz)"
    )

    # Per-camera analysis
    cameras = {f["camera"] for f in frames}
    for cam in sorted(cameras):
        cam_frames = [f for f in frames if f["camera"] == cam]
        hashes = [f["hash"] for f in cam_frames]
        unique = len(set(hashes))
        sizes = {f["size"] for f in cam_frames}
        frame_ids = [f["frame_id"] for f in cam_frames]

        print(f"\n  Camera: {cam}")
        print(f"    Frames: {len(cam_frames)}, Unique: {unique}")
        print(f"    Size range: {min(sizes)}-{max(sizes)} bytes")
        print(f"    Frame IDs: {frame_ids[:5]}...{frame_ids[-3:]}")
        if unique <= 1 and len(cam_frames) > 1:
            print(f"    ⚠ ALL {len(cam_frames)} FRAMES ARE IDENTICAL!")
        elif unique < len(cam_frames):
            print(f"    ⚠ Only {unique}/{len(cam_frames)} frames are unique")

    return {
        "count": len(frames),
        "unique_hashes": len({f["hash"] for f in frames}),
        "cameras": list(cameras),
        "hz": len(frames) / elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Full VLN test with frame capture
# ═══════════════════════════════════════════════════════════════════════════


def _write_pending_frames(pending_frames: list[tuple[str, int, bytes]]) -> None:
    """Write collected frame images to disk synchronously."""
    out_dir = Path("diag_frames")
    out_dir.mkdir(exist_ok=True)
    for camera, fid, img_bytes in pending_frames:
        fname = out_dir / f"frame_{camera}_{fid:06d}.jpg"
        fname.write_bytes(img_bytes)
    print(f"    Saved {len(pending_frames)} frames to diag_frames/")


async def phase4_full_test(
    task: str,
    web_url: str,
    nats_url: str,
    timeout_sec: float = 120.0,
    save_frames: bool = False,
):
    """Full VLN test: web turn → monitor events → save frames → analyze."""
    import aiohttp

    nc = await nats.connect(nats_url)
    topics = Topics()

    # Collect everything
    skill_events: list[dict] = []
    skill_results: list[dict] = []
    agent_replies: list[dict] = []
    robot_statuses: list[dict] = []
    frame_hashes: list[tuple[float, str, int]] = []  # (ts, hash, frame_id)
    pending_frames: list[tuple[str, int, bytes]] = []  # deferred sync I/O

    async def on_event(msg):
        skill_events.append(json.loads(msg.data.decode()))

    async def on_result(msg):
        skill_results.append(json.loads(msg.data.decode()))

    async def on_reply(msg):
        agent_replies.append(json.loads(msg.data.decode()))

    async def on_status(msg):
        robot_statuses.append(json.loads(msg.data.decode()))

    async def on_frame(msg):
        try:
            header_size = struct.unpack("!I", msg.data[:4])[0]
            image_bytes = msg.data[4 + header_size :]
            metadata = json.loads(msg.data[4 : 4 + header_size].decode("utf-8"))
            h = hashlib.sha256(image_bytes).hexdigest()[:8]
            frame_hashes.append((time.time(), h, metadata.get("frame_id", 0)))
            if save_frames:
                pending_frames.append(
                    (
                        metadata.get("camera", "unknown"),
                        metadata.get("frame_id", 0),
                        image_bytes,
                    )
                )
        except Exception as exc:
            print(f"[phase4] frame parse warning: {exc}")

    await nc.subscribe(topics.skill_event, cb=on_event)
    await nc.subscribe(topics.skill_result, cb=on_result)
    await nc.subscribe(topics.agent_reply, cb=on_reply)
    await nc.subscribe(topics.robot_status, cb=on_status)
    await nc.subscribe("robot.camera.frame.>", cb=on_frame)

    # Wait a moment for frame collection to start
    await asyncio.sleep(1.0)

    # Record frame hashes before the turn
    pre_hashes = {h for _, h, _ in frame_hashes}

    # Send web turn
    print(f"\n  Sending task: {task}")
    async with (
        aiohttp.ClientSession() as s,
        s.post(
            f"{web_url}/turn",
            json={"text": task, "chat_id": "full-test", "sender_id": "diag"},
        ) as resp,
    ):
        turn_resp = await resp.json()
    trace_id = turn_resp.get("trace_id", "")
    t_send = time.time()
    print(f"  trace_id={trace_id}")

    # Wait for agent reply
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if agent_replies:
            break
        await asyncio.sleep(0.5)

    t_end = time.time()

    # Post-turn frame hashes
    post_hashes = {h for _, h, _ in frame_hashes}
    new_hashes = post_hashes - pre_hashes

    if save_frames and pending_frames:
        _write_pending_frames(pending_frames)

    # ── Analysis ──

    print(f"\n{'=' * 50}")
    print(f"RESULTS FOR: {task}")
    print(f"{'=' * 50}")

    # Frame analysis
    print("\n  [Frames]")
    print(f"    Total frames captured: {len(frame_hashes)}")
    print(f"    Unique hashes: {len({h for _, h, _ in frame_hashes})}")
    print(f"    New frames after turn: {len(new_hashes)}")
    if not new_hashes:
        print("    ⚠ NO NEW FRAMES after turn!")

    # VLN event analysis
    vln_planner_events = []
    for e in skill_events:
        meta = e.get("metadata", {}) or {}
        ux = meta.get("ux", {})
        planner = ux.get("planner", {})
        if planner:
            vln_planner_events.append(
                {
                    "name": e.get("name"),
                    "mode": planner.get("mode"),
                    "heading": planner.get("heading_deg"),
                    "pixel": planner.get("pixel_goal"),
                    "stop": planner.get("stop"),
                    "primitive": ux.get("primitive"),
                }
            )

    print(f"\n  [VLN Planner Events: {len(vln_planner_events)}]")
    modes = []
    headings = []
    for ve in vln_planner_events:
        print(
            f"    mode={ve['mode']} heading={ve['heading']} pixel={ve['pixel']} "
            f"stop={ve['stop']} → {ve['primitive']}"
        )
        if ve["mode"]:
            modes.append(ve["mode"])
        if ve["heading"] is not None:
            headings.append(ve["heading"])

    mode_counts = dict(Counter(modes))
    print(f"\n  Mode distribution: {mode_counts}")
    if headings:
        print(f"  Headings: {[f'{h:.0f}°' for h in headings]}")

    # Check if model only outputs one mode
    issues = []
    if len(modes) > 0 and len(set(modes)) == 1:
        issues.append(f"Model ONLY outputs '{modes[0]}' mode — no variation")
    if len(headings) > 0 and len({round(h) for h in headings}) == 1:
        h = headings[0]
        issues.append(f"Model ONLY outputs heading {h:.0f}° — robot just spins")

    # Skill results
    print(f"\n  [Skill Results: {len(skill_results)}]")
    for r in skill_results:
        print(
            f"    success={r.get('success')} status={r.get('status')} "
            f"failure={r.get('failure_mode', '')} "
            f"summary={(r.get('summary') or '')[:80]}"
        )

    # Agent replies
    print(f"\n  [Agent Replies: {len(agent_replies)}]")
    for r in agent_replies:
        text = r.get("text", "") or ""
        print(f"    {text[:200]}")

    # Position changes
    positions = []
    for s in robot_statuses:
        pos = s.get("position")
        if pos:
            positions.append(
                (
                    s.get("envelope", {}).get("timestamp", 0),
                    pos.get("x"),
                    pos.get("y"),
                    pos.get("yaw"),
                )
            )
    if positions:
        print(f"\n  [Position Changes: {len(positions)}]")
        first = positions[0]
        last = positions[-1]
        print(f"    Start: x={first[1]:.2f}, y={first[2]:.2f}, yaw={first[3]:.2f}")
        print(f"    End:   x={last[1]:.2f}, y={last[2]:.2f}, yaw={last[3]:.2f}")
        dx = (last[1] or 0) - (first[1] or 0)
        dy = (last[2] or 0) - (first[2] or 0)
        dyaw = (last[3] or 0) - (first[3] or 0)
        print(f"    Delta: dx={dx:.2f}m, dy={dy:.2f}m, dyaw={dyaw:.1f}°")

    # Suggestions
    print("\n  [Issues Found]")
    if issues:
        for issue in issues:
            print(f"    ⚠ {issue}")
    else:
        print("    No critical issues detected")

    await nc.close()
    return {
        "task": task,
        "trace_id": trace_id,
        "issues": issues,
        "vln_modes": modes,
        "vln_headings": headings,
        "mode_distribution": mode_counts,
        "frame_count": len(frame_hashes),
        "new_frames_after_turn": len(new_hashes),
        "elapsed": t_end - t_send,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VLN Debugging Tool")
    parser.add_argument(
        "phase",
        choices=["phase1", "phase2", "phase3", "phase4"],
        help="Which debug phase to run",
    )
    parser.add_argument("--task", default="导航去厨房")
    parser.add_argument("--web", default="http://127.0.0.1:8080")
    parser.add_argument("--nats", default="nats://127.0.0.1:4222")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument(
        "--duration",
        type=float,
        default=15.0,
        help="Duration for frame collection (phase3)",
    )
    args = parser.parse_args()

    if args.phase == "phase1":
        print("=" * 60)
        print("PHASE 1: Testing basic movement primitives")
        print("=" * 60)
        asyncio.run(phase1_test_movement(args.nats))

    elif args.phase == "phase2":
        print("=" * 60)
        print("PHASE 2: Mock VLN test")
        print("=" * 60)
        print("(Make sure VLN service has mock_mode: true in config)")
        result = asyncio.run(phase2_mock_vln(args.web, args.nats, args.timeout))
        print(f"\n{json.dumps(result, indent=2, ensure_ascii=False)}")

    elif args.phase == "phase3":
        print("=" * 60)
        print("PHASE 3: Frame uniqueness check")
        print("=" * 60)
        result = asyncio.run(phase3_check_frames(args.nats, args.duration))
        print(f"\n{json.dumps(result, indent=2, ensure_ascii=False)}")

    elif args.phase == "phase4":
        print("=" * 60)
        print("PHASE 4: Full VLN test")
        print("=" * 60)
        result = asyncio.run(
            phase4_full_test(
                args.task, args.web, args.nats, args.timeout, args.save_frames
            )
        )
        print(f"\n{json.dumps(result, indent=2, ensure_ascii=False, default=str)}")
