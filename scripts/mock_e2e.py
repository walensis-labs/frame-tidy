"""Run the whole workflow against the mock Aura API and score the face matching.

Starts the mock server, launches the real MCP server over stdio (exactly as
Claude Code does) pointed at the mock, then: list frames -> sync -> find the
target person -> preview -> remove from one frame -> delete from the account.
Compares the matches with mock/ground_truth.json.

Needs mock frames first:  uv run python scripts/build_mock_frames.py
Usage:                    uv run python scripts/mock_e2e.py [--threshold 0.45]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]
MOCK_HOME = ROOT / "mock"


def payload(result):
    """The tool's JSON result, or raise with the tool's error text."""
    text = "".join(getattr(c, "text", "") for c in result.content)
    if getattr(result, "is_error", False):
        raise RuntimeError(text)
    structured = getattr(result, "structured_content", None)
    if isinstance(structured, dict):
        return structured.get("result", structured) if set(structured) == {"result"} else structured
    return json.loads(text)


async def run(base: str, threshold: float) -> None:
    truth = json.loads((MOCK_HOME / "ground_truth.json").read_text())
    targets = set(truth["target_file_names"])
    env = {
        "PATH": os.environ["PATH"],
        "HOME": os.environ["HOME"],
        "AURA_MCP_HOME": str(MOCK_HOME),
        "AURA_API_BASE_URL": f"{base}/v5",
        "AURA_IMAGE_BASE_URL": f"{base}/img",
        "AURA_EMAIL": "mock@example.com",
        "AURA_PASSWORD": "mock",
    }
    params = StdioServerParameters(
        command="uv", args=["run", "--directory", str(ROOT), "aura-mcp"], env=env
    )

    async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        async def call(tool: str, **arguments):
            return payload(await session.call_tool(tool, arguments))

        frames = {f["name"]: f for f in await call("list_frames")}
        print("frames:", {name: f["asset_count"] for name, f in frames.items()})
        living, kitchen = frames["Living Room"]["id"], frames["Kitchen"]["id"]

        for name, frame_id in (("Living Room", living), ("Kitchen", kitchen)):
            started = time.time()
            synced = await call("sync_frame_photos", frame_id=frame_id)
            print(f"sync {name}: {synced['total_synced']} photos "
                  f"({synced['downloaded']} downloaded, {len(synced['failed'])} failed) "
                  f"in {time.time() - started:.0f}s")

        started = time.time()
        found = await call("find_person_in_frame", frame_id=living,
                           reference_image_path=truth["reference"],
                           similarity_threshold=threshold)
        flagged = {m["file_name"]: m["similarity"] for m in found["matches"]}
        on_frame = targets  # every target photo is on the Living Room frame
        hits = set(flagged) & on_frame
        false_pos = sorted(set(flagged) - on_frame)
        missed = sorted(on_frame - set(flagged))
        print(f"\nfind {truth['person']} on Living Room at threshold {threshold} "
              f"({time.time() - started:.0f}s, {found['photos_scanned']} photos scanned, "
              f"{found['photos_with_faces']} with faces)")
        print(f"  flagged {len(flagged)}; correct {len(hits)} of {len(on_frame)} target photos")
        print(f"  precision {len(hits) / max(1, len(flagged)):.1%}  "
              f"recall {len(hits) / len(on_frame):.1%}")
        print(f"  wrongly flagged: {false_pos or 'none'}")
        print(f"  missed: {missed or 'none'}")
        if hits:
            print(f"  lowest similarity among correct matches: {min(flagged[h] for h in hits):.3f}")

        ids = [m["asset_id"] for m in found["matches"]]
        preview = await call("delete_assets", frame_id=living, asset_ids=ids)
        state = httpx.get(f"{base}/mock/state").json()
        print(f"\npreview: status={preview['status']} scope={preview['scope']} "
              f"would_delete={preview['would_delete_count']}; "
              f"mock still has {state['total_assets']} photos")

        report = await call("delete_assets", frame_id=living, asset_ids=ids, confirm=True)
        state = httpx.get(f"{base}/mock/state").json()
        print(f"frame-scope removal: {report['deleted_count']} removed, "
              f"{report['failed_count']} failed -> {[(f['name'], f['num_assets']) for f in state['frames']]}")

        found_k = await call("find_person_in_frame", frame_id=kitchen,
                             reference_image_path=truth["reference"],
                             similarity_threshold=threshold)
        print(f"Kitchen still shows the target in {found_k['match_count']} photos "
              f"(expected {truth['frames']['Kitchen']['target']})")
        report = await call("delete_assets", frame_id=kitchen, scope="account", confirm=True,
                            asset_ids=[m["asset_id"] for m in found_k["matches"]])
        state = httpx.get(f"{base}/mock/state").json()
        print(f"account-scope deletion: {report['deleted_count']} deleted -> "
              f"{[(f['name'], f['num_assets']) for f in state['frames']]}, "
              f"{state['total_assets']} photos left in the account")

        again = await call("find_person_in_frame", frame_id=living,
                           reference_image_path=truth["reference"],
                           similarity_threshold=threshold)
        print(f"re-scan of Living Room finds {again['match_count']} matches")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    mock = subprocess.Popen(
        ["uv", "run", "--directory", str(ROOT), "aura-mock", "--port", str(args.port)],
        stdout=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                httpx.post(f"{base}/mock/reset").raise_for_status()
                break
            except httpx.HTTPError:
                if mock.poll() is not None:
                    sys.exit("The mock server failed to start (is the port in use?).")
                time.sleep(0.2)
        else:
            sys.exit("The mock server did not come up.")
        asyncio.run(run(base, args.threshold))
    finally:
        mock.terminate()
        mock.wait(timeout=10)


if __name__ == "__main__":
    main()
