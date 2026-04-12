#!/usr/bin/env python3
"""
Test client for the Parakeet EOU streaming ASR server.

Usage:
    # Local testing
    python scripts/test_client.py test.wav

    # Over Tailscale
    python scripts/test_client.py test.wav --server ws://100.64.0.5:3030/asr

    # With timing simulation disabled (fast mode)
    python scripts/test_client.py test.wav --fast
"""
import argparse
import asyncio
import json
import struct
import time
import wave

import websockets


async def test_streaming(wav_path: str, server_url: str, fast: bool = False):
    with wave.open(wav_path, "rb") as wf:
        assert wf.getsampwidth() == 2 and wf.getnchannels() == 1, \
            f"Expected 16-bit mono WAV, got {wf.getsampwidth()*8}-bit {wf.getnchannels()}ch"
        sr = wf.getframerate()
        if sr != 16000:
            print(f"WARNING: WAV sample rate is {sr}Hz, server expects 16000Hz")
        frames = wf.readframes(wf.getnframes())

    samples = struct.unpack(f"<{len(frames)//2}h", frames)
    f32_bytes = struct.pack(f"<{len(samples)}f", *[s / 32768.0 for s in samples])

    chunk_bytes = 2560 * 4  # 160ms of f32 samples
    total_chunks = (len(f32_bytes) + chunk_bytes - 1) // chunk_bytes
    audio_duration = len(samples) / 16000

    print(f"Audio: {audio_duration:.1f}s, {total_chunks} chunks")
    print(f"Server: {server_url}")
    print(f"Mode: {'fast (no timing)' if fast else 'realtime (160ms spacing)'}")
    print("---")

    t0 = time.monotonic()

    async with websockets.connect(server_url) as ws:
        for i in range(0, len(f32_bytes), chunk_bytes):
            chunk = f32_bytes[i : i + chunk_bytes]
            await ws.send(chunk)

            if not fast:
                await asyncio.sleep(0.16)

            try:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=0.01)
                    event = json.loads(msg)
                    elapsed = time.monotonic() - t0
                    etype = event["type"]
                    text = event.get("text", event.get("message", ""))
                    print(f"[{elapsed:6.2f}s] [{etype:7s}] {text}")
                    if etype == "eou":
                        print("--- EOU detected ---")
            except (asyncio.TimeoutError, Exception):
                pass

        await ws.send(json.dumps({"type": "eos"}))

        try:
            async for msg in ws:
                event = json.loads(msg)
                elapsed = time.monotonic() - t0
                etype = event["type"]
                text = event.get("text", event.get("message", ""))
                print(f"[{elapsed:6.2f}s] [{etype:7s}] {text}")
                if etype == "eou":
                    print("--- EOU detected ---")
                    break
        except websockets.exceptions.ConnectionClosed:
            pass

    total = time.monotonic() - t0
    print(f"\nDone in {total:.2f}s (audio was {audio_duration:.1f}s, RTF={total/audio_duration:.2f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Parakeet EOU ASR server")
    parser.add_argument("wav", help="Path to 16kHz mono WAV file")
    parser.add_argument("--server", default="ws://localhost:3030/asr", help="WebSocket URL")
    parser.add_argument("--fast", action="store_true", help="Send chunks without timing delay")
    args = parser.parse_args()
    asyncio.run(test_streaming(args.wav, args.server, args.fast))
