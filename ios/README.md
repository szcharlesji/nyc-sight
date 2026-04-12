# Spark Sight — iOS Client

Native iOS application for the Spark Sight blind navigation assistant. Replaces the browser-based reference client with a production-quality native app optimized for visually impaired users.

## Architecture

```
iOS Device (Client)                        DGX Spark GB10 (Server)
┌──────────────────────────┐              ┌──────────────────────────────┐
│                          │              │                              │
│  Microphone              │              │  WebSocket Server (port 3000)│
│    → WhisperKit (base)   │              │    ↓                         │
│    → on-device STT       │              │  Orchestrator                │
│    → transcript text ────┼── WebSocket──┼──→ Planning Agent            │
│                          │  (0x05)      │      → Nemotron (LLM)       │
│  Camera                  │              │                              │
│    → AVCaptureSession    │              │  Ambient Agent               │
│    → JPEG frames (0.7q) ─┼── WebSocket──┼──→ Cosmos Reason (VLM)      │
│    → 10 FPS              │  (0x01)      │      → frame analysis        │
│                          │              │                              │
│  Speaker/Headphones      │              │  Response text ──────────────┤
│    ← AVSpeechSynthesizer─┼── WebSocket──┼── (0x06) text response      │
│    ← on-device TTS       │              │── (0x07) warning text        │
│                          │              │── (0x04) status updates      │
│  GPS                     │              │                              │
│    → CLLocationManager   │              │                              │
│    → coordinates ────────┼── WebSocket──┼──→ NYC Open Data queries     │
│                          │  (0x08)      │                              │
└──────────────────────────┘              └──────────────────────────────┘
```

## Binary Protocol

All WebSocket messages use a 1-byte type prefix:

| Tag  | Direction | Payload | Description |
|------|-----------|---------|-------------|
| 0x01 | iOS → Server | JPEG bytes | Camera frame (640×480, quality 0.7) |
| 0x05 | iOS → Server | UTF-8 text | Transcript from WhisperKit |
| 0x08 | iOS → Server | UTF-8 JSON | GPS: `{"lat": float, "lng": float}` |
| 0x04 | Server → iOS | UTF-8 JSON | Status update (signal, mode, goal) |
| 0x06 | Server → iOS | UTF-8 JSON | Text response: `{"text": str, "final": bool}` |
| 0x07 | Server → iOS | UTF-8 JSON | Warning: `{"text": str, "urgency": str}` |

## Setup

### Prerequisites
- Xcode 15.0+ with iOS 17.0 SDK
- iPhone 13 or newer (for WhisperKit performance)
- DGX Spark running the Spark Sight server

### Build Steps

1. Open Xcode → File → New → Project → iOS App (SwiftUI)
2. Name it "SparkSight", set minimum deployment to iOS 17.0
3. Delete the auto-generated ContentView.swift
4. Copy all files from `SparkSight/` into the Xcode project
5. Add WhisperKit via SPM: File → Add Package Dependencies
   - URL: `https://github.com/argmaxinc/WhisperKit`
   - Branch: `main`
6. Set the DGX Spark IP in `AppCoordinator.swift` → `serverHost`
7. Build and run on a physical device (camera/mic require real hardware)

### Permissions
The app requests Camera, Microphone, and Location permissions on first launch.
All permissions are required for full functionality.

## Latency Masking

The app uses several strategies to eliminate dead silence during VLM inference (2-5s):

1. **Earcon** — instant chime when speech ends ("I heard you")
2. **Filler phrases** — contextual phrases spoken before VLM responds
3. **Wake-word echo** — compressed restatement of intent ("Checking for construction ahead...")
4. **On-device responses** — time, battery, location answered without server
5. **Streaming chunks** — TTS starts on first sentence, queues rest

## Accessibility

- Full VoiceOver support on all elements
- Single tap: start/stop listening
- Double tap: repeat last response  
- Long press: "Where am I?" (reverse geocode)
- 88×88pt minimum touch targets
- Haptic feedback for all state changes
- No visual-only feedback anywhere
- Auto-start on launch (no setup wizard)

## Files

```
SparkSight/
├── SparkSightApp.swift           # Entry point, audio session config
├── Models/
│   └── MessageModels.swift       # Protocol types, pack/unpack, enums
├── Managers/
│   ├── AppCoordinator.swift      # Central orchestrator (wires everything)
│   ├── SparkClient.swift         # WebSocket client with reconnection
│   ├── CameraManager.swift       # AVCaptureSession at 10 FPS
│   ├── TranscriptionManager.swift # WhisperKit on-device STT + VAD
│   ├── TTSManager.swift          # AVSpeechSynthesizer + priority queue
│   ├── LocationManager.swift     # CoreLocation + reverse geocoding
│   ├── EarconPlayer.swift        # Audio cues + haptic feedback
│   └── FillerPhraseEngine.swift  # Contextual filler phrases + local queries
├── Views/
│   └── MainView.swift            # Single-screen accessible UI
├── Resources/
│   └── (empty — earcons generated programmatically)
└── Info.plist                    # Permissions, capabilities, ATS
```
