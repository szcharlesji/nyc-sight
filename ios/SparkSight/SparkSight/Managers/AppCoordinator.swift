// AppCoordinator.swift
// Spark Sight — iOS Client
//
// Central orchestrator that wires all managers together and implements
// the full end-to-end data flow:
//
//   Camera → JPEG frames → WebSocket → DGX Spark
//   Mic → WhisperKit STT → transcript → WebSocket → DGX Spark
//   GPS → CoreLocation → periodic updates → WebSocket → DGX Spark
//   WebSocket response → text → AVSpeechSynthesizer → Speaker
//   WebSocket warning → interrupt → AVSpeechSynthesizer → Speaker
//
// Also handles latency masking (filler phrases, earcons, local queries).

import SwiftUI
import Combine
import UIKit

@MainActor
final class AppCoordinator: ObservableObject {

    // MARK: - Published State (consumed by MainView)

    @Published var connectionState: ConnectionState = .disconnected
    @Published var isListening = false
    @Published var isSpeaking = false
    @Published var currentMode = "patrol"
    @Published var activeGoal: String?
    @Published var currentTranscript = ""
    @Published var currentAddress = "Locating..."
    @Published var framesDelivered = 0

    // MARK: - Settings (persisted in UserDefaults)

    @AppStorage("sparkHost") var serverHost = "192.168.1.100"
    @AppStorage("sparkPort") var serverPort = 3000

    // MARK: - Managers

    let sparkClient = SparkClient()
    let camera = CameraManager()
    let transcription = TranscriptionManager()
    let tts = TTSManager()
    let location = LocationManager()
    let earcon = EarconPlayer()
    let filler = FillerPhraseEngine()

    // MARK: - Private

    private var cancellables = Set<AnyCancellable>()
    private var isWaitingForResponse = false

    // MARK: - Startup

    /// Initialize all subsystems and connect to the DGX Spark.
    func start() async {
        // Enable battery monitoring for on-device battery queries.
        UIDevice.current.isBatteryMonitoringEnabled = true

        // Setup earcons (haptics + tones).
        earcon.setup()

        // Load WhisperKit model (async, may take a few seconds).
        tts.speak("Spark Sight starting. Loading speech model.", priority: .normal)
        await transcription.setup()

        // Request location permission and start GPS.
        location.requestPermission()

        // Wire up all callbacks and observation.
        wireCallbacks()
        wireObservation()

        // Start camera capture.
        camera.startCapture()

        // Connect to the server.
        sparkClient.connect(host: serverHost, port: serverPort)

        tts.speak("Ready. Tap anywhere to speak.", priority: .normal)
    }

    // MARK: - User Actions (triggered by MainView gestures)

    /// Toggle microphone listening on/off.
    func toggleListening() {
        if isListening {
            transcription.stopListening()
            isListening = false
        } else {
            transcription.startListening()
            isListening = true
            // Subtle haptic to confirm listening started.
            earcon.hapticModeChange()
        }
    }

    /// Repeat the last VLM response.
    func repeatLastResponse() {
        tts.repeatLast()
    }

    /// Speak the current reverse-geocoded location.
    func speakCurrentLocation() {
        Task {
            let description = await location.describeCurrentLocation()
            tts.speak(description, priority: .normal)
        }
    }

    /// Toggle the rear camera torch (flashlight).
    func toggleTorch() {
        camera.toggleTorch()
        let state = camera.isTorchOn ? "on" : "off"
        tts.speak("Flashlight \(state).", priority: .normal)
    }

    // MARK: - Callback Wiring

    private func wireCallbacks() {
        // ── Camera → WebSocket ──
        camera.onFrame = { [weak self] jpegData in
            self?.sparkClient.sendFrame(jpegData)
        }

        // ── WhisperKit → Process Utterance ──
        transcription.onUtterance = { [weak self] text in
            Task { @MainActor in
                self?.handleUtterance(text)
            }
        }

        // ── GPS → WebSocket ──
        location.onLocationUpdate = { [weak self] lat, lng in
            self?.sparkClient.sendLocation(lat: lat, lng: lng)
        }

        // ── WebSocket → Text Response → TTS ──
        sparkClient.onTextResponse = { [weak self] text, isFinal in
            Task { @MainActor in
                self?.handleTextResponse(text, isFinal: isFinal)
            }
        }

        // ── WebSocket → Warning → Urgent TTS ──
        sparkClient.onWarning = { [weak self] text, urgency in
            Task { @MainActor in
                self?.handleWarning(text, urgency: urgency)
            }
        }

        // ── WebSocket → Status Update ──
        sparkClient.onStatus = { [weak self] status in
            Task { @MainActor in
                self?.handleStatus(status)
            }
        }
    }

    // MARK: - Observation Wiring (Combine)

    private func wireObservation() {
        // Mirror SparkClient state.
        sparkClient.$connectionState
            .receive(on: RunLoop.main)
            .sink { [weak self] state in
                let oldState = self?.connectionState
                self?.connectionState = state
                // Haptic + audio feedback on connection changes.
                if oldState != state {
                    self?.earcon.hapticConnectionChange(connected: state == .connected)
                    if state == .connected {
                        self?.tts.speak("Connected to Spark.", priority: .normal)
                    } else if state == .reconnecting && oldState == .connected {
                        self?.tts.speak("Connection lost. Reconnecting.", priority: .high)
                    }
                }
            }
            .store(in: &cancellables)

        sparkClient.$framesDelivered
            .receive(on: RunLoop.main)
            .assign(to: &$framesDelivered)

        // Mirror TTS state.
        tts.$isSpeaking
            .receive(on: RunLoop.main)
            .assign(to: &$isSpeaking)

        // Mirror transcription state.
        transcription.$currentTranscript
            .receive(on: RunLoop.main)
            .assign(to: &$currentTranscript)

        // Mirror location.
        location.$currentAddress
            .receive(on: RunLoop.main)
            .assign(to: &$currentAddress)
    }

    // MARK: - Utterance Processing (the core latency-masking flow)

    /// Handle a complete utterance from WhisperKit.
    ///
    /// Flow:
    /// 1. Play earcon (instant "I heard you" confirmation)
    /// 2. Check for on-device quick responses (time, battery, repeat)
    /// 3. If server-bound: speak filler phrase, then send transcript
    private func handleUtterance(_ text: String) {
        guard !text.isEmpty else { return }

        // 1. Earcon — instant audio confirmation.
        earcon.playAcknowledgment()

        // 2. Check for "repeat" request.
        if filler.isRepeatRequest(text) {
            tts.repeatLast()
            return
        }

        // 3. Check for on-device quick response.
        if let localAnswer = filler.localResponse(for: text) {
            tts.speak(localAnswer, priority: .normal)
            return
        }

        // 4. Server-bound query — mask latency.
        isWaitingForResponse = true

        // 4a. Try wake-word echo first (more specific than filler).
        if let echo = filler.wakeWordEcho(for: text) {
            tts.speak(echo, priority: .normal)
        } else {
            // 4b. Generic filler phrase.
            let fillerText = filler.fillerPhrase(for: text)
            tts.speak(fillerText, priority: .normal)
        }

        // 5. Send transcript to DGX Spark.
        sparkClient.sendTranscript(text)
    }

    // MARK: - Server Response Handling

    /// Handle a text response from the server (for on-device TTS).
    private func handleTextResponse(_ text: String, isFinal: Bool) {
        isWaitingForResponse = false
        tts.speakChunk(text, isFinal: isFinal)
    }

    /// Handle an urgent warning from the active agent.
    private func handleWarning(_ text: String, urgency: String) {
        // Haptic feedback first (instant).
        earcon.hapticWarning(urgency: urgency)

        // Interrupt all current speech for critical/high warnings.
        let priority: MessagePriority = urgency == "critical" ? .critical : .high
        tts.speak(text, priority: priority)
    }

    /// Handle a status update from the ambient agent.
    private func handleStatus(_ status: StatusUpdate) {
        if let mode = status.mode {
            let oldMode = currentMode
            currentMode = mode
            if oldMode != mode {
                earcon.hapticModeChange()
            }
        }
        if let goal = status.goal {
            activeGoal = goal.isEmpty ? nil : goal
        }
    }
}
