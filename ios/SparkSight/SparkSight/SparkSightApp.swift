// SparkSightApp.swift
// Spark Sight — iOS Client for Blind Navigation Assistant
//
// Entry point. Configures the audio session for simultaneous playback
// and recording, then launches the single-screen accessible interface.

import SwiftUI
import AVFoundation

@main
struct SparkSightApp: App {

    @StateObject private var coordinator = AppCoordinator()

    init() {
        Self.configureAudioSession()
    }

    var body: some Scene {
        WindowGroup {
            MainView()
                .environmentObject(coordinator)
                .onAppear {
                    Task { await coordinator.start() }
                }
                .sheet(isPresented: $coordinator.showSettings) {
                    SettingsView(
                        serverHost: $coordinator.serverHost,
                        serverPort: Binding(
                            get: { String(coordinator.serverPort) },
                            set: { coordinator.serverPort = Int($0) ?? 3000 }
                        ),
                        onConnect: { coordinator.reconnect() }
                    )
                    .interactiveDismissDisabled()
                }
        }
    }

    // MARK: - Audio Session

    /// Configure AVAudioSession for simultaneous mic input + speaker output.
    /// Must be called before WhisperKit or AVSpeechSynthesizer are used.
    private static func configureAudioSession() {
        let session = AVAudioSession.sharedInstance()
        do {
            try session.setCategory(
                .playAndRecord,
                mode: .voiceChat,
                options: [.defaultToSpeaker, .allowBluetooth, .mixWithOthers]
            )
            try session.setActive(true)
        } catch {
            print("[AudioSession] Configuration failed: \(error)")
        }
    }
}
