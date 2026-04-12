// EarconPlayer.swift
// Spark Sight — iOS Client
//
// Plays short audio cues (earcons) and haptic feedback for non-visual
// state communication. The acknowledgment chime plays instantly when the
// user finishes speaking, confirming "I heard you" before any response.

import AVFoundation
import UIKit

@MainActor
final class EarconPlayer: ObservableObject {

    // MARK: - Private

    private var acknowledgmentPlayer: AVAudioPlayer?
    private var processingPlayer: AVAudioPlayer?

    // Haptic generators (pre-warmed for zero-latency feedback).
    private let impactLight = UIImpactFeedbackGenerator(style: .light)
    private let impactMedium = UIImpactFeedbackGenerator(style: .medium)
    private let impactHeavy = UIImpactFeedbackGenerator(style: .heavy)
    private let notification = UINotificationFeedbackGenerator()

    // MARK: - Setup

    func setup() {
        // Pre-warm haptic generators.
        impactLight.prepare()
        impactMedium.prepare()
        impactHeavy.prepare()
        notification.prepare()

        // Generate earcon tones programmatically (no bundled audio files needed).
        acknowledgmentPlayer = generateTone(
            frequency: 880,  // A5 — gentle high chime
            duration: 0.15,
            volume: 0.3
        )
        acknowledgmentPlayer?.prepareToPlay()

        processingPlayer = generateTone(
            frequency: 440,  // A4 — subtle processing indicator
            duration: 0.08,
            volume: 0.1
        )
        processingPlayer?.prepareToPlay()
    }

    // MARK: - Earcon Playback

    /// Play a short chime confirming "I heard your speech".
    /// Called immediately when VAD detects end-of-utterance.
    func playAcknowledgment() {
        acknowledgmentPlayer?.currentTime = 0
        acknowledgmentPlayer?.play()
        impactLight.impactOccurred()
    }

    /// Play a subtle tone indicating processing is happening.
    func playProcessing() {
        processingPlayer?.currentTime = 0
        processingPlayer?.play()
    }

    // MARK: - Haptic Feedback

    /// Connection state changed (connected/disconnected).
    func hapticConnectionChange(connected: Bool) {
        if connected {
            notification.notificationOccurred(.success)
        } else {
            notification.notificationOccurred(.warning)
        }
    }

    /// Warning arrived from the active agent.
    func hapticWarning(urgency: String) {
        switch urgency {
        case "critical":
            impactHeavy.impactOccurred()
            // Double-tap haptic for critical warnings.
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                self.impactHeavy.impactOccurred()
            }
        case "high":
            impactHeavy.impactOccurred()
        default:
            impactMedium.impactOccurred()
        }
    }

    /// Mode changed (patrol ↔ goal).
    func hapticModeChange() {
        impactMedium.impactOccurred()
    }

    // MARK: - Tone Generation

    /// Generate a simple sine-wave tone as an AVAudioPlayer.
    private func generateTone(
        frequency: Double,
        duration: Double,
        volume: Float
    ) -> AVAudioPlayer? {
        let sampleRate = 44100.0
        let numSamples = Int(sampleRate * duration)

        var audioData = Data()

        for i in 0..<numSamples {
            let t = Double(i) / sampleRate
            // Sine wave with fade-in/fade-out envelope.
            let envelope: Double
            let fadeLength = 0.02  // 20ms fade
            if t < fadeLength {
                envelope = t / fadeLength
            } else if t > duration - fadeLength {
                envelope = (duration - t) / fadeLength
            } else {
                envelope = 1.0
            }

            let sample = sin(2.0 * .pi * frequency * t) * envelope
            var int16Sample = Int16(sample * Double(Int16.max) * Double(volume))
            audioData.append(Data(bytes: &int16Sample, count: 2))
        }

        // Wrap in a WAV header.
        let wavData = createWAVData(
            pcm: audioData,
            sampleRate: Int(sampleRate),
            channels: 1,
            bitsPerSample: 16
        )

        return try? AVAudioPlayer(data: wavData)
    }

    /// Create a minimal WAV file from raw PCM data.
    private func createWAVData(
        pcm: Data,
        sampleRate: Int,
        channels: Int,
        bitsPerSample: Int
    ) -> Data {
        var data = Data()

        let byteRate = sampleRate * channels * bitsPerSample / 8
        let blockAlign = channels * bitsPerSample / 8
        let dataSize = pcm.count
        let fileSize = 36 + dataSize

        // RIFF header
        data.append(contentsOf: "RIFF".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(fileSize).littleEndian) { Array($0) })
        data.append(contentsOf: "WAVE".utf8)

        // fmt chunk
        data.append(contentsOf: "fmt ".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(16).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(1).littleEndian) { Array($0) })  // PCM
        data.append(contentsOf: withUnsafeBytes(of: UInt16(channels).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(sampleRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt32(byteRate).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(blockAlign).littleEndian) { Array($0) })
        data.append(contentsOf: withUnsafeBytes(of: UInt16(bitsPerSample).littleEndian) { Array($0) })

        // data chunk
        data.append(contentsOf: "data".utf8)
        data.append(contentsOf: withUnsafeBytes(of: UInt32(dataSize).littleEndian) { Array($0) })
        data.append(pcm)

        return data
    }
}
