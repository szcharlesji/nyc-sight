// TTSManager.swift
// Spark Sight — iOS Client
//
// On-device text-to-speech using AVSpeechSynthesizer with priority-based
// interruption. Critical warnings (obstacles, hazards) interrupt any
// current speech immediately. Normal responses queue after current speech.

import AVFoundation

@MainActor
final class TTSManager: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {

    // MARK: - Published State

    @Published private(set) var isSpeaking = false

    // MARK: - Private

    private let synthesizer = AVSpeechSynthesizer()
    private var lastSpokenText = ""
    private var pendingTexts: [(String, MessagePriority)] = []

    // Use the best available voice.
    private var preferredVoice: AVSpeechSynthesisVoice? {
        // Try premium/enhanced voices first, then fall back to default.
        let voices = AVSpeechSynthesisVoice.speechVoices()
            .filter { $0.language.hasPrefix("en-US") }
            .sorted { v1, v2 in
                // Higher quality enum values are better.
                v1.quality.rawValue > v2.quality.rawValue
            }
        return voices.first ?? AVSpeechSynthesisVoice(language: "en-US")
    }

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    // MARK: - Public API

    /// Speak text with the given priority.
    ///
    /// - `.critical`: Interrupts any current speech immediately (warnings).
    /// - `.high` / `.normal`: Queues after current speech.
    func speak(_ text: String, priority: MessagePriority = .normal) {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        if priority == .critical {
            // Interrupt everything for safety warnings.
            synthesizer.stopSpeaking(at: .immediate)
            pendingTexts.removeAll()
            speakImmediately(text)
            return
        }

        if synthesizer.isSpeaking {
            // Queue for later.
            pendingTexts.append((text, priority))
        } else {
            speakImmediately(text)
        }
    }

    /// Speak a chunked/streaming response. Each chunk is queued as it arrives.
    func speakChunk(_ text: String, isFinal: Bool) {
        // Split into sentences for progressive TTS.
        let sentences = text
            .components(separatedBy: CharacterSet(charactersIn: ".!?"))
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }

        for sentence in sentences {
            speak(sentence, priority: .normal)
        }
    }

    /// Re-speak the last response (accessibility: double-tap to repeat).
    func repeatLast() {
        guard !lastSpokenText.isEmpty else {
            speak("Nothing to repeat yet.", priority: .normal)
            return
        }
        speak(lastSpokenText, priority: .normal)
    }

    /// Stop all speech immediately.
    func stop() {
        synthesizer.stopSpeaking(at: .immediate)
        pendingTexts.removeAll()
        isSpeaking = false
    }

    // MARK: - Private

    private func speakImmediately(_ text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = preferredVoice
        utterance.rate = 0.52   // Slightly above default for efficiency
        utterance.pitchMultiplier = 1.0
        utterance.preUtteranceDelay = 0.05
        utterance.postUtteranceDelay = 0.1

        lastSpokenText = text
        isSpeaking = true
        synthesizer.speak(utterance)
    }

    // MARK: - AVSpeechSynthesizerDelegate

    nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didFinish utterance: AVSpeechUtterance
    ) {
        Task { @MainActor in
            // Speak the next queued text, if any.
            if let next = self.pendingTexts.first {
                self.pendingTexts.removeFirst()
                self.speakImmediately(next.0)
            } else {
                self.isSpeaking = false
            }
        }
    }

    nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didCancel utterance: AVSpeechUtterance
    ) {
        Task { @MainActor in
            self.isSpeaking = !self.pendingTexts.isEmpty
        }
    }
}
