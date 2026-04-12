// TranscriptionManager.swift
// Spark Sight — iOS Client
//
// On-device speech-to-text using WhisperKit (base model, 74M params).
// Handles microphone capture, VAD-based utterance boundary detection,
// and transcription. Delivers complete utterances via a callback.

import AVFoundation
import WhisperKit

@MainActor
final class TranscriptionManager: ObservableObject {

    // MARK: - Published State

    @Published private(set) var isListening = false
    @Published private(set) var isModelLoaded = false
    @Published private(set) var currentTranscript = ""

    // MARK: - Callback

    /// Called with each complete transcribed utterance.
    var onUtterance: ((String) -> Void)?

    // MARK: - Private

    private var whisperKit: WhisperKit?
    private var audioEngine: AVAudioEngine?
    private var audioBufferQueue: [Float] = []

    // VAD parameters
    private let silenceThreshold: Float = 0.015    // RMS below this = silence
    private let speechThreshold: Float = 0.025     // RMS above this = speech
    private var isSpeechDetected = false
    private var silenceFrameCount = 0
    private let silenceFramesToEnd = 25            // ~1.5s at 60 buffers/sec
    private let minSpeechFrames = 5                // ~300ms minimum utterance

    private var speechFrameCount = 0

    // MARK: - Setup

    /// Load the WhisperKit "base" model on-device.
    func setup() async {
        do {
            whisperKit = try await WhisperKit(
                model: "base",
                verbose: false,
                prewarm: true
            )
            isModelLoaded = true
            print("[Whisper] Model loaded (base, 74M params)")
        } catch {
            print("[Whisper] Model load failed: \(error)")
        }
    }

    // MARK: - Start / Stop Listening

    func startListening() {
        guard isModelLoaded, !isListening else { return }

        audioEngine = AVAudioEngine()
        guard let audioEngine else { return }

        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)

        // Install a tap to capture microphone audio.
        inputNode.installTap(onBus: 0, bufferSize: 4096, format: format) {
            [weak self] buffer, _ in
            self?.processAudioBuffer(buffer)
        }

        do {
            try audioEngine.start()
            isListening = true
            print("[Whisper] Listening started (sample rate: \(format.sampleRate))")
        } catch {
            print("[Whisper] Audio engine start failed: \(error)")
        }
    }

    func stopListening() {
        audioEngine?.inputNode.removeTap(onBus: 0)
        audioEngine?.stop()
        audioEngine = nil
        isListening = false

        // Flush any pending speech.
        if !audioBufferQueue.isEmpty && speechFrameCount >= minSpeechFrames {
            Task { await transcribeBufferedAudio() }
        }
        audioBufferQueue.removeAll()
    }

    // MARK: - Audio Processing (VAD)

    private func processAudioBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData?[0] else { return }
        let frameCount = Int(buffer.frameLength)

        // Compute RMS energy.
        var sumSquares: Float = 0
        for i in 0..<frameCount {
            let sample = channelData[i]
            sumSquares += sample * sample
        }
        let rms = sqrt(sumSquares / Float(frameCount))

        // Accumulate audio samples.
        let samples = Array(UnsafeBufferPointer(start: channelData, count: frameCount))

        if rms >= speechThreshold {
            // Speech detected.
            if !isSpeechDetected {
                isSpeechDetected = true
                speechFrameCount = 0
                print("[VAD] Speech start (RMS=\(String(format: "%.4f", rms)))")
            }
            silenceFrameCount = 0
            speechFrameCount += 1
            audioBufferQueue.append(contentsOf: samples)
        } else if isSpeechDetected {
            // Silence while recording speech.
            silenceFrameCount += 1
            audioBufferQueue.append(contentsOf: samples)  // Include trailing silence.

            if silenceFrameCount >= silenceFramesToEnd {
                // End of utterance detected.
                print("[VAD] Speech end (\(speechFrameCount) frames)")
                if speechFrameCount >= minSpeechFrames {
                    let audioToTranscribe = audioBufferQueue
                    audioBufferQueue.removeAll()
                    Task { await transcribe(samples: audioToTranscribe) }
                } else {
                    audioBufferQueue.removeAll()
                }
                isSpeechDetected = false
                silenceFrameCount = 0
                speechFrameCount = 0
            }
        }
        // Else: silence and no active speech — discard.
    }

    // MARK: - Transcription

    private func transcribeBufferedAudio() async {
        let samples = audioBufferQueue
        audioBufferQueue.removeAll()
        await transcribe(samples: samples)
    }

    private func transcribe(samples: [Float]) async {
        guard let whisperKit, !samples.isEmpty else { return }

        do {
            // WhisperKit expects 16kHz mono Float32 audio.
            // The input may be at the device's native sample rate (typically 48kHz).
            // Resample if needed.
            let targetRate = 16000
            let inputRate = Int(AVAudioSession.sharedInstance().sampleRate)
            let resampled: [Float]

            if inputRate != targetRate {
                resampled = resample(samples, from: inputRate, to: targetRate)
            } else {
                resampled = samples
            }

            let results = try await whisperKit.transcribe(audioArray: resampled)
            let text = results.compactMap { $0.text }
                .joined(separator: " ")
                .trimmingCharacters(in: .whitespacesAndNewlines)

            guard !text.isEmpty else { return }

            print("[Whisper] Transcript: \(text)")

            await MainActor.run {
                self.currentTranscript = text
                self.onUtterance?(text)
            }
        } catch {
            print("[Whisper] Transcription failed: \(error)")
        }
    }

    // MARK: - Resampling

    /// Simple linear interpolation resampling from one sample rate to another.
    private func resample(_ samples: [Float], from inputRate: Int, to outputRate: Int) -> [Float] {
        let ratio = Double(inputRate) / Double(outputRate)
        let outputLength = Int(Double(samples.count) / ratio)
        var output = [Float](repeating: 0, count: outputLength)

        for i in 0..<outputLength {
            let srcIndex = Double(i) * ratio
            let lower = Int(srcIndex)
            let upper = min(lower + 1, samples.count - 1)
            let fraction = Float(srcIndex - Double(lower))
            output[i] = samples[lower] * (1 - fraction) + samples[upper] * fraction
        }

        return output
    }
}
