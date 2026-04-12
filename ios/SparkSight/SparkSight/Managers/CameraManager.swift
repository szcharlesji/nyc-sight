// CameraManager.swift
// Spark Sight — iOS Client
//
// Captures camera frames from the rear camera at ~10 FPS, compresses
// to JPEG (quality 0.7), and delivers them via a callback for WebSocket
// transmission to the DGX Spark server.

import AVFoundation
import UIKit

@MainActor
final class CameraManager: NSObject, ObservableObject {

    // MARK: - Published State

    @Published private(set) var isRunning = false
    @Published private(set) var isTorchOn = false

    // MARK: - Callback

    /// Called on a background queue with each JPEG frame (~10 FPS).
    var onFrame: ((Data) -> Void)?

    // MARK: - Private

    private let session = AVCaptureSession()
    private let outputQueue = DispatchQueue(label: "com.sparksight.camera", qos: .userInitiated)
    private var frameCount: UInt64 = 0
    private var device: AVCaptureDevice?

    // CIContext is expensive to create — reuse one instance.
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    // MARK: - Setup & Start

    func startCapture() {
        guard !isRunning else { return }

        session.beginConfiguration()
        session.sessionPreset = .medium  // 640×480

        // Rear wide-angle camera.
        guard let camera = AVCaptureDevice.default(
            .builtInWideAngleCamera, for: .video, position: .back
        ) else {
            print("[Camera] No back camera available")
            return
        }
        device = camera

        // Lock to 30 FPS so we can reliably sample every 3rd frame → ~10 FPS.
        do {
            try camera.lockForConfiguration()
            camera.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
            camera.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
            camera.unlockForConfiguration()
        } catch {
            print("[Camera] Frame rate lock failed: \(error)")
        }

        guard let input = try? AVCaptureDeviceInput(device: camera) else {
            print("[Camera] Could not create input")
            return
        }
        if session.canAddInput(input) {
            session.addInput(input)
        }

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: outputQueue)

        if session.canAddOutput(output) {
            session.addOutput(output)
        }

        session.commitConfiguration()

        // Start on a background thread to avoid blocking the main thread.
        Task.detached(priority: .userInitiated) { [session] in
            session.startRunning()
        }
        isRunning = true
        print("[Camera] Capture started (640×480 @ 30 FPS, sampling every 3rd → ~10 FPS)")
    }

    func stopCapture() {
        guard isRunning else { return }
        Task.detached { [session] in
            session.stopRunning()
        }
        isRunning = false
    }

    // MARK: - Torch Control

    func toggleTorch() {
        guard let device, device.hasTorch else { return }
        do {
            try device.lockForConfiguration()
            let newMode: AVCaptureDevice.TorchMode = device.torchMode == .on ? .off : .on
            device.torchMode = newMode
            device.unlockForConfiguration()
            isTorchOn = newMode == .on
        } catch {
            print("[Camera] Torch toggle failed: \(error)")
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {

    nonisolated func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // Sample every 3rd frame → ~10 FPS from a 30 FPS source.
        let count = OSAtomicIncrement64(unsafeFrameCountPointer)
        guard count % 3 == 0 else { return }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return }

        let uiImage = UIImage(cgImage: cgImage)
        guard let jpegData = uiImage.jpegData(compressionQuality: 0.7) else { return }

        onFrame?(jpegData)
    }

    // Atomic counter for thread-safe frame counting.
    // OSAtomicIncrement64 needs a pointer to Int64.
    nonisolated private var unsafeFrameCountPointer: UnsafeMutablePointer<Int64> {
        // This is a workaround: we use the actor-isolated frameCount
        // by bridging through an unmanaged pointer stored once.
        // In production, use os_unfair_lock or Atomics package.
        // For the hackathon, a simple static suffices.
        struct Static {
            static var counter: Int64 = 0
        }
        return withUnsafeMutablePointer(to: &Static.counter) { $0 }
    }
}
