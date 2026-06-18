import AVFoundation
import Vision
import CoreImage
import UIKit

// MARK: - Data models

struct PosePoint {
    let name: String
    let x: CGFloat
    let y: CGFloat
    let confidence: Float
}

struct PersonPose {
    let points: [PosePoint]
}

// MARK: - CameraManager

final class CameraManager: NSObject, ObservableObject {

    @Published var poses: [PersonPose] = []
    @Published var previewLayer: AVCaptureVideoPreviewLayer?
    @Published var fps: Double = 0

    private let session = AVCaptureSession()
    private let videoOutput = AVCaptureVideoDataOutput()
    private let processingQueue = DispatchQueue(label: "pose.processing", qos: .userInteractive)

    // FPS tracking
    private var frameCount = 0
    private var fpsTimer: Timer?
    private var lastFPSUpdate = Date()

    // Vision request — reused every frame
    private let bodyPoseRequest = VNDetectHumanBodyPoseRequest()

    private static let coco17Joints: [(String, VNHumanBodyPoseObservation.JointName)] = [
        ("nose",           .nose),
        ("left_eye",       .leftEye),
        ("right_eye",      .rightEye),
        ("left_ear",       .leftEar),
        ("right_ear",      .rightEar),
        ("left_shoulder",  .leftShoulder),
        ("right_shoulder", .rightShoulder),
        ("left_elbow",     .leftElbow),
        ("right_elbow",    .rightElbow),
        ("left_wrist",     .leftWrist),
        ("right_wrist",    .rightWrist),
        ("left_hip",       .leftHip),
        ("right_hip",      .rightHip),
        ("left_knee",      .leftKnee),
        ("right_knee",     .rightKnee),
        ("left_ankle",     .leftAnkle),
        ("right_ankle",    .rightAnkle)
    ]

    override init() {
        super.init()
        setupSession()
        startFPSTimer()
    }

    // MARK: - Setup

    private func setupSession() {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        // Input — front camera
        guard
            let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front),
            let input  = try? AVCaptureDeviceInput(device: device),
            session.canAddInput(input)
        else {
            print("Camera unavailable.")
            session.commitConfiguration()
            return
        }

        // Lock at 30 FPS
        try? device.lockForConfiguration()
        device.activeVideoMinFrameDuration = CMTime(value: 1, timescale: 30)
        device.activeVideoMaxFrameDuration = CMTime(value: 1, timescale: 30)
        device.unlockForConfiguration()

        session.addInput(input)

        // Output
        videoOutput.setSampleBufferDelegate(self, queue: processingQueue)
        videoOutput.alwaysDiscardsLateVideoFrames = true

        guard session.canAddOutput(videoOutput) else {
            session.commitConfiguration()
            return
        }
        session.addOutput(videoOutput)

        // Match preview orientation
        if let connection = videoOutput.connection(with: .video) {
            connection.videoRotationAngle = 90
            if connection.isVideoMirroringSupported {
                connection.isVideoMirrored = true
            }
        }

        session.commitConfiguration()

        // Preview layer
        let layer = AVCaptureVideoPreviewLayer(session: session)
        layer.videoGravity = .resizeAspectFill
        DispatchQueue.main.async { self.previewLayer = layer }
    }

    // MARK: - Session control

    func start() {
        guard !session.isRunning else { return }
        processingQueue.async { self.session.startRunning() }
    }

    func stop() {
        guard session.isRunning else { return }
        processingQueue.async { self.session.stopRunning() }
    }

    // MARK: - FPS counter

    private func startFPSTimer() {
        fpsTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            let now = Date()
            let elapsed = now.timeIntervalSince(self.lastFPSUpdate)
            DispatchQueue.main.async {
                self.fps = Double(self.frameCount) / elapsed
            }
            self.frameCount = 0
            self.lastFPSUpdate = now
        }
    }

    // MARK: - Vision inference

    private func detectPoses(in sampleBuffer: CMSampleBuffer) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])

        do {
            try handler.perform([bodyPoseRequest])
        } catch {
            print("Vision error: \(error)")
            return
        }

        let observations = bodyPoseRequest.results ?? []
        let detected = observations.prefix(4).compactMap { parsePose($0) }

        DispatchQueue.main.async {
            self.poses = detected
        }
    }

    private func parsePose(_ observation: VNHumanBodyPoseObservation) -> PersonPose? {
        guard let recognizedPoints = try? observation.recognizedPoints(.all) else { return nil }

        let points = Self.coco17Joints.compactMap { name, joint -> PosePoint? in
            guard
                let point = recognizedPoints[joint],
                point.confidence > 0.2
            else { return nil }

            return PosePoint(
                name: name,
                x: point.location.x,
                y: point.location.y,
                confidence: point.confidence
            )
        }

        return PersonPose(points: points)
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        frameCount += 1
        detectPoses(in: sampleBuffer)
    }
}
