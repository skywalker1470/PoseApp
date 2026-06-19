import Foundation
import AVFoundation
import Vision
import CoreGraphics
import CoreImage

// MARK: - Skeleton

private let skeleton: [(String, String)] = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
    ("nose",           "left_eye"),
    ("nose",           "right_eye"),
    ("left_eye",       "left_ear"),
    ("right_eye",      "right_ear")
]

private let coco17: [(String, VNHumanBodyPoseObservation.JointName)] = [
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

// MARK: - Pipeline

func runPipeline() async throws {
    let args       = CommandLine.arguments
    let inputPath  = args.count > 1 ? args[1] : "video.mp4"
    let outputPath = args.count > 2 ? args[2] : "output.mp4"

    guard FileManager.default.fileExists(atPath: inputPath) else {
        fputs("Error: input file not found at \(inputPath)\n", stderr); exit(1)
    }
    try? FileManager.default.removeItem(at: URL(fileURLWithPath: outputPath))

    // --- Reader ---
    let asset  = AVAsset(url: URL(fileURLWithPath: inputPath))
    let reader = try AVAssetReader(asset: asset)

    guard let track = try await asset.loadTracks(withMediaType: .video).first else {
        fputs("Error: no video track.\n", stderr); exit(1)
    }

    let trackFPS    = try await track.load(.nominalFrameRate)
    let naturalSize = try await track.load(.naturalSize)
    let transform   = try await track.load(.preferredTransform)

    let displaySize = CGRect(origin: .zero, size: naturalSize).applying(transform)
    let width  = Int(abs(displaySize.width))
    let height = Int(abs(displaySize.height))

    print("Input: \(inputPath)  FPS: \(trackFPS)  Size: \(width)x\(height)")

    let readerOut = AVAssetReaderTrackOutput(track: track, outputSettings: [
        kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
    ])
    reader.add(readerOut)
    reader.startReading()

    // --- Writer at 30 FPS ---
    let outputFPS: Int32 = 30
    let writer = try AVAssetWriter(
        outputURL: URL(fileURLWithPath: outputPath), fileType: .mp4
    )
    let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
        AVVideoCodecKey:  AVVideoCodecType.h264,
        AVVideoWidthKey:  width,
        AVVideoHeightKey: height
    ])
    writerInput.expectsMediaDataInRealTime = false

    let adaptor = AVAssetWriterInputPixelBufferAdaptor(
        assetWriterInput: writerInput, sourcePixelBufferAttributes: nil
    )
    writer.add(writerInput)
    writer.startWriting()
    writer.startSession(atSourceTime: .zero)

    // --- Vision ---
    let poseRequest = VNDetectHumanBodyPoseRequest()

    // --- Process ---
    var frameIndex: Int64 = 0
    var totalPeople = 0
    let t0 = Date()
    print("Processing...")

    while let sample = readerOut.copyNextSampleBuffer() {
        guard let pb = CMSampleBufferGetImageBuffer(sample) else { frameIndex += 1; continue }

        let handler = VNImageRequestHandler(cvPixelBuffer: pb, orientation: .up, options: [:])
        try? handler.perform([poseRequest])
        let poses = (poseRequest.results ?? []).prefix(4).compactMap { parsePose($0) }
        totalPeople += poses.count

        let outPB = annotate(poses, pixelBuffer: pb, width: width, height: height) ?? pb
        let pts   = CMTime(value: frameIndex, timescale: outputFPS)

        while !writerInput.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.005) }
        adaptor.append(outPB, withPresentationTime: pts)

        frameIndex += 1
        if frameIndex % 60 == 0 {
            print("  frame \(frameIndex)  elapsed \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")
        }
    }

    writerInput.markAsFinished()
    await writer.finishWriting()

    print("Done — \(frameIndex) frames, \(totalPeople) detections, \(String(format: "%.1f", Date().timeIntervalSince(t0)))s")
    print("Output: \(outputPath)")
}

// MARK: - Helpers

func parsePose(_ obs: VNHumanBodyPoseObservation) -> [(String, CGPoint)]? {
    guard let pts = try? obs.recognizedPoints(.all) else { return nil }
    let result = coco17.compactMap { name, joint -> (String, CGPoint)? in
        guard let p = pts[joint], p.confidence > 0.2 else { return nil }
        return (name, p.location)
    }
    return result.isEmpty ? nil : result
}

func annotate(_ poses: [[(String, CGPoint)]], pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
    let ci = CIImage(cvPixelBuffer: pixelBuffer)
    guard let cg = CIContext().createCGImage(ci, from: ci.extent) else { return nil }

    guard let ctx = CGContext(
        data: nil, width: width, height: height,
        bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }

    // Draw image (flip so pixels are right-side up in CGContext top-left space)
    ctx.translateBy(x: 0, y: CGFloat(height)); ctx.scaleBy(x: 1, y: -1)
    ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
    ctx.scaleBy(x: 1, y: -1); ctx.translateBy(x: 0, y: -CGFloat(height))

    ctx.setLineWidth(3)
    ctx.setStrokeColor(CGColor(red: 0, green: 1, blue: 0, alpha: 1))
    ctx.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))

    for pose in poses {
        var map: [String: CGPoint] = [:]
        for (name, loc) in pose {
            map[name] = CGPoint(x: loc.x * CGFloat(width), y: (1 - loc.y) * CGFloat(height))
        }
        for (a, b) in skeleton {
            guard let p1 = map[a], let p2 = map[b] else { continue }
            ctx.move(to: p1); ctx.addLine(to: p2); ctx.strokePath()
        }
        for pt in map.values {
            ctx.fillEllipse(in: CGRect(x: pt.x-5, y: pt.y-5, width: 10, height: 10))
        }
    }

    guard let composed = ctx.makeImage() else { return nil }

    // Rotate 180 + horizontal flip to correct final orientation
    guard let rotCtx = CGContext(
        data: nil, width: width, height: height,
        bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }
    rotCtx.translateBy(x: CGFloat(width), y: CGFloat(height))
    rotCtx.rotate(by: .pi)
    rotCtx.translateBy(x: CGFloat(width), y: 0)
    rotCtx.scaleBy(x: -1, y: 1)
    rotCtx.draw(composed, in: CGRect(x: 0, y: 0, width: width, height: height))
    guard let final = rotCtx.makeImage() else { return nil }

    // Back to CVPixelBuffer
    var out: CVPixelBuffer?
    CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
        [kCVPixelBufferCGImageCompatibilityKey: true,
         kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary, &out)
    guard let out else { return nil }
    CVPixelBufferLockBaseAddress(out, [])
    CGContext(
        data: CVPixelBufferGetBaseAddress(out), width: width, height: height,
        bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(out),
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
    )?.draw(final, in: CGRect(x: 0, y: 0, width: width, height: height))
    CVPixelBufferUnlockBaseAddress(out, [])
    return out
}

// MARK: - Entry point

let sema = DispatchSemaphore(value: 0)
Task {
    do { try await runPipeline() }
    catch { fputs("Fatal: \(error)\n", stderr); exit(1) }
    sema.signal()
}
sema.wait()
