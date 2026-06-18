import Foundation
import AVFoundation
import Vision
import CoreGraphics
import ImageIO
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

// MARK: - Main

let args = CommandLine.arguments
let inputPath  = args.count > 1 ? args[1] : "video.mp4"
let outputPath = args.count > 2 ? args[2] : "output.mp4"

let inputURL  = URL(fileURLWithPath: inputPath)
let outputURL = URL(fileURLWithPath: outputPath)

guard FileManager.default.fileExists(atPath: inputPath) else {
    fputs("Error: input file not found at \(inputPath)\n", stderr)
    exit(1)
}

// Remove existing output
try? FileManager.default.removeItem(at: outputURL)

// MARK: - Read input video

let asset  = AVAsset(url: inputURL)
let reader = try! AVAssetReader(asset: asset)

guard let videoTrack = try! asset.loadTracks(withMediaType: .video).first else {
    fputs("Error: no video track found.\n", stderr)
    exit(1)
}

let trackFPS    = try! videoTrack.load(.nominalFrameRate)
let naturalSize = try! videoTrack.load(.naturalSize)
let transform   = try! videoTrack.load(.preferredTransform)

// Apply track transform to get display size
let displaySize: CGSize = {
    let rect = CGRect(origin: .zero, size: naturalSize).applying(transform)
    return CGSize(width: abs(rect.width), height: abs(rect.height))
}()

let width  = Int(displaySize.width)
let height = Int(displaySize.height)

print("Input: \(inputPath)")
print("Track FPS: \(trackFPS), Display size: \(width)x\(height)")

let readerSettings: [String: Any] = [
    kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
]

let readerOutput = AVAssetReaderTrackOutput(
    track: videoTrack,
    outputSettings: readerSettings
)
reader.add(readerOutput)
reader.startReading()

// MARK: - Write output video at 30 FPS

let writer = try! AVAssetWriter(outputURL: outputURL, fileType: .mp4)

let outputFPS: Int32 = 30
let videoSettings: [String: Any] = [
    AVVideoCodecKey:  AVVideoCodecType.h264,
    AVVideoWidthKey:  width,
    AVVideoHeightKey: height
]

let writerInput = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
writerInput.expectsMediaDataInRealTime = false

let adaptor = AVAssetWriterInputPixelBufferAdaptor(
    assetWriterInput: writerInput,
    sourcePixelBufferAttributes: nil
)

writer.add(writerInput)
writer.startWriting()
writer.startSession(atSourceTime: .zero)

// MARK: - Vision request

let bodyPoseRequest = VNDetectHumanBodyPoseRequest()

// MARK: - Process frames

var frameIndex: Int64 = 0
var totalPeople = 0
let startTime = Date()

print("Processing frames...")

while let sampleBuffer = readerOutput.copyNextSampleBuffer() {
    guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
        frameIndex += 1
        continue
    }

    // Run Vision
    let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
    try? handler.perform([bodyPoseRequest])
    let observations = bodyPoseRequest.results ?? []
    let poses = observations.prefix(4).compactMap { parsePose($0) }
    totalPeople += poses.count

    // Draw annotations
    let annotated = drawPoses(poses, on: pixelBuffer, width: width, height: height)

    // Write frame at 30 FPS
    let pts = CMTime(value: frameIndex, timescale: outputFPS)

    while !writerInput.isReadyForMoreMediaData {
        Thread.sleep(forTimeInterval: 0.01)
    }

    if let annotated = annotated {
        adaptor.append(annotated, withPresentationTime: pts)
    } else {
        adaptor.append(pixelBuffer, withPresentationTime: pts)
    }

    frameIndex += 1

    if frameIndex % 30 == 0 {
        let elapsed = Date().timeIntervalSince(startTime)
        print("Frame \(frameIndex) | elapsed: \(String(format: "%.1f", elapsed))s")
    }
}

// MARK: - Finish writing

writerInput.markAsFinished()
let writeGroup = DispatchGroup()
writeGroup.enter()
writer.finishWriting {
    writeGroup.leave()
}
writeGroup.wait()

let totalTime = Date().timeIntervalSince(startTime)
print("Done.")
print("Frames processed : \(frameIndex)")
print("Total detections : \(totalPeople)")
print("Output           : \(outputPath)")
print("Time             : \(String(format: "%.1f", totalTime))s")

// MARK: - Helpers

func parsePose(_ observation: VNHumanBodyPoseObservation) -> [(String, CGPoint, Float)]? {
    guard let points = try? observation.recognizedPoints(.all) else { return nil }

    let coco17: [(String, VNHumanBodyPoseObservation.JointName)] = [
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

    return coco17.compactMap { name, joint -> (String, CGPoint, Float)? in
        guard let p = points[joint], p.confidence > 0.2 else { return nil }
        return (name, p.location, p.confidence)
    }
}

func drawPoses(
    _ poses: [[(String, CGPoint, Float)]],
    on pixelBuffer: CVPixelBuffer,
    width: Int,
    height: Int
) -> CVPixelBuffer? {
    // Create CGImage from pixel buffer
    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    let ciContext = CIContext()
    guard let cgImage = ciContext.createCGImage(ciImage, from: ciImage.extent) else { return nil }

    guard let context = CGContext(
        data: nil,
        width: width, height: height,
        bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }

    // Draw image right-side up
    context.translateBy(x: 0, y: CGFloat(height))
    context.scaleBy(x: 1, y: -1)
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))

    // Reset transform for keypoints (top-left origin)
    context.scaleBy(x: 1, y: -1)
    context.translateBy(x: 0, y: -CGFloat(height))

    context.setLineWidth(3)
    context.setStrokeColor(CGColor(red: 0, green: 1, blue: 0, alpha: 1))
    context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1))

    for pose in poses {
        // Build point map — Vision bottom-left → top-left
        var pts: [String: CGPoint] = [:]
        for (name, loc, _) in pose {
            pts[name] = CGPoint(
                x: loc.x * CGFloat(width),
                y: (1.0 - loc.y) * CGFloat(height)
            )
        }

        // Bones
        for (a, b) in skeleton {
            guard let p1 = pts[a], let p2 = pts[b] else { continue }
            context.move(to: p1)
            context.addLine(to: p2)
            context.strokePath()
        }

        // Joints
        for pt in pts.values {
            context.fillEllipse(in: CGRect(x: pt.x - 5, y: pt.y - 5, width: 10, height: 10))
        }
    }

    guard let annotatedImage = context.makeImage() else { return nil }

    // Rotate 180 + horizontal flip to correct orientation
    guard let rotCtx = CGContext(
        data: nil,
        width: width, height: height,
        bitsPerComponent: 8, bytesPerRow: 0,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    ) else { return nil }

    rotCtx.translateBy(x: CGFloat(width), y: CGFloat(height))
    rotCtx.rotate(by: .pi)
    rotCtx.translateBy(x: CGFloat(width), y: 0)
    rotCtx.scaleBy(x: -1, y: 1)
    rotCtx.draw(annotatedImage, in: CGRect(x: 0, y: 0, width: width, height: height))

    guard let finalImage = rotCtx.makeImage() else { return nil }

    // Convert back to CVPixelBuffer
    var outBuffer: CVPixelBuffer?
    CVPixelBufferCreate(
        kCFAllocatorDefault,
        width, height,
        kCVPixelFormatType_32BGRA,
        [kCVPixelBufferCGImageCompatibilityKey: true,
         kCVPixelBufferCGBitmapContextCompatibilityKey: true] as CFDictionary,
        &outBuffer
    )

    guard let outBuffer else { return nil }

    CVPixelBufferLockBaseAddress(outBuffer, [])
    let pxCtx = CGContext(
        data: CVPixelBufferGetBaseAddress(outBuffer),
        width: width, height: height,
        bitsPerComponent: 8,
        bytesPerRow: CVPixelBufferGetBytesPerRow(outBuffer),
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
    )
    pxCtx?.draw(finalImage, in: CGRect(x: 0, y: 0, width: width, height: height))
    CVPixelBufferUnlockBaseAddress(outBuffer, [])

    return outBuffer
}
