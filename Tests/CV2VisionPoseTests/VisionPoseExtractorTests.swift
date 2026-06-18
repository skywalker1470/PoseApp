import XCTest
import Foundation
import CoreGraphics
@testable import CV2VisionPose

final class VisionPoseExtractorTests: XCTestCase {
    func testVisionRequestRunsOnBlankImage() throws {
        let width = 128
        let height = 128
        let colorSpace = CGColorSpaceCreateDeviceRGB()

        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: colorSpace,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else {
            XCTFail("Could not create CGContext.")
            return
        }

        context.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))

        guard let image = context.makeImage() else {
            XCTFail("Could not create CGImage.")
            return
        }

        let poses = try VisionPoseExtractor.detect(in: image)

        // A blank image should usually have zero people.
        // This test proves Vision imports and the request runs.
        XCTAssertLessThanOrEqual(poses.count, 4)
    }

    func testVisionPoseOnOptionalPeopleFixture() throws {
        guard let url = Bundle.module.url(
            forResource: "people",
            withExtension: "jpg"
        ) else {
            throw XCTSkip("Add Tests/CV2VisionPoseTests/Fixtures/people.jpg to test real people detection.")
        }

        let data = try Data(contentsOf: url)

        guard
            let provider = CGDataProvider(data: data as CFData),
            let image = CGImage(
                jpegDataProviderSource: provider,
                decode: nil,
                shouldInterpolate: true,
                intent: .defaultIntent
            )
        else {
            XCTFail("Could not load people.jpg.")
            return
        }

        let poses = try VisionPoseExtractor.detect(in: image)

        XCTAssertLessThanOrEqual(poses.count, 4)

        for pose in poses {
            XCTAssertEqual(pose.points.count, 17)
        }

        print("Detected \(poses.count) people.")
    }
}