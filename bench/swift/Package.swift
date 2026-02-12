// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "mlx-swift-bench",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift-lm/",
            from: "2.30.3"
        ),
    ],
    targets: [
        .executableTarget(
            name: "mlx-swift-bench",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources"
        ),
    ]
)
