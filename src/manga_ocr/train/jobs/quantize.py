from ._common import placeholder_main


def main() -> None:
    placeholder_main(
        "prune-and-qat",
        "Apply structured pruning and quantization-aware training for ONNX INT8 export.",
    )


if __name__ == "__main__":
    main()
