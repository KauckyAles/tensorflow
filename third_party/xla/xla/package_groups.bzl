"""XLA package_group definitions"""

def xla_package_groups(name = "xla_package_groups"):
    """Defines visibility groups for XLA.

    Args:
     name: package groups name
    """

    native.package_group(
        name = "friends",
        packages = ["//..."],
    )

    native.package_group(
        name = "internal",
        packages = ["//..."],
    )

    native.package_group(
        name = "backends",
        packages = ["//..."],
    )

    native.package_group(
        name = "codegen",
        packages = ["//..."],
    )

    native.package_group(
        name = "collectives",
        packages = ["//..."],
    )

    native.package_group(
        name = "runtime",
        packages = ["//..."],
    )

    native.package_group(
        name = "restricted",
        packages = [
            "//xla/service/cpu/restricted",
            "//xla/service/cpu/tests/restricted",
            "//xla/service/debug/restricted",
            "//xla/service/llvm_ir/restricted",
            "//xla/service/restricted",
            "//xla/stream_executor/sycl/restricted",
            "//xla/tests/restricted",
            "//xla/tools/hlo_bisect/restricted",
            "//xla/service/gpu",
            "//xla/backends/gpu/tests",
            "//xla/backends/gpu/codegen/triton/tests",
            "//xla/backends/gpu/transforms",
            "//xla/backends/gpu/codegen/triton",
            "//xla/backends/gpu/codegen",
            "//xla/service/gpu/model",
            "//xla/service/gpu/autotuning",
            "//xla/backends/gpu/runtime",
        ],
    )

def xla_test_friend_package_group(name):
    """Defines visibility group for XLA tests.

    Args:
     name: package group name
    """

    native.package_group(
        name = name,
        packages = ["//..."],
    )
