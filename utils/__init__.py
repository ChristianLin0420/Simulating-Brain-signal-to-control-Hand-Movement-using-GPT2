from packaging import version

__version__ = "4.9.0.dev0"


def check_min_version(min_version):
    if version.parse(__version__) < version.parse(min_version):
        if "dev" in min_version:
            error_message = (
                "This example requires a source install from HuggingFace Transformers (see "
                "`https://huggingface.co/transformers/installation.html#installing-from-source`),"
            )
        else:
            error_message = f"This example requires a minimum version of {min_version},"
        error_message += f" but the version found is {__version__}.\n"
        raise ImportError(
            error_message
            + (
                "Check out https://huggingface.co/transformers/examples.html for the examples corresponding to other "
                "versions of HuggingFace Transformers."
            )
        )
