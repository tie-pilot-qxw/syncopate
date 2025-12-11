from pathlib import Path
from setuptools import find_packages, setup

ROOT = Path(__file__).parent
README_PATH = ROOT / "README.md"
LONG_DESCRIPTION = (
    README_PATH.read_text(encoding="utf-8")
    if README_PATH.exists()
    else ""
)

MODULE_NAME = "syncopate"
SOURCE_DIR = "syncopate"

packages = [MODULE_NAME]
packages += [
    f"{MODULE_NAME}.{pkg}"
    for pkg in find_packages(where=SOURCE_DIR)
]

setup(
    name="syncopate",
    version="0.1.0",
    description="Compiler for experimenting with fine-grained communication/computation overlap.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown" if LONG_DESCRIPTION else "text/plain",
    author="",
    python_requires=">=3.10",
    packages=packages,
    package_dir={MODULE_NAME: SOURCE_DIR},
    include_package_data=True,
    install_requires=[],
    extras_require={
        "dev": ["pytest>=8.4"],
    },
)
