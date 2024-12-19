from pathlib import Path
from __version__ import __version__
from setuptools import find_packages, setup

# Get the project root directory
root_dir = Path(__file__).parent

# Add the package directory to the Python path
package_dir = root_dir / "kan"

# Read the requirements from the requirements.txt file
# First try to read from the root directory, if not found, try the package directory
requirements_file = "./requirements.txt"

if requirements_file.exists():
    with open(requirements_file) as fid:
        requirements = [l.strip() for l in fid.readlines()]
else:
    print("Warning: requirements.txt not found. Proceeding without dependencies.")

# Setup configuration
setup(
    name="OpenAudioAPI",
    version=__version__,
    description="A versatile Text-to-Speech and Speech-To-Text API that supports multiple TTS architectures including F5-TTS-MLX, XTTS, and Piper. This API provides high-quality speech synthesis with various voice customization options and audio enhancements.",
    long_description=open(root_dir / "README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author_email="goekdenizguelmez@gmail.com",
    author="Gökdeniz Gülmez",
    url="https://github.com/Goekdeniz-Guelmez/OpenAudioAPI.git",
    # license="Apache-2.0",
    install_requires=requirements,
    packages=find_packages(),
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Operating System :: MacOS :: MacOS X"
    ],
)