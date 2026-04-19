"""
Helix - Self-Replicating Code Evolution Engine

Setup script for pip installation.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="helix-evolution",
    version="1.0.0",
    author="moggan1337",
    author_email="",
    description="Biological-inspired genetic programming framework for evolving code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moggan1337/Helix",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
        ],
        "viz": [
            "pillow>=9.0.0",
            "cairosvg>=2.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "helix=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
