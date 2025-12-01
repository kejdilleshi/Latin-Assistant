"""Setup script for SFT Training package."""

from setuptools import setup, find_packages
import os

# Read the README file if it exists
readme_path = os.path.join(os.path.dirname(__file__), "Readme.md")
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="sft-training",
    version="0.1.0",
    author="Kejdi Lleshi",
    author_email="kejdi.lleshi@unil.ch",
    description="A package for Supervised Fine-Tuning with DeepSpeed",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/klleshi/sft-training",
    packages=find_packages(exclude=["scripts", "exsess", "data", "results", "outputs", "wandb"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "deepspeed>=0.9.0",
        "trl>=0.7.0",
        "wandb>=0.15.0",
        "accelerate>=0.20.0",
        "peft>=0.4.0",
        "openai>=1.40.0",
        "pydantic>=2.5.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "sft-train=scripts.train_sft:main",
            "sft-sweep=scripts.run_hyperparameter_sweep:main",
            "sft-build-data=scripts.build_sft_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.jinja", "*.json"],
    },
)
