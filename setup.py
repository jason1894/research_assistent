"""Setup configuration for the research_assistent package."""
from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="research_assistent",
    version="0.1.0",
    author="Research Assistant Contributors",
    author_email="research-assistant@example.com",
    description="An AI-powered research assistant with RAG, LoRA fine-tuning, and multi-model support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/research_assistent",
    project_urls={
        "Bug Tracker": "https://github.com/example/research_assistent/issues",
        "Documentation": "https://github.com/example/research_assistent/wiki",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=find_packages(exclude=["tests*", "scripts*", "data*"]),
    python_requires=">=3.10",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-core>=0.1.0",
        "ollama>=0.1.0",
        "chromadb>=0.4.0",
        "sentence-transformers>=2.2.0",
        "PyMuPDF>=1.23.0",
        "pypdf>=3.0.0",
        "sqlalchemy>=2.0.0",
        "click>=8.1.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm>=4.65.0",
        "requests>=2.31.0",
    ],
    extras_require={
        "lora": [
            "torch>=2.0.0",
            "transformers>=4.35.0",
            "peft>=0.6.0",
            "datasets>=2.14.0",
            "accelerate>=0.24.0",
            "bitsandbytes>=0.41.0",
        ],
        "web": ["streamlit>=1.28.0"],
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "research-assistant=ui.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
