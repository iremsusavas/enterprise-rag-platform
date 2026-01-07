"""
Setup script for Enterprise RAG Platform
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="enterprise-rag-platform",
    version="1.0.0",
    description="Production-grade RAG system with multi-index routing and automated quality evaluation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Enterprise RAG Platform Contributors",
    url="https://github.com/yourusername/enterprise-rag-platform",
    packages=find_packages(),
    install_requires=[
        "faiss-cpu>=1.7.4",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pypdf>=4.0.0",
        "python-docx>=1.1.0",
        "beautifulsoup4>=4.12.0",
        "markdown>=3.5.0",
        "sentence-transformers>=2.3.0",
        "torch>=2.0.0",
        "transformers>=4.40.0",
        "accelerate>=0.28.0",
        "streamlit>=1.31.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
        "pydantic>=2.5.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)

