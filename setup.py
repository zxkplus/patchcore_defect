"""PatchCore Defect 安装配置"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() 
        for line in fh 
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="patchcore-defect",
    version="1.0.0",
    author="PatchCore Defect Team",
    author_email="contact@example.com",
    description="基于 PatchCore 的缺陷检测与分类系统，支持分割和分类双重功能",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/patchcore-defect",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.0",
            "torch>=1.9.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "patchcore-train=patchcore_defect.bin.run_patchcore:main",
            "patchcore-train-defect=patchcore_defect.bin.run_patchcore_defect:main",
        ],
    },
)
