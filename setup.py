from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="Hessian_Aware_Sampling_in_High_Dimensions",
    version="1.0.0",
    author="Tanisha Gupta",
    author_email="tanisha.gupta008@gmail.com",
    description="Advanced MCMC sampling using Hessian information for high-dimensional distributions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Tani843/Hessian_Aware_Sampling_in_High_Dimensions",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hessian-benchmark=examples.comprehensive_benchmark:main",
            "hessian-experiment=scripts.run_complete_experiment:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)