"""Setup script for f1_predictor package."""

from setuptools import setup, find_packages

setup(
    name="f1_predictor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "requests>=2.25.0",
        "PyYAML>=6.0",
        "xgboost>=1.5.0",
        "lightgbm>=3.3.0",
        "matplotlib>=3.4.0",
        "joblib>=1.1.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "fastf1": ["fastf1>=2.3.0"],
        "dev": [
            "pytest>=6.0.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "f1-predict=f1_predictor.run_predict:main",
            "f1-train=f1_predictor.run_training:main",
        ],
    },
    python_requires=">=3.8",
    author="Formula 1 Prediction Team",
    author_email="example@example.com",
    description="A modular and extensible system for predicting Formula 1 race outcomes",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/formula1-prediction",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 