from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="downstreamtask",
    version="0.1",
    author="Eunsoo Kang",
    author_email="woo569628@gmail.com",
    description="This library is based on simpletransformers and HuggingFace's Transformers library.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eunsour/nlp-downstreamtask/",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "tqdm>=4.47.0",
        "regex",
        "transformers>=4.6.0",
        "datasets",
        "scikit-learn",
        "pandas",
        "tokenizers",
        "wandb>=0.10.32",
        "streamlit",
        "sentencepiece",
    ],
)
