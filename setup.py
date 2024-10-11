from setuptools import setup, find_packages

setup(
    name="axcer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        # "spacy",
        # "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz",
    ],
    description="A package for extracting keywords from text using spaCy",
    author="Your Name",
    author_email="pransermi@gmail.com",
    url="https://github.com/itz-amethyst/axcer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
