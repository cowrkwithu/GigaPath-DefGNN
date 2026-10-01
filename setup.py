from setuptools import setup, find_packages

setup(
    name="vetgigagraph",
    version="0.1.0",
    description="Prov-GigaPath x GNN hybrid models for canine cutaneous tumor WSI classification",
    author="VetGigaGraph contributors",
    license="MIT",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
