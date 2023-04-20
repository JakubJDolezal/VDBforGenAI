import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="VDBforGenAI",
    version="0.1",
    author="Jakub Dolezal",
    author_email="jakubdolezal93@gmail.com",
    description="A package for generating and querying Vector Databases for Genomic Data using AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JakubJDolezal/VDBforGenAI",
    packages=setuptools.find_packages(),
    install_requires=[
        "faiss",
        "transformers",
        "torch",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)