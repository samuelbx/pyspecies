import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PySpecies",
    version="0.1.1",
    author="Samuel Boïté, Mathias Grau",
    author_email="boite.samuel@gmail.com",
    description="Blazing-fast simulation of self-organized patterns in reaction-diffusion systems.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/samuel-boite/pyspecies",
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
