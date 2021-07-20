import setuptools


# version.py defines the VERSION variable.
# We use exec here so we don't import repro whilst setting up.
VERSION = {}
with open("repro/version.py", "r") as version_file:
    exec(version_file.read(), VERSION)


setuptools.setup(
    name="repro",
    version=VERSION["VERSION"],
    author="Daniel Deutsch",
    description="An open-source library for reproducing results from research papers",
    url="https://github.com/danieldeutsch/repro",
    packages=setuptools.find_packages(),
    entry_points={"console_scripts": ["repro=repro.__main__:main"]},
    python_requires=">=3.6",
    install_requires=[
        "black==21.7b0",
        "pytest==6.2.4",
    ],
)
