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
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
    entry_points={"console_scripts": ["repro=repro.__main__:main"]},
    python_requires=">=3.6",
    include_package_data=True,
    install_requires=[
        "black==21.7b0",
        "datasets==1.9.0",
        "docker==5.0.0",
        "overrides==3.1.0",
        "parameterized==0.8.1",
        "pytest==6.2.4",
        "six==1.16.0",
    ],
)
