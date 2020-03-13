import setuptools
import pdb
stop = pdb.set_trace

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pychell_rvs",
    version="2.0.0",
    author="Bryson Cale",
    author_email="bryson.cale1@gmail.com",
    description="Extract radial velocities from echelle spectra.",
    longdescription=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    include_package_data=True,
    url="https://github.com/astrobc1/pychell_rvs",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: GPLv3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0'
)