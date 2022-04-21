from setuptools import setup, find_packages
from py_sc_fermi import __version__ as VERSION

readme = "README.md"
long_description = open(readme).read()

config = {
    "description": "working with rocksalts",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "author" : "alexsquires",
    "author_email": "alexsquires@gmail.com",
    "url": "https://github.com/alexsquires/dwayne", 
    "version": VERSION, 
    "python_requires": ">=3.9",
    "license": "MIT",
    "packages": ["dwayne","stanley","tracy"],
    "name": "dwayne",
}

setup(**config)
