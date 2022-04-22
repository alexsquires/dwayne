from setuptools import setup, find_packages

readme = "README.md"
long_description = open(readme).read()

config = {
    "description": "working with rocksalts",
    "long_description": long_description,
    "long_description_content_type": "text/markdown",
    "author" : "alexsquires",
    "author_email": "alexsquires@gmail.com",
    "url": "https://github.com/alexsquires/dwayne", 
    "version": "0.0.1", 
    "python_requires": ">=3.9",
    "license": "MIT",
    "packages": ["dwayne","stanley","tracy"],
    "name": "dwayne",
}

setup(**config)
