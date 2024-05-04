from setuptools import setup, find_packages

setup(
    name="colors",
    version="0.1.0",
    description="A package for working with color spaces.",
    author="Yunis Yilmaz",
    author_email="yunis@yilmaz.nyc",
    maintainer="Yunis Yilmaz",
    maintainer_email="yunis@yilmaz.nyc",
    url="https://github.com/yoonthegoon/colors",
    license="GPLv3",
    packages=find_packages(),
    install_requires=[
        "numpy~=1.26.4",
    ],
)
