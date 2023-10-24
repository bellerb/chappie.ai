from setuptools import setup, find_packages

setup(
    name="chappie_ai",
    version="0.0.1",
    author="Ben Bellerose",
    author_email="benbellerose@gmail.com",
    description="Machine learning agent",
    long_description=open('README.md').read(),  # Read the README file
    long_description_content_type='text/markdown',
    packages=find_packages(),
    exclude_package_data={
        "": ["deploy.sh", "notebooks/*", "examples/*", ".vscode/*"]},
    install_requires=[
        "torch",
        "pandas",
        "numpy",
        "tdqm",
        "einops"
    ],
    keywords=[
        "python",
        "Chappie",
        "MuZero",
        "RETRO"
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ],
    url="https://github.com/bellerb/chappie.ai",
    bugtrack_url="https://github.com/bellerb/chappie.ai/issues",
    license="MIT",
    platforms=["Any"],
    license_file="LICENSE.txt",
)
