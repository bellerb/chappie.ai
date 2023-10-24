# Generate distribution files
python3 setup.py sdist bdist_wheel

# Upload to PyPi
twine upload dist/*