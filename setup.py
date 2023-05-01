from distutils.core import setup
import os
from setuptools import find_packages

# User-friendly description from README.md
current_directory = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(current_directory, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except Exception:
    long_description = ''

setup(
	# Name of the package 
	name='VDBforGenAI',
	# Packages to include into the distribution 
	packages=find_packages('.'),
	# Start with a small number and increase it with 
	# every change you make https://semver.org 
	version='0.2',
	# Chose a license from here: https: // 
	# help.github.com / articles / licensing - a - 
	# repository. For example: MIT 
	license='',
	# Short description of your library 
	description='A simple package for generating and querying Vector Databases for Generative AI as well any other reason',
	# Long description of your library 
	long_description=long_description,
	long_description_content_type='text/markdown',
	# Your name 
	author='Jakub Dolezal',
	# Your email 
	author_email='jakubdolezal93@gmail.com',
	# Either the link to your github or to your website 
	url='https://github.com/JakubJDolezal/VDBforGenAI',
	# List of keywords 
	keywords=[],
	# List of packages to install with this one 
	install_requires=[
		"faiss-cpu",
        "transformers",
        "torch",
        "numpy","PyPDF2",'docx','python-docx'],
	# https://pypi.org/classifiers/ 
	classifiers=[ "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"],
python_requires = ">=3.6",
)
