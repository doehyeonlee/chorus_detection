import os

# List of required packages
required_packages = [
    "numpy",
    "chardet",
    "nltk",
    "scipy",
    "fastdtw",
    "gensim",
    "python-Levenshtein", # Levenshtein module
    "sentence-transformers",
    "hyphenate",
#    "re",   # This is part of the Python standard library. No need to install.
#    "functools",  # This is part of the Python standard library. No need to install.
    "pandas",
    # Add any additional packages/modules you need to install here.
]

# Install each package
for package in required_packages:
    os.system(f"pip install {package}")

# Download NLTK datasets
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')