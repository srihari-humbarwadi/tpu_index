import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='tpu_index',
    version='0.0.4',
    author='Srihari Humbarwadi',
    author_email='sriharihumbarwadi97@gmail.com',
    description='TPU index is a package for fast similarity search over large collections of high dimension vectors on Google Cloud TPUs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/srihari-humbarwadi/tpu_index',
    packages=setuptools.find_packages(),
    install_requires=['tensorflow>=2.0.0'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.5',
)
