from setuptools import setup, find_packages

setup(
    name='llmftax',
    version='0.1.0',
    description='Fine Tuing for Causal Inference in Jax',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/pharringtonp19/llmftax',  # URL to the repository
    author='Patrick Power',
    author_email='pharringtonp19@gmail.com',
    license='MIT',  # Or whatever license you choose
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here, e.g.,
        # 'numpy',
        # 'pandas',
    ]
)
