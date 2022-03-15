from distutils.core import setup


def readme():
    """Import the README.md Markdown file and try to convert it to RST format."""
    try:
        import pypandoc
        return pypandoc.convert('README.md', 'rst')
    except(IOError, ImportError):
        with open('README.md') as readme_file:
            return readme_file.read()


setup(
    name='CryptoSentimentAnalysis',
    version='0.1',
    description='Sentiment Analysis of Crytocurrency Tweets',
    long_description=readme(),
    classifiers=[
        'Programming Language :: Python :: 3',
    ],
    # Substitute <github_account> with the name of your GitHub account
    url='https://github.com/ilksenece/CryptoSentimentAnalysis',
    author='Ece Ay',  # Substitute your name
    author_email='eceicyuz@outlook.com',  # Substitute your email
    license='XXX',
    packages=['titanic'],
    install_requires=[
        'pypandoc>=1.7.2',
        'numpy>=1.18.5',
        'pandas>=1.4.1',
        'torch>=1.6.0',
        'transformers>=3.5.1',
        'seaborn>=0.11.2',
        'matplotlib>=3.3.4'
    ]
)