from setuptools import setup 

setup(
    name='wired_cli',
    version='1.4',
    py_modules=['wired_cli'],
    include_package_data=True,
    install_requires=[
        'click',
        'tqdm',
        'tabulate',
        'gensim',
        'pandas',
        'emoji',
        'spacy',
        'newspaper3k',
        'PyInquirer',
        'pyfiglet'
    ],
    entry_points='''
        [console_scripts]
        wired_cli=wired_cli:cli
    ''',
)