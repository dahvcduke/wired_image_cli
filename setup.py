from setuptools import setup 

setup(
    name='wired_image_cli',
    version='1.4',
    py_modules=['wired_image_cli'],
    include_package_data=True,
    install_requires=[
        'click',
        'tqdm',
        'cv2',
        'gensim',
        'pandas',
        'emoji',
        'keras',
        'scipy',
        'PyInquirer',
        'pyfiglet',
        'tensorflow',
        'numpy',
        'pickle',
    ],
    entry_points='''
        [console_scripts]
        wired_image_cli=wired_image_clis:cli
    ''',
)