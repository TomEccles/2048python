from setuptools import setup, Extension


setup(name='roller', version='1.0',  \
      ext_modules=[Extension('roller', ['Python2048Extension.cpp', 'roller.cpp', 'board.cpp'])])