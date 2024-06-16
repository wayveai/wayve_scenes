from setuptools import setup, find_packages

setup(
    name='wayve_scenes',
    version='0.3',
    description='Evaluation API for the WayveScenes101 Dataset',
    url='http://wayve.ai/science/wayvescenes101', 
    author='Wayve Technologies Ltd',  
    author_email='hello@wayve.ai', 
    license='MIT',  
    packages=find_packages(),
    
    classifiers=[
        # Choose classifiers from https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.10',
        "Environment :: GPU"
    ],
    keywords='Autonomous Driving, Scene Reconstruction, Gaussian Splatting, Neural Radiance Fields, Benchmark, Evaluation',
    zip_safe=False
)
