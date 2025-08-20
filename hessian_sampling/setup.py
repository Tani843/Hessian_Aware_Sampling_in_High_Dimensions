"""
Setup script for Hessian Aware Sampling in High Dimensions package.
"""

from setuptools import setup, find_packages
import os


def read_requirements():
    """Read requirements from requirements.txt file."""
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    try:
        with open(req_path, 'r') as f:
            lines = f.readlines()
        
        # Filter out comments and empty lines
        requirements = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Extract package name (before any version specifiers)
                pkg_name = line.split('>=')[0].split('==')[0].split('<')[0]
                requirements.append(line)
        
        return requirements
    except FileNotFoundError:
        return []


def read_long_description():
    """Read long description from README.md."""
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Hessian Aware Sampling in High Dimensions - A package for efficient MCMC sampling using Hessian information."


# Core requirements (essential for basic functionality)
core_requirements = [
    'numpy>=1.20.0',
    'scipy>=1.7.0',
    'matplotlib>=3.3.0'
]

# Optional requirements grouped by functionality
optional_requirements = {
    'autodiff': [
        'torch>=1.9.0',
        'jax>=0.3.0',
        'jaxlib>=0.3.0'
    ],
    'visualization': [
        'seaborn>=0.11.0'
    ],
    'dev': [
        'pytest>=6.0.0',
        'pytest-cov>=2.12.0',
        'black>=21.0.0',
        'flake8>=3.9.0',
        'mypy>=0.910'
    ],
    'docs': [
        'sphinx>=4.0.0',
        'sphinx-rtd-theme>=1.0.0',
        'nbsphinx>=0.8.0'
    ],
    'profiling': [
        'line-profiler>=3.3.0',
        'memory-profiler>=0.60.0'
    ],
    'data': [
        'pandas>=1.3.0',
        'h5py>=3.1.0'
    ]
}

# All optional requirements
all_optional = []
for deps in optional_requirements.values():
    all_optional.extend(deps)

setup(
    name='hessian-sampling',
    version='0.1.0',
    author='Hessian Sampling Team',
    author_email='hessian-sampling@example.com',
    description='Efficient MCMC sampling using Hessian information for high-dimensional spaces',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-username/hessian-sampling',
    
    # Package configuration
    packages=find_packages(exclude=['tests*', 'examples*', 'docs*']),
    package_dir={'': '.'},
    python_requires='>=3.7',
    
    # Dependencies
    install_requires=core_requirements,
    extras_require={
        **optional_requirements,
        'all': all_optional
    },
    
    # Entry points
    entry_points={
        'console_scripts': [
            'hessian-sampling=hessian_sampling.cli:main',
        ],
    },
    
    # Package metadata
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    
    keywords=[
        'mcmc', 'sampling', 'hessian', 'bayesian', 'statistics',
        'high-dimensional', 'langevin', 'monte-carlo'
    ],
    
    # Additional metadata
    project_urls={
        'Bug Reports': 'https://github.com/your-username/hessian-sampling/issues',
        'Source': 'https://github.com/your-username/hessian-sampling',
        'Documentation': 'https://hessian-sampling.readthedocs.io/',
    },
    
    # Include additional files
    include_package_data=True,
    package_data={
        'hessian_sampling': [
            'data/*.json',
            'examples/*.py',
            'examples/*.ipynb'
        ],
    },
    
    # Test configuration
    test_suite='tests',
    tests_require=[
        'pytest>=6.0.0',
        'pytest-cov>=2.12.0',
        'numpy>=1.20.0',
        'scipy>=1.7.0'
    ],
    
    # Zip safety
    zip_safe=False,
    
    # License
    license='MIT',
    
    # Platform compatibility
    platforms=['any'],
)