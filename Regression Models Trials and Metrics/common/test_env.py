"""Based on https://machinelearningmastery.com/machine-learning-in-python-step-by-step/"""
import sys
import importlib

MISSING_MODULE_STR = 'Required module missing: '


def python_version():
    """Test and print python version"""
    try:
        if sys.version_info < (3, 8, 8):
            # Inline with anaconda3:2021.05
            sys.exit('You need Python 3.8.8 or newer')
        else:
            print('Python: {}'.format(sys.version))

    except ImportError as error:
        sys.exit(MISSING_MODULE_STR + str(error))


def module_version(module_str):
    """Print module versions from module string"""
    try:
        module = importlib.import_module(module_str)
        print('{}: {}'.format(module_str, module.__version__))
    except ImportError as error:
        sys.exit(MISSING_MODULE_STR + str(error))


def versions(modules):
    """Print python and modules versions from modules list"""
    print('# Python and modules versions')
    python_version()

    for module in modules:
        module_version(module)

    print('\n')
