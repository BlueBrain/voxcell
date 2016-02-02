#!/usr/bin/env python
''' install utility for voxcellview ipython extension '''

import argparse
from os.path import dirname, abspath, join
from notebook.nbextensions import install_nbextension # pylint: disable=F0401


def install(user=False, symlink=False, **kwargs):
    '''Install the voxcellview nbextension.

    Args:
      user (bool): Install for current user instead of system-wide.
      symlink (bool): Symlink instead of copy (for development).
      **kwargs (keyword arguments): Other keyword arguments
    passed to the install_nbextension command
    '''
    directory = join(dirname(abspath(__file__)), 'nbextension')
    install_nbextension(directory, destination='voxcellview',
                        symlink=symlink, user=user, **kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Installs the voxcellview widget')
    parser.add_argument('-u', '--user',
                        help='Install as current user instead of system-wide',
                        action='store_true')
    parser.add_argument('-s', '--symlink',
                        help='Symlink instead of copying files',
                        action='store_true')
    parser.add_argument('-f', '--force',
                        help='Overwrite any previously-installed files for this extension',
                        action='store_true')
    args = parser.parse_args()
    install(user=args.user, symlink=args.symlink, overwrite=args.force)
