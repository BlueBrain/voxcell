#!/usr/bin/env bash
set -eu

BASH=/bin/bash
CONDA_URL=https://repo.continuum.io/miniconda/

if [[ "$OSTYPE" == "linux-gnu" ]]; then
  CONDA_URL=$CONDA_URL/Miniconda-latest-Linux-x86_64.sh
elif [[ "$OSTYPE" == "darwin"* ]]; then
  CONDA_URL=$CONDA_URL/Miniconda-latest-MacOSX-x86_64.sh
fi

INSTALL=/tmp/miniconda.sh
if [ ! -f $INSTALL ]; then
    echo Getting Install...
    curl $CONDA_URL > $INSTALL
fi

CONDA_PATH=~/miniconda2
if [ ! -f $CONDA_PATH ]; then
    echo Running Install...
    $BASH $INSTALL -b
fi

echo Creating environment...
$CONDA_PATH/bin/conda create -n brainbuilder jupyter numpy scipy pandas h5py matplotlib

echo Installing BrainBuilder
$CONDA_PATH/envs/brainbuilder/bin/pip install --no-deps -e voxcell
$CONDA_PATH/envs/brainbuilder/bin/pip install --no-deps -e voxcellview
$CONDA_PATH/envs/brainbuilder/bin/pip install --no-deps -e brainbuilder

echo "To activate the environment, do:"
echo "  'source $CONDA_PATH/envs/brainbuilder/bin/activate brainbuilder'"
echo "To directly run the jupyter notebook in the current directory, do:"
echo "  '$CONDA_PATH/envs/brainbuilder/bin/jupyter-notebook'"
