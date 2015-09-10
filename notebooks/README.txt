ipython notebooks to illustrate the Brain Builder Modules

Installation
============

install ipython in your virtual environment:
pip install ipython[notebook]==3.2.1

notes:

you may need to escape the square brackets: \[notebook\]

you may experience a error saying it cannot import path when launching the ipython notebook server. In that case, remove the files path.py and path.pyc that are located in the site-packages directory in your virtualenv. I (JDC) haven't found a better workaround for now.

The morphology viewer requires a ascii symbolic link to the neurolucida files in order to load them.
In viewer/js directory, run for instance:
ln -s /gpfs/bbp.cscs.ch/release/l2/2012.07.23/morphologies/ascii ascii

Starting the server
===================

ipython notebook
----------------
The notebook server will serve the notebooks and the static content that is located in the "viewer" directory.

within the virtualenv, in the "BrainBuilder" directory, launch the ipython server:

ipython notebook --NotebookApp.extra_static_paths="['${PWD}/viewer/']" --ip=put_your_ip_here --port=put_your_port_here



