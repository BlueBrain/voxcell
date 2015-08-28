ipython notebooks to illustrate the Brain Builder Modules

Installation
============

install ipython in your virtual environment:
pip install ipython[notebook]==3.2.1

notes:

you may need to escape the square brackets: \[notebook\]

you may experience a error saying it cannot import path when launching the ipython notebook server. In that case, remove the files path.py and path.pyc that are located in the site-packages directory in your virtualenv. I (JDC) haven't found a better workaround for now.


Starting the servers
====================

ipython notebook
----------------
within the virtualenv, in the "BrainBuilder" directory, launch the ipython server:

ipython notebook --ip=put_your_ip_here --port=put_your_port_here

viewers
-------
in the "viewer" directory:
python -m SimpleHTTPServer put_your_viewer_server_port_here

Configuring the notebook
========================
open the BrainBuilderModules.ipynb notebook by navigating to http://your_notebook_ip:your_notebook_port/tree/notebooks
change the display_server variable to "http://your_viewer_ip:your_viewer_port/"
change the output_directory variable.
