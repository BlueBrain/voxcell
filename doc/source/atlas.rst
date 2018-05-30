.. |name| replace:: ``voxcell``

Atlas access
============

Each VoxelBrain atlas is a collection of volumetric datasets defining some values in space + definition of brain regions hierarchy.

VoxelBrain provides `REST API <https://bbpteam.epfl.ch/project/spaces/display/NRINF/Voxel+Brain+REST+API>`_ for fetching and querying these data.

Here is an example of such an atlas for Hippocampus:

http://voxels.nexus.apps.bbp.epfl.ch/api/analytics/atlas/releases/9B1F97DD-13B8-4FCF-B9B1-59E4EBE4B5D8/

Let us describe how to access these data using |name|.


Volumetric datasets
~~~~~~~~~~~~~~~~~~~

Volumetric datasets are stored in VoxelBrain in `NRRD <http://teem.sourceforge.net/nrrd/format.html>`_ format, which is basically a grid of values for each voxel + some metadata defining voxel size and location in the global atlas space.

The values stored in volumetric datasets could be `scalar <https://bbpteam.epfl.ch/project/spaces/display/NRINF/Scalar+Value+Image>`_ (e.g., brain region ID, cell density); as well as vector (e.g., morphology `orientation field <https://bbpteam.epfl.ch/project/spaces/display/NRINF/Orientation+Field>`_).

For the atlas mentioned above, NRRDs for each volumetric dataset available could be fetched from `here <http://voxels.nexus.apps.bbp.epfl.ch/api/analytics/atlas/releases/9B1F97DD-13B8-4FCF-B9B1-59E4EBE4B5D8/data>`_.

|name| provides ``VoxelData`` class for transparent access to these files.

.. code-block:: python

    >> from voxcell import VoxelData
    >> vd = VoxelData.load_nrrd(<path-to-NRRD>)

Under the hood ``VoxelData`` is a combination of:

 **raw**
    | :math:`(N + K)`-dimensional NumPy array, where first :math:`N` dimensions are spatial ones; and last :math:`K` -- payload ones.
    | In our case :math:`N = 3`; and `K` ranges from :math:`0` for scalar values to :math:`2` for rotation matrices.

 **voxel_dimensions**
    | 1D NumPy array of length :math:`N` defining sides of each voxel.
    | NB: despite the naming, the values here could be negative, since they encode also axis *direction*.

 **offset**
    1D NumPy array of length :math:`N` defining XYZ-position of the "lower left" corner of voxel :math:`(0,..., 0)`

Once ``VoxelData`` is instantiated, the most typical operation is querying values for given XYZ-position(s):

.. code-block:: python

    # Lookup value for a single position
    >> value = vd.lookup([1., 2., 3.])

    # Lookup value for an array of positions
    >> values = vd.lookup([[1., 2., 3], [4., 5. 6.]])

If some of the coordinates are outside of volumetric dataset domain, ``VoxcellError`` would be raised.
Alternatively, one can provide ``outer_value`` optional argument to return a stub value for such coordinates:

.. code-block:: python

    >> vd.lookup(xyz, outer_value=-1)


TODO: document other methods


OrientationField
^^^^^^^^^^^^^^^^

|name| provides also a subclass of ``VoxelData`` for transparently converting quaternions stored in orientation fields to rotation matrices form.

Thus, given an NRRD of the specific `format <https://bbpteam.epfl.ch/project/spaces/display/NRINF/Orientation+Field>`_, one can:

.. code-block:: python

    >> from voxcell import OrientationField
    >> of = OrientationField.load_nrrd(<path-to-NRRD>)
    >> of.lookup(xyz)  # returns `xyz.shape` x 3 x 3 array with orientation matrices


Brain region hierarchy
~~~~~~~~~~~~~~~~~~~~~~

Brain region IDs are organized into hierarchy (for instance, ``CA`` region in Hippocampus consists of ``CA1``, ``CA2`` and ``CA3``).

This hierarchy, along with brain region names corresponding to IDs, is stored in a JSON file of the following form (full example could be found `here <http://voxels.nexus.apps.bbp.epfl.ch/api/analytics/atlas/releases/9B1F97DD-13B8-4FCF-B9B1-59E4EBE4B5D8/filters/brain_region/65535>`_):

.. code-block:: console

    {
        "id" : 103,
        "name" : "CA1a",
        "children" : [ {
          "id" : 1,
          "name" : "SLM"
        }, {
          "id" : 8,
          "name" : "SR"
        }, {
          "id" : 15,
          "name" : "SP"
        }
    }


|name| provides ``Hierarchy`` class for transparent access to these files.

.. code-block:: python

    hierarchy = Hierarchy.load_json(<path-to-JSON>)

Each element stored in ``Hierarchy`` stores the attributes from corresponding JSON part.

The most used method for ``Hierarchy`` is *collecting* some attribute, given the value of another one.
Note that it returns a set of values for the region(s) matching the searched attribute, *and all of their children recursively*. For instance, collecting region name(s) for a given ID:

.. code-block:: python

    >> hierarchy.collect('id', 101, 'name')
    {u'CA2', u'SLM', u'SO', u'SP', u'SR'}

or the other way around (region IDs for given region name):

.. code-block:: python

    >> hierarchy.collect('name', 'SLM', 'id')
    {1, 2, 3, 4, 5, 6, 7}

.. note::

    We consider changing this interface.
    For instance, sometimes it might make more sense to return a list of the attributes for the given region and its *parents*, rather than *children*.
    Please let us know your opinion by dropping an `email <mailto: bbp-ou-nse@groupes.epfl.ch>`_.


Fetching data
~~~~~~~~~~~~~

When working with a VoxelBrain atlas, there is no need to instanstiate ``VoxelData`` and ``Hierarchy`` directly.

|name| provides ``Atlas`` class to transparently fetch data from VoxelBrain in the form of ``VoxelData`` and ``Hierarchy`` objects.

For instance,

.. code-block:: python

    >> from voxcell.nexus.voxelbrain import Atlas
    >> atlas = Atlas.open(
        'http://voxels.nexus.apps.bbp.epfl.ch/api/analytics/atlas/releases/568F4549-82D2-464F-9844-C163FA0C8F8A',
        cache_dir='.'
       )

``cache_dir`` specifies where NRRD and JSON files would be stored. Once they are fetched, they would be reused later without redownloading again. A subfolder with atlas ID (for instance, ``568F4549-82D2-464F-9844-C163FA0C8F8A``) would be created in ``cache_dir``.

.. note::

    At the moment the caching is implemented in a naive way.
    We assume that each VoxelBrain atlas is immutable; and thus once some dataset is fetched and stored locally, we won't check for updates or invalidate the cache.
    To invalidate the cache manually, just remove the corresponding the folder with atlas ID from the cache folder.

By checking the `list <http://voxels.nexus.apps.bbp.epfl.ch/api/analytics/atlas/releases/568F4549-82D2-464F-9844-C163FA0C8F8A>`_ stored for this atlas, we can see that there are `brain_regions`, `longitude` and `orientation`.

We can load any of those with:

.. code-block:: python

    >> brain_regions = atlas.load_data('brain_regions')
    >> longitude = atlas.load_data('longitude')

as well as brain region hierarchy:

.. code-block:: python

    >> hierarchy = atlas.load_hierarchy()

By default, ``VoxelData`` class is used for loading NRRD. To change it to ``OrientationField``, please specify it with:

.. code-block:: python

    >> from voxcell import OrientationField
    >> orientation = atlas.load_data('orientation', cls=OrientationField)

Locally-stored atlas
^^^^^^^^^^^^^^^^^^^^

For development purposes one can use a locally-stored "atlas", which is simply a folder with a collection of NRRD files + JSON file with brain region hierarchy.

For instance:

.. code-block:: console

    $ ls -1 /gpfs/bbp.cscs.ch/project/proj67/entities/dev/atlas/O1-230/

    astrocytes.nrrd
    brain_regions.nrrd
    hierarchy.json
    orientation.nrrd

In this case there is no need to specify ``cache-dir`` when instantiating ``Atlas``:

.. code-block:: python

    >> from voxcell.nexus.voxelbrain import Atlas

    >> atlas = Atlas.open('/gpfs/bbp.cscs.ch/project/proj67/entities/dev/atlas/O1-230/')

    >> hierarchy = atlas.load_hierarchy()
    >> brain_regions = atlas.load_data('brain_regions')
