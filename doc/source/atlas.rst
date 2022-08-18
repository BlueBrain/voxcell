.. |name| replace:: ``voxcell``

Atlas access
============

Each VoxelBrain atlas is a collection of volumetric datasets defining some values in space + definition of brain regions hierarchy.

Let us describe how to access these data using |name|.


Volumetric datasets
~~~~~~~~~~~~~~~~~~~

Volumetric datasets are stored in VoxelBrain in `NRRD <http://teem.sourceforge.net/nrrd/format.html>`_ format, which is basically a grid of values for each voxel + some metadata defining voxel size and location in the global atlas space.

The values stored in volumetric datasets could be :ref:`Scalar Image File Format` (e.g., brain region ID, cell density); as well as vector (e.g., morphology orientation `OrientationField`_).

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


.. todo::

   document other methods


OrientationField
^^^^^^^^^^^^^^^^

|name| provides this subclass of ``VoxelData`` for transparently converting quaternions stored in orientation fields to rotation matrices form.

Thus, given a NRRD in the :ref:`Orientation Field File Format`, one can:

.. code-block:: python

    >> from voxcell import OrientationField
    >> of = OrientationField.load_nrrd(<path-to-NRRD>)
    >> of.lookup(xyz)  # returns `xyz.shape` x 3 x 3 array with orientation matrices


ROIMask
^^^^^^^

|name| provides this subclass of ``VoxelData`` for transparently loading masks and converting values from ``int8`` or ``uint8`` to ``bool``.

Thus, given a NRRD in the specific :ref:`Mask Image for Region of Interest (ROI)`, one can:

.. code-block:: python

    >> from voxcell import ROIMask
    >> data = ROIMask.load_nrrd(<path-to-NRRD>)
    >> data.lookup(xyz)  # return array with boolean values


Brain region hierarchy
~~~~~~~~~~~~~~~~~~~~~~

Brain region IDs are organized into hierarchy (for instance, ``CA`` region in Hippocampus consists of ``CA1``, ``CA2`` and ``CA3``).

This hierarchy, along with different fields for each region, is stored in a JSON file of the
following form (full example could be found `here <http://api.brain-map.org/api/v2/structure_graph_download/1.json>`_):

.. code-block:: console

    {
        "id" : 382,
        "acronym" : "CA1a",
        "name": "Field CA1",
        "children" : [ {
          "id" : 391,
          "acronym" : "CA1slm"
          "name": "Field CA1, stratum lacunosum-moleculare"
        }, {
          "id" : 399,
          "acronym" : "CA1so"
          "name": "Field CA1, stratum oriens"
        }, {
          "id" : 407,
          "acronym" : "CA1sp"
          "name": "Field CA1, pyramidal layer",
        },{
          "id" : 415,
          "acronym" : "CA1sr",
          "name": "Field CA1, stratum radiatum"
        }
    }


|name| provides the ``RegionMap`` class for transparent access to these files.

.. note::
    This interface replaces the historical ``Hierarchy`` which has been removed
    in the |name| version 3.0.0.

One can use the ``load_json`` method to load hierarchy file and instantiate a ``RegionMap`` object.

.. code-block:: python

    from voxcell import RegionMap
    region_map = RegionMap.load_json('<path-to-JSON>')
    # Or you can instantiate directly from a nested directory :
    region_map = RegionMap.from_dict(hierarchy_dict)

Each element stored in ``RegionMap`` stores the attributes from the corresponding JSON part.

Then you can use this object to retrieve information from the hierarchy :

.. code-block:: python

    >> region_map.get(382, "name")
       'Field CA1'

With 382 being the Allen Brain id for the 'CA1'.

With this function, you can also retrieve the ``name`` field for all the parent regions :

.. code-block:: python

    >> region_map.get(382, "name", with_ascendants=True)
    ['Field CA1', "Ammon's horn", 'Hippocampal region', 'Hippocampal formation',
    'Cortical plate', 'Cerebral cortex', 'Cerebrum']

This means the 'Field CA1' is included in the 'Ammon's horn', itself included in the 'Hippocampal region'
etc...

You can also retrieve an ID using a any kind of field. If you know the acronym of the CA1, then you can use it to
get the CA1 ID :

.. code-block:: python

    >> region_map.find('CA1', "acronym")
    {382}

You can also mix everything to retrieve information using something else than the id :

.. code-block:: python

    >> region_map.get(region_map.find('Field CA1', "name").pop(), "acronym")
    'CA1'

You can also check if a brain region possesses a brain sub-region or not :

.. code-block:: python

    >> region_map.is_leaf_id(382)
    False
    >> region_map.is_leaf_id(399)
    True

Fetching data
~~~~~~~~~~~~~

When working with a VoxelBrain atlas, there is no need to instanstiate ``VoxelData`` directly.

|name| provides ``Atlas`` class to transparently fetch data from VoxelBrain in the form of ``VoxelData`` objects.

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

as well as brain region map :

.. code-block:: python

    >> region_map = atlas.load_region_map()

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

    >> region_map = atlas.load_region_map()
    >> brain_regions = atlas.load_data('brain_regions')
