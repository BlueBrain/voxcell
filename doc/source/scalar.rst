Scalar Image File Format
========================


Scalar value image includes the following data types:

- brain_region: label image with integer values as voxel type
- gray_level: raw imaging data, integer or float values as voxel type
- longitude: computed depth information of the atlas, integer values as voxel type
- hemisphere: hemisphere dataset with int8 or uint8 values as voxel type (0: undefined, 1: left, 2: right)

.. bbp_table_section:: Specification
    :description:
     The meta data stored in the NRRD file should follow the specification below:

    .. bbp_table_value:: dimension
        :type: integer
        :value: 3
        :comment: dimension of the image is always 3 for volumetric data

    .. bbp_table_value:: encoding
        :type: string
        :value: gzip
        :comment: use 'gzip' for compressing the data

    .. bbp_table_value:: endian
        :type: string
        :value: little
        :comment: use 'little' for x86 machines

    .. bbp_table_value:: kinds
        :type: list of string
        :value: ['domain', 'domain', 'domain']
        :comment: use 'domain' to indicate that the axis is image axis

    .. bbp_table_value:: sizes
        :type: list of integer
        :value: [{size_x}, {size_y}, {size_z}]
        :comment: the size of the volume in x, y, z dimension

    .. bbp_table_value:: space directions
        :type: 3d matrix
        :value:
        :comment: A 3D matrix indicating the orientation of the image data, with the value indicating the spacing of the voxel

    .. bbp_table_value:: space origin
        :type: list of float
        :value: [{origin_x}, {origin_y}, {origin_z}]
        :comment: physical coordinates of the image origin

    .. bbp_table_value:: space
        :type: string
        :value: {x_orientation}-{y_orientation}-{z_orientation}
        :comment: the orientation of the image data, should be consistent with the space direction

    .. bbp_table_value:: type
        :type: string
        :value: {voxel type}
        :comment: the type of the values stored in the voxels (e.g. unsigned/signed int, float, etc.)

Example:

::

  {
    u'space origin': ['-46.540000915527344', '-152.15999984741211', '-152'],
    u'space directions': [['16', '0', '0'], ['0', '16', '0'], ['0', '0', '16']],
    u'sizes': [308, 495, 464],
    u'space': 'left-posterior-superior',
    u'encoding': 'gzip',
    u'endian': 'little',
    u'kinds': ['domain', 'domain', 'domain'],
    u'type': 'unsigned short',
    u'dimension': 3
  }
