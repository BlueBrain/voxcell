Orientation Field File Format
=============================


The orientation field indicates the orientation of the morphology at each voxel position in the form of quaternion. The coefficients of the quaternion is encoded in the order of (w, x, y, z).Therefore, each voxel stores a 4d vector of float number in the NRRD file.

.. bbp_table_section:: Specification
    :description:
     The meta data stored in the NRRD file should follow the specification below:

    .. bbp_table_value:: dimension
        :type: integer
        :value: 4
        :comment: dimension of the image is 4, with the 4th dimension indicating the quaternion 4d vector

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
        :value: ['quaternion', 'domain', 'domain', 'domain']
        :comment: use 'quaternion' to indicate the type of the first data dimension, use 'domain' to indicate the other dimensions are image axis

    .. bbp_table_value:: sizes
        :type: list of integer
        :value: [4, {size_x}, {size_y}, {size_z}]
        :comment: the 1st element is the size of quaternion vector which is 4, together with the size of the voxels in x, y, z dimension

    .. bbp_table_value:: space directions
        :type: 3d matrix
        :value: [u'none', {3d matrix}]
        :comment: 'none' indicate the first dimension does not represent the image space. the second part is a 3D matrix indicating the orientation of the image data, with the value indicating the spacing of the voxel

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
        :value: float (default) or int8 (optional)
        :comment: the default type of the values stored in the voxels is float. alternatively, the type of the values stored in the voxels can be signed 1-byte integer to minimize the file size

Example:

::

  {
    u'space origin': ['-46.540000915527344', '-152.15999984741211', '-152'],
    u'space directions': [u'none', ['16', '0', '0'], ['0', '16', '0'], ['0', '0', '16']],
    u'sizes': [4, 308, 495, 464],
    u'space': 'left-posterior-superior',
    u'encoding': 'gzip',
    u'endian': 'little',
    u'kinds': ['quaternion', 'domain', 'domain', 'domain'],
    u'type': 'int8',
    u'dimension': 4
  }
