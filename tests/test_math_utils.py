import numpy as np
import numpy.testing as npt
import pytest

import voxcell.math_utils as test_module
from voxcell import VoxelData


def test_clip():
    r = test_module.clip(
        np.array([
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]),
        (np.array([1, 1]), np.array([1, 1]))
    )
    npt.assert_equal(r, np.array([[1]]))


def test_is_diagonal_true():
    A = np.array([
        [2, 0],
        [0, 3]
    ])
    assert test_module.is_diagonal(A)


def test_is_diagonal_false():
    A = np.array([
        [2, 0],
        [1, 3]
    ])
    assert not test_module.is_diagonal(A)


def test_lcmm():
    npt.assert_equal(12, test_module.lcmm([2, 3, 4]))


def test_angles_to_matrices_x():
    angles = [np.pi / 2]
    expected = [[
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0],
    ]]
    result = test_module.angles_to_matrices(angles, 'x')
    npt.assert_almost_equal(expected, result)


def test_angles_to_matrices_y():
    angles = [np.pi / 2]
    expected = [[
        [0, 0, 1],
        [0, 1, 0],
        [-1, 0, 0],
    ]]
    result = test_module.angles_to_matrices(angles, 'y')
    npt.assert_almost_equal(expected, result)


def test_angles_to_matrices_z():
    angles = [np.pi / 2]
    expected = [[
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1],
    ]]
    result = test_module.angles_to_matrices(angles, 'z')
    npt.assert_almost_equal(expected, result)


def test_normalize_empty():
    npt.assert_equal(test_module.normalize([]), [])


def test_normalize_0():
    npt.assert_equal(test_module.normalize([1, 0, 0]), [1, 0, 0])


def test_normalize_1():
    npt.assert_equal(test_module.normalize([2, 2, 1]), [2./3, 2./3, 1./3])


def test_normalize_3():
    npt.assert_equal(test_module.normalize([[1, 0, 0], [0, 0, 0]]), [[1, 0, 0], [0, 0, 0]])


def test_isin_empty():
    npt.assert_equal(test_module.isin([], [1, 2]), [])


def test_isin_0():
    npt.assert_equal(
        test_module.isin(
            [[1, 2], [2, 3]],
            []
        ),
        [[False, False], [False, False]]
    )


def test_isin_1():
    npt.assert_equal(
        test_module.isin(
            [[1, 2], [2, 3]],
            [2, 4, 4]
        ),
        [[False, True], [True, False]]
    )


def test_isin_2():
    npt.assert_equal(
        test_module.isin(
            [[1, 2], [2, 3]],
            set([1, 2, 3, 5])
        ),
        [[True, True], [True, True]]
    )


def test_euler2mat():
    pi2 = np.pi / 2
    pi3 = np.pi / 3
    pi4 = np.pi / 4
    pi6 = np.pi / 6
    actual = test_module.euler2mat(
        [0.0, pi2, pi3],  # rotation_angle_z
        [pi2, 0.0, pi4],  # rotation_angle_y
        [pi2, pi2, pi6],  # rotation_angle_x
    )
    expected = np.array([
        [
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
        ],
        [
            [0., -1.,  0.],
            [0.,  0., -1.],
            [1.,  0.,  0.],
        ],
        [
            [0.35355339, -0.61237244,  0.70710678],
            [0.92677670,  0.12682648, -0.35355339],
            [0.12682648,  0.78033009,  0.61237244],
        ]
    ])
    npt.assert_almost_equal(actual, expected)


def test_mat2euler_roundtrip():
    original = np.asarray([
        [
            # ay = pi / 2
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.],
        ],
        [
            # ay = -pi / 2
            [ 0., 0., -1.],
            [ 0., 1.,  0.],
            [ 1., 0.,  0.],
        ],
        [
            # ay = pi / 4
            [0.35355339, -0.61237244,  0.70710678],
            [0.92677670,  0.12682648, -0.35355339],
            [0.12682648,  0.78033009,  0.61237244],
        ]
    ])
    actual = test_module.euler2mat(
        *test_module.mat2euler(original)
    )
    npt.assert_almost_equal(original, actual)


def _random_angles(n):
    return 2 * np.pi * np.random.random(n)


def test_mat2euler_roundtrip_random():
    n = 100
    az = _random_angles(n)
    ay = _random_angles(n)
    ax = _random_angles(n)
    mm = test_module.euler2mat(az, ay, ax)
    actual = test_module.euler2mat(
        *test_module.mat2euler(mm)
    )


def test_mat2euler_raises_1():
    with pytest.raises(AssertionError):
        test_module.mat2euler(np.asarray([
            [0., -1.,  0.],
            [0.,  0., -1.],
            [1.,  0.,  0.],
        ]))


def test_mat2euler_raises_2():
    with pytest.raises(AssertionError):
        test_module.mat2euler(np.asarray([
            [
                [0., -1.],
                [1.,  0.],
            ]
        ]))


@pytest.fixture
def simple_voxel_data():
    return VoxelData(np.ones((3, 4, 5)),
                     voxel_dimensions=[1, 1, 1],
                     offset=[0, 0, 0])


@pytest.mark.parametrize(
    "with_sub_segments",
    [
        pytest.param(True, id="Get indices and cut the segment according to voxel planes"),
        pytest.param(False, id="Get indices only"),
    ],
)
@pytest.mark.parametrize(
    "segment, expected_indices, expected_sub_segments",
    [
        pytest.param(
            [[-1, -1, -1], [-2, -2, -2]],
            np.zeros((0, 3)),
            [[-1, -1, -1, -2, -2, -2]],
            id="The segment does not intersect any voxel",
        ),
        pytest.param(
            [[1.5, 0, 3], [0, 2.5, 0]],
            [
                [1, 0, 3],
                [1, 0, 2],
                [0, 0, 1],
                [0, 1, 1],
                [0, 1, 0],
                [0, 2, 0],
            ],
            [
                [1.5, 0.0, 3.0, 1.5, 0.0, 3.0],
                [1.5, 0.0, 3.0, 1.0, 0.83333333, 2.0],
                [1.0, 0.83333333, 2.0, 0.9, 1.0, 1.8],
                [0.9, 1.0, 1.8, 0.5, 1.66666667, 1.0],
                [0.5, 1.66666667, 1.0, 0.3, 2.0, 0.6],
                [0.3, 2.0, 0.6, 0.0, 2.5, 0.0],
            ],
            id="The segment intersects several voxels",
        ),
        pytest.param(
            [[0.25, 0.25, 0.25], [0.75, 0.75, 0.75]],
            [[0, 0, 0]],
            [
                [0.25, 0.25, 0.25, 0.75, 0.75, 0.75],
            ],
            id="The segment is entirely inside the voxel",
        ),
        pytest.param(
            [[0, 0.5, 0.5], [0.5, 0.5, 0.5]],
            [[0, 0, 0]],
            [
                [0.0, 0.5, 0.5, 0.5, 0.5, 0.5],
            ],
            id="The segment touches the xmin plane and is inside the first voxel",
        ),
        pytest.param(
            [[1, 0.25, 0.25], [1, 0.75, 0.75]],
            [[1, 0, 0]],
            [
                [1, 0.25, 0.25, 1, 0.75, 0.75],
            ],
            id="The segment is contained in the xmin plane",
        ),
        pytest.param(
            [[1, 0.5, 0.5], [1.5, 0.5, 0.5]],
            [[1, 0, 0]],
            [
                [1.0, 0.5, 0.5, 1.5, 0.5, 0.5],
            ],
            id="The segment touches the xmin plane and is inside the second voxel",
        ),
        pytest.param(
            [[0.5, 0.5, 0.5], [1, 0.5, 0.5]],
            [[0, 0, 0], [1, 0, 0]],
            [
                [0.5, 0.5, 0.5, 1, 0.5, 0.5],
                [1, 0.5, 0.5, 1, 0.5, 0.5],
            ],
            id="The segment touches the xmax plane and is inside the voxel",
        ),
        pytest.param(
            [[0.5, 0, 0.5], [0.5, 0.5, 0.5]],
            [[0, 0, 0]],
            [
                [0.5, 0.0, 0.5, 0.5, 0.5, 0.5],
            ],
            id="The segment touches the ymin plane and is inside the first voxel",
        ),
        pytest.param(
            [[0.5, 1, 0.5], [0.5, 1.5, 0.5]],
            [[0, 1, 0]],
            [
                [0.5, 1.0, 0.5, 0.5, 1.5, 0.5],
            ],
            id="The segment touches the ymin plane and is inside the second voxel",
        ),
        pytest.param(
            [[0.5, 0.5, 0.5], [0.5, 1, 0.5]],
            [[0, 0, 0], [0, 1, 0]],
            [
                [0.5, 0.5, 0.5, 0.5, 1.0, 0.5],
                [0.5, 1.0, 0.5, 0.5, 1.0, 0.5],
            ],
            id="The segment touches the ymax plane and is inside the voxel",
        ),
        pytest.param(
            [[0.25, 1, 0.25], [0.75, 1, 0.75]],
            [[0, 1, 0]],
            [
                [0.25, 1.0, 0.25, 0.75, 1.0, 0.75],
            ],
            id="The segment is contained in the ymin plane",
        ),
        pytest.param(
            [[0.5, 0.5, 0], [0.5, 0.5, 0.5]],
            [[0, 0, 0]],
            [
                [0.5, 0.5, 0.0, 0.5, 0.5, 0.5],
            ],
            id="The segment touches the zmin plane and is inside the first voxel",
        ),
        pytest.param(
            [[0.5, 0.5, 1], [0.5, 0.5, 1.5]],
            [[0, 0, 1]],
            [
                [0.5, 0.5, 1.0, 0.5, 0.5, 1.5],
            ],
            id="The segment touches the zmin plane and is inside the second voxel",
        ),
        pytest.param(
            [[0.5, 0.5, 0.5], [0.5, 0.5, 1]],
            [[0, 0, 0], [0, 0, 1]],
            [
                [0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
                [0.5, 0.5, 1.0, 0.5, 0.5, 1.0],
            ],
            id="The segment touches the zmax plane and is inside the voxel",
        ),
        pytest.param(
            [[0.25, 0.25, 1], [0.75, 0.75, 1]],
            [[0, 0, 1]],
            [
                [0.25, 0.25, 1.0, 0.75, 0.75, 1.0],
            ],
            id="The segment is contained in the zmin plane",
        ),
        pytest.param(
            [[0.25, 0.25, 0.25], [1, 1, 1]],
            [[0, 0, 0], [1, 1, 1]],
            [
                [0.25, 0.25, 0.25, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            id="The segment touches the corner of several voxels and is inside one voxel",
        ),
        pytest.param(
            [[0.25, 0.25, 0.25], [2.25, 0.25, 0.25]],
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            [
                [0.25, 0.25, 0.25, 1.0, 0.25, 0.25],
                [1.0, 0.25, 0.25, 2.0, 0.25, 0.25],
                [2.0, 0.25, 0.25, 2.25, 0.25, 0.25],
            ],
            id="The segment crosses several voxels along X",
        ),
        pytest.param(
            [[0.25, 0.25, 0.25], [0.25, 2.25, 0.25]],
            [[0, 0, 0], [0, 1, 0], [0, 2, 0]],
            [
                [0.25, 0.25, 0.25, 0.25, 1.0, 0.25],
                [0.25, 1.0, 0.25, 0.25, 2.0, 0.25],
                [0.25, 2.0, 0.25, 0.25, 2.25, 0.25],
            ],
            id="The segment crosses several voxels along Y",
        ),
        pytest.param(
            [[0.25, 0.25, 0.25], [0.25, 0.25, 2.25]],
            [[0, 0, 0], [0, 0, 1], [0, 0, 2]],
            [
                [0.25, 0.25, 0.25, 0.25, 0.25, 1.0],
                [0.25, 0.25, 1.0, 0.25, 0.25, 2.0],
                [0.25, 0.25, 2.0, 0.25, 0.25, 2.25],
            ],
            id="The segment crosses several voxels along Z",
        ),
        pytest.param(
            [[0, 1, 1], [3, 1, 1]],
            [
                [0, 1, 1],
                [1, 1, 1],
                [2, 1, 1],
            ],
            [
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                [2.0, 1.0, 1.0, 3.0, 1.0, 1.0],
            ],
            id="The segment touches the boundaries of several voxels along X",
        ),
        pytest.param(
            [[3, 1, 1], [0, 1, 1]],
            [
                [2, 1, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            [
                [3.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                [2.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            ],
            id="The segment touches the boundaries of several voxels along X and is reversed",
        ),
        pytest.param(
            [[1, 0, 1], [1, 3, 1]],
            [
                [1, 0, 1],
                [1, 1, 1],
                [1, 2, 1],
                [1, 3, 1],
            ],
            [
                [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 2.0, 1.0],
                [1.0, 2.0, 1.0, 1.0, 3.0, 1.0],
                [1.0, 3.0, 1.0, 1.0, 3.0, 1.0],
            ],
            id="The segment touches the boundaries of several voxels along Y",
        ),
        pytest.param(
            [[1, 3, 1], [1, 0, 1]],
            [
                [1, 3, 1],
                [1, 2, 1],
                [1, 1, 1],
                [1, 0, 1],
            ],
            [
                [1.0, 3.0, 1.0, 1.0, 3.0, 1.0],
                [1.0, 3.0, 1.0, 1.0, 2.0, 1.0],
                [1.0, 2.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
            ],
            id="The segment touches the boundaries of several voxels along Y and is reversed",
        ),
        pytest.param(
            [[1, 1, 0], [1, 1, 3]],
            [
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 2],
                [1, 1, 3],
            ],
            [
                [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 2.0],
                [1.0, 1.0, 2.0, 1.0, 1.0, 3.0],
                [1.0, 1.0, 3.0, 1.0, 1.0, 3.0],
            ],
            id="The segment touches the boundaries of several voxels along Z",
        ),
        pytest.param(
            [[1, 1, 3], [1, 1, 0]],
            [
                [1, 1, 3],
                [1, 1, 2],
                [1, 1, 1],
                [1, 1, 0],
            ],
            [
                [1.0, 1.0, 3.0, 1.0, 1.0, 3.0],
                [1.0, 1.0, 3.0, 1.0, 1.0, 2.0],
                [1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            ],
            id="The segment touches the boundaries of several voxels along Z and is reversed",
        ),
        pytest.param(
            [[1.5, 1.5, 1.5], [0.5, 0.5, 0.5]],
            [
                [1, 1, 1],
                [0, 0, 0],
            ],
            [
                [1.5, 1.5, 1.5, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.5, 0.5, 0.5],
            ],
            id="The segment is oblique and passes only in the upper right corner of the voxel",
        ),
    ],
)
@pytest.mark.parametrize(
    "translation",
    [
        pytest.param(
            None,
            id="No translation",
        ),
        pytest.param(
            [-0.5, -0.5, -0.5],
            id="Slight negative translation",
        ),
        pytest.param(
            [0.5, 0.5, 0.5],
            id="Slight positive translation",
        ),
        pytest.param(
            [-1.5e6, -1.5e6, -1.5e6],
            id="Large negative translation",
        ),
        pytest.param(
            [1.5e6, 1.5e6, 1.5e6],
            id="Large positive translation",
        ),
    ]
)
@pytest.mark.parametrize(
    "scale",
    [
        pytest.param(
            None,
            id="No rescale",
        ),
        pytest.param(
            0.75,
            id="Slight downscale",
        ),
        pytest.param(
            1.5,
            id="Slight upscale",
        ),
        pytest.param(
            1.5e6,
            id="Large upscale",
        ),
    ]
)
def test_voxel_intersection(
    simple_voxel_data,
    segment,
    expected_indices,
    expected_sub_segments,
    with_sub_segments,
    translation,
    scale,
):
    if translation is not None:
        simple_voxel_data.offset += np.array(translation, dtype=simple_voxel_data.offset.dtype)
        segment = (np.array(segment, dtype=float) + np.array(translation, dtype=float)).tolist()
        expected_sub_segments = (
            np.array(expected_sub_segments, dtype=float)
            + np.concatenate([translation] * 2, dtype=float)
        ).tolist()

    if scale is not None:
        simple_voxel_data.voxel_dimensions *= scale
        segment = (
            simple_voxel_data.offset + (
                np.array(segment, dtype=float) - simple_voxel_data.offset
            ) * scale
        ).tolist()
        expected_sub_segments = (
            np.concatenate([simple_voxel_data.offset] * 2) + (
                np.array(expected_sub_segments, dtype=float)
                - np.concatenate([simple_voxel_data.offset] * 2)
            ) * scale
        ).tolist()

    result = test_module.voxel_intersection(
        segment, simple_voxel_data, return_sub_segments=with_sub_segments
    )
    if with_sub_segments:
        indices, sub_segments = result
        np.testing.assert_allclose(sub_segments, expected_sub_segments, atol=abs((scale or 1) / 1e6))
    else:
        indices = result
    np.testing.assert_array_equal(indices, expected_indices)
