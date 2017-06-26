import numpy as np
import pandas as pd

from nose import tools as nt

from voxcellview import viewer

SIZE = 3


def _get_df():
    data = np.array([[14, 2, 44],
                     [50, 17, 3],
                     [19, 64, 0]])

    df = pd.DataFrame(data, columns=['x', 'y', 'z'])
    return df


def _default_color_map(e):
    return {'L1_HAC': ['1', '1', '1'],
            'L4_SP': ['0', '1', '0'],
            }.get(e, ['0.5', '0.5', '1'])


def test_get_cell_color_with_map():
    df = _get_df()
    default_attribute = 'x_type'
    df[default_attribute] = np.random.choice(['L1_HAC', 'L4_SP'], SIZE)
    colors = viewer.get_cell_color(df, default_attribute, input_color_map=_default_color_map)
    unique_colors = pd.DataFrame(colors).drop_duplicates()
    nt.eq_(len(colors), SIZE)
    nt.eq_(len(colors[0]), 4)  # RGBA
    unique_types = df[default_attribute].drop_duplicates()
    nt.eq_(len(unique_colors), len(unique_types))


def test_get_cell_color_without_map():
    df = _get_df()
    default_attribute = 'x_type'
    df[default_attribute] = np.random.choice(['L1_HAC', 'L4_SP'], SIZE)
    colors = viewer.get_cell_color(df, default_attribute, input_color_map=None)
    unique_colors = pd.DataFrame(colors).drop_duplicates()
    nt.eq_(len(colors), SIZE)
    nt.eq_(len(colors[0]), 4)  # RGBA
    unique_types = df[default_attribute].drop_duplicates()
    nt.eq_(len(unique_colors), len(unique_types))


def test_get_cell_color_with_positions():
    df = _get_df()
    colors = viewer.get_cell_color(df, 'position', input_color_map=None)
    nt.ok_(np.allclose(np.ones([SIZE, 4]), colors))

    def color_mapper(row):
        if row[0] < 25:
            return [0, 0, 0]
        return [1, 1, 1]

    colors = viewer.get_cell_color(df, 'position', input_color_map=color_mapper)
    nt.ok_(np.allclose(np.array([0., 1., 0.]), colors[:, 0]))


def test_non_callable_colormap():
    df = _get_df()
    nt.assert_raises(AssertionError, viewer.get_cell_color, df, 'x', {'not': 'callable'})
