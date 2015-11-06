'''algorithm to assign morphologies to a group of cells'''
import logging
L = logging.getLogger(__name__)


def assign_morphology(positions, metypes, sdist):
    '''for every cell in positions, assign a morphology to each cell based on its metype

    Args:
        positions: list of positions for soma centers (x, y, z).
        metypes: dataframe with the mtype and etype values that correspond to each position.
        sdist: SpatialDistribution containing at least the properties:
            mtype, etype, morphology.

    Returns:
        A pandas DataFrame with one row for each position and one column: morphology.
        For those positions whose morphology could not be determined, nan is used.
    '''
    chosen = sdist.assign_conditional(positions, metypes)
    return sdist.collect_traits(chosen, ('morphology',))
