'''export results to a format usable by the rest of the pipeline
    (touch detector, functionalizer, bluepy, etc..)'''


# pylint: disable=W0613
def export_for_bbp(positions, orientations, chosen_traits):
    '''
    Rest of mesobuilder steps to make the circuit readable by bluepy:
    create a CircutConfig, create sqlite, etc.

    Accepts:
        positions: list of positions for soma centers (x, y, z).
        orientations: list of orientations (3 vectors: right, up, fwd).
        chosen_traits: list of chosen properties: morphology, mtype, etype, mClass, sClass
    Returns:
        circuit: collection of files importable by bluepy
    '''
    # TODO do
    raise NotImplementedError
