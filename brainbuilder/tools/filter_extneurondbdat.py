"""Script that
(1) loads extneurondb.dat file and list of morphologies to exclude,
(2) filters morphologies out,
(3) prints a warning for every (layer, m-type, e-type) that is thus filtered
    out,
(4) stores the resulting extneurondb.dat file.
"""
from __future__ import print_function

import sys
import argparse
import pandas


def parse_args():
    """Parse arguments"""
    parser = argparse.ArgumentParser("Filter extneurondb.dat")
    parser.add_argument(
        "--input",
        help="Path to extneurondb.dat",
        required=True
    )
    parser.add_argument(
        "--exclude",
        help="Path to list of morphologies to exclude",
        required=True
    )
    parser.add_argument(
        "--output",
        help="Path to filtered database",
        default="filtered_extneurondb.dat",
        required=False
    )
    parser.add_argument(
        "--strict",
        help="Script fails if (m-type, e-type, layer)-combinations are missing"
             " after the filtering operation",
        action="store_true"
    )
    return parser.parse_args()


def load_list(path):
    """Load list from text file"""
    with open(path) as file_:
        data = file_.read().split()
    return data


def load_extneurondbdat(path):
    """Load extneurondb.dat with headers"""
    names = ["morphology", "layer", "mtype", "etype", "mecombo"]
    return pandas.read_csv(path, delimiter=r'\s+', names=names)


def filter_extneurondbdat(data, morphs):
    """Filter out morphologies that are in a given list"""
    return data[~data["morphology"].isin(morphs)]


def verify_result(extneurondbdat, filtered_extneurondbdat, strict):
    """Verify filtered database and print a warning for all (layer, m-type,
    e-type)-combinations that have been filtered out.
    """
    def _get_mel_types(data):
        """Find all (layer, m-type, e-type)-combinations"""
        return set(
            data[['layer', 'mtype', 'etype']].drop_duplicates().itertuples(
                index=False, name=None))

    original_mel_types = _get_mel_types(extneurondbdat)
    filtered_mel_types = _get_mel_types(filtered_extneurondbdat)

    for mel_type in original_mel_types.difference(filtered_mel_types):
        template = "WARNING: no instances of type {} in filtered database"
        warning = template.format(mel_type)
        if strict:
            raise Exception(warning)
        print(warning, file=sys.stderr)


def save_extneurondbdat(data, path):
    """Save filtered extneurondb.dat to path"""
    data.to_csv(path, sep=' ', header=False, index=False)


def main():
    """Main function"""
    args = parse_args()

    # load list with morphologies to exclude
    morphs_to_exclude = load_list(args.exclude)

    # load extneurondbdat
    extneurondbdat = load_extneurondbdat(args.input)

    # filter extneurondbdat
    filtered_extneurondbdat = filter_extneurondbdat(extneurondbdat,
                                                    morphs_to_exclude)
    nb_excluded = len(extneurondbdat) - len(filtered_extneurondbdat)
    print("{} entries were excluded.".format(nb_excluded), file=sys.stderr)

    # verify me-types, m-types, e-types, layers
    verify_result(extneurondbdat, filtered_extneurondbdat, args.strict)

    # save filtered extneurondbdat
    save_extneurondbdat(filtered_extneurondbdat, args.output)


if __name__ == "__main__":
    main()
