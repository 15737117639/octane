import argparse
import csv

from molecular_groups import (
    split_molecule_to_substructures,
    molecular_groups_count
)
from model import Model


def main():
    """ entry point """

    parser = argparse.ArgumentParser(
        description='Octane number prediction'
    )
    parser.add_argument(
        '--data-set',
        metavar='data_file',
        # TODO: os.path.join
        default='..\data\data_set.csv',
        type=str,
        nargs='?',
        help='path to data set file'
    )
    args = parser.parse_args()

    try:
        structures = []
        with open(args.data_set) as data_set:
            csv_reader = csv.reader(data_set, delimiter=',')
            for row in csv_reader:
                structure, ron, mon = row
                r = split_molecule_to_substructures(structure)
                structures.append((list(r.values()), ron))
        x_data = [struct for struct, ron in structures]
        y_data = [[ron] for struct, ron in structures]

        model = Model(
            [molecular_groups_count, 3, 1],
            learning_rate=0.001,
            epochs=2000
        )

        with model:
            model.train([x_data, y_data])
            for x in x_data:
                print(x, model.feed_through(x))

    except FileNotFoundError:
        print('No such file')

if __name__ == '__main__':
    main()
