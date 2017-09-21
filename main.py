import argparse
import csv

import matplotlib.pyplot as plt
from scipy import stats

from octane.molecular_groups import (
    generate_fingerprint
)
from octane.model import Model

def check_equals(structures):

    structs_ = [s for s, _ in structures]

    ss = []
    for s in structs_:
        s = ''.join([str(item) for item in s])
        ss.append(s)
    print(len(set(ss)))
    print(len(ss))

    if len(set(ss)) != len(ss):
        return False

def main():
    """ entry point """

    parser = argparse.ArgumentParser(
        description='Octane number prediction'
    )
    parser.add_argument(
        '--data-set',
        metavar='data_file',
        # TODO: os.path.join
        default='data/data_set.csv',
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
                fp = [0 for i in range(1024)]
                r = generate_fingerprint(structure, 'maccs')
                for bit_index in r:
                    fp[bit_index] = 1
                structures.append((fp, ron))
        x_data = [struct for struct, ron in structures]
        y_data = [[ron] for struct, ron in structures]

        if not check_equals(structures):
            print('Equals')

        model = Model(
            [1024, 800, 1],
            learning_rate=0.0001,
            epochs=1000
        )

        pred = []

        with model:
            model.train([x_data, y_data])
            for x in x_data:
                pred.append(model.feed_through(x)[0][0])

        y_toplot = []
        for r in y_data:
            for j in r:
                y_toplot.append(float(j))

        diff = []

        for i, j in zip(y_toplot, pred):
            print(i, j)
            diff.append(i-j)
        #plt.plot(pred, 'ro')
        plt.plot(y_toplot, pred, 'bo', list(range(100)), list(range(100)), 'r--')

        slope, intercept, r_value, p_value, std_err = stats.linregress(y_toplot, pred)
        print('r-squared value: {}'.format(r_value))

        plt.show()

    except FileNotFoundError:
        print('No such file')

if __name__ == '__main__':
    main()
