import csv
import argparse
import logging

# import matplotlib.pyplot as plt
# from scipy import stats

from octane.molecular_groups import (
    get_fingerprint_bitarray
)
from octane.model import Model

main_logger = logging.getLogger('Main')
main_logger.setLevel(logging.DEBUG)
logging.info('Main started')


def prepare_csv_data_set(data_file):
    """ """

    logging.info('reading f{data_file} data set')
    structures = []
    try:
        with open(data_file) as data_set:
            csv_reader = csv.reader(data_set, delimiter=',')
            for row in csv_reader:
                structure, ron, mon = row
                fp = get_fingerprint_bitarray(structure, size=1024, fp_fmt='ecfp0')
                structures.append((fp, ron))
    except FileNotFoundError:
        main_logger.error('No such file f{data_file}')

    x_data = [struct for struct, ron in structures]
    y_data = [[ron] for struct, ron in structures]
    return x_data, y_data

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
    parser.add_argument(
        '--train-and-save',
        metavar='to_train',
        # TODO: make as flag
        default=False,
        type=bool,
        nargs='?',
        help='train and save trained model to a file specified by --model-file'
    )
    parser.add_argument(
        '--model-file',
        metavar='saved_model_file',
        # TODO: os.path.join
        default='.\\model\model.ckpt',
        type=str,
        nargs='?',
        help='path to model file, for saving/restoring model'
    )
    parser.add_argument(
        '--feed-through',
        metavar='feed_trough',
        type=str,
        nargs='?',
        help='smiles structure for octane number prediction'
    )
    parser.add_argument(
        '--epochs',
        metavar='saved_model_file',
        default=280,
        type=int,
        nargs='?',
        help='number of epochs'
    )
    args = parser.parse_args()

    prepared_data = prepare_csv_data_set(args.data_set)
    x_data, y_data = prepared_data

    model = Model(
        [1024, 64, 1],
        learning_rate=0.00001,
        epochs=args.epochs
    )

    if args.train_and_save:
        with model:
            model.fit_model([x_data, y_data])
            model.save(args.model_file)
    elif args.feed_through:
        with model:
            model.restore(args.model_file)
            fp = get_fingerprint_bitarray(args.feed_through, size=1024, fp_fmt='ecfp0')
            prediction = model.feed_through(fp)
            main_logger.info(f'Prediction for {args.feed_through}: {prediction[0][0]}')

    # y_toplot = []
    # for r in y_data:
    #     for j in r:
    #         y_toplot.append(float(j))

    # diff = []

    # for i, j in zip(y_toplot, pred):
    #     print(i, j)
    #     diff.append(i-j)
    # plt.plot(pred, 'ro')
    # plt.plot(y_toplot, pred, 'bo', list(range(-20,120)), list(range(-20,120)), 'r--')
    # plt.plot(list(range(len(y_toplot))), y_toplot, 'r--', list(range(len(y_toplot))), pred, 'g--')

    # slope, intercept, r_value, p_value, std_err = stats.linregress(y_toplot, pred)
    # print('r-squared value: {}'.format(r_value))



    # plt.show()

if __name__ == '__main__':
    main()
