

def check_results(feature_list, y_pred, y_test):
    print('Predicted\t\tExpected')
    for x, a, b in zip(feature_list, y_pred, y_test):
        print('\t{:4.2f}\t\t{:4.2f}'.format(a, b))
        print('[', end='')
        for f in x:
            print('{:4.2f}\t'.format(f), end='')
        print(']', end='')
