import numpy as np


class NaiveSelect:
    def fit(self, x, y):
        pass

    def predict(self, X: np.array):
        #print(X)
        min_mse = 9999999
        for value in range(int(np.min(X)), int(np.max(X)) + 1):
            tmp_mse = sum([(x - value)**2 for x in X.flat])
            if(tmp_mse < min_mse):
                min_mse = tmp_mse
                ret_value = value

        return np.array([ret_value])


class NaiveSelectWeight:
    def fit(self, x, y):
        pass

    def predict(self, X: np.array):
        #print(X)
        min_mse = 9999999
        sum_value = 0
        list_weight1 = [1 /(len(X[0])+1 - x)  for x in range(1, len(X[0])+1)]
        list_weight2 = [x * 1 for x in range(1, len(X[0]) + 1)]
        list_weight = map(lambda x,y : x*0.1 + y*0.9 , list_weight1 ,list_weight2)
        #print(list_weight)
        for value in range(int(np.min(X)), int(np.max(X)) + 1):

            tmp_mse = sum(map(lambda x,y : x* y,list_weight2,[(x - value)**2 for x in X.flat]))
            # for x in X.flat
            #     weight =
            #     sum_value += (x - value)**2 * weight
            if(tmp_mse < min_mse):
                min_mse = tmp_mse
                ret_value = value

        return np.array([ret_value])




