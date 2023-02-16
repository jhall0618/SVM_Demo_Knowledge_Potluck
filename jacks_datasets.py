from sklearn.utils import Bunch
import numpy as np


def retrieve_jacks_data(d_set_name):
    match d_set_name:
        case 'xor':
            Nsamps = 1000
            A = 2 * np.random.rand(1, Nsamps) - 1.0
            B = 2 * np.random.rand(1, Nsamps) - 1.0
            target = (((A > 0) | (B > 0)) & ~((A > 0) & (B > 0))).astype(int)
            target = target.reshape(target.shape[1:])
            data = np.concatenate([A, B]).transpose()
            my_data = Bunch(data=data, target=target, feature_names=['A', 'B'])
        case 'rings':
            Nsamps = 500
            thetas = 2 * np.pi * np.random.rand(1, Nsamps // 2) - np.pi
            rescale_factors1 = np.random.uniform(low=0.9, high=1.1, size=(1, thetas.shape[1]))
            A = np.multiply(np.cos(thetas), rescale_factors1)
            B = np.multiply(np.sin(thetas), rescale_factors1)
            target = np.ones((1, Nsamps))
            target[:, :Nsamps // 2] = 0
            target = target.reshape(target.shape[1:])
            data1 = np.concatenate([A, B]).transpose()
            data2 = np.concatenate([1.75 * A, 1.75 * B]).transpose()
            data = np.concatenate([data1, data2], axis=0)
            my_data = Bunch(data=data, target=target)
            my_data = Bunch(data=data, target=target, feature_names=['x', 'y'])
    return my_data