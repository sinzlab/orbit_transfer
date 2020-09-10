import h5py
import numpy as np
from sklearn.datasets import fetch_openml


def load_mauna_loa_atmospheric_co2():
    ml_data = fetch_openml(data_id=41187)
    months = []
    ppmv_sums = []
    counts = []

    y = ml_data.data[:, 0]
    m = ml_data.data[:, 1]
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            # aggregate monthly sum to produce average
            ppmv_sums[-1] += ppmv
            counts[-1] += 1

    months = np.asarray(months).reshape(-1, 1)
    avg_ppmvs = np.asarray(ppmv_sums) / counts
    # normalize:
    avg_ppmvs -= np.mean(avg_ppmvs)
    avg_ppmvs /= np.std(avg_ppmvs)
    X_plot = months
    Y_plot = avg_ppmvs
    X_train = np.concatenate((X_plot[:120], X_plot[150:300], X_plot[380:450]))
    Y_train = np.concatenate((Y_plot[:120], Y_plot[150:300], Y_plot[380:450]))

    return X_plot, Y_plot, X_train, Y_train


def load_co2():
    f = h5py.File('co2_data.h5', 'r')
    data_train = np.concatenate((f['data'].value, f['label'].value), axis=1)
    f.close()
    X_train = data_train[:, 0].reshape(-1, 1)
    Y_train = data_train[:, 1].reshape(-1)

    X_plot = np.concatenate((X_train,np.arange(1.73, 3.51, 0.01).reshape(-1, 1)))
    Y_plot = np.concatenate((Y_train,np.zeros((int((3.51-1.73)//0.01 + 1),))))
    X_train = np.concatenate((X_train[:120], X_train[150:]))
    Y_train = np.concatenate((Y_train[:120], Y_train[150:]))

    return X_plot, Y_plot, X_train, Y_train

def load_sinusoid_data(noisy=False):
    def f(x):
        return (x + np.sin(2 * x)).ravel()

    rng = np.random.RandomState(0)
    X_plot = np.linspace(-10, 40, 1000).reshape(-1, 1)
    X_train = np.sort(10 * rng.rand(100, 1), axis=0)
    X_train = np.concatenate((X_train, (np.sort(10 * rng.rand(100, 1) + 20, axis=0))))
    Y_train = f(X_train)
    Y_plot = f(X_plot)
    if noisy:
        Y_train = Y_train + 1 * (0.5 - rng.rand(X_train.shape[0]))
    return X_plot, Y_plot, X_train, Y_train
