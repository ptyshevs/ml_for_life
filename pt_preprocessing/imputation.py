import numpy as np
from .scaling import standardize
from .models import LinearRegression, KNN

def drop_na_naive(a, axis=0):
    if axis == 0:
        return a[~np.isnan(a).any(1), :]
    else:
        return a[:, ~np.isnan(a).any(0).T]

def isna_obj(obj):
  return obj != obj
      
def fill_na_mode(a):
    if len(a.shape) == 1:
        a = a[:, np.newaxis]
    for i in range(a.shape[1]):
        cnts = dict()
        for j in range(a.shape[0]):
            if a[j, i] in cnts:
                cnts[a[j, i]] += 1
            elif not isna_obj(a[j, i]):
                cnts[a[j, i]] = 1
        a[np.where(isna_obj(a[:, i])), i] = max(cnts, key=cnts.get)
    return a

def fill_na(a, value='mean', inplace=False):
    if not inplace:
        a = np.copy(a)
    if len(a.shape) == 1:
      a = a[:, np.newaxis]
    f = np.nanmean
    if value == 'median':
        f = np.nanmedian
    elif value == 'mode':
        return fill_na_mode(a)
    for i in range(a.shape[1]):
        col_mean = f(a[:, i])
        a[np.where(np.isnan(a[:, i])), i] = col_mean
    return a

def fill_lr(predictors, target):
    """
    Assuming predictors to have all data necessary
    """
#     print("target shape:", target.shape)
    not_nan_idx = ~np.isnan(target)
    if len(predictors.shape) == 1:
        predictors = predictors[:, np.newaxis]
    if len(target.shape) == 1:
        target = target[:, np.newaxis]
#     print("NOT NAN IDX:", not_nan_idx)
#     print("SHAPE:", predictors.squeeze().shape)
    X_train, y_train = predictors[not_nan_idx, :], target[not_nan_idx, :]
    X_train, train_mean, train_std = standardize(X_train)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    X_test, _, _ = standardize(predictors[~not_nan_idx, :], train_mean, train_std)
    target[~not_nan_idx] = lr.predict(X_test).squeeze()

def fill_knn(predictors, target, k=3):
    not_nan_idx = ~np.array([True if x != x else False for x in target])
    if len(target[~not_nan_idx]) == 0:
        return None
    if len(predictors.shape) == 1:
        predictors = predictors[:, np.newaxis]
    if len(target.shape) == 1:
        target = target.values[:, np.newaxis]
    X_train, y_train = predictors[not_nan_idx, :], np.array([x for x in target if x == x])
    knn = KNN(k)
    knn.fit(X_train, y_train)
    X_test = predictors[~not_nan_idx, :]
#     print("X_test.shape:", X_test.shape)
    target[~not_nan_idx, :] = knn.predict(X_test)
    return target

def fill_lr_all(a):
    full_columns = a[:, np.where(~np.isnan(a).any(0))].squeeze()
    if len(full_columns.shape) == 1:
        full_columns = full_columns[:, np.newaxis]
#     print("shape:", full_columns.shape)
    non_full_columns_idx = np.where(np.isnan(a).any(0))
#     print("non_full_columns_idx:", non_full_columns_idx[0].shape)
    for idx in non_full_columns_idx[0]:
#         print("idx:", idx)
#         print("full_columns shape:", full_columns.shape, "missed column:", a[:, idx].shape)
        fill_lr(full_columns, a[:, idx])
        full_columns = np.hstack([full_columns, a[:, idx][:, np.newaxis]])
    return a