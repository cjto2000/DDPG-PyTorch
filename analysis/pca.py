from sklearn.decomposition import PCA

def pcs(time_series):
    pca = PCA(n_components=2)
    pca.fit(time_series)
    components = pca.components_
    res = components @ time_series.transpose(1, 0)
    res[0] = res[0]
    res[1] = res[1]
    return res[0], res[1]

def pc_components(time_series):
    """
    Gets the first two PC components
    """
    pca = PCA(n_components=2)
    pca.fit(time_series)
    components = pca.components_
    return components

def demean(time_series):
    mean = time_series.mean(axis=0)
    return time_series - mean




