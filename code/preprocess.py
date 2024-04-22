import scipy.io as scipy


def get_feature_dicts():
    gist_filepath = "data/zappos-gist.mat"
    color_filepath = "data/zappos-color.mat"

    gist_dict = scipy.loadmat(gist_filepath)
    color_dict = scipy.loadmat(color_filepath)

    return gist_dict, color_dict
