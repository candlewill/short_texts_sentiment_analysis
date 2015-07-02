from load_data import load_pickle
feature_name = load_pickle('./data/features/feature_names.p')
vec = load_pickle('./data/transformed_data/transformed_train.p')
print(feature_name)
print(vec[0])