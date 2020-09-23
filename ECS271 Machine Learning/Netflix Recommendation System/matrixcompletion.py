import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

df_ratings = pd.read_csv(
    'train.csv',
    usecols=['customer-id', 'movie-id', 'rating'],
    dtype={'customer-id': 'int32', 'movie-id': 'int32', 'rating': 'int32'})

df_test = pd.read_csv('test.csv')
# print(df_ratings)

# pivot ratings into movie features
df_ratings = df_ratings.drop_duplicates(['customer-id', 'movie-id'])
R_df = df_ratings.pivot(
    index='customer-id',
    columns='movie-id',
    values='rating'
)

user_id = list(set(df_ratings['customer-id']))
user_id.sort()
users = {}
for i, j in enumerate(user_id):
    users[j] = i
movie_id = list(set(df_ratings['movie-id']))
movie_id.sort()
movies = {}
for i, j in enumerate(movie_id):
    movies[j] = i

mean = R_df.mean(skipna=True)
R = R_df.fillna(mean)

R = R.to_numpy()

# SVD
U, sigma, Vt = svds(R, k=20)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=R_df.columns)
#print(preds_df)

test = df_test.to_numpy()
for i in range(len(test)):
    if test[i, 0] in movies and test[i, 1] in users:
        movie_idx = movies[test[i, 0]]
        user_idx = users[test[i, 1]]
        test[i, 2] = int(round(preds_df.iloc[user_idx, movie_idx]))
    else:
        test[i, 2] = int(3)
print(test)
np.savetxt('haitongli_preds_matrix.txt', test, fmt='%s')


def split_dataset(csvdata, percentage_to_hold_out):
    n = len(csvdata)
    k = round(percentage_to_hold_out * n)  # number of held out examples
    idxs = np.arange(n)
    held_idxs = np.random.choice(idxs, k, replace=False)
    idxs_mask = np.ones(n, np.bool)
    idxs_mask[held_idxs] = False
    held_data = [csvdata[i] for i, v in enumerate(idxs_mask) if v == False]
    return held_data


def evaluate(ratings_csv, users, movies, ratings):
    err = 0
    count = 0
    correct = 0
    for i, row in ratings_csv.iterrows():
        movie_idx = movies[row[0]]
        user_idx = users[row[1]]
        rating_label = row[2]
        #print(user_idx, movie_idx, rating_label)
        rating_predict = round(ratings.iloc[user_idx, movie_idx])
        err += (rating_label - rating_predict) ** 2
        if rating_label == rating_predict:
            correct += 1

    print(err, (len(ratings_csv)), correct/(len(ratings_csv)))
    return err / (len(ratings_csv))


def validate(test, low, high):
    test = split_dataset(df_ratings.to_numpy(), 0.2)
    df_test = pd.DataFrame(test, columns=df_ratings.columns)
    mean = np.mean(df_test['rating'])
    for i, row in df_test.iterrows():
        R_df.loc[row[1], row[0]] = mean
        
    means = R_df.mean(skipna=True)
    R = R_df.fillna(means)
    R = R.to_numpy()

    for i in range(low,high):
        print(i)
        U, sigma, Vt = svds(R, k=i)
        sigma = np.diag(sigma)

        all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)
        preds_df
        print(evaluate(df_test, users,movies, preds_df))