# Import Libraries
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

df_ratings = pd.read_csv(
    'train.csv',
    usecols=['customer-id', 'movie-id', 'rating'],
    dtype={'customer-id': 'int32', 'movie-id': 'int32', 'rating': 'int32'})

test = pd.read_csv('test.csv').to_numpy()

# pivot ratings into movie features
df_ratings = df_ratings.drop_duplicates(['customer-id','movie-id'])
R_df = df_ratings.pivot_table(
    index='customer-id',
    columns='movie-id',
    values='rating'
)

def split_dataset(csvdata, percentage_to_hold_out):
    n = len(csvdata)
    k = round(percentage_to_hold_out * n)  # number of held out examples
    idxs = np.arange(n)
    held_idxs = np.random.choice(idxs, k, replace=False)
    idxs_mask = np.ones(n, np.bool)
    idxs_mask[held_idxs] = False
    used_data = [csvdata[i] for i, v in enumerate(idxs_mask) if v == True]
    held_data = [csvdata[i] for i, v in enumerate(idxs_mask) if v == False]
    return used_data, held_data


def evaluate(ratings_csv, users, movies, ratings, user_cluster):
    err = 0
    count = 0
    correct = 0
    for i, row in ratings_csv.iterrows():
        if row[0] in movies and row[1] in users:
            movie_idx = movies[row[0]]
            user_idx = users[row[1]]
            rating_label = row[2]
            #print(user_idx, movie_idx, rating_label)
            rating_predict = ratings[user_cluster[user_idx], movie_idx]
            err += (rating_label - rating_predict) ** 2
            if rating_label == rating_predict:
                correct += 1
        else:
            count += 1
    print(err, (len(ratings_csv)-count), correct/(len(ratings_csv)-count))
    return err / (len(ratings_csv)-count)


def get_all_ratings(data, K, labels, movieidx, means):
    predicted_ratings = np.zeros((K, len(movieidx.keys())))
    for i in range(K):
        user_in_cluster = np.where(labels == i)[0]
        #print(i, user_in_cluster)
        for j in range(len(movieidx.keys())):
            rs = data.iloc[user_in_cluster, j].to_numpy()
            rs = rs[(rs.nonzero()[0])]
            #print(j, rs)
            rounded = round(np.nanmean(rs))
            if len(rs) == 0 or np.isnan(rounded):
                predicted_ratings[i, j] = round(
                    means[list(movieidx.keys())[j]])
                # print(predicted_ratings[i,j])
            else:
                # print(rounded)
                if rounded > 5:
                    predicted_ratings[i, j] = 5
                else:
                    predicted_ratings[i, j] = rounded
                # print('rounded')
    return predicted_ratings


def getWbyKNN(dis_matrix, k):
    W = np.zeros((len(dis_matrix), len(dis_matrix)))
    for idx, each in enumerate(dis_matrix):
        index = np.argsort(each)
        for i in index[len(index)-k:len(index)-1]:
            W[idx][i] = dis_matrix[idx][i]
    tmp_W = np.transpose(W)
    W = (tmp_W+W)/2
    return W


def get_eigen(df):
    # pivot data
    df = df.drop_duplicates(['customer-id', 'movie-id'])
    user_movie = df.pivot(
        index='customer-id',
        columns='movie-id',
        values='rating'
    )
    R = user_movie.apply(lambda x: x.fillna(
        x.mean(skipna=True)), axis=1).to_numpy()
    # cosine similarity matrix
    print('similarity matrix')
    W_cos = cosine_similarity(R)
    print('affinity matrix by KNN')
    W = getWbyKNN(W_cos, k=25)
    #print(W)
    # degree matrix
    D = np.diag(np.sum(W, axis=1))
    # laplacian matrix
    print('laplacian')
    L = D - W
    #print(L)
    e, v = np.linalg.eig(L)
    return e, v


def kmeans(eigenvalue, eigenvector, K):
    v_s = eigenvector[:, np.argsort(eigenvalue)]
    e_s = eigenvalue[np.argsort(eigenvalue)]
    U = v_s[:, 0:K]

    km = KMeans(init='k-means++', n_clusters=K)
    km.fit(U)
    return km.labels_

def predict(train, test):
    movie_id = list(set(train['movie-id']))
    movie_id.sort()
    user_id = list(set(train['customer-id']))
    user_id.sort()
    users = {}
    movies = {}
    for i, j in enumerate(user_id):
        users[j] = i
    for i, j in enumerate(movie_id):
        movies[j] = i
        
    e, v = get_eigen(train)
    
    train_df = train.drop_duplicates(['customer-id', 'movie-id'])
    train_df = train_df.pivot(
        index='customer-id',
        columns='movie-id',
        values='rating'
    )
    means = train_df.mean(axis=0, skipna=True)
    
    labels = kmeans(e, v, 20)
    #print(labels)
    np.savetxt('label.txt', labels, fmt='%s')
    ratings = get_all_ratings(train_df, 20, labels, movies, means)
    #print(ratings)
    for i in range(len(test)):
        if test[i,0] in movies and test[i,1] in users:
            movie_idx = movies[test[i,0]]
            user_idx = users[test[i,1]]
            test[i,2] = int(ratings[labels[user_idx], movie_idx])
        else:
            test[i,2] = int(3)
    print(test)
    np.savetxt('haitongli_preds_clustering1.txt', test, fmt='%s')

def validate(per, low, high):
    train, test = split_dataset(df_ratings.to_numpy(), per)
    train = pd.DataFrame(train, columns=df_ratings.columns)
    movie_id = list(set(train['movie-id']))
    movie_id.sort()
    user_id = list(set(train['customer-id']))
    user_id.sort()
    users_t = {}
    movies_t = {}
    for i, j in enumerate(user_id):
        users_t[j] = i
    for i, j in enumerate(movie_id):
        movies_t[j] = i
    e, v = get_eigen(train)

    train_df = train.drop_duplicates(['customer-id', 'movie-id'])
    train_df = train_df.pivot(
        index='customer-id',
        columns='movie-id',
        values='rating'
    )
    means = train_df.mean(axis=0, skipna=True)
    test = pd.DataFrame(test, columns=df_ratings.columns)
    for k in range(low, high):
        print(k)
        labels = kmeans(e, v, k)
        print(labels)
        ratings = get_all_ratings(train_df, k, labels, movies_t, means)
        print(ratings)
        print(evaluate(test, users_t, movies_t, ratings, labels))


predict(df_ratings, test)
