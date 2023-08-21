from numba import cuda, float32, jit, int32
import numpy as np
import pandas as pd
from scipy.sparse import dok_matrix
import xgboost as xgb
from sklearn.metrics import accuracy_score

@cuda.jit('void(uint16[:,:], uint16[:,:], uint16[:,:])')
def kernel_function(X, Y, Z):
    x, y = cuda.grid(2)
    if x < X.shape[0] and y < Y.shape[0]:
        str1 = X[x]
        str2 = Y[y]
        m, n = min(len(str1),150), min(len(str2),150)
        dp = cuda.local.array((150, 150), dtype=int32)

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                char1 = str1[i-1]
                char2 = str2[j-1] 

                if char1 == char2:
                    cost = 0
                else:
                    cost = 1

                dp1 = dp[i - 1][j] + 1
                dp2 = dp[i][j - 1] + 1
                dp3 = dp[i - 1][j - 1] + cost
                dp[i][j] = min(dp1, dp2, dp3)

        levenshtein_distance = dp[m][n]

        # TODO: understand why x and y need to be flipped here... is this just a numpy feature?
        Z[y][x] = levenshtein_distance

def transform_str_array(array, fixed_length = 135):
    num_texts = len(array)
    X = dok_matrix((num_texts, fixed_length), dtype=np.uint16)
    for i, s in enumerate(array):
        padded_string = s.ljust(fixed_length, '\x00')
        char_codes = np.array(list(map(ord, padded_string)), dtype=np.uint16)
        X[i, :] = char_codes

    return X

X_df = pd.read_csv("./train.csv", header = None)
Y_df = pd.read_csv("./test.csv", header = None)
X_df.columns = ["topic", "question_title", "question_description", "top_answer"]
Y_df.columns = ["topic", "question_title", "question_description", "top_answer"]
X_df["topic"] = X_df["topic"] - 1
Y_df["topic"] = Y_df["topic"] - 1

X1_df = X_df[:5000]
X2_df = X_df[5000:10000]
Y1_df = Y_df[:100]

print(X1_df)
print(X2_df)
print(Y1_df)

X1_strings = X1_df["question_title"].to_numpy(dtype=np.str_)
X2_strings = X2_df["question_title"].to_numpy(dtype=np.str_)
Y1_strings = Y1_df["question_title"].to_numpy(dtype=np.str_)

X1_matrix = transform_str_array(X1_strings)
X2_matrix = transform_str_array(X2_strings)
Y1_matrix = transform_str_array(Y1_strings)

X1_array = X1_matrix.toarray()
X2_array = X2_matrix.toarray()
Y1_array = Y1_matrix.toarray()

print(X1_array.shape)
print(X2_array.shape)
print(Y1_array.shape)

X1_n_rows = X1_array.shape[0]
X2_n_rows = X2_array.shape[0]
Y1_n_rows = Y1_array.shape[0]

batch_size = 10
n_classes = X_df["topic"].nunique()

threadsperblock = (16, 16)
blockspergrid_x = X1_n_rows

distances = []
for start in range(0, X2_n_rows, batch_size):
    print(f"Processing training set: {start}/{X2_n_rows}")
    end = min(start + batch_size, X2_n_rows)
    X2_array_batch = X2_array[start:end, :]
    blockspergrid_y = end-start
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    # TODO: understand why batch_size and X1_n_rows need to be flipped here... is this just a numpy feature?
    batch_distances = np.zeros((end-start, X1_n_rows), dtype=np.uint16)
    kernel_function[blockspergrid, threadsperblock](X1_array, X2_array_batch, batch_distances)
    distances.extend(batch_distances)

params = {
    "objective":"multi:softmax", 
    "num_class":10, 
    "random_state":42,
    "colsample_bytree": 0.4,
    "gamma": 0.5,
    "learning_rate": 0.1, 
    "max_depth": 6,
    "n_estimators": 200
}

X_distances = pd.DataFrame(distances)
print(X_distances)

clf_distances = xgb.XGBClassifier(**params)
clf_distances.fit(X_distances, X2_df["topic"])

distances = []
for start in range(0, Y1_n_rows, batch_size):
    print(f"Processing testing set: {start}/{Y1_n_rows}")
    end = min(start + batch_size, Y1_n_rows)
    Y1_array_batch = Y1_array[start:end, :]
    blockspergrid_y = end-start
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    # TODO: understand why batch_size and X1_n_rows need to be flipped here... is this just a numpy feature?
    batch_distances = np.zeros((batch_size, X1_n_rows), dtype=np.uint16)
    kernel_function[blockspergrid, threadsperblock](X1_array, Y1_array_batch, batch_distances)
    distances.extend(batch_distances)

clf_distances_pred = clf_distances.predict(np.array(distances))
accuracy = accuracy_score(Y1_df["topic"], clf_distances_pred)
print(f"Accuracy: {accuracy:.4f}")