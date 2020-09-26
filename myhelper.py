import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def extract_column_names(df, regex):
    cols = df.filter(regex=regex, axis=1).columns.tolist()
    return cols


def data_prep(df):

    # lower case column names
    df.columns = df.columns.str.lower()
    # first set of new features
    df['a_follow_rate'] = df.a_following_count.div(df.a_follower_count+1)
    df['a_mention_rate'] = df.a_mentions_received.div(df.a_mentions_sent+1)
    df['a_retweet_rate'] = df.a_retweets_received.div(df.a_retweets_sent+1)
    df['a_popularity_rate'] = df.a_listed_count.div(df.a_posts+1)
    df['b_follow_rate'] = df.b_following_count.div(df.b_follower_count+1)
    df['b_mention_rate'] = df.b_mentions_received.div(df.b_mentions_sent+1)
    df['b_retweet_rate'] = df.b_retweets_received.div(df.b_retweets_sent+1)
    df['b_popularity_rate'] = df.b_listed_count.div(df.b_posts+1)

    # common columns between a and b
    common_cols = [col[2:] for col in extract_column_names(df, '^a_')]

    # adding columns
    for col in common_cols:
        col_a, col_b = 'a_{}'.format(col), 'b_{}'.format(col),
        df['fe1__a_ratio_b_{}'.format(col)] = df[col_a].div(df[col_b]+1)
        df['fe2__a_gt_b_{}'.format(col)] = (df[col_a]>df[col_b]).astype(int)

    # mark original columns as fe0
    cols = extract_column_names(df, '^(a_|b_)')
    df.rename(columns=dict(zip(cols, ['fe0__' + col for col in cols])), inplace=True)

    return df


def sampling(X, y, sample_size, seed):
    # randomly selct N indices without replacement
    idx = np.random.choice(range(0, len(X)), sample_size, False)
    X_ = X.loc[idx, :]
    y_ = y[idx]

    return X_, y_


class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, regex):
        self.regex = regex

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.loc[:, extract_column_names(X, self.regex)]










