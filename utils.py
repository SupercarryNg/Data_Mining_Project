import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegressionCV
# from sklearn.metrics import roc_curve, auc
# from sklearn.preprocessing import label_binarize

from pprint import pprint
from visualization import *

import warnings

warnings.filterwarnings('ignore')


# data preprocess filling the symbol codes, transform  pd into datetime format
def preprocessing(df):
    df = df.reset_index(drop=True)
    # Drop the duplicates rows
    df = df.drop_duplicates(keep='first')

    # process the table with TICKER SYMBOL
    if 'TICKER_SYMBOL' in df.columns:
        df['TICKER_SYMBOL'] = df['TICKER_SYMBOL'].map(lambda x: str(x).zfill(6)[-6:])

    #
    df['TRADE_DATE'] = pd.to_datetime(df.TRADE_DATE)
    df['TRADE_DATE'] = df['TRADE_DATE'].dt.date
    return df


def dataloader(start_train, end_train, end_val, df):
    # get the data in a specific period and return train set and val set respectively
    # dataset structure >>> start_train ----1---- end_train --2-- end_val
    # Use df_factor_1 as our default set
    # start_train means the start date of train set (str)
    # start val means the start date of valid set, maybe a split factor could be better (str)
    # val_end means the whole length of the period >>> train + val, also the end of valid set (str)

    df = df.copy()
    mask_train = (df['TRADE_DATE'] >= start_train) & (df['TRADE_DATE'] < end_train)  # row filter condition for 1
    mask_val = (df['TRADE_DATE'] >= end_train) & (df['TRADE_DATE'] < end_val)  # row filter condition for 2

    train = df.loc[mask_train]
    val = df.loc[mask_val]

    X_train, X_val = train.drop(['ret'], axis=1), val.drop(['ret'], axis=1)
    y_train, y_val = train['ret'], val['ret']

    return X_train, y_train, X_val, y_val


# transform the close price into true label of rise
def get_ret(df):
    df = df.pivot_table(index='TRADE_DATE', columns='TICKER_SYMBOL',
                        values='closePrice').ffill().pct_change().shift(-1)
    df = df.T.stack().reset_index().rename(columns={0: 'ret'})

    return df


def training(X_train, X_val, y_train, y_val, param_grid=None, method='rf'):
    # have to drop DATE & SYMBOL, they are meaningless in training
    drop_col = ['TRADE_DATE', 'TICKER_SYMBOL']
    pred = None

    if method == 'rf':
        base = RandomForestClassifier()
        if param_grid is not None:
            grid = GridSearchCV(base, param_grid=param_grid, cv=5)
            grid.fit(X_train.drop(drop_col, axis=1), y_train)
            estimator_ = grid.best_estimator_
        else:
            estimator_ = base.fit(X_train.drop(drop_col, axis=1), y_train)

        # predict probability of stock rising
        pred = estimator_.predict_proba(X_val.drop(drop_col, axis=1))

    if method == 'lr':
        estimator_ = LogisticRegressionCV(Cs=100, max_iter=1000)
        estimator_.fit(X_train.drop(drop_col, axis=1), y_train.ravel())
        pred = estimator_.predict_proba(X_val.drop(drop_col, axis=1))

    # make sure the model works well
    if pred is not None:
        rise = X_val[['TRADE_DATE', 'TICKER_SYMBOL']].copy()
        rise.reset_index(drop=True, inplace=True)
        rise['rise'] = pred[:, 1]

        return rise


# get the rank of each stock per day (the output of out_sample)
def stock_rank(df):
    val_ins = df.industryName1.unique()
    df_val_estimator_rank = pd.DataFrame(columns=['TICKER_SYMBOL', 'TRADE_DATE', 'rank_'])
    for ins in val_ins:
        tmp = df[df['industryName1'] == ins].copy()
        tmp = tmp.pivot_table(index='TICKER_SYMBOL', columns='TRADE_DATE', values='rise')
        tmp = tmp.apply(lambda x: pd.qcut(x, 5, labels=False, duplicates='drop'), axis=0).stack().reset_index()
        tmp = tmp.rename(columns={0: 'rank_'})
        df_val_estimator_rank = df_val_estimator_rank.append(tmp)

    return df_val_estimator_rank


# factor test
def factor_test(df_rank, index_cum, period_idx, method, freq='BM', tier_num=5):
    freq_dict = {'BM': 12, 'B': 252, 'Q': 4, 'W': 52, '6M': 2}

    num = freq_dict[freq]

    df = df_rank.copy()  # use deep copy here to prevent copy-warning
    df['return_ranking'] = df.groupby('TRADE_DATE')['ret'].rank(ascending=False)  # To be used in calculating IC
    df['factor_ranking'] = df.groupby('TRADE_DATE')['rise'].rank(ascending=True)  # To be used in calculating IC
    df[['rise', 'ret']] = df[['rise', 'ret']].astype('float32')

    # evaluation
    # IC
    IC_list = df.groupby('TRADE_DATE').apply(lambda x: x['rise'].corr(x.ret))
    print_evaluation(IC_list, 'IC')

    # Rank IC
    RIC_list = df.groupby('TRADE_DATE').apply(lambda x: x.factor_ranking.corr(x.return_ranking, method='pearson'))
    print_evaluation(RIC_list, 'RIC')

    y_test = df['ret'].copy()
    y_pred = df['rise'].copy()
    y_test = y_test.apply(lambda x: 1 if x >= 0 else 0)

    df_tier_ret = plot_all_evaluation(y_test, y_pred, df, index_cum, tier_num, period_idx=period_idx, method=method)
    print_long_short(df, tier_num, num, is_period=True)

    return df_tier_ret


def test_all_period(index_cum, tier_ret, auc_all, method, freq='BM', tier_num=5):
    freq_dict = {'BM': 12, 'B': 252, 'Q': 4, 'W': 52, '6M': 2}
    num = freq_dict[freq]

    plot_test_all_period(index_cum, tier_ret, tier_num, auc_all, method=method)
    print('#'*15 + ' All Period Results ' + '#'*15)
    print_long_short(tier_ret, tier_num, num, is_period=False)



