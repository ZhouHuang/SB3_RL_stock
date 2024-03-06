import numpy as np
import pandas as pd
import calendar

try:
    from pyqts.pytsdb import PyTsdb
except ModuleNotFoundError as e:
    print('pytsdb not found')
    exit(1)

TSDB_ROOT = '/home/huangzh/TSDB/stk/'

def _extractInd(code, level=1):
    """
    TSDB行业代码转换为行业分类

    0代表上市不足或未分类。
    """
    kflag = (1 << 8) - 1
    code = (code >> (8 * (3-level)))
    return (code & kflag)

def get_data_pack(tp_beg='20050101 00:00:00.000', tp_end='TODAY 15:00:00.000'):
    db = PyTsdb(TSDB_ROOT)
    col_types = {
        'close': np.float32,
        'open': np.float32,
        'high': np.float32,
        'low': np.float32,
        'adjfactor': np.float32,
        'vol': np.float32,
        'amount': np.float32,
        'avgprice': np.float32,
        'preclose': np.float32,
        'share.total': np.float32,
        'ind.sw': np.int32,
        'universe.all': np.int8,
    }
    data = db.read_columns(tbl="d01e", cols=list(col_types.keys()), tp_beg=tp_beg, tp_end=tp_end, dtypes=col_types)
    df_d01 = pd.DataFrame(data)
    df_d01 = df_d01[df_d01['universe.all'] == 1]
    df_d01['ticker'] = df_d01['ticker'].apply(lambda s : s.decode('utf-8')).astype(str)
    # 沪深两市，科创板(688开头)和创业板(300)股票，日涨跌幅限制为20%，其余为10%
    df_d01['limit_pct'] = df_d01['ticker'].apply(lambda s : 0.2 if s[:3] in ('688', '300') else 0.1) 
    df_d01['uplimit'] = df_d01['preclose'] * (1.0 + df_d01['limit_pct'])
    df_d01['downlimit'] = df_d01['preclose'] * (1.0 - df_d01['limit_pct'])
    df_d01['O'] = df_d01['open'] * df_d01['adjfactor']
    df_d01['H'] = df_d01['high'] * df_d01['adjfactor']
    df_d01['L'] = df_d01['low'] * df_d01['adjfactor']
    df_d01['C'] = df_d01['close'] * df_d01['adjfactor']
    df_d01['A'] = df_d01['avgprice'] * df_d01['adjfactor']
    df_d01['V'] = df_d01['vol']

    df_d01['UD'] = (df_d01['close'] < df_d01['uplimit']) & (df_d01['close'] > df_d01['downlimit']) 
    df_d01['IND'] = df_d01['ind.sw'].apply(lambda x : _extractInd(x))
    df_d01['CAP'] = df_d01['share.total'] * df_d01['close'] * 1e4
    df_d01['date'] = df_d01['date'].astype(str).astype('datetime64[ns]')

    df_d01 = df_d01[['date', 'ticker', 'open', 'high', 'low', 'close', 'avgprice', 'vol', 'amount']]
    # df_d01 = df_d01[['date', 'ticker', 'O', 'H', 'L', 'C', 'A', 'V', 'amount']]
    # df_d01.rename({'O' : 'open', 'H' : 'high', 'L' : 'low', 'C' : 'close', 'A' : 'avgprice', 'V' : 'vol'}, axis=1, inplace=True)

    return df_d01
