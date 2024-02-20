import pandas as pd


def save_df(df, name, cols):

    dir = '../data/' + name

    if cols is not None:
        for col in cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ','.join(map(str, x)))

    df.to_csv(dir, index=False)


def unravel_df(df, cols):
    for col in cols:
        df[col] = df[col].apply(lambda x: x.replace("[","").replace("]",""))
        df[col] = df[col].apply(lambda x: [float(x) for x in x.split(',')])
    return df