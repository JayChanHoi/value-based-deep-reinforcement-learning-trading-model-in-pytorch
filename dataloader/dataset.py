import pandas as pd

def read_data(path, start_date, split_date):
    data = pd.read_csv(path)
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date')

    if start_date:
        train_data = data[start_date:split_date]
    else:
        train_data = data[:split_date]
    test_data = data[split_date:]

    return train_data, test_data

if __name__ == '__main__':
    train_data, test_data = read_data(
        '/Users/jaychan/Desktop/youtube_project/rl_trading_agent/Data/Stocks/a.us.txt',
        start_date='2006-01-01',
        split_date='2016-01-01',
    )

    print('train\n',train_data)
    print('test\n',test_data)