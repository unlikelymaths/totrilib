import datetime

def datetime_str():
    return datetime.datetime.now().strftime("%-y.%m.%d-%-H.%M.%S")
