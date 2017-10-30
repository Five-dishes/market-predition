import datetime
import numpy as np


class DateUtil:
    @staticmethod
    def date_to_int(date: datetime.datetime) -> np.int64:
        return np.int64(date.strftime("%Y%m%d"))

    @staticmethod
    def int_to_date(date: np.int64) -> datetime.datetime:
        return datetime.datetime.strptime(str(date), "%Y%m%d")

    @staticmethod
    def minus_no_skip(subtrahend: int, day: datetime.datetime) -> np.int64:
        date = day - datetime.timedelta(subtrahend)
        return DateUtil.date_to_int(date)

    def __init__(self, skip_festival_=True):
        self.skip_festival = skip_festival_
        spring_festival_end = np.int64(20150227)
        spring_festival_start =\
            DateUtil.minus_no_skip(13, DateUtil.int_to_date(spring_festival_end))
        self.festival_dates = [
            (spring_festival_start, spring_festival_end, 14)
        ]

    def minus(self, subtrahend: int, day: np.int64 or int) -> np.int64 or None:
        for s, e, p in self.festival_dates:
            if s <= day <= e:
                return None
        day = DateUtil.int_to_date(day)
        no_skip = DateUtil.minus_no_skip(subtrahend, day)
        if not self.skip_festival:
            return no_skip
        for s, e, p in self.festival_dates:
            if s <= no_skip <= e:
                return DateUtil.minus_no_skip(p, DateUtil.int_to_date(no_skip))
            else:
                return no_skip


if __name__ == '__main__':
    du = DateUtil()
    for date in [20150228, 20150301, 20150227, 20150214, 20150213]:
        print('{} - 1 = {}'.format(date, du.minus(1, date)))


