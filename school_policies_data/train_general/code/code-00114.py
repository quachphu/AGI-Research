            # The str() method on weekday objects doesn't generate
            # RFC5545-compliant strings, so we should modify that.
            original_rule = dict(self._original_rule)
            wday_strings = []
            for wday in original_rule['byweekday']:
                if wday.n:
                    wday_strings.append('{n:+d}{wday}'.format(
                        n=wday.n,
                        wday=repr(wday)[0:2]))
                else:
                    wday_strings.append(repr(wday))

            original_rule['byweekday'] = wday_strings
        else:
            original_rule = self._original_rule

        partfmt = '{name}={vals}'
        for name, key in [('BYSETPOS', 'bysetpos'),
                          ('BYMONTH', 'bymonth'),
                          ('BYMONTHDAY', 'bymonthday'),
                          ('BYYEARDAY', 'byyearday'),
                          ('BYWEEKNO', 'byweekno'),
                          ('BYDAY', 'byweekday'),
                          ('BYHOUR', 'byhour'),
                          ('BYMINUTE', 'byminute'),
                          ('BYSECOND', 'bysecond'),
                          ('BYEASTER', 'byeaster')]:
            value = original_rule.get(key)
            if value:
                parts.append(partfmt.format(name=name, vals=(','.join(str(v)
                                                             for v in value))))

        output.append('RRULE:' + ';'.join(parts))
        return '\n'.join(output)

    def replace(self, **kwargs):
        """Return new rrule with same attributes except for those attributes given new
           values by whichever keyword arguments are specified."""
        new_kwargs = {"interval": self._interval,
                      "count": self._count,
                      "dtstart": self._dtstart,
                      "freq": self._freq,
                      "until": self._until,
                      "wkst": self._wkst,
                      "cache": False if self._cache is None else True }
        new_kwargs.update(self._original_rule)
        new_kwargs.update(kwargs)
        return rrule(**new_kwargs)

    def _iter(self):
        year, month, day, hour, minute, second, weekday, yearday, _ = \
            self._dtstart.timetuple()

        # Some local variables to speed things up a bit
        freq = self._freq
        interval = self._interval
        wkst = self._wkst
        until = self._until
        bymonth = self._bymonth
        byweekno = self._byweekno
        byyearday = self._byyearday
        byweekday = self._byweekday
        byeaster = self._byeaster
        bymonthday = self._bymonthday
        bynmonthday = self._bynmonthday
        bysetpos = self._bysetpos
        byhour = self._byhour
        byminute = self._byminute
        bysecond = self._bysecond

        ii = _iterinfo(self)
        ii.rebuild(year, month)

        getdayset = {YEARLY: ii.ydayset,
                     MONTHLY: ii.mdayset,
                     WEEKLY: ii.wdayset,
                     DAILY: ii.ddayset,
                     HOURLY: ii.ddayset,
                     MINUTELY: ii.ddayset,
                     SECONDLY: ii.ddayset}[freq]

        if freq < HOURLY:
            timeset = self._timeset
        else:
            gettimeset = {HOURLY: ii.htimeset,
                          MINUTELY: ii.mtimeset,
                          SECONDLY: ii.stimeset}[freq]
            if ((freq >= HOURLY and
                 self._byhour and hour not in self._byhour) or
                (freq >= MINUTELY and
                 self._byminute and minute not in self._byminute) or
                (freq >= SECONDLY and
                 self._bysecond and second not in self._bysecond)):
                timeset = ()
            else:
                timeset = gettimeset(hour, minute, second)

        total = 0
        count = self._count
        while True:
            # Get dayset with the right frequency
            dayset, start, end = getdayset(year, month, day)

            # Do the "hard" work ;-)
            filtered = False
            for i in dayset[start:end]:
                if ((bymonth and ii.mmask[i] not in bymonth) or
                    (byweekno and not ii.wnomask[i]) or
                    (byweekday and ii.wdaymask[i] not in byweekday) or
                    (ii.nwdaymask and not ii.nwdaymask[i]) or
                    (byeaster and not ii.eastermask[i]) or
                    ((bymonthday or bynmonthday) and
                     ii.mdaymask[i] not in bymonthday and
                     ii.nmdaymask[i] not in bynmonthday) or
                    (byyearday and
                     ((i < ii.yearlen and i+1 not in byyearday and
                       -ii.yearlen+i not in byyearday) or
                      (i >= ii.yearlen and i+1-ii.yearlen not in byyearday and
                       -ii.nextyearlen+i-ii.yearlen not in byyearday)))):
                    dayset[i] = None
                    filtered = True

            # Output results
            if bysetpos and timeset:
                poslist = []
                for pos in bysetpos:
                    if pos < 0:
                        daypos, timepos = divmod(pos, len(timeset))
                    else:
                        daypos, timepos = divmod(pos-1, len(timeset))
                    try:
                        i = [x for x in dayset[start:end]
                             if x is not None][daypos]
                        time = timeset[timepos]
                    except IndexError:
                        pass
                    else:
                        date = datetime.date.fromordinal(ii.yearordinal+i)
                        res = datetime.datetime.combine(date, time)
                        if res not in poslist:
                            poslist.append(res)
                poslist.sort()
                for res in poslist:
                    if until and res > until:
                        self._len = total
                        return
                    elif res >= self._dtstart:
                        if count is not None:
                            count -= 1
                            if count < 0:
                                self._len = total
                                return
                        total += 1
                        yield res
            else:
                for i in dayset[start:end]:
                    if i is not None:
                        date = datetime.date.fromordinal(ii.yearordinal + i)
                        for time in timeset:
                            res = datetime.datetime.combine(date, time)
                            if until and res > until:
                                self._len = total
                                return
                            elif res >= self._dtstart:
                                if count is not None:
                                    count -= 1
                                    if count < 0:
                                        self._len = total
                                        return

                                total += 1
                                yield res

            # Handle frequency and interval
            fixday = False
            if freq == YEARLY:
                year += interval
                if year > datetime.MAXYEAR:
                    self._len = total
                    return
                ii.rebuild(year, month)
            elif freq == MONTHLY:
                month += interval
                if month > 12:
                    div, mod = divmod(month, 12)
                    month = mod
                    year += div
                    if month == 0:
                        month = 12
                        year -= 1
                    if year > datetime.MAXYEAR:
                        self._len = total
                        return
                ii.rebuild(year, month)
            elif freq == WEEKLY:
                if wkst > weekday:
                    day += -(weekday+1+(6-wkst))+self._interval*7
                else:
                    day += -(weekday-wkst)+self._interval*7
                weekday = wkst
                fixday = True
            elif freq == DAILY:
                day += interval
                fixday = True
            elif freq == HOURLY:
                if filtered:
                    # Jump to one iteration before next day
                    hour += ((23-hour)//interval)*interval

                if byhour:
                    ndays, hour = self.__mod_distance(value=hour,
                                                      byxxx=self._byhour,
                                                      base=24)
                else:
                    ndays, hour = divmod(hour+interval, 24)

                if ndays:
                    day += ndays
                    fixday = True

                timeset = gettimeset(hour, minute, second)
            elif freq == MINUTELY:
                if filtered:
                    # Jump to one iteration before next day
                    minute += ((1439-(hour*60+minute))//interval)*interval

                valid = False
                rep_rate = (24*60)
                for j in range(rep_rate // gcd(interval, rep_rate)):
                    if byminute:
                        nhours, minute = \
                            self.__mod_distance(value=minute,
                                                byxxx=self._byminute,
                                                base=60)
                    else:
                        nhours, minute = divmod(minute+interval, 60)

                    div, hour = divmod(hour+nhours, 24)
                    if div:
                        day += div
                        fixday = True
                        filtered = False

                    if not byhour or hour in byhour:
                        valid = True
                        break

                if not valid:
                    raise ValueError('Invalid combination of interval and ' +
                                     'byhour resulting in empty rule.')

                timeset = gettimeset(hour, minute, second)
            elif freq == SECONDLY:
                if filtered:
                    # Jump to one iteration before next day
                    second += (((86399 - (hour * 3600 + minute * 60 + second))
                                // interval) * interval)

                rep_rate = (24 * 3600)
                valid = False
                for j in range(0, rep_rate // gcd(interval, rep_rate)):
                    if bysecond:
                        nminutes, second = \
                            self.__mod_distance(value=second,
                                                byxxx=self._bysecond,
                                                base=60)
                    else:
                        nminutes, second = divmod(second+interval, 60)

                    div, minute = divmod(minute+nminutes, 60)
                    if div:
                        hour += div
                        div, hour = divmod(hour, 24)
                        if div:
                            day += div
                            fixday = True

                    if ((not byhour or hour in byhour) and
                            (not byminute or minute in byminute) and
                            (not bysecond or second in bysecond)):
                        valid = True
                        break

                if not valid:
                    raise ValueError('Invalid combination of interval, ' +
                                     'byhour and byminute resulting in empty' +
                                     ' rule.')

                timeset = gettimeset(hour, minute, second)

            if fixday and day > 28:
                daysinmonth = calendar.monthrange(year, month)[1]
                if day > daysinmonth:
                    while day > daysinmonth:
                        day -= daysinmonth
                        month += 1
                        if month == 13:
                            month = 1
                            year += 1
                            if year > datetime.MAXYEAR:
                                self._len = total
                                return
                        daysinmonth = calendar.monthrange(year, month)[1]
                    ii.rebuild(year, month)

    def __construct_byset(self, start, byxxx, base):
        """
        If a `BYXXX` sequence is passed to the constructor at the same level as
        `FREQ` (e.g. `FREQ=HOURLY,BYHOUR={2,4,7},INTERVAL=3`), there are some
        specifications which cannot be reached given some starting conditions.

        This occurs whenever the interval is not coprime with the base of a
        given unit and the difference between the starting position and the
        ending position is not coprime with the greatest common denominator
        between the interval and the base. For example, with a FREQ of hourly
        starting at 17:00 and an interval of 4, the only valid values for
        BYHOUR would be {21, 1, 5, 9, 13, 17}, because 4 and 24 are not
        coprime.

        :param start:
            Specifies the starting position.
        :param byxxx:
            An iterable containing the list of allowed values.
        :param base:
            The largest allowable value for the specified frequency (e.g.
            24 hours, 60 minutes).

        This does not preserve the type of the iterable, returning a set, since
        the values should be unique and the order is irrelevant, this will
        speed up later lookups.

        In the event of an empty set, raises a :exception:`ValueError`, as this
        results in an empty rrule.
        """

        cset = set()

        # Support a single byxxx value.
        if isinstance(byxxx, integer_types):
            byxxx = (byxxx, )

        for num in byxxx:
            i_gcd = gcd(self._interval, base)
            # Use divmod rather than % because we need to wrap negative nums.
            if i_gcd == 1 or divmod(num - start, i_gcd)[1] == 0:
                cset.add(num)

        if len(cset) == 0:
            raise ValueError("Invalid rrule byxxx generates an empty set.")

        return cset

    def __mod_distance(self, value, byxxx, base):
        """
        Calculates the next value in a sequence where the `FREQ` parameter is
        specified along with a `BYXXX` parameter at the same "level"
        (e.g. `HOURLY` specified with `BYHOUR`).

        :param value:
            The old value of the component.
        :param byxxx:
            The `BYXXX` set, which should have been generated by
            `rrule._construct_byset`, or something else which checks that a
            valid rule is present.
        :param base:
            The largest allowable value for the specified frequency (e.g.
            24 hours, 60 minutes).

        If a valid value is not found after `base` iterations (the maximum
        number before the sequence would start to repeat), this raises a
        :exception:`ValueError`, as no valid values were found.

        This returns a tuple of `divmod(n*interval, base)`, where `n` is the
        smallest number of `interval` repetitions until the next specified
        value in `byxxx` is found.
        """
        accumulator = 0
        for ii in range(1, base + 1):
            # Using divmod() over % to account for negative intervals
            div, value = divmod(value + self._interval, base)
            accumulator += div
            if value in byxxx:
                return (accumulator, value)


class _iterinfo(object):
    __slots__ = ["rrule", "lastyear", "lastmonth",
                 "yearlen", "nextyearlen", "yearordinal", "yearweekday",
                 "mmask", "mrange", "mdaymask", "nmdaymask",
                 "wdaymask", "wnomask", "nwdaymask", "eastermask"]

    def __init__(self, rrule):
        for attr in self.__slots__:
            setattr(self, attr, None)
        self.rrule = rrule

    def rebuild(self, year, month):
        # Every mask is 7 days longer to handle cross-year weekly periods.
        rr = self.rrule
        if year != self.lastyear:
            self.yearlen = 365 + calendar.isleap(year)
            self.nextyearlen = 365 + calendar.isleap(year + 1)
            firstyday = datetime.date(year, 1, 1)
            self.yearordinal = firstyday.toordinal()
            self.yearweekday = firstyday.weekday()

            wday = datetime.date(year, 1, 1).weekday()
            if self.yearlen == 365:
                self.mmask = M365MASK
                self.mdaymask = MDAY365MASK
                self.nmdaymask = NMDAY365MASK
                self.wdaymask = WDAYMASK[wday:]
                self.mrange = M365RANGE
            else:
                self.mmask = M366MASK
                self.mdaymask = MDAY366MASK
                self.nmdaymask = NMDAY366MASK
                self.wdaymask = WDAYMASK[wday:]
                self.mrange = M366RANGE

            if not rr._byweekno:
                self.wnomask = None
            else:
                self.wnomask = [0]*(self.yearlen+7)
                # no1wkst = firstwkst = self.wdaymask.index(rr._wkst)
                no1wkst = firstwkst = (7-self.yearweekday+rr._wkst) % 7
                if no1wkst >= 4:
                    no1wkst = 0
                    # Number of days in the year, plus the days we got
                    # from last year.
                    wyearlen = self.yearlen+(self.yearweekday-rr._wkst) % 7
                else:
                    # Number of days in the year, minus the days we
                    # left in last year.
                    wyearlen = self.yearlen-no1wkst
                div, mod = divmod(wyearlen, 7)
                numweeks = div+mod//4
                for n in rr._byweekno:
                    if n < 0:
                        n += numweeks+1
                    if not (0 < n <= numweeks):
                        continue
                    if n > 1:
                        i = no1wkst+(n-1)*7
                        if no1wkst != firstwkst:
                            i -= 7-firstwkst
                    else:
                        i = no1wkst
                    for j in range(7):
                        self.wnomask[i] = 1
                        i += 1
                        if self.wdaymask[i] == rr._wkst:
                            break
                if 1 in rr._byweekno:
                    # Check week number 1 of next year as well
                    # TODO: Check -numweeks for next year.
                    i = no1wkst+numweeks*7
                    if no1wkst != firstwkst:
                        i -= 7-firstwkst
                    if i < self.yearlen:
                        # If week starts in next year, we
                        # don't care about it.
                        for j in range(7):
                            self.wnomask[i] = 1
                            i += 1
                            if self.wdaymask[i] == rr._wkst:
                                break
                if no1wkst:
                    # Check last week number of last year as
                    # well. If no1wkst is 0, either the year
                    # started on week start, or week number 1
                    # got days from last year, so there are no
                    # days from last year's last week number in
                    # this year.
                    if -1 not in rr._byweekno:
                        lyearweekday = datetime.date(year-1, 1, 1).weekday()
                        lno1wkst = (7-lyearweekday+rr._wkst) % 7
                        lyearlen = 365+calendar.isleap(year-1)
                        if lno1wkst >= 4:
                            lno1wkst = 0
                            lnumweeks = 52+(lyearlen +
                                            (lyearweekday-rr._wkst) % 7) % 7//4
                        else:
                            lnumweeks = 52+(self.yearlen-no1wkst) % 7//4
                    else:
                        lnumweeks = -1
                    if lnumweeks in rr._byweekno:
                        for i in range(no1wkst):
                            self.wnomask[i] = 1

        if (rr._bynweekday and (month != self.lastmonth or
                                year != self.lastyear)):
            ranges = []
            if rr._freq == YEARLY:
                if rr._bymonth:
                    for month in rr._bymonth:
                        ranges.append(self.mrange[month-1:month+1])
                else:
                    ranges = [(0, self.yearlen)]
            elif rr._freq == MONTHLY:
                ranges = [self.mrange[month-1:month+1]]
            if ranges:
                # Weekly frequency won't get here, so we may not
                # care about cross-year weekly periods.
                self.nwdaymask = [0]*self.yearlen
                for first, last in ranges:
                    last -= 1
                    for wday, n in rr._bynweekday:
                        if n < 0:
                            i = last+(n+1)*7
                            i -= (self.wdaymask[i]-wday) % 7
                        else:
                            i = first+(n-1)*7
                            i += (7-self.wdaymask[i]+wday) % 7
                        if first <= i <= last:
                            self.nwdaymask[i] = 1

        if rr._byeaster:
            self.eastermask = [0]*(self.yearlen+7)
            eyday = easter.easter(year).toordinal()-self.yearordinal
            for offset in rr._byeaster:
                self.eastermask[eyday+offset] = 1

        self.lastyear = year
        self.lastmonth = month

    def ydayset(self, year, month, day):
        return list(range(self.yearlen)), 0, self.yearlen

    def mdayset(self, year, month, day):
        dset = [None]*self.yearlen
        start, end = self.mrange[month-1:month+1]
        for i in range(start, end):
            dset[i] = i
        return dset, start, end

    def wdayset(self, year, month, day):
        # We need to handle cross-year weeks here.
        dset = [None]*(self.yearlen+7)
        i = datetime.date(year, month, day).toordinal()-self.yearordinal
        start = i
        for j in range(7):
            dset[i] = i
            i += 1
            # if (not (0 <= i < self.yearlen) or
            #    self.wdaymask[i] == self.rrule._wkst):
            # This will cross the year boundary, if necessary.
            if self.wdaymask[i] == self.rrule._wkst:
                break
        return dset, start, i

    def ddayset(self, year, month, day):
        dset = [None] * self.yearlen
        i = datetime.date(year, month, day).toordinal() - self.yearordinal
        dset[i] = i
        return dset, i, i + 1

    def htimeset(self, hour, minute, second):
        tset = []
        rr = self.rrule
        for minute in rr._byminute:
            for second in rr._bysecond:
                tset.append(datetime.time(hour, minute, second,
                                          tzinfo=rr._tzinfo))
        tset.sort()
        return tset

    def mtimeset(self, hour, minute, second):
        tset = []
        rr = self.rrule
        for second in rr._bysecond:
            tset.append(datetime.time(hour, minute, second, tzinfo=rr._tzinfo))
        tset.sort()
        return tset

    def stimeset(self, hour, minute, second):
        return (datetime.time(hour, minute, second,
                tzinfo=self.rrule._tzinfo),)


class rruleset(rrulebase):
    """ The rruleset type allows more complex recurrence setups, mixing
    multiple rules, dates, exclusion rules, and exclusion dates. The type
    constructor takes the following keyword arguments:

    :param cache: If True, caching of results will be enabled, improving
                  performance of multiple queries considerably. """

    class _genitem(object):
        def __init__(self, genlist, gen):
            try:
                self.dt = advance_iterator(gen)
                genlist.append(self)
            except StopIteration:
                pass
            self.genlist = genlist
            self.gen = gen

        def __next__(self):
            try:
                self.dt = advance_iterator(self.gen)
            except StopIteration:
                if self.genlist[0] is self:
                    heapq.heappop(self.genlist)
                else:
                    self.genlist.remove(self)
                    heapq.heapify(self.genlist)

        next = __next__

        def __lt__(self, other):
            return self.dt < other.dt

        def __gt__(self, other):
            return self.dt > other.dt

        def __eq__(self, other):
            return self.dt == other.dt

        def __ne__(self, other):
            return self.dt != other.dt

    def __init__(self, cache=False):
        super(rruleset, self).__init__(cache)
        self._rrule = []
        self._rdate = []
        self._exrule = []
        self._exdate = []

    @_invalidates_cache
    def rrule(self, rrule):
        """ Include the given :py:class:`rrule` instance in the recurrence set
            generation. """
        self._rrule.append(rrule)

    @_invalidates_cache
    def rdate(self, rdate):
        """ Include the given :py:class:`datetime` instance in the recurrence
            set generation. """
        self._rdate.append(rdate)

    @_invalidates_cache
    def exrule(self, exrule):
        """ Include the given rrule instance in the recurrence set exclusion
            list. Dates which are part of the given recurrence rules will not
            be generated, even if some inclusive rrule or rdate matches them.
        """
        self._exrule.append(exrule)

    @_invalidates_cache
    def exdate(self, exdate):
        """ Include the given datetime instance in the recurrence set
            exclusion list. Dates included that way will not be generated,
            even if some inclusive rrule or rdate matches them. """
        self._exdate.append(exdate)

    def _iter(self):
        rlist = []
        self._rdate.sort()
        self._genitem(rlist, iter(self._rdate))
        for gen in [iter(x) for x in self._rrule]:
            self._genitem(rlist, gen)
        exlist = []
        self._exdate.sort()
        self._genitem(exlist, iter(self._exdate))
        for gen in [iter(x) for x in self._exrule]:
            self._genitem(exlist, gen)
        lastdt = None
        total = 0
        heapq.heapify(rlist)
        heapq.heapify(exlist)
        while rlist:
            ritem = rlist[0]
            if not lastdt or lastdt != ritem.dt:
                while exlist and exlist[0] < ritem:
                    exitem = exlist[0]
                    advance_iterator(exitem)
                    if exlist and exlist[0] is exitem:
                        heapq.heapreplace(exlist, exitem)
                if not exlist or ritem != exlist[0]:
                    total += 1
                    yield ritem.dt
                lastdt = ritem.dt
            advance_iterator(ritem)
            if rlist and rlist[0] is ritem:
                heapq.heapreplace(rlist, ritem)
        self._len = total




class _rrulestr(object):
    """ Parses a string representation of a recurrence rule or set of
    recurrence rules.

    :param s:
        Required, a string defining one or more recurrence rules.

    :param dtstart:
        If given, used as the default recurrence start if not specified in the
        rule string.

    :param cache:
        If set ``True`` caching of results will be enabled, improving
        performance of multiple queries considerably.

    :param unfold:
        If set ``True`` indicates that a rule string is split over more
        than one line and should be joined before processing.

    :param forceset:
        If set ``True`` forces a :class:`dateutil.rrule.rruleset` to
        be returned.

    :param compatible:
        If set ``True`` forces ``unfold`` and ``forceset`` to be ``True``.

    :param ignoretz:
        If set ``True``, time zones in parsed strings are ignored and a naive
        :class:`datetime.datetime` object is returned.

    :param tzids:
        If given, a callable or mapping used to retrieve a
        :class:`datetime.tzinfo` from a string representation.
        Defaults to :func:`dateutil.tz.gettz`.

    :param tzinfos:
        Additional time zone names / aliases which may be present in a string
        representation.  See :func:`dateutil.parser.parse` for more
        information.

    :return:
        Returns a :class:`dateutil.rrule.rruleset` or
        :class:`dateutil.rrule.rrule`
    """

    _freq_map = {"YEARLY": YEARLY,
                 "MONTHLY": MONTHLY,
                 "WEEKLY": WEEKLY,
                 "DAILY": DAILY,
                 "HOURLY": HOURLY,
                 "MINUTELY": MINUTELY,
                 "SECONDLY": SECONDLY}

    _weekday_map = {"MO": 0, "TU": 1, "WE": 2, "TH": 3,
                    "FR": 4, "SA": 5, "SU": 6}

    def _handle_int(self, rrkwargs, name, value, **kwargs):
        rrkwargs[name.lower()] = int(value)

    def _handle_int_list(self, rrkwargs, name, value, **kwargs):
        rrkwargs[name.lower()] = [int(x) for x in value.split(',')]

    _handle_INTERVAL = _handle_int
    _handle_COUNT = _handle_int
    _handle_BYSETPOS = _handle_int_list
    _handle_BYMONTH = _handle_int_list
    _handle_BYMONTHDAY = _handle_int_list
    _handle_BYYEARDAY = _handle_int_list
    _handle_BYEASTER = _handle_int_list
    _handle_BYWEEKNO = _handle_int_list
    _handle_BYHOUR = _handle_int_list
    _handle_BYMINUTE = _handle_int_list
    _handle_BYSECOND = _handle_int_list

    def _handle_FREQ(self, rrkwargs, name, value, **kwargs):
        rrkwargs["freq"] = self._freq_map[value]

    def _handle_UNTIL(self, rrkwargs, name, value, **kwargs):
        global parser
        if not parser:
            from dateutil import parser
        try:
            rrkwargs["until"] = parser.parse(value,
                                             ignoretz=kwargs.get("ignoretz"),
                                             tzinfos=kwargs.get("tzinfos"))
        except ValueError:
            raise ValueError("invalid until date")

    def _handle_WKST(self, rrkwargs, name, value, **kwargs):
        rrkwargs["wkst"] = self._weekday_map[value]

    def _handle_BYWEEKDAY(self, rrkwargs, name, value, **kwargs):
        """
        Two ways to specify this: +1MO or MO(+1)
        """
        l = []
        for wday in value.split(','):
            if '(' in wday:
                # If it's of the form TH(+1), etc.
                splt = wday.split('(')
                w = splt[0]
                n = int(splt[1][:-1])
            elif len(wday):
                # If it's of the form +1MO
                for i in range(len(wday)):
                    if wday[i] not in '+-0123456789':
                        break
                n = wday[:i] or None
                w = wday[i:]
                if n:
                    n = int(n)
            else:
                raise ValueError("Invalid (empty) BYDAY specification.")

            l.append(weekdays[self._weekday_map[w]](n))
        rrkwargs["byweekday"] = l

    _handle_BYDAY = _handle_BYWEEKDAY

    def _parse_rfc_rrule(self, line,
                         dtstart=None,
                         cache=False,
                         ignoretz=False,
                         tzinfos=None):
        if line.find(':') != -1:
            name, value = line.split(':')
            if name != "RRULE":
                raise ValueError("unknown parameter name")
        else:
            value = line
        rrkwargs = {}
        for pair in value.split(';'):
            name, value = pair.split('=')
            name = name.upper()
            value = value.upper()
            try:
                getattr(self, "_handle_"+name)(rrkwargs, name, value,
                                               ignoretz=ignoretz,
                                               tzinfos=tzinfos)
            except AttributeError:
                raise ValueError("unknown parameter '%s'" % name)
            except (KeyError, ValueError):
                raise ValueError("invalid '%s': %s" % (name, value))
        return rrule(dtstart=dtstart, cache=cache, **rrkwargs)

    def _parse_date_value(self, date_value, parms, rule_tzids,
                          ignoretz, tzids, tzinfos):
        global parser
        if not parser:
            from dateutil import parser

        datevals = []
        value_found = False
        TZID = None

        for parm in parms:
            if parm.startswith("TZID="):
                try:
                    tzkey = rule_tzids[parm.split('TZID=')[-1]]
                except KeyError:
                    continue
                if tzids is None:
                    from . import tz
                    tzlookup = tz.gettz
                elif callable(tzids):
                    tzlookup = tzids
                else:
                    tzlookup = getattr(tzids, 'get', None)
                    if tzlookup is None:
                        msg = ('tzids must be a callable, mapping, or None, '
                               'not %s' % tzids)
                        raise ValueError(msg)

                TZID = tzlookup(tzkey)
                continue

            # RFC 5445 3.8.2.4: The VALUE parameter is optional, but may be found
            # only once.
            if parm not in {"VALUE=DATE-TIME", "VALUE=DATE"}:
                raise ValueError("unsupported parm: " + parm)
            else:
                if value_found:
                    msg = ("Duplicate value parameter found in: " + parm)
                    raise ValueError(msg)
                value_found = True

        for datestr in date_value.split(','):
            date = parser.parse(datestr, ignoretz=ignoretz, tzinfos=tzinfos)
            if TZID is not None:
                if date.tzinfo is None:
                    date = date.replace(tzinfo=TZID)
                else:
                    raise ValueError('DTSTART/EXDATE specifies multiple timezone')
            datevals.append(date)

        return datevals

    def _parse_rfc(self, s,
                   dtstart=None,
                   cache=False,
                   unfold=False,
                   forceset=False,
                   compatible=False,
                   ignoretz=False,
                   tzids=None,
                   tzinfos=None):
        global parser
        if compatible:
            forceset = True
            unfold = True

        TZID_NAMES = dict(map(
            lambda x: (x.upper(), x),
            re.findall('TZID=(?P<name>[^:]+):', s)
        ))
        s = s.upper()
        if not s.strip():
            raise ValueError("empty string")
        if unfold:
            lines = s.splitlines()
            i = 0
            while i < len(lines):
                line = lines[i].rstrip()
                if not line:
                    del lines[i]
                elif i > 0 and line[0] == " ":
                    lines[i-1] += line[1:]
                    del lines[i]
                else:
                    i += 1
        else:
            lines = s.split()
        if (not forceset and len(lines) == 1 and (s.find(':') == -1 or
                                                  s.startswith('RRULE:'))):
            return self._parse_rfc_rrule(lines[0], cache=cache,
                                         dtstart=dtstart, ignoretz=ignoretz,
                                         tzinfos=tzinfos)
        else:
            rrulevals = []
            rdatevals = []
            exrulevals = []
            exdatevals = []
            for line in lines:
                if not line:
                    continue
                if line.find(':') == -1:
                    name = "RRULE"
                    value = line
                else:
                    name, value = line.split(':', 1)
                parms = name.split(';')
                if not parms:
                    raise ValueError("empty property name")
                name = parms[0]
                parms = parms[1:]
                if name == "RRULE":
                    for parm in parms:
                        raise ValueError("unsupported RRULE parm: "+parm)
                    rrulevals.append(value)
                elif name == "RDATE":
                    for parm in parms:
                        if parm != "VALUE=DATE-TIME":
                            raise ValueError("unsupported RDATE parm: "+parm)
                    rdatevals.append(value)
                elif name == "EXRULE":
                    for parm in parms:
                        raise ValueError("unsupported EXRULE parm: "+parm)
                    exrulevals.append(value)
                elif name == "EXDATE":
                    exdatevals.extend(
                        self._parse_date_value(value, parms,
                                               TZID_NAMES, ignoretz,
                                               tzids, tzinfos)
                    )
                elif name == "DTSTART":
                    dtvals = self._parse_date_value(value, parms, TZID_NAMES,
                                                    ignoretz, tzids, tzinfos)
                    if len(dtvals) != 1:
                        raise ValueError("Multiple DTSTART values specified:" +
                                         value)
                    dtstart = dtvals[0]
                else:
                    raise ValueError("unsupported property: "+name)
            if (forceset or len(rrulevals) > 1 or rdatevals
                    or exrulevals or exdatevals):
                if not parser and (rdatevals or exdatevals):
                    from dateutil import parser
                rset = rruleset(cache=cache)
                for value in rrulevals:
                    rset.rrule(self._parse_rfc_rrule(value, dtstart=dtstart,
                                                     ignoretz=ignoretz,
                                                     tzinfos=tzinfos))
                for value in rdatevals:
                    for datestr in value.split(','):
                        rset.rdate(parser.parse(datestr,
                                                ignoretz=ignoretz,
                                                tzinfos=tzinfos))
                for value in exrulevals:
                    rset.exrule(self._parse_rfc_rrule(value, dtstart=dtstart,
                                                      ignoretz=ignoretz,
                                                      tzinfos=tzinfos))
                for value in exdatevals:
                    rset.exdate(value)
                if compatible and dtstart:
                    rset.rdate(dtstart)
                return rset
            else:
                return self._parse_rfc_rrule(rrulevals[0],
                                             dtstart=dtstart,
                                             cache=cache,
                                             ignoretz=ignoretz,
                                             tzinfos=tzinfos)

    def __call__(self, s, **kwargs):
        return self._parse_rfc(s, **kwargs)


rrulestr = _rrulestr()

# vim:ts=4:sw=4:et
<|endoftext|># -*- coding: utf-8 -*-
import datetime
import calendar

import operator
from math import copysign

from six import integer_types
from warnings import warn

from ._common import weekday

MO, TU, WE, TH, FR, SA, SU = weekdays = tuple(weekday(x) for x in range(7))

__all__ = ["relativedelta", "MO", "TU", "WE", "TH", "FR", "SA", "SU"]


class relativedelta(object):
    """
    The relativedelta type is designed to be applied to an existing datetime and
    can replace specific components of that datetime, or represents an interval
    of time.

    It is based on the specification of the excellent work done by M.-A. Lemburg
    in his
    `mx.DateTime <https://www.egenix.com/products/python/mxBase/mxDateTime/>`_ extension.
    However, notice that this type does *NOT* implement the same algorithm as
    his work. Do *NOT* expect it to behave like mx.DateTime's counterpart.

    There are two different ways to build a relativedelta instance. The
    first one is passing it two date/datetime classes::

        relativedelta(datetime1, datetime2)

    The second one is passing it any number of the following keyword arguments::

        relativedelta(arg1=x,arg2=y,arg3=z...)

        year, month, day, hour, minute, second, microsecond:
            Absolute information (argument is singular); adding or subtracting a
            relativedelta with absolute information does not perform an arithmetic
            operation, but rather REPLACES the corresponding value in the
            original datetime with the value(s) in relativedelta.

        years, months, weeks, days, hours, minutes, seconds, microseconds:
            Relative information, may be negative (argument is plural); adding
            or subtracting a relativedelta with relative information performs
            the corresponding arithmetic operation on the original datetime value
            with the information in the relativedelta.

        weekday:
            One of the weekday instances (MO, TU, etc) available in the
            relativedelta module. These instances may receive a parameter N,
            specifying the Nth weekday, which could be positive or negative
            (like MO(+1) or MO(-2)). Not specifying it is the same as specifying
            +1. You can also use an integer, where 0=MO. This argument is always
            relative e.g. if the calculated date is already Monday, using MO(1)
            or MO(-1) won't change the day. To effectively make it absolute, use
            it in combination with the day argument (e.g. day=1, MO(1) for first
            Monday of the month).

        leapdays:
            Will add given days to the date found, if year is a leap
            year, and the date found is post 28 of february.

        yearday, nlyearday:
            Set the yearday or the non-leap year day (jump leap days).
            These are converted to day/month/leapdays information.

    There are relative and absolute forms of the keyword
    arguments. The plural is relative, and the singular is
    absolute. For each argument in the order below, the absolute form
    is applied first (by setting each attribute to that value) and
    then the relative form (by adding the value to the attribute).

    The order of attributes considered when this relativedelta is
    added to a datetime is:

    1. Year
    2. Month
    3. Day
    4. Hours
    5. Minutes
    6. Seconds
    7. Microseconds

    Finally, weekday is applied, using the rule described above.

    For example

    >>> from datetime import datetime
    >>> from dateutil.relativedelta import relativedelta, MO
    >>> dt = datetime(2018, 4, 9, 13, 37, 0)
    >>> delta = relativedelta(hours=25, day=1, weekday=MO(1))
    >>> dt + delta
    datetime.datetime(2018, 4, 2, 14, 37)

    First, the day is set to 1 (the first of the month), then 25 hours
    are added, to get to the 2nd day and 14th hour, finally the
    weekday is applied, but since the 2nd is already a Monday there is
    no effect.

    """

    def __init__(self, dt1=None, dt2=None,
                 years=0, months=0, days=0, leapdays=0, weeks=0,
                 hours=0, minutes=0, seconds=0, microseconds=0,
                 year=None, month=None, day=None, weekday=None,
                 yearday=None, nlyearday=None,
                 hour=None, minute=None, second=None, microsecond=None):

        if dt1 and dt2:
            # datetime is a subclass of date. So both must be date
            if not (isinstance(dt1, datetime.date) and
                    isinstance(dt2, datetime.date)):
                raise TypeError("relativedelta only diffs datetime/date")

            # We allow two dates, or two datetimes, so we coerce them to be
            # of the same type
            if (isinstance(dt1, datetime.datetime) !=
                    isinstance(dt2, datetime.datetime)):
                if not isinstance(dt1, datetime.datetime):
                    dt1 = datetime.datetime.fromordinal(dt1.toordinal())
                elif not isinstance(dt2, datetime.datetime):
                    dt2 = datetime.datetime.fromordinal(dt2.toordinal())

            self.years = 0
            self.months = 0
            self.days = 0
            self.leapdays = 0
            self.hours = 0
            self.minutes = 0
            self.seconds = 0
            self.microseconds = 0
            self.year = None
            self.month = None
            self.day = None
            self.weekday = None
            self.hour = None
            self.minute = None
            self.second = None
            self.microsecond = None
            self._has_time = 0

            # Get year / month delta between the two
            months = (dt1.year - dt2.year) * 12 + (dt1.month - dt2.month)
            self._set_months(months)

            # Remove the year/month delta so the timedelta is just well-defined
            # time units (seconds, days and microseconds)
            dtm = self.__radd__(dt2)

            # If we've overshot our target, make an adjustment
            if dt1 < dt2:
                compare = operator.gt
                increment = 1
            else:
                compare = operator.lt
                increment = -1

            while compare(dt1, dtm):
                months += increment
                self._set_months(months)
                dtm = self.__radd__(dt2)

            # Get the timedelta between the "months-adjusted" date and dt1
            delta = dt1 - dtm
            self.seconds = delta.seconds + delta.days * 86400
            self.microseconds = delta.microseconds
        else:
            # Check for non-integer values in integer-only quantities
            if any(x is not None and x != int(x) for x in (years, months)):
                raise ValueError("Non-integer years and months are "
                                 "ambiguous and not currently supported.")

            # Relative information
            self.years = int(years)
            self.months = int(months)
            self.days = days + weeks * 7
            self.leapdays = leapdays
            self.hours = hours
            self.minutes = minutes
            self.seconds = seconds
            self.microseconds = microseconds

            # Absolute information
            self.year = year
            self.month = month
            self.day = day
            self.hour = hour
            self.minute = minute
            self.second = second
            self.microsecond = microsecond

            if any(x is not None and int(x) != x
                   for x in (year, month, day, hour,
                             minute, second, microsecond)):
                # For now we'll deprecate floats - later it'll be an error.
                warn("Non-integer value passed as absolute information. " +
                     "This is not a well-defined condition and will raise " +
                     "errors in future versions.", DeprecationWarning)

            if isinstance(weekday, integer_types):
                self.weekday = weekdays[weekday]
            else:
                self.weekday = weekday

            yday = 0
            if nlyearday:
                yday = nlyearday
            elif yearday:
                yday = yearday
                if yearday > 59:
                    self.leapdays = -1
            if yday:
                ydayidx = [31, 59, 90, 120, 151, 181, 212,
                           243, 273, 304, 334, 366]
                for idx, ydays in enumerate(ydayidx):
                    if yday <= ydays:
                        self.month = idx+1
                        if idx == 0:
                            self.day = yday
                        else:
                            self.day = yday-ydayidx[idx-1]
                        break
                else:
                    raise ValueError("invalid year day (%d)" % yday)

        self._fix()

    def _fix(self):
        if abs(self.microseconds) > 999999:
            s = _sign(self.microseconds)
            div, mod = divmod(self.microseconds * s, 1000000)
            self.microseconds = mod * s
            self.seconds += div * s
        if abs(self.seconds) > 59:
            s = _sign(self.seconds)
            div, mod = divmod(self.seconds * s, 60)
            self.seconds = mod * s
            self.minutes += div * s
        if abs(self.minutes) > 59:
            s = _sign(self.minutes)
            div, mod = divmod(self.minutes * s, 60)
            self.minutes = mod * s
            self.hours += div * s
        if abs(self.hours) > 23:
            s = _sign(self.hours)
            div, mod = divmod(self.hours * s, 24)
            self.hours = mod * s
            self.days += div * s
        if abs(self.months) > 11:
            s = _sign(self.months)
            div, mod = divmod(self.months * s, 12)
            self.months = mod * s
            self.years += div * s
        if (self.hours or self.minutes or self.seconds or self.microseconds
                or self.hour is not None or self.minute is not None or
                self.second is not None or self.microsecond is not None):
            self._has_time = 1
        else:
            self._has_time = 0

    @property
    def weeks(self):
        return int(self.days / 7.0)

    @weeks.setter
    def weeks(self, value):
        self.days = self.days - (self.weeks * 7) + value * 7

    def _set_months(self, months):
        self.months = months
        if abs(self.months) > 11:
            s = _sign(self.months)
            div, mod = divmod(self.months * s, 12)
            self.months = mod * s
            self.years = div * s
        else:
            self.years = 0

    def normalized(self):
        """
        Return a version of this object represented entirely using integer
        values for the relative attributes.

        >>> relativedelta(days=1.5, hours=2).normalized()
        relativedelta(days=+1, hours=+14)

        :return:
            Returns a :class:`dateutil.relativedelta.relativedelta` object.
        """
        # Cascade remainders down (rounding each to roughly nearest microsecond)
        days = int(self.days)

        hours_f = round(self.hours + 24 * (self.days - days), 11)
        hours = int(hours_f)

        minutes_f = round(self.minutes + 60 * (hours_f - hours), 10)
        minutes = int(minutes_f)

        seconds_f = round(self.seconds + 60 * (minutes_f - minutes), 8)
        seconds = int(seconds_f)

        microseconds = round(self.microseconds + 1e6 * (seconds_f - seconds))

        # Constructor carries overflow back up with call to _fix()
        return self.__class__(years=self.years, months=self.months,
                              days=days, hours=hours, minutes=minutes,
                              seconds=seconds, microseconds=microseconds,
                              leapdays=self.leapdays, year=self.year,
                              month=self.month, day=self.day,
                              weekday=self.weekday, hour=self.hour,
                              minute=self.minute, second=self.second,
                              microsecond=self.microsecond)

    def __add__(self, other):
        if isinstance(other, relativedelta):
            return self.__class__(years=other.years + self.years,
                                 months=other.months + self.months,
                                 days=other.days + self.days,
                                 hours=other.hours + self.hours,
                                 minutes=other.minutes + self.minutes,
                                 seconds=other.seconds + self.seconds,
                                 microseconds=(other.microseconds +
                                               self.microseconds),
                                 leapdays=other.leapdays or self.leapdays,
                                 year=(other.year if other.year is not None
                                       else self.year),
                                 month=(other.month if other.month is not None
                                        else self.month),
                                 day=(other.day if other.day is not None
                                      else self.day),
                                 weekday=(other.weekday if other.weekday is not None
                                          else self.weekday),
                                 hour=(other.hour if other.hour is not None
                                       else self.hour),
                                 minute=(other.minute if other.minute is not None
                                         else self.minute),
                                 second=(other.second if other.second is not None
                                         else self.second),
                                 microsecond=(other.microsecond if other.microsecond
                                              is not None else
                                              self.microsecond))
        if isinstance(other, datetime.timedelta):
            return self.__class__(years=self.years,
                                  months=self.months,
                                  days=self.days + other.days,
                                  hours=self.hours,
                                  minutes=self.minutes,
                                  seconds=self.seconds + other.seconds,
                                  microseconds=self.microseconds + other.microseconds,
                                  leapdays=self.leapdays,
                                  year=self.year,
                                  month=self.month,
                                  day=self.day,
                                  weekday=self.weekday,
                                  hour=self.hour,
                                  minute=self.minute,
                                  second=self.second,
                                  microsecond=self.microsecond)
        if not isinstance(other, datetime.date):
            return NotImplemented
        elif self._has_time and not isinstance(other, datetime.datetime):
            other = datetime.datetime.fromordinal(other.toordinal())
        year = (self.year or other.year)+self.years
        month = self.month or other.month
        if self.months:
            assert 1 <= abs(self.months) <= 12
            month += self.months
            if month > 12:
                year += 1
                month -= 12
            elif month < 1:
                year -= 1
                month += 12
        day = min(calendar.monthrange(year, month)[1],
                  self.day or other.day)
        repl = {"year": year, "month": month, "day": day}
        for attr in ["hour", "minute", "second", "microsecond"]:
            value = getattr(self, attr)
            if value is not None:
                repl[attr] = value
        days = self.days
        if self.leapdays and month > 2 and calendar.isleap(year):
            days += self.leapdays
        ret = (other.replace(**repl)
               + datetime.timedelta(days=days,
                                    hours=self.hours,
                                    minutes=self.minutes,
                                    seconds=self.seconds,
                                    microseconds=self.microseconds))
        if self.weekday:
            weekday, nth = self.weekday.weekday, self.weekday.n or 1
            jumpdays = (abs(nth) - 1) * 7
            if nth > 0:
                jumpdays += (7 - ret.weekday() + weekday) % 7
            else:
                jumpdays += (ret.weekday() - weekday) % 7
                jumpdays *= -1
            ret += datetime.timedelta(days=jumpdays)
        return ret

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return self.__neg__().__radd__(other)

    def __sub__(self, other):
        if not isinstance(other, relativedelta):
            return NotImplemented   # In case the other object defines __rsub__
        return self.__class__(years=self.years - other.years,
                             months=self.months - other.months,
                             days=self.days - other.days,
                             hours=self.hours - other.hours,
                             minutes=self.minutes - other.minutes,
                             seconds=self.seconds - other.seconds,
                             microseconds=self.microseconds - other.microseconds,
                             leapdays=self.leapdays or other.leapdays,
                             year=(self.year if self.year is not None
                                   else other.year),
                             month=(self.month if self.month is not None else
                                    other.month),
                             day=(self.day if self.day is not None else
                                  other.day),
                             weekday=(self.weekday if self.weekday is not None else
                                      other.weekday),
                             hour=(self.hour if self.hour is not None else
                                   other.hour),
                             minute=(self.minute if self.minute is not None else
                                     other.minute),
                             second=(self.second if self.second is not None else
                                     other.second),
                             microsecond=(self.microsecond if self.microsecond
                                          is not None else
                                          other.microsecond))

    def __abs__(self):
        return self.__class__(years=abs(self.years),
                              months=abs(self.months),
                              days=abs(self.days),
                              hours=abs(self.hours),
                              minutes=abs(self.minutes),
                              seconds=abs(self.seconds),
                              microseconds=abs(self.microseconds),
                              leapdays=self.leapdays,
                              year=self.year,
                              month=self.month,
                              day=self.day,
                              weekday=self.weekday,
                              hour=self.hour,
                              minute=self.minute,
                              second=self.second,
                              microsecond=self.microsecond)

    def __neg__(self):
        return self.__class__(years=-self.years,
                             months=-self.months,
                             days=-self.days,
                             hours=-self.hours,
                             minutes=-self.minutes,
                             seconds=-self.seconds,
                             microseconds=-self.microseconds,
                             leapdays=self.leapdays,
                             year=self.year,
                             month=self.month,
                             day=self.day,
                             weekday=self.weekday,
                             hour=self.hour,
                             minute=self.minute,
                             second=self.second,
                             microsecond=self.microsecond)

    def __bool__(self):
        return not (not self.years and
                    not self.months and
                    not self.days and
                    not self.hours and
                    not self.minutes and
                    not self.seconds and
                    not self.microseconds and
                    not self.leapdays and
                    self.year is None and
                    self.month is None and
                    self.day is None and
                    self.weekday is None and
                    self.hour is None and
                    self.minute is None and
                    self.second is None and
                    self.microsecond is None)
    # Compatibility with Python 2.x
    __nonzero__ = __bool__

    def __mul__(self, other):
        try:
            f = float(other)
        except TypeError:
            return NotImplemented

        return self.__class__(years=int(self.years * f),
                             months=int(self.months * f),
                             days=int(self.days * f),
                             hours=int(self.hours * f),
                             minutes=int(self.minutes * f),
                             seconds=int(self.seconds * f),
                             microseconds=int(self.microseconds * f),
                             leapdays=self.leapdays,
                             year=self.year,
                             month=self.month,
                             day=self.day,
                             weekday=self.weekday,
                             hour=self.hour,
                             minute=self.minute,
                             second=self.second,
                             microsecond=self.microsecond)

    __rmul__ = __mul__

    def __eq__(self, other):
        if not isinstance(other, relativedelta):
            return NotImplemented
        if self.weekday or other.weekday:
            if not self.weekday or not other.weekday:
                return False
            if self.weekday.weekday != other.weekday.weekday:
                return False
            n1, n2 = self.weekday.n, other.weekday.n
            if n1 != n2 and not ((not n1 or n1 == 1) and (not n2 or n2 == 1)):
                return False
        return (self.years == other.years and
                self.months == other.months and
                self.days == other.days and
                self.hours == other.hours and
                self.minutes == other.minutes and
                self.seconds == other.seconds and
                self.microseconds == other.microseconds and
                self.leapdays == other.leapdays and
                self.year == other.year and
                self.month == other.month and
                self.day == other.day and
                self.hour == other.hour and
                self.minute == other.minute and
                self.second == other.second and
                self.microsecond == other.microsecond)

    def __hash__(self):
        return hash((
            self.weekday,
            self.years,
            self.months,
            self.days,
            self.hours,
            self.minutes,
            self.seconds,
            self.microseconds,
            self.leapdays,
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
            self.microsecond,
        ))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __div__(self, other):
        try:
            reciprocal = 1 / float(other)
        except TypeError:
            return NotImplemented

        return self.__mul__(reciprocal)

    __truediv__ = __div__

    def __repr__(self):
        l = []
        for attr in ["years", "months", "days", "leapdays",
                     "hours", "minutes", "seconds", "microseconds"]:
            value = getattr(self, attr)
            if value:
                l.append("{attr}={value:+g}".format(attr=attr, value=value))
        for attr in ["year", "month", "day", "weekday",
                     "hour", "minute", "second", "microsecond"]:
            value = getattr(self, attr)
            if value is not None:
                l.append("{attr}={value}".format(attr=attr, value=repr(value)))
        return "{classname}({attrs})".format(classname=self.__class__.__name__,
                                             attrs=", ".join(l))


def _sign(x):
    return int(copysign(1, x))

# vim:ts=4:sw=4:et
<|endoftext|># -*- coding: utf-8 -*-
import sys

try:
    from ._version import version as __version__
except ImportError:
    __version__ = 'unknown'

__all__ = ['easter', 'parser', 'relativedelta', 'rrule', 'tz',
           'utils', 'zoneinfo']

def __getattr__(name):
    import importlib

    if name in __all__:
        return importlib.import_module("." + name, __name__)
    raise AttributeError(
        "module {!r} has not attribute {!r}".format(__name__, name)
    )


def __dir__():
    # __dir__ should include all the lazy-importable modules as well.
    return [x for x in globals() if x not in sys.modules] + __all__
<|endoftext|>"""
Common code used in multiple modules.
"""


class weekday(object):
    __slots__ = ["weekday", "n"]

    def __init__(self, weekday, n=None):
        self.weekday = weekday
        self.n = n

    def __call__(self, n):
        if n == self.n:
            return self
        else:
            return self.__class__(self.weekday, n)

    def __eq__(self, other):
        try:
            if self.weekday != other.weekday or self.n != other.n:
                return False
        except AttributeError:
            return False
        return True

    def __hash__(self):
        return hash((
          self.weekday,
          self.n,
        ))

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        s = ("MO", "TU", "WE", "TH", "FR", "SA", "SU")[self.weekday]
        if not self.n:
            return s
        else:
            return "%s(%+d)" % (s, self.n)

# vim:ts=4:sw=4:et
<|endoftext|># -*- coding: utf-8 -*-
"""
This module provides an interface to the native time zone data on Windows,
including :py:class:`datetime.tzinfo` implementations.

Attempting to import this module on a non-Windows platform will raise an
:py:obj:`ImportError`.
"""
# This code was originally contributed by Jeffrey Harris.
import datetime
import struct

from six.moves import winreg
from six import text_type

try:
    import ctypes
    from ctypes import wintypes
except ValueError:
    # ValueError is raised on non-Windows systems for some horrible reason.
    raise ImportError("Running tzwin on non-Windows system")

from ._common import tzrangebase

__all__ = ["tzwin", "tzwinlocal", "tzres"]

ONEWEEK = datetime.timedelta(7)

TZKEYNAMENT = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Time Zones"
TZKEYNAME9X = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Time Zones"
TZLOCALKEYNAME = r"SYSTEM\CurrentControlSet\Control\TimeZoneInformation"


def _settzkeyname():
    handle = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
    try:
        winreg.OpenKey(handle, TZKEYNAMENT).Close()
        TZKEYNAME = TZKEYNAMENT
    except WindowsError:
        TZKEYNAME = TZKEYNAME9X
    handle.Close()
    return TZKEYNAME


TZKEYNAME = _settzkeyname()


class tzres(object):
    """
    Class for accessing ``tzres.dll``, which contains timezone name related
    resources.

    .. versionadded:: 2.5.0
    """
    p_wchar = ctypes.POINTER(wintypes.WCHAR)        # Pointer to a wide char

    def __init__(self, tzres_loc='tzres.dll'):
        # Load the user32 DLL so we can load strings from tzres
        user32 = ctypes.WinDLL('user32')

        # Specify the LoadStringW function
        user32.LoadStringW.argtypes = (wintypes.HINSTANCE,
                                       wintypes.UINT,
                                       wintypes.LPWSTR,
                                       ctypes.c_int)

        self.LoadStringW = user32.LoadStringW
        self._tzres = ctypes.WinDLL(tzres_loc)
        self.tzres_loc = tzres_loc

    def load_name(self, offset):
        """
        Load a timezone name from a DLL offset (integer).

        >>> from dateutil.tzwin import tzres
        >>> tzr = tzres()
        >>> print(tzr.load_name(112))
        'Eastern Standard Time'

        :param offset:
            A positive integer value referring to a string from the tzres dll.

        .. note::

            Offsets found in the registry are generally of the form
            ``@tzres.dll,-114``. The offset in this case is 114, not -114.

        """
        resource = self.p_wchar()
        lpBuffer = ctypes.cast(ctypes.byref(resource), wintypes.LPWSTR)
        nchar = self.LoadStringW(self._tzres._handle, offset, lpBuffer, 0)
        return resource[:nchar]

    def name_from_string(self, tzname_str):
        """
        Parse strings as returned from the Windows registry into the time zone
        name as defined in the registry.

        >>> from dateutil.tzwin import tzres
        >>> tzr = tzres()
        >>> print(tzr.name_from_string('@tzres.dll,-251'))
        'Dateline Daylight Time'
        >>> print(tzr.name_from_string('Eastern Standard Time'))
        'Eastern Standard Time'

        :param tzname_str:
            A timezone name string as returned from a Windows registry key.

        :return:
            Returns the localized timezone string from tzres.dll if the string
            is of the form `@tzres.dll,-offset`, else returns the input string.
        """
        if not tzname_str.startswith('@'):
            return tzname_str

        name_splt = tzname_str.split(',-')
        try:
            offset = int(name_splt[1])
        except:
            raise ValueError("Malformed timezone string.")

        return self.load_name(offset)


class tzwinbase(tzrangebase):
    """tzinfo class based on win32's timezones available in the registry."""
    def __init__(self):
        raise NotImplementedError('tzwinbase is an abstract base class')

    def __eq__(self, other):
        # Compare on all relevant dimensions, including name.
        if not isinstance(other, tzwinbase):
            return NotImplemented

        return  (self._std_offset == other._std_offset and
                 self._dst_offset == other._dst_offset and
                 self._stddayofweek == other._stddayofweek and
                 self._dstdayofweek == other._dstdayofweek and
                 self._stdweeknumber == other._stdweeknumber and
                 self._dstweeknumber == other._dstweeknumber and
                 self._stdhour == other._stdhour and
                 self._dsthour == other._dsthour and
                 self._stdminute == other._stdminute and
                 self._dstminute == other._dstminute and
                 self._std_abbr == other._std_abbr and
                 self._dst_abbr == other._dst_abbr)

    @staticmethod
    def list():
        """Return a list of all time zones known to the system."""
        with winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE) as handle:
            with winreg.OpenKey(handle, TZKEYNAME) as tzkey:
                result = [winreg.EnumKey(tzkey, i)
                          for i in range(winreg.QueryInfoKey(tzkey)[0])]
        return result

    def display(self):
        """
        Return the display name of the time zone.
        """
        return self._display

    def transitions(self, year):
        """
        For a given year, get the DST on and off transition times, expressed
        always on the standard time side. For zones with no transitions, this
        function returns ``None``.

        :param year:
            The year whose transitions you would like to query.

        :return:
            Returns a :class:`tuple` of :class:`datetime.datetime` objects,
            ``(dston, dstoff)`` for zones with an annual DST transition, or
            ``None`` for fixed offset zones.
        """

        if not self.hasdst:
            return None

        dston = picknthweekday(year, self._dstmonth, self._dstdayofweek,
                               self._dsthour, self._dstminute,
                               self._dstweeknumber)

        dstoff = picknthweekday(year, self._stdmonth, self._stddayofweek,
                                self._stdhour, self._stdminute,
                                self._stdweeknumber)

        # Ambiguous dates default to the STD side
        dstoff -= self._dst_base_offset

        return dston, dstoff

    def _get_hasdst(self):
        return self._dstmonth != 0

    @property
    def _dst_base_offset(self):
        return self._dst_base_offset_


class tzwin(tzwinbase):
    """
    Time zone object created from the zone info in the Windows registry

    These are similar to :py:class:`dateutil.tz.tzrange` objects in that
    the time zone data is provided in the format of a single offset rule
    for either 0 or 2 time zone transitions per year.

    :param: name
        The name of a Windows time zone key, e.g. "Eastern Standard Time".
        The full list of keys can be retrieved with :func:`tzwin.list`.
    """

    def __init__(self, name):
        self._name = name

        with winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE) as handle:
            tzkeyname = text_type("{kn}\\{name}").format(kn=TZKEYNAME, name=name)
            with winreg.OpenKey(handle, tzkeyname) as tzkey:
                keydict = valuestodict(tzkey)

        self._std_abbr = keydict["Std"]
        self._dst_abbr = keydict["Dlt"]

        self._display = keydict["Display"]

        # See http://ww_winreg.jsiinc.com/SUBA/tip0300/rh0398.htm
        tup = struct.unpack("=3l16h", keydict["TZI"])
        stdoffset = -tup[0]-tup[1]          # Bias + StandardBias * -1
        dstoffset = stdoffset-tup[2]        # + DaylightBias * -1
        self._std_offset = datetime.timedelta(minutes=stdoffset)
        self._dst_offset = datetime.timedelta(minutes=dstoffset)

        # for the meaning see the win32 TIME_ZONE_INFORMATION structure docs
        # http://msdn.microsoft.com/en-us/library/windows/desktop/ms725481(v=vs.85).aspx
        (self._stdmonth,
         self._stddayofweek,   # Sunday = 0
         self._stdweeknumber,  # Last = 5
         self._stdhour,
         self._stdminute) = tup[4:9]

        (self._dstmonth,
         self._dstdayofweek,   # Sunday = 0
         self._dstweeknumber,  # Last = 5
         self._dsthour,
         self._dstminute) = tup[12:17]

        self._dst_base_offset_ = self._dst_offset - self._std_offset
        self.hasdst = self._get_hasdst()

    def __repr__(self):
        return "tzwin(%s)" % repr(self._name)

    def __reduce__(self):
        return (self.__class__, (self._name,))


class tzwinlocal(tzwinbase):
    """
    Class representing the local time zone information in the Windows registry

    While :class:`dateutil.tz.tzlocal` makes system calls (via the :mod:`time`
    module) to retrieve time zone information, ``tzwinlocal`` retrieves the
    rules directly from the Windows registry and creates an object like
    :class:`dateutil.tz.tzwin`.

    Because Windows does not have an equivalent of :func:`time.tzset`, on
    Windows, :class:`dateutil.tz.tzlocal` instances will always reflect the
    time zone settings *at the time that the process was started*, meaning
    changes to the machine's time zone settings during the run of a program
    on Windows will **not** be reflected by :class:`dateutil.tz.tzlocal`.
    Because ``tzwinlocal`` reads the registry directly, it is unaffected by
    this issue.
    """
    def __init__(self):
        with winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE) as handle:
            with winreg.OpenKey(handle, TZLOCALKEYNAME) as tzlocalkey:
                keydict = valuestodict(tzlocalkey)

            self._std_abbr = keydict["StandardName"]
            self._dst_abbr = keydict["DaylightName"]

            try:
                tzkeyname = text_type('{kn}\\{sn}').format(kn=TZKEYNAME,
                                                          sn=self._std_abbr)
                with winreg.OpenKey(handle, tzkeyname) as tzkey:
                    _keydict = valuestodict(tzkey)
                    self._display = _keydict["Display"]
            except OSError:
                self._display = None

        stdoffset = -keydict["Bias"]-keydict["StandardBias"]
        dstoffset = stdoffset-keydict["DaylightBias"]

        self._std_offset = datetime.timedelta(minutes=stdoffset)
        self._dst_offset = datetime.timedelta(minutes=dstoffset)

        # For reasons unclear, in this particular key, the day of week has been
        # moved to the END of the SYSTEMTIME structure.
        tup = struct.unpack("=8h", keydict["StandardStart"])

        (self._stdmonth,
         self._stdweeknumber,  # Last = 5
         self._stdhour,
         self._stdminute) = tup[1:5]

        self._stddayofweek = tup[7]

        tup = struct.unpack("=8h", keydict["DaylightStart"])

        (self._dstmonth,
         self._dstweeknumber,  # Last = 5
         self._dsthour,
         self._dstminute) = tup[1:5]

        self._dstdayofweek = tup[7]

        self._dst_base_offset_ = self._dst_offset - self._std_offset
        self.hasdst = self._get_hasdst()

    def __repr__(self):
        return "tzwinlocal()"

    def __str__(self):
        # str will return the standard name, not the daylight name.
        return "tzwinlocal(%s)" % repr(self._std_abbr)

    def __reduce__(self):
        return (self.__class__, ())


def picknthweekday(year, month, dayofweek, hour, minute, whichweek):
    """ dayofweek == 0 means Sunday, whichweek 5 means last instance """
    first = datetime.datetime(year, month, 1, hour, minute)

    # This will work if dayofweek is ISO weekday (1-7) or Microsoft-style (0-6),
    # Because 7 % 7 = 0
    weekdayone = first.replace(day=((dayofweek - first.isoweekday()) % 7) + 1)
    wd = weekdayone + ((whichweek - 1) * ONEWEEK)
    if (wd.month != month):
        wd -= ONEWEEK

    return wd


def valuestodict(key):
    """Convert a registry key's values to a dictionary."""
    dout = {}
    size = winreg.QueryInfoKey(key)[1]
    tz_res = None

    for i in range(size):
        key_name, value, dtype = winreg.EnumValue(key, i)
        if dtype == winreg.REG_DWORD or dtype == winreg.REG_DWORD_LITTLE_ENDIAN:
            # If it's a DWORD (32-bit integer), it's stored as unsigned - convert
            # that to a proper signed integer
            if value & (1 << 31):
                value = value - (1 << 32)
        elif dtype == winreg.REG_SZ:
            # If it's a reference to the tzres DLL, load the actual string
            if value.startswith('@tzres'):
                tz_res = tz_res or tzres()
                value = tz_res.name_from_string(value)

            value = value.rstrip('\x00')    # Remove trailing nulls

        dout[key_name] = value

    return dout
<|endoftext|>from datetime import timedelta
import weakref
from collections import OrderedDict

from six.moves import _thread


class _TzSingleton(type):
    def __init__(cls, *args, **kwargs):
        cls.__instance = None
        super(_TzSingleton, cls).__init__(*args, **kwargs)

    def __call__(cls):
        if cls.__instance is None:
            cls.__instance = super(_TzSingleton, cls).__call__()
        return cls.__instance


class _TzFactory(type):
    def instance(cls, *args, **kwargs):
        """Alternate constructor that returns a fresh instance"""
        return type.__call__(cls, *args, **kwargs)


class _TzOffsetFactory(_TzFactory):
    def __init__(cls, *args, **kwargs):
        cls.__instances = weakref.WeakValueDictionary()
        cls.__strong_cache = OrderedDict()
        cls.__strong_cache_size = 8

        cls._cache_lock = _thread.allocate_lock()

    def __call__(cls, name, offset):
        if isinstance(offset, timedelta):
            key = (name, offset.total_seconds())
        else:
            key = (name, offset)

        instance = cls.__instances.get(key, None)
        if instance is None:
            instance = cls.__instances.setdefault(key,
                                                  cls.instance(name, offset))

        # This lock may not be necessary in Python 3. See GH issue #901
        with cls._cache_lock:
            cls.__strong_cache[key] = cls.__strong_cache.pop(key, instance)

            # Remove an item if the strong cache is overpopulated
            if len(cls.__strong_cache) > cls.__strong_cache_size:
                cls.__strong_cache.popitem(last=False)

        return instance


class _TzStrFactory(_TzFactory):
    def __init__(cls, *args, **kwargs):
        cls.__instances = weakref.WeakValueDictionary()
        cls.__strong_cache = OrderedDict()
        cls.__strong_cache_size = 8

        cls.__cache_lock = _thread.allocate_lock()

    def __call__(cls, s, posix_offset=False):
        key = (s, posix_offset)
        instance = cls.__instances.get(key, None)

        if instance is None:
            instance = cls.__instances.setdefault(key,
                cls.instance(s, posix_offset))

        # This lock may not be necessary in Python 3. See GH issue #901
        with cls.__cache_lock:
            cls.__strong_cache[key] = cls.__strong_cache.pop(key, instance)

            # Remove an item if the strong cache is overpopulated
            if len(cls.__strong_cache) > cls.__strong_cache_size:
                cls.__strong_cache.popitem(last=False)

        return instance

<|endoftext|># -*- coding: utf-8 -*-
from .tz import *
from .tz import __doc__

__all__ = ["tzutc", "tzoffset", "tzlocal", "tzfile", "tzrange",
           "tzstr", "tzical", "tzwin", "tzwinlocal", "gettz",
           "enfold", "datetime_ambiguous", "datetime_exists",
           "resolve_imaginary", "UTC", "DeprecatedTzFormatWarning"]


class DeprecatedTzFormatWarning(Warning):
    """Warning raised when time zones are parsed from deprecated formats."""
<|endoftext|># -*- coding: utf-8 -*-
"""
This module offers timezone implementations subclassing the abstract
:py:class:`datetime.tzinfo` type. There are classes to handle tzfile format
files (usually are in :file:`/etc/localtime`, :file:`/usr/share/zoneinfo`,
etc), TZ environment string (in all known formats), given ranges (with help
from relative deltas), local machine timezone, fixed offset timezone, and UTC
timezone.
"""
import datetime
import struct
import time
import sys
import os
import bisect
import weakref
from collections import OrderedDict

import six
from six import string_types
from six.moves import _thread
from ._common import tzname_in_python2, _tzinfo
from ._common import tzrangebase, enfold
from ._common import _validate_fromutc_inputs

from ._factories import _TzSingleton, _TzOffsetFactory
from ._factories import _TzStrFactory
try:
    from .win import tzwin, tzwinlocal
except ImportError:
    tzwin = tzwinlocal = None

# For warning about rounding tzinfo
from warnings import warn

ZERO = datetime.timedelta(0)
EPOCH = datetime.datetime(1970, 1, 1, 0, 0)
EPOCHORDINAL = EPOCH.toordinal()


@six.add_metaclass(_TzSingleton)
class tzutc(datetime.tzinfo):
    """
    This is a tzinfo object that represents the UTC time zone.

    **Examples:**

    .. doctest::

        >>> from datetime import *
        >>> from dateutil.tz import *

        >>> datetime.now()
        datetime.datetime(2003, 9, 27, 9, 40, 1, 521290)

        >>> datetime.now(tzutc())
        datetime.datetime(2003, 9, 27, 12, 40, 12, 156379, tzinfo=tzutc())

        >>> datetime.now(tzutc()).tzname()
        'UTC'

    .. versionchanged:: 2.7.0
        ``tzutc()`` is now a singleton, so the result of ``tzutc()`` will
        always return the same object.

        .. doctest::

            >>> from dateutil.tz import tzutc, UTC
            >>> tzutc() is tzutc()
            True
            >>> tzutc() is UTC
            True
    """
    def utcoffset(self, dt):
        return ZERO

    def dst(self, dt):
        return ZERO

    @tzname_in_python2
    def tzname(self, dt):
        return "UTC"

    def is_ambiguous(self, dt):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.


        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """
        return False

    @_validate_fromutc_inputs
    def fromutc(self, dt):
        """
        Fast track version of fromutc() returns the original ``dt`` object for
        any valid :py:class:`datetime.datetime` object.
        """
        return dt

    def __eq__(self, other):
        if not isinstance(other, (tzutc, tzoffset)):
            return NotImplemented

        return (isinstance(other, tzutc) or
                (isinstance(other, tzoffset) and other._offset == ZERO))

    __hash__ = None

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "%s()" % self.__class__.__name__

    __reduce__ = object.__reduce__


#: Convenience constant providing a :class:`tzutc()` instance
#:
#: .. versionadded:: 2.7.0
UTC = tzutc()


@six.add_metaclass(_TzOffsetFactory)
class tzoffset(datetime.tzinfo):
    """
    A simple class for representing a fixed offset from UTC.

    :param name:
        The timezone name, to be returned when ``tzname()`` is called.
    :param offset:
        The time zone offset in seconds, or (since version 2.6.0, represented
        as a :py:class:`datetime.timedelta` object).
    """
    def __init__(self, name, offset):
        self._name = name

        try:
            # Allow a timedelta
            offset = offset.total_seconds()
        except (TypeError, AttributeError):
            pass

        self._offset = datetime.timedelta(seconds=_get_supported_offset(offset))

    def utcoffset(self, dt):
        return self._offset

    def dst(self, dt):
        return ZERO

    @tzname_in_python2
    def tzname(self, dt):
        return self._name

    @_validate_fromutc_inputs
    def fromutc(self, dt):
        return dt + self._offset

    def is_ambiguous(self, dt):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.
        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """
        return False

    def __eq__(self, other):
        if not isinstance(other, tzoffset):
            return NotImplemented

        return self._offset == other._offset

    __hash__ = None

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               repr(self._name),
                               int(self._offset.total_seconds()))

    __reduce__ = object.__reduce__


class tzlocal(_tzinfo):
    """
    A :class:`tzinfo` subclass built around the ``time`` timezone functions.
    """
    def __init__(self):
        super(tzlocal, self).__init__()

        self._std_offset = datetime.timedelta(seconds=-time.timezone)
        if time.daylight:
            self._dst_offset = datetime.timedelta(seconds=-time.altzone)
        else:
            self._dst_offset = self._std_offset

        self._dst_saved = self._dst_offset - self._std_offset
        self._hasdst = bool(self._dst_saved)
        self._tznames = tuple(time.tzname)

    def utcoffset(self, dt):
        if dt is None and self._hasdst:
            return None

        if self._isdst(dt):
            return self._dst_offset
        else:
            return self._std_offset

    def dst(self, dt):
        if dt is None and self._hasdst:
            return None

        if self._isdst(dt):
            return self._dst_offset - self._std_offset
        else:
            return ZERO

    @tzname_in_python2
    def tzname(self, dt):
        return self._tznames[self._isdst(dt)]

    def is_ambiguous(self, dt):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.


        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """
        naive_dst = self._naive_is_dst(dt)
        return (not naive_dst and
                (naive_dst != self._naive_is_dst(dt - self._dst_saved)))

    def _naive_is_dst(self, dt):
        timestamp = _datetime_to_timestamp(dt)
        return time.localtime(timestamp + time.timezone).tm_isdst

    def _isdst(self, dt, fold_naive=True):
        # We can't use mktime here. It is unstable when deciding if
        # the hour near to a change is DST or not.
        #
        # timestamp = time.mktime((dt.year, dt.month, dt.day, dt.hour,
        #                         dt.minute, dt.second, dt.weekday(), 0, -1))
        # return time.localtime(timestamp).tm_isdst
        #
        # The code above yields the following result:
        #
        # >>> import tz, datetime
        # >>> t = tz.tzlocal()
        # >>> datetime.datetime(2003,2,15,23,tzinfo=t).tzname()
        # 'BRDT'
        # >>> datetime.datetime(2003,2,16,0,tzinfo=t).tzname()
        # 'BRST'
        # >>> datetime.datetime(2003,2,15,23,tzinfo=t).tzname()
        # 'BRST'
        # >>> datetime.datetime(2003,2,15,22,tzinfo=t).tzname()
        # 'BRDT'
        # >>> datetime.datetime(2003,2,15,23,tzinfo=t).tzname()
        # 'BRDT'
        #
        # Here is a more stable implementation:
        #
        if not self._hasdst:
            return False

        # Check for ambiguous times:
        dstval = self._naive_is_dst(dt)
        fold = getattr(dt, 'fold', None)

        if self.is_ambiguous(dt):
            if fold is not None:
                return not self._fold(dt)
            else:
                return True

        return dstval

    def __eq__(self, other):
        if isinstance(other, tzlocal):
            return (self._std_offset == other._std_offset and
                    self._dst_offset == other._dst_offset)
        elif isinstance(other, tzutc):
            return (not self._hasdst and
                    self._tznames[0] in {'UTC', 'GMT'} and
                    self._std_offset == ZERO)
        elif isinstance(other, tzoffset):
            return (not self._hasdst and
                    self._tznames[0] == other._name and
                    self._std_offset == other._offset)
        else:
            return NotImplemented

    __hash__ = None

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "%s()" % self.__class__.__name__

    __reduce__ = object.__reduce__


class _ttinfo(object):
    __slots__ = ["offset", "delta", "isdst", "abbr",
                 "isstd", "isgmt", "dstoffset"]

    def __init__(self):
        for attr in self.__slots__:
            setattr(self, attr, None)

    def __repr__(self):
        l = []
        for attr in self.__slots__:
            value = getattr(self, attr)
            if value is not None:
                l.append("%s=%s" % (attr, repr(value)))
        return "%s(%s)" % (self.__class__.__name__, ", ".join(l))

    def __eq__(self, other):
        if not isinstance(other, _ttinfo):
            return NotImplemented

        return (self.offset == other.offset and
                self.delta == other.delta and
                self.isdst == other.isdst and
                self.abbr == other.abbr and
                self.isstd == other.isstd and
                self.isgmt == other.isgmt and
                self.dstoffset == other.dstoffset)

    __hash__ = None

    def __ne__(self, other):
        return not (self == other)

    def __getstate__(self):
        state = {}
        for name in self.__slots__:
            state[name] = getattr(self, name, None)
        return state

    def __setstate__(self, state):
        for name in self.__slots__:
            if name in state:
                setattr(self, name, state[name])


class _tzfile(object):
    """
    Lightweight class for holding the relevant transition and time zone
    information read from binary tzfiles.
    """
    attrs = ['trans_list', 'trans_list_utc', 'trans_idx', 'ttinfo_list',
             'ttinfo_std', 'ttinfo_dst', 'ttinfo_before', 'ttinfo_first']

    def __init__(self, **kwargs):
        for attr in self.attrs:
            setattr(self, attr, kwargs.get(attr, None))


class tzfile(_tzinfo):
    """
    This is a ``tzinfo`` subclass that allows one to use the ``tzfile(5)``
    format timezone files to extract current and historical zone information.

    :param fileobj:
        This can be an opened file stream or a file name that the time zone
        information can be read from.

    :param filename:
        This is an optional parameter specifying the source of the time zone
        information in the event that ``fileobj`` is a file object. If omitted
        and ``fileobj`` is a file stream, this parameter will be set either to
        ``fileobj``'s ``name`` attribute or to ``repr(fileobj)``.

    See `Sources for Time Zone and Daylight Saving Time Data
    <https://data.iana.org/time-zones/tz-link.html>`_ for more information.
    Time zone files can be compiled from the `IANA Time Zone database files
    <https://www.iana.org/time-zones>`_ with the `zic time zone compiler
    <https://www.freebsd.org/cgi/man.cgi?query=zic&sektion=8>`_

    .. note::

        Only construct a ``tzfile`` directly if you have a specific timezone
        file on disk that you want to read into a Python ``tzinfo`` object.
        If you want to get a ``tzfile`` representing a specific IANA zone,
        (e.g. ``'America/New_York'``), you should call
        :func:`dateutil.tz.gettz` with the zone identifier.


    **Examples:**

    Using the US Eastern time zone as an example, we can see that a ``tzfile``
    provides time zone information for the standard Daylight Saving offsets:

    .. testsetup:: tzfile

        from dateutil.tz import gettz
        from datetime import datetime

    .. doctest:: tzfile

        >>> NYC = gettz('America/New_York')
        >>> NYC
        tzfile('/usr/share/zoneinfo/America/New_York')

        >>> print(datetime(2016, 1, 3, tzinfo=NYC))     # EST
        2016-01-03 00:00:00-05:00

        >>> print(datetime(2016, 7, 7, tzinfo=NYC))     # EDT
        2016-07-07 00:00:00-04:00


    The ``tzfile`` structure contains a fully history of the time zone,
    so historical dates will also have the right offsets. For example, before
    the adoption of the UTC standards, New York used local solar  mean time:

    .. doctest:: tzfile

       >>> print(datetime(1901, 4, 12, tzinfo=NYC))    # LMT
       1901-04-12 00:00:00-04:56

    And during World War II, New York was on "Eastern War Time", which was a
    state of permanent daylight saving time:

    .. doctest:: tzfile

        >>> print(datetime(1944, 2, 7, tzinfo=NYC))    # EWT
        1944-02-07 00:00:00-04:00

    """

    def __init__(self, fileobj, filename=None):
        super(tzfile, self).__init__()

        file_opened_here = False
        if isinstance(fileobj, string_types):
            self._filename = fileobj
            fileobj = open(fileobj, 'rb')
            file_opened_here = True
        elif filename is not None:
            self._filename = filename
        elif hasattr(fileobj, "name"):
            self._filename = fileobj.name
        else:
            self._filename = repr(fileobj)

        if fileobj is not None:
            if not file_opened_here:
                fileobj = _nullcontext(fileobj)

            with fileobj as file_stream:
                tzobj = self._read_tzfile(file_stream)

            self._set_tzdata(tzobj)

    def _set_tzdata(self, tzobj):
        """ Set the time zone data of this object from a _tzfile object """
        # Copy the relevant attributes over as private attributes
        for attr in _tzfile.attrs:
            setattr(self, '_' + attr, getattr(tzobj, attr))

    def _read_tzfile(self, fileobj):
        out = _tzfile()

        # From tzfile(5):
        #
        # The time zone information files used by tzset(3)
        # begin with the magic characters "TZif" to identify
        # them as time zone information files, followed by
        # sixteen bytes reserved for future use, followed by
        # six four-byte values of type long, written in a
        # ``standard'' byte order (the high-order  byte
        # of the value is written first).
        if fileobj.read(4).decode() != "TZif":
            raise ValueError("magic not found")

        fileobj.read(16)

        (
            # The number of UTC/local indicators stored in the file.
            ttisgmtcnt,

            # The number of standard/wall indicators stored in the file.
            ttisstdcnt,

            # The number of leap seconds for which data is
            # stored in the file.
            leapcnt,

            # The number of "transition times" for which data
            # is stored in the file.
            timecnt,

            # The number of "local time types" for which data
            # is stored in the file (must not be zero).
            typecnt,

            # The  number  of  characters  of "time zone
            # abbreviation strings" stored in the file.
            charcnt,

        ) = struct.unpack(">6l", fileobj.read(24))

        # The above header is followed by tzh_timecnt four-byte
        # values  of  type long,  sorted  in ascending order.
        # These values are written in ``standard'' byte order.
        # Each is used as a transition time (as  returned  by
        # time(2)) at which the rules for computing local time
        # change.

        if timecnt:
            out.trans_list_utc = list(struct.unpack(">%dl" % timecnt,
                                                    fileobj.read(timecnt*4)))
        else:
            out.trans_list_utc = []

        # Next come tzh_timecnt one-byte values of type unsigned
        # char; each one tells which of the different types of
        # ``local time'' types described in the file is associated
        # with the same-indexed transition time. These values
        # serve as indices into an array of ttinfo structures that
        # appears next in the file.

        if timecnt:
            out.trans_idx = struct.unpack(">%dB" % timecnt,
                                          fileobj.read(timecnt))
        else:
            out.trans_idx = []

        # Each ttinfo structure is written as a four-byte value
        # for tt_gmtoff  of  type long,  in  a  standard  byte
        # order, followed  by a one-byte value for tt_isdst
        # and a one-byte  value  for  tt_abbrind.   In  each
        # structure, tt_gmtoff  gives  the  number  of
        # seconds to be added to UTC, tt_isdst tells whether
        # tm_isdst should be set by  localtime(3),  and
        # tt_abbrind serves  as an index into the array of
        # time zone abbreviation characters that follow the
        # ttinfo structure(s) in the file.

        ttinfo = []

        for i in range(typecnt):
            ttinfo.append(struct.unpack(">lbb", fileobj.read(6)))

        abbr = fileobj.read(charcnt).decode()

        # Then there are tzh_leapcnt pairs of four-byte
        # values, written in  standard byte  order;  the
        # first  value  of  each pair gives the time (as
        # returned by time(2)) at which a leap second
        # occurs;  the  second  gives the  total  number of
        # leap seconds to be applied after the given time.
        # The pairs of values are sorted in ascending order
        # by time.

        # Not used, for now (but seek for correct file position)
        if leapcnt:
            fileobj.seek(leapcnt * 8, os.SEEK_CUR)

        # Then there are tzh_ttisstdcnt standard/wall
        # indicators, each stored as a one-byte value;
        # they tell whether the transition times associated
        # with local time types were specified as standard
        # time or wall clock time, and are used when
        # a time zone file is used in handling POSIX-style
        # time zone environment variables.

        if ttisstdcnt:
            isstd = struct.unpack(">%db" % ttisstdcnt,
                                  fileobj.read(ttisstdcnt))

        # Finally, there are tzh_ttisgmtcnt UTC/local
        # indicators, each stored as a one-byte value;
        # they tell whether the transition times associated
        # with local time types were specified as UTC or
        # local time, and are used when a time zone file
        # is used in handling POSIX-style time zone envi-
        # ronment variables.

        if ttisgmtcnt:
            isgmt = struct.unpack(">%db" % ttisgmtcnt,
                                  fileobj.read(ttisgmtcnt))

        # Build ttinfo list
        out.ttinfo_list = []
        for i in range(typecnt):
            gmtoff, isdst, abbrind = ttinfo[i]
            gmtoff = _get_supported_offset(gmtoff)
            tti = _ttinfo()
            tti.offset = gmtoff
            tti.dstoffset = datetime.timedelta(0)
            tti.delta = datetime.timedelta(seconds=gmtoff)
            tti.isdst = isdst
            tti.abbr = abbr[abbrind:abbr.find('\x00', abbrind)]
            tti.isstd = (ttisstdcnt > i and isstd[i] != 0)
            tti.isgmt = (ttisgmtcnt > i and isgmt[i] != 0)
            out.ttinfo_list.append(tti)

        # Replace ttinfo indexes for ttinfo objects.
        out.trans_idx = [out.ttinfo_list[idx] for idx in out.trans_idx]

        # Set standard, dst, and before ttinfos. before will be
        # used when a given time is before any transitions,
        # and will be set to the first non-dst ttinfo, or to
        # the first dst, if all of them are dst.
        out.ttinfo_std = None
        out.ttinfo_dst = None
        out.ttinfo_before = None
        if out.ttinfo_list:
            if not out.trans_list_utc:
                out.ttinfo_std = out.ttinfo_first = out.ttinfo_list[0]
            else:
                for i in range(timecnt-1, -1, -1):
                    tti = out.trans_idx[i]
                    if not out.ttinfo_std and not tti.isdst:
                        out.ttinfo_std = tti
                    elif not out.ttinfo_dst and tti.isdst:
                        out.ttinfo_dst = tti

                    if out.ttinfo_std and out.ttinfo_dst:
                        break
                else:
                    if out.ttinfo_dst and not out.ttinfo_std:
                        out.ttinfo_std = out.ttinfo_dst

                for tti in out.ttinfo_list:
                    if not tti.isdst:
                        out.ttinfo_before = tti
                        break
                else:
                    out.ttinfo_before = out.ttinfo_list[0]

        # Now fix transition times to become relative to wall time.
        #
        # I'm not sure about this. In my tests, the tz source file
        # is setup to wall time, and in the binary file isstd and
        # isgmt are off, so it should be in wall time. OTOH, it's
        # always in gmt time. Let me know if you have comments
        # about this.
        lastdst = None
        lastoffset = None
        lastdstoffset = None
        lastbaseoffset = None
        out.trans_list = []

        for i, tti in enumerate(out.trans_idx):
            offset = tti.offset
            dstoffset = 0

            if lastdst is not None:
                if tti.isdst:
                    if not lastdst:
                        dstoffset = offset - lastoffset

                    if not dstoffset and lastdstoffset:
                        dstoffset = lastdstoffset

                    tti.dstoffset = datetime.timedelta(seconds=dstoffset)
                    lastdstoffset = dstoffset

            # If a time zone changes its base offset during a DST transition,
            # then you need to adjust by the previous base offset to get the
            # transition time in local time. Otherwise you use the current
            # base offset. Ideally, I would have some mathematical proof of
            # why this is true, but I haven't really thought about it enough.
            baseoffset = offset - dstoffset
            adjustment = baseoffset
            if (lastbaseoffset is not None and baseoffset != lastbaseoffset
                    and tti.isdst != lastdst):
                # The base DST has changed
                adjustment = lastbaseoffset

            lastdst = tti.isdst
            lastoffset = offset
            lastbaseoffset = baseoffset

            out.trans_list.append(out.trans_list_utc[i] + adjustment)

        out.trans_idx = tuple(out.trans_idx)
        out.trans_list = tuple(out.trans_list)
        out.trans_list_utc = tuple(out.trans_list_utc)

        return out

    def _find_last_transition(self, dt, in_utc=False):
        # If there's no list, there are no transitions to find
        if not self._trans_list:
            return None

        timestamp = _datetime_to_timestamp(dt)

        # Find where the timestamp fits in the transition list - if the
        # timestamp is a transition time, it's part of the "after" period.
        trans_list = self._trans_list_utc if in_utc else self._trans_list
        idx = bisect.bisect_right(trans_list, timestamp)

        # We want to know when the previous transition was, so subtract off 1
        return idx - 1

    def _get_ttinfo(self, idx):
        # For no list or after the last transition, default to _ttinfo_std
        if idx is None or (idx + 1) >= len(self._trans_list):
            return self._ttinfo_std

        # If there is a list and the time is before it, return _ttinfo_before
        if idx < 0:
            return self._ttinfo_before

        return self._trans_idx[idx]

    def _find_ttinfo(self, dt):
        idx = self._resolve_ambiguous_time(dt)

        return self._get_ttinfo(idx)

    def fromutc(self, dt):
        """
        The ``tzfile`` implementation of :py:func:`datetime.tzinfo.fromutc`.

        :param dt:
            A :py:class:`datetime.datetime` object.

        :raises TypeError:
            Raised if ``dt`` is not a :py:class:`datetime.datetime` object.

        :raises ValueError:
            Raised if this is called with a ``dt`` which does not have this
            ``tzinfo`` attached.

        :return:
            Returns a :py:class:`datetime.datetime` object representing the
            wall time in ``self``'s time zone.
        """
        # These isinstance checks are in datetime.tzinfo, so we'll preserve
        # them, even if we don't care about duck typing.
        if not isinstance(dt, datetime.datetime):
            raise TypeError("fromutc() requires a datetime argument")

        if dt.tzinfo is not self:
            raise ValueError("dt.tzinfo is not self")

        # First treat UTC as wall time and get the transition we're in.
        idx = self._find_last_transition(dt, in_utc=True)
        tti = self._get_ttinfo(idx)

        dt_out = dt + datetime.timedelta(seconds=tti.offset)

        fold = self.is_ambiguous(dt_out, idx=idx)

        return enfold(dt_out, fold=int(fold))

    def is_ambiguous(self, dt, idx=None):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.


        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """
        if idx is None:
            idx = self._find_last_transition(dt)

        # Calculate the difference in offsets from current to previous
        timestamp = _datetime_to_timestamp(dt)
        tti = self._get_ttinfo(idx)

        if idx is None or idx <= 0:
            return False

        od = self._get_ttinfo(idx - 1).offset - tti.offset
        tt = self._trans_list[idx]          # Transition time

        return timestamp < tt + od

    def _resolve_ambiguous_time(self, dt):
        idx = self._find_last_transition(dt)

        # If we have no transitions, return the index
        _fold = self._fold(dt)
        if idx is None or idx == 0:
            return idx

        # If it's ambiguous and we're in a fold, shift to a different index.
        idx_offset = int(not _fold and self.is_ambiguous(dt, idx))

        return idx - idx_offset

    def utcoffset(self, dt):
        if dt is None:
            return None

        if not self._ttinfo_std:
            return ZERO

        return self._find_ttinfo(dt).delta

    def dst(self, dt):
        if dt is None:
            return None

        if not self._ttinfo_dst:
            return ZERO

        tti = self._find_ttinfo(dt)

        if not tti.isdst:
            return ZERO

        # The documentation says that utcoffset()-dst() must
        # be constant for every dt.
        return tti.dstoffset

    @tzname_in_python2
    def tzname(self, dt):
        if not self._ttinfo_std or dt is None:
            return None
        return self._find_ttinfo(dt).abbr

    def __eq__(self, other):
        if not isinstance(other, tzfile):
            return NotImplemented
        return (self._trans_list == other._trans_list and
                self._trans_idx == other._trans_idx and
                self._ttinfo_list == other._ttinfo_list)

    __hash__ = None

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, repr(self._filename))

    def __reduce__(self):
        return self.__reduce_ex__(None)

    def __reduce_ex__(self, protocol):
        return (self.__class__, (None, self._filename), self.__dict__)


class tzrange(tzrangebase):
    """
    The ``tzrange`` object is a time zone specified by a set of offsets and
    abbreviations, equivalent to the way the ``TZ`` variable can be specified
    in POSIX-like systems, but using Python delta objects to specify DST
    start, end and offsets.

    :param stdabbr:
        The abbreviation for standard time (e.g. ``'EST'``).

    :param stdoffset:
        An integer or :class:`datetime.timedelta` object or equivalent
        specifying the base offset from UTC.

        If unspecified, +00:00 is used.

    :param dstabbr:
        The abbreviation for DST / "Summer" time (e.g. ``'EDT'``).

        If specified, with no other DST information, DST is assumed to occur
        and the default behavior or ``dstoffset``, ``start`` and ``end`` is
        used. If unspecified and no other DST information is specified, it
        is assumed that this zone has no DST.

        If this is unspecified and other DST information is *is* specified,
        DST occurs in the zone but the time zone abbreviation is left
        unchanged.

    :param dstoffset:
        A an integer or :class:`datetime.timedelta` object or equivalent
        specifying the UTC offset during DST. If unspecified and any other DST
        information is specified, it is assumed to be the STD offset +1 hour.

    :param start:
        A :class:`relativedelta.relativedelta` object or equivalent specifying
        the time and time of year that daylight savings time starts. To
        specify, for example, that DST starts at 2AM on the 2nd Sunday in
        March, pass:

            ``relativedelta(hours=2, month=3, day=1, weekday=SU(+2))``

        If unspecified and any other DST information is specified, the default
        value is 2 AM on the first Sunday in April.

    :param end:
        A :class:`relativedelta.relativedelta` object or equivalent
        representing the time and time of year that daylight savings time
        ends, with the same specification method as in ``start``. One note is
        that this should point to the first time in the *standard* zone, so if
        a transition occurs at 2AM in the DST zone and the clocks are set back
        1 hour to 1AM, set the ``hours`` parameter to +1.


    **Examples:**

    .. testsetup:: tzrange

        from dateutil.tz import tzrange, tzstr

    .. doctest:: tzrange

        >>> tzstr('EST5EDT') == tzrange("EST", -18000, "EDT")
        True

        >>> from dateutil.relativedelta import *
        >>> range1 = tzrange("EST", -18000, "EDT")
        >>> range2 = tzrange("EST", -18000, "EDT", -14400,
        ...                  relativedelta(hours=+2, month=4, day=1,
        ...                                weekday=SU(+1)),
        ...                  relativedelta(hours=+1, month=10, day=31,
        ...                                weekday=SU(-1)))
        >>> tzstr('EST5EDT') == range1 == range2
        True

    """
    def __init__(self, stdabbr, stdoffset=None,
                 dstabbr=None, dstoffset=None,
                 start=None, end=None):

        global relativedelta
        from dateutil import relativedelta

        self._std_abbr = stdabbr
        self._dst_abbr = dstabbr

        try:
            stdoffset = stdoffset.total_seconds()
        except (TypeError, AttributeError):
            pass

        try:
            dstoffset = dstoffset.total_seconds()
        except (TypeError, AttributeError):
            pass

        if stdoffset is not None:
            self._std_offset = datetime.timedelta(seconds=stdoffset)
        else:
            self._std_offset = ZERO

        if dstoffset is not None:
            self._dst_offset = datetime.timedelta(seconds=dstoffset)
        elif dstabbr and stdoffset is not None:
            self._dst_offset = self._std_offset + datetime.timedelta(hours=+1)
        else:
            self._dst_offset = ZERO

        if dstabbr and start is None:
            self._start_delta = relativedelta.relativedelta(
                hours=+2, month=4, day=1, weekday=relativedelta.SU(+1))
        else:
            self._start_delta = start

        if dstabbr and end is None:
            self._end_delta = relativedelta.relativedelta(
                hours=+1, month=10, day=31, weekday=relativedelta.SU(-1))
        else:
            self._end_delta = end

        self._dst_base_offset_ = self._dst_offset - self._std_offset
        self.hasdst = bool(self._start_delta)

    def transitions(self, year):
        """
        For a given year, get the DST on and off transition times, expressed
        always on the standard time side. For zones with no transitions, this
        function returns ``None``.

        :param year:
            The year whose transitions you would like to query.

        :return:
            Returns a :class:`tuple` of :class:`datetime.datetime` objects,
            ``(dston, dstoff)`` for zones with an annual DST transition, or
            ``None`` for fixed offset zones.
        """
        if not self.hasdst:
            return None

        base_year = datetime.datetime(year, 1, 1)

        start = base_year + self._start_delta
        end = base_year + self._end_delta

        return (start, end)

    def __eq__(self, other):
        if not isinstance(other, tzrange):
            return NotImplemented

        return (self._std_abbr == other._std_abbr and
                self._dst_abbr == other._dst_abbr and
                self._std_offset == other._std_offset and
                self._dst_offset == other._dst_offset and
                self._start_delta == other._start_delta and
                self._end_delta == other._end_delta)

    @property
    def _dst_base_offset(self):
        return self._dst_base_offset_


@six.add_metaclass(_TzStrFactory)
class tzstr(tzrange):
    """
    ``tzstr`` objects are time zone objects specified by a time-zone string as
    it would be passed to a ``TZ`` variable on POSIX-style systems (see
    the `GNU C Library: TZ Variable`_ for more details).

    There is one notable exception, which is that POSIX-style time zones use an
    inverted offset format, so normally ``GMT+3`` would be parsed as an offset
    3 hours *behind* GMT. The ``tzstr`` time zone object will parse this as an
    offset 3 hours *ahead* of GMT. If you would like to maintain the POSIX
    behavior, pass a ``True`` value to ``posix_offset``.

    The :class:`tzrange` object provides the same functionality, but is
    specified using :class:`relativedelta.relativedelta` objects. rather than
    strings.

    :param s:
        A time zone string in ``TZ`` variable format. This can be a
        :class:`bytes` (2.x: :class:`str`), :class:`str` (2.x:
        :class:`unicode`) or a stream emitting unicode characters
        (e.g. :class:`StringIO`).

    :param posix_offset:
        Optional. If set to ``True``, interpret strings such as ``GMT+3`` or
        ``UTC+3`` as being 3 hours *behind* UTC rather than ahead, per the
        POSIX standard.

    .. caution::

        Prior to version 2.7.0, this function also supported time zones
        in the format:

            * ``EST5EDT,4,0,6,7200,10,0,26,7200,3600``
            * ``EST5EDT,4,1,0,7200,10,-1,0,7200,3600``

        This format is non-standard and has been deprecated; this function
        will raise a :class:`DeprecatedTZFormatWarning` until
        support is removed in a future version.

    .. _`GNU C Library: TZ Variable`:
        https://www.gnu.org/software/libc/manual/html_node/TZ-Variable.html
    """
    def __init__(self, s, posix_offset=False):
        global parser
        from dateutil.parser import _parser as parser

        self._s = s

        res = parser._parsetz(s)
        if res is None or res.any_unused_tokens:
            raise ValueError("unknown string format")

        # Here we break the compatibility with the TZ variable handling.
        # GMT-3 actually *means* the timezone -3.
        if res.stdabbr in ("GMT", "UTC") and not posix_offset:
            res.stdoffset *= -1

        # We must initialize it first, since _delta() needs
        # _std_offset and _dst_offset set. Use False in start/end
        # to avoid building it two times.
        tzrange.__init__(self, res.stdabbr, res.stdoffset,
                         res.dstabbr, res.dstoffset,
                         start=False, end=False)

        if not res.dstabbr:
            self._start_delta = None
            self._end_delta = None
        else:
            self._start_delta = self._delta(res.start)
            if self._start_delta:
                self._end_delta = self._delta(res.end, isend=1)

        self.hasdst = bool(self._start_delta)

    def _delta(self, x, isend=0):
        from dateutil import relativedelta
        kwargs = {}
        if x.month is not None:
            kwargs["month"] = x.month
            if x.weekday is not None:
                kwargs["weekday"] = relativedelta.weekday(x.weekday, x.week)
                if x.week > 0:
                    kwargs["day"] = 1
                else:
                    kwargs["day"] = 31
            elif x.day:
                kwargs["day"] = x.day
        elif x.yday is not None:
            kwargs["yearday"] = x.yday
        elif x.jyday is not None:
            kwargs["nlyearday"] = x.jyday
        if not kwargs:
            # Default is to start on first sunday of april, and end
            # on last sunday of october.
            if not isend:
                kwargs["month"] = 4
                kwargs["day"] = 1
                kwargs["weekday"] = relativedelta.SU(+1)
            else:
                kwargs["month"] = 10
                kwargs["day"] = 31
                kwargs["weekday"] = relativedelta.SU(-1)
        if x.time is not None:
            kwargs["seconds"] = x.time
        else:
            # Default is 2AM.
            kwargs["seconds"] = 7200
        if isend:
            # Convert to standard time, to follow the documented way
            # of working with the extra hour. See the documentation
            # of the tzinfo class.
            delta = self._dst_offset - self._std_offset
            kwargs["seconds"] -= delta.seconds + delta.days * 86400
        return relativedelta.relativedelta(**kwargs)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, repr(self._s))


class _tzicalvtzcomp(object):
    def __init__(self, tzoffsetfrom, tzoffsetto, isdst,
                 tzname=None, rrule=None):
        self.tzoffsetfrom = datetime.timedelta(seconds=tzoffsetfrom)
        self.tzoffsetto = datetime.timedelta(seconds=tzoffsetto)
        self.tzoffsetdiff = self.tzoffsetto - self.tzoffsetfrom
        self.isdst = isdst
        self.tzname = tzname
        self.rrule = rrule


class _tzicalvtz(_tzinfo):
    def __init__(self, tzid, comps=[]):
        super(_tzicalvtz, self).__init__()

        self._tzid = tzid
        self._comps = comps
        self._cachedate = []
        self._cachecomp = []
        self._cache_lock = _thread.allocate_lock()

    def _find_comp(self, dt):
        if len(self._comps) == 1:
            return self._comps[0]

        dt = dt.replace(tzinfo=None)

        try:
            with self._cache_lock:
                return self._cachecomp[self._cachedate.index(
                    (dt, self._fold(dt)))]
        except ValueError:
            pass

        lastcompdt = None
        lastcomp = None

        for comp in self._comps:
            compdt = self._find_compdt(comp, dt)

            if compdt and (not lastcompdt or lastcompdt < compdt):
                lastcompdt = compdt
                lastcomp = comp

        if not lastcomp:
            # RFC says nothing about what to do when a given
            # time is before the first onset date. We'll look for the
            # first standard component, or the first component, if
            # none is found.
            for comp in self._comps:
                if not comp.isdst:
                    lastcomp = comp
                    break
            else:
                lastcomp = comp[0]

        with self._cache_lock:
            self._cachedate.insert(0, (dt, self._fold(dt)))
            self._cachecomp.insert(0, lastcomp)

            if len(self._cachedate) > 10:
                self._cachedate.pop()
                self._cachecomp.pop()

        return lastcomp

    def _find_compdt(self, comp, dt):
        if comp.tzoffsetdiff < ZERO and self._fold(dt):
            dt -= comp.tzoffsetdiff

        compdt = comp.rrule.before(dt, inc=True)

        return compdt

    def utcoffset(self, dt):
        if dt is None:
            return None

        return self._find_comp(dt).tzoffsetto

    def dst(self, dt):
        comp = self._find_comp(dt)
        if comp.isdst:
            return comp.tzoffsetdiff
        else:
            return ZERO

    @tzname_in_python2
    def tzname(self, dt):
        return self._find_comp(dt).tzname

    def __repr__(self):
        return "<tzicalvtz %s>" % repr(self._tzid)

    __reduce__ = object.__reduce__


class tzical(object):
    """
    This object is designed to parse an iCalendar-style ``VTIMEZONE`` structure
    as set out in `RFC 5545`_ Section 4.6.5 into one or more `tzinfo` objects.

    :param `fileobj`:
        A file or stream in iCalendar format, which should be UTF-8 encoded
        with CRLF endings.

    .. _`RFC 5545`: https://tools.ietf.org/html/rfc5545
    """
    def __init__(self, fileobj):
        global rrule
        from dateutil import rrule

        if isinstance(fileobj, string_types):
            self._s = fileobj
            # ical should be encoded in UTF-8 with CRLF
            fileobj = open(fileobj, 'r')
        else:
            self._s = getattr(fileobj, 'name', repr(fileobj))
            fileobj = _nullcontext(fileobj)

        self._vtz = {}

        with fileobj as fobj:
            self._parse_rfc(fobj.read())

    def keys(self):
        """
        Retrieves the available time zones as a list.
        """
        return list(self._vtz.keys())

    def get(self, tzid=None):
        """
        Retrieve a :py:class:`datetime.tzinfo` object by its ``tzid``.

        :param tzid:
            If there is exactly one time zone available, omitting ``tzid``
            or passing :py:const:`None` value returns it. Otherwise a valid
            key (which can be retrieved from :func:`keys`) is required.

        :raises ValueError:
            Raised if ``tzid`` is not specified but there are either more
            or fewer than 1 zone defined.

        :returns:
            Returns either a :py:class:`datetime.tzinfo` object representing
            the relevant time zone or :py:const:`None` if the ``tzid`` was
            not found.
        """
        if tzid is None:
            if len(self._vtz) == 0:
                raise ValueError("no timezones defined")
            elif len(self._vtz) > 1:
                raise ValueError("more than one timezone available")
            tzid = next(iter(self._vtz))

        return self._vtz.get(tzid)

    def _parse_offset(self, s):
        s = s.strip()
        if not s:
            raise ValueError("empty offset")
        if s[0] in ('+', '-'):
            signal = (-1, +1)[s[0] == '+']
            s = s[1:]
        else:
            signal = +1
        if len(s) == 4:
            return (int(s[:2]) * 3600 + int(s[2:]) * 60) * signal
        elif len(s) == 6:
            return (int(s[:2]) * 3600 + int(s[2:4]) * 60 + int(s[4:])) * signal
        else:
            raise ValueError("invalid offset: " + s)

    def _parse_rfc(self, s):
        lines = s.splitlines()
        if not lines:
            raise ValueError("empty string")

        # Unfold
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            if not line:
                del lines[i]
            elif i > 0 and line[0] == " ":
                lines[i-1] += line[1:]
                del lines[i]
            else:
                i += 1

        tzid = None
        comps = []
        invtz = False
        comptype = None
        for line in lines:
            if not line:
                continue
            name, value = line.split(':', 1)
            parms = name.split(';')
            if not parms:
                raise ValueError("empty property name")
            name = parms[0].upper()
            parms = parms[1:]
            if invtz:
                if name == "BEGIN":
                    if value in ("STANDARD", "DAYLIGHT"):
                        # Process component
                        pass
                    else:
                        raise ValueError("unknown component: "+value)
                    comptype = value
                    founddtstart = False
                    tzoffsetfrom = None
                    tzoffsetto = None
                    rrulelines = []
                    tzname = None
                elif name == "END":
                    if value == "VTIMEZONE":
                        if comptype:
                            raise ValueError("component not closed: "+comptype)
                        if not tzid:
                            raise ValueError("mandatory TZID not found")
                        if not comps:
                            raise ValueError(
                                "at least one component is needed")
                        # Process vtimezone
                        self._vtz[tzid] = _tzicalvtz(tzid, comps)
                        invtz = False
                    elif value == comptype:
                        if not founddtstart:
                            raise ValueError("mandatory DTSTART not found")
                        if tzoffsetfrom is None:
                            raise ValueError(
                                "mandatory TZOFFSETFROM not found")
                        if tzoffsetto is None:
                            raise ValueError(
                                "mandatory TZOFFSETFROM not found")
                        # Process component
                        rr = None
                        if rrulelines:
                            rr = rrule.rrulestr("\n".join(rrulelines),
                                                compatible=True,
                                                ignoretz=True,
                                                cache=True)
                        comp = _tzicalvtzcomp(tzoffsetfrom, tzoffsetto,
                                              (comptype == "DAYLIGHT"),
                                              tzname, rr)
                        comps.append(comp)
                        comptype = None
                    else:
                        raise ValueError("invalid component end: "+value)
                elif comptype:
                    if name == "DTSTART":
                        # DTSTART in VTIMEZONE takes a subset of valid RRULE
                        # values under RFC 5545.
                        for parm in parms:
                            if parm != 'VALUE=DATE-TIME':
                                msg = ('Unsupported DTSTART param in ' +
                                       'VTIMEZONE: ' + parm)
                                raise ValueError(msg)
                        rrulelines.append(line)
                        founddtstart = True
                    elif name in ("RRULE", "RDATE", "EXRULE", "EXDATE"):
                        rrulelines.append(line)
                    elif name == "TZOFFSETFROM":
                        if parms:
                            raise ValueError(
                                "unsupported %s parm: %s " % (name, parms[0]))
                        tzoffsetfrom = self._parse_offset(value)
                    elif name == "TZOFFSETTO":
                        if parms:
                            raise ValueError(
                                "unsupported TZOFFSETTO parm: "+parms[0])
                        tzoffsetto = self._parse_offset(value)
                    elif name == "TZNAME":
                        if parms:
                            raise ValueError(
                                "unsupported TZNAME parm: "+parms[0])
                        tzname = value
                    elif name == "COMMENT":
                        pass
                    else:
                        raise ValueError("unsupported property: "+name)
                else:
                    if name == "TZID":
                        if parms:
                            raise ValueError(
                                "unsupported TZID parm: "+parms[0])
                        tzid = value
                    elif name in ("TZURL", "LAST-MODIFIED", "COMMENT"):
                        pass
                    else:
                        raise ValueError("unsupported property: "+name)
            elif name == "BEGIN" and value == "VTIMEZONE":
                tzid = None
                comps = []
                invtz = True

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, repr(self._s))


if sys.platform != "win32":
    TZFILES = ["/etc/localtime", "localtime"]
    TZPATHS = ["/usr/share/zoneinfo",
               "/usr/lib/zoneinfo",
               "/usr/share/lib/zoneinfo",
               "/etc/zoneinfo"]
else:
    TZFILES = []
    TZPATHS = []


def __get_gettz():
    tzlocal_classes = (tzlocal,)
    if tzwinlocal is not None:
        tzlocal_classes += (tzwinlocal,)

    class GettzFunc(object):
        """
        Retrieve a time zone object from a string representation

        This function is intended to retrieve the :py:class:`tzinfo` subclass
        that best represents the time zone that would be used if a POSIX
        `TZ variable`_ were set to the same value.

        If no argument or an empty string is passed to ``gettz``, local time
        is returned:

        .. code-block:: python3

            >>> gettz()
            tzfile('/etc/localtime')

        This function is also the preferred way to map IANA tz database keys
        to :class:`tzfile` objects:

        .. code-block:: python3

            >>> gettz('Pacific/Kiritimati')
            tzfile('/usr/share/zoneinfo/Pacific/Kiritimati')

        On Windows, the standard is extended to include the Windows-specific
        zone names provided by the operating system:

        .. code-block:: python3

            >>> gettz('Egypt Standard Time')
            tzwin('Egypt Standard Time')

        Passing a GNU ``TZ`` style string time zone specification returns a
        :class:`tzstr` object:

        .. code-block:: python3

            >>> gettz('AEST-10AEDT-11,M10.1.0/2,M4.1.0/3')
            tzstr('AEST-10AEDT-11,M10.1.0/2,M4.1.0/3')

        :param name:
            A time zone name (IANA, or, on Windows, Windows keys), location of
            a ``tzfile(5)`` zoneinfo file or ``TZ`` variable style time zone
            specifier. An empty string, no argument or ``None`` is interpreted
            as local time.

        :return:
            Returns an instance of one of ``dateutil``'s :py:class:`tzinfo`
            subclasses.

        .. versionchanged:: 2.7.0

            After version 2.7.0, any two calls to ``gettz`` using the same
            input strings will return the same object:

            .. code-block:: python3

                >>> tz.gettz('America/Chicago') is tz.gettz('America/Chicago')
                True

            In addition to improving performance, this ensures that
            `"same zone" semantics`_ are used for datetimes in the same zone.


        .. _`TZ variable`:
            https://www.gnu.org/software/libc/manual/html_node/TZ-Variable.html

        .. _`"same zone" semantics`:
            https://blog.ganssle.io/articles/2018/02/aware-datetime-arithmetic.html
        """
        def __init__(self):

            self.__instances = weakref.WeakValueDictionary()
            self.__strong_cache_size = 8
            self.__strong_cache = OrderedDict()
            self._cache_lock = _thread.allocate_lock()

        def __call__(self, name=None):
            with self._cache_lock:
                rv = self.__instances.get(name, None)

                if rv is None:
                    rv = self.nocache(name=name)
                    if not (name is None
                            or isinstance(rv, tzlocal_classes)
                            or rv is None):
                        # tzlocal is slightly more complicated than the other
                        # time zone providers because it depends on environment
                        # at construction time, so don't cache that.
                        #
                        # We also cannot store weak references to None, so we
                        # will also not store that.
                        self.__instances[name] = rv
                    else:
                        # No need for strong caching, return immediately
                        return rv

                self.__strong_cache[name] = self.__strong_cache.pop(name, rv)

                if len(self.__strong_cache) > self.__strong_cache_size:
                    self.__strong_cache.popitem(last=False)

            return rv

        def set_cache_size(self, size):
            with self._cache_lock:
                self.__strong_cache_size = size
                while len(self.__strong_cache) > size:
                    self.__strong_cache.popitem(last=False)

        def cache_clear(self):
            with self._cache_lock:
                self.__instances = weakref.WeakValueDictionary()
                self.__strong_cache.clear()

        @staticmethod
        def nocache(name=None):
            """A non-cached version of gettz"""
            tz = None
            if not name:
                try:
                    name = os.environ["TZ"]
                except KeyError:
                    pass
            if name is None or name in ("", ":"):
                for filepath in TZFILES:
                    if not os.path.isabs(filepath):
                        filename = filepath
                        for path in TZPATHS:
                            filepath = os.path.join(path, filename)
                            if os.path.isfile(filepath):
                                break
                        else:
                            continue
                    if os.path.isfile(filepath):
                        try:
                            tz = tzfile(filepath)
                            break
                        except (IOError, OSError, ValueError):
                            pass
                else:
                    tz = tzlocal()
            else:
                try:
                    if name.startswith(":"):
                        name = name[1:]
                except TypeError as e:
                    if isinstance(name, bytes):
                        new_msg = "gettz argument should be str, not bytes"
                        six.raise_from(TypeError(new_msg), e)
                    else:
                        raise
                if os.path.isabs(name):
                    if os.path.isfile(name):
                        tz = tzfile(name)
                    else:
                        tz = None
                else:
                    for path in TZPATHS:
                        filepath = os.path.join(path, name)
                        if not os.path.isfile(filepath):
                            filepath = filepath.replace(' ', '_')
                            if not os.path.isfile(filepath):
                                continue
                        try:
                            tz = tzfile(filepath)
                            break
                        except (IOError, OSError, ValueError):
                            pass
                    else:
                        tz = None
                        if tzwin is not None:
                            try:
                                tz = tzwin(name)
                            except (WindowsError, UnicodeEncodeError):
                                # UnicodeEncodeError is for Python 2.7 compat
                                tz = None

                        if not tz:
                            from dateutil.zoneinfo import get_zonefile_instance
                            tz = get_zonefile_instance().get(name)

                        if not tz:
                            for c in name:
                                # name is not a tzstr unless it has at least
                                # one offset. For short values of "name", an
                                # explicit for loop seems to be the fastest way
                                # To determine if a string contains a digit
                                if c in "0123456789":
                                    try:
                                        tz = tzstr(name)
                                    except ValueError:
                                        pass
                                    break
                            else:
                                if name in ("GMT", "UTC"):
                                    tz = UTC
                                elif name in time.tzname:
                                    tz = tzlocal()
            return tz

    return GettzFunc()


gettz = __get_gettz()
del __get_gettz


def datetime_exists(dt, tz=None):
    """
    Given a datetime and a time zone, determine whether or not a given datetime
    would fall in a gap.

    :param dt:
        A :class:`datetime.datetime` (whose time zone will be ignored if ``tz``
        is provided.)

    :param tz:
        A :class:`datetime.tzinfo` with support for the ``fold`` attribute. If
        ``None`` or not provided, the datetime's own time zone will be used.

    :return:
        Returns a boolean value whether or not the "wall time" exists in
        ``tz``.

    .. versionadded:: 2.7.0
    """
    if tz is None:
        if dt.tzinfo is None:
            raise ValueError('Datetime is naive and no time zone provided.')
        tz = dt.tzinfo

    dt = dt.replace(tzinfo=None)

    # This is essentially a test of whether or not the datetime can survive
    # a round trip to UTC.
    dt_rt = dt.replace(tzinfo=tz).astimezone(UTC).astimezone(tz)
    dt_rt = dt_rt.replace(tzinfo=None)

    return dt == dt_rt


def datetime_ambiguous(dt, tz=None):
    """
    Given a datetime and a time zone, determine whether or not a given datetime
    is ambiguous (i.e if there are two times differentiated only by their DST
    status).

    :param dt:
        A :class:`datetime.datetime` (whose time zone will be ignored if ``tz``
        is provided.)

    :param tz:
        A :class:`datetime.tzinfo` with support for the ``fold`` attribute. If
        ``None`` or not provided, the datetime's own time zone will be used.

    :return:
        Returns a boolean value whether or not the "wall time" is ambiguous in
        ``tz``.

    .. versionadded:: 2.6.0
    """
    if tz is None:
        if dt.tzinfo is None:
            raise ValueError('Datetime is naive and no time zone provided.')

        tz = dt.tzinfo

    # If a time zone defines its own "is_ambiguous" function, we'll use that.
    is_ambiguous_fn = getattr(tz, 'is_ambiguous', None)
    if is_ambiguous_fn is not None:
        try:
            return tz.is_ambiguous(dt)
        except Exception:
            pass

    # If it doesn't come out and tell us it's ambiguous, we'll just check if
    # the fold attribute has any effect on this particular date and time.
    dt = dt.replace(tzinfo=tz)
    wall_0 = enfold(dt, fold=0)
    wall_1 = enfold(dt, fold=1)

    same_offset = wall_0.utcoffset() == wall_1.utcoffset()
    same_dst = wall_0.dst() == wall_1.dst()

    return not (same_offset and same_dst)


def resolve_imaginary(dt):
    """
    Given a datetime that may be imaginary, return an existing datetime.

    This function assumes that an imaginary datetime represents what the
    wall time would be in a zone had the offset transition not occurred, so
    it will always fall forward by the transition's change in offset.

    .. doctest::

        >>> from dateutil import tz
        >>> from datetime import datetime
        >>> NYC = tz.gettz('America/New_York')
        >>> print(tz.resolve_imaginary(datetime(2017, 3, 12, 2, 30, tzinfo=NYC)))
        2017-03-12 03:30:00-04:00

        >>> KIR = tz.gettz('Pacific/Kiritimati')
        >>> print(tz.resolve_imaginary(datetime(1995, 1, 1, 12, 30, tzinfo=KIR)))
        1995-01-02 12:30:00+14:00

    As a note, :func:`datetime.astimezone` is guaranteed to produce a valid,
    existing datetime, so a round-trip to and from UTC is sufficient to get
    an extant datetime, however, this generally "falls back" to an earlier time
    rather than falling forward to the STD side (though no guarantees are made
    about this behavior).

    :param dt:
        A :class:`datetime.datetime` which may or may not exist.

    :return:
        Returns an existing :class:`datetime.datetime`. If ``dt`` was not
        imaginary, the datetime returned is guaranteed to be the same object
        passed to the function.

    .. versionadded:: 2.7.0
    """
    if dt.tzinfo is not None and not datetime_exists(dt):

        curr_offset = (dt + datetime.timedelta(hours=24)).utcoffset()
        old_offset = (dt - datetime.timedelta(hours=24)).utcoffset()

        dt += curr_offset - old_offset

    return dt


def _datetime_to_timestamp(dt):
    """
    Convert a :class:`datetime.datetime` object to an epoch timestamp in
    seconds since January 1, 1970, ignoring the time zone.
    """
    return (dt.replace(tzinfo=None) - EPOCH).total_seconds()


if sys.version_info >= (3, 6):
    def _get_supported_offset(second_offset):
        return second_offset
else:
    def _get_supported_offset(second_offset):
        # For python pre-3.6, round to full-minutes if that's not the case.
        # Python's datetime doesn't accept sub-minute timezones. Check
        # http://python.org/sf/1447945 or https://bugs.python.org/issue5288
        # for some information.
        old_offset = second_offset
        calculated_offset = 60 * ((second_offset + 30) // 60)
        return calculated_offset


try:
    # Python 3.7 feature
    from contextlib import nullcontext as _nullcontext
except ImportError:
    class _nullcontext(object):
        """
        Class for wrapping contexts so that they are passed through in a
        with statement.
        """
        def __init__(self, context):
            self.context = context

        def __enter__(self):
            return self.context

        def __exit__(*args, **kwargs):
            pass

# vim:ts=4:sw=4:et
<|endoftext|>from six import PY2

from functools import wraps

from datetime import datetime, timedelta, tzinfo


ZERO = timedelta(0)

__all__ = ['tzname_in_python2', 'enfold']


def tzname_in_python2(namefunc):
    """Change unicode output into bytestrings in Python 2

    tzname() API changed in Python 3. It used to return bytes, but was changed
    to unicode strings
    """
    if PY2:
        @wraps(namefunc)
        def adjust_encoding(*args, **kwargs):
            name = namefunc(*args, **kwargs)
            if name is not None:
                name = name.encode()

            return name

        return adjust_encoding
    else:
        return namefunc


# The following is adapted from Alexander Belopolsky's tz library
# https://github.com/abalkin/tz
if hasattr(datetime, 'fold'):
    # This is the pre-python 3.6 fold situation
    def enfold(dt, fold=1):
        """
        Provides a unified interface for assigning the ``fold`` attribute to
        datetimes both before and after the implementation of PEP-495.

        :param fold:
            The value for the ``fold`` attribute in the returned datetime. This
            should be either 0 or 1.

        :return:
            Returns an object for which ``getattr(dt, 'fold', 0)`` returns
            ``fold`` for all versions of Python. In versions prior to
            Python 3.6, this is a ``_DatetimeWithFold`` object, which is a
            subclass of :py:class:`datetime.datetime` with the ``fold``
            attribute added, if ``fold`` is 1.

        .. versionadded:: 2.6.0
        """
        return dt.replace(fold=fold)

else:
    class _DatetimeWithFold(datetime):
        """
        This is a class designed to provide a PEP 495-compliant interface for
        Python versions before 3.6. It is used only for dates in a fold, so
        the ``fold`` attribute is fixed at ``1``.

        .. versionadded:: 2.6.0
        """
        __slots__ = ()

        def replace(self, *args, **kwargs):
            """
            Return a datetime with the same attributes, except for those
            attributes given new values by whichever keyword arguments are
            specified. Note that tzinfo=None can be specified to create a naive
            datetime from an aware datetime with no conversion of date and time
            data.

            This is reimplemented in ``_DatetimeWithFold`` because pypy3 will
            return a ``datetime.datetime`` even if ``fold`` is unchanged.
            """
            argnames = (
                'year', 'month', 'day', 'hour', 'minute', 'second',
                'microsecond', 'tzinfo'
            )

            for arg, argname in zip(args, argnames):
                if argname in kwargs:
                    raise TypeError('Duplicate argument: {}'.format(argname))

                kwargs[argname] = arg

            for argname in argnames:
                if argname not in kwargs:
                    kwargs[argname] = getattr(self, argname)

            dt_class = self.__class__ if kwargs.get('fold', 1) else datetime

            return dt_class(**kwargs)

        @property
        def fold(self):
            return 1

    def enfold(dt, fold=1):
        """
        Provides a unified interface for assigning the ``fold`` attribute to
        datetimes both before and after the implementation of PEP-495.

        :param fold:
            The value for the ``fold`` attribute in the returned datetime. This
            should be either 0 or 1.

        :return:
            Returns an object for which ``getattr(dt, 'fold', 0)`` returns
            ``fold`` for all versions of Python. In versions prior to
            Python 3.6, this is a ``_DatetimeWithFold`` object, which is a
            subclass of :py:class:`datetime.datetime` with the ``fold``
            attribute added, if ``fold`` is 1.

        .. versionadded:: 2.6.0
        """
        if getattr(dt, 'fold', 0) == fold:
            return dt

        args = dt.timetuple()[:6]
        args += (dt.microsecond, dt.tzinfo)

        if fold:
            return _DatetimeWithFold(*args)
        else:
            return datetime(*args)


def _validate_fromutc_inputs(f):
    """
    The CPython version of ``fromutc`` checks that the input is a ``datetime``
    object and that ``self`` is attached as its ``tzinfo``.
    """
    @wraps(f)
    def fromutc(self, dt):
        if not isinstance(dt, datetime):
            raise TypeError("fromutc() requires a datetime argument")
        if dt.tzinfo is not self:
            raise ValueError("dt.tzinfo is not self")

        return f(self, dt)

    return fromutc


class _tzinfo(tzinfo):
    """
    Base class for all ``dateutil`` ``tzinfo`` objects.
    """

    def is_ambiguous(self, dt):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.


        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """

        dt = dt.replace(tzinfo=self)

        wall_0 = enfold(dt, fold=0)
        wall_1 = enfold(dt, fold=1)

        same_offset = wall_0.utcoffset() == wall_1.utcoffset()
        same_dt = wall_0.replace(tzinfo=None) == wall_1.replace(tzinfo=None)

        return same_dt and not same_offset

    def _fold_status(self, dt_utc, dt_wall):
        """
        Determine the fold status of a "wall" datetime, given a representation
        of the same datetime as a (naive) UTC datetime. This is calculated based
        on the assumption that ``dt.utcoffset() - dt.dst()`` is constant for all
        datetimes, and that this offset is the actual number of hours separating
        ``dt_utc`` and ``dt_wall``.

        :param dt_utc:
            Representation of the datetime as UTC

        :param dt_wall:
            Representation of the datetime as "wall time". This parameter must
            either have a `fold` attribute or have a fold-naive
            :class:`datetime.tzinfo` attached, otherwise the calculation may
            fail.
        """
        if self.is_ambiguous(dt_wall):
            delta_wall = dt_wall - dt_utc
            _fold = int(delta_wall == (dt_utc.utcoffset() - dt_utc.dst()))
        else:
            _fold = 0

        return _fold

    def _fold(self, dt):
        return getattr(dt, 'fold', 0)

    def _fromutc(self, dt):
        """
        Given a timezone-aware datetime in a given timezone, calculates a
        timezone-aware datetime in a new timezone.

        Since this is the one time that we *know* we have an unambiguous
        datetime object, we take this opportunity to determine whether the
        datetime is ambiguous and in a "fold" state (e.g. if it's the first
        occurrence, chronologically, of the ambiguous datetime).

        :param dt:
            A timezone-aware :class:`datetime.datetime` object.
        """

        # Re-implement the algorithm from Python's datetime.py
        dtoff = dt.utcoffset()
        if dtoff is None:
            raise ValueError("fromutc() requires a non-None utcoffset() "
                             "result")

        # The original datetime.py code assumes that `dst()` defaults to
        # zero during ambiguous times. PEP 495 inverts this presumption, so
        # for pre-PEP 495 versions of python, we need to tweak the algorithm.
        dtdst = dt.dst()
        if dtdst is None:
            raise ValueError("fromutc() requires a non-None dst() result")
        delta = dtoff - dtdst

        dt += delta
        # Set fold=1 so we can default to being in the fold for
        # ambiguous dates.
        dtdst = enfold(dt, fold=1).dst()
        if dtdst is None:
            raise ValueError("fromutc(): dt.dst gave inconsistent "
                             "results; cannot convert")
        return dt + dtdst

    @_validate_fromutc_inputs
    def fromutc(self, dt):
        """
        Given a timezone-aware datetime in a given timezone, calculates a
        timezone-aware datetime in a new timezone.

        Since this is the one time that we *know* we have an unambiguous
        datetime object, we take this opportunity to determine whether the
        datetime is ambiguous and in a "fold" state (e.g. if it's the first
        occurrence, chronologically, of the ambiguous datetime).

        :param dt:
            A timezone-aware :class:`datetime.datetime` object.
        """
        dt_wall = self._fromutc(dt)

        # Calculate the fold status given the two datetimes.
        _fold = self._fold_status(dt, dt_wall)

        # Set the default fold value for ambiguous dates
        return enfold(dt_wall, fold=_fold)


class tzrangebase(_tzinfo):
    """
    This is an abstract base class for time zones represented by an annual
    transition into and out of DST. Child classes should implement the following
    methods:

        * ``__init__(self, *args, **kwargs)``
        * ``transitions(self, year)`` - this is expected to return a tuple of
          datetimes representing the DST on and off transitions in standard
          time.

    A fully initialized ``tzrangebase`` subclass should also provide the
    following attributes:
        * ``hasdst``: Boolean whether or not the zone uses DST.
        * ``_dst_offset`` / ``_std_offset``: :class:`datetime.timedelta` objects
          representing the respective UTC offsets.
        * ``_dst_abbr`` / ``_std_abbr``: Strings representing the timezone short
          abbreviations in DST and STD, respectively.
        * ``_hasdst``: Whether or not the zone has DST.

    .. versionadded:: 2.6.0
    """
    def __init__(self):
        raise NotImplementedError('tzrangebase is an abstract base class')

    def utcoffset(self, dt):
        isdst = self._isdst(dt)

        if isdst is None:
            return None
        elif isdst:
            return self._dst_offset
        else:
            return self._std_offset

    def dst(self, dt):
        isdst = self._isdst(dt)

        if isdst is None:
            return None
        elif isdst:
            return self._dst_base_offset
        else:
            return ZERO

    @tzname_in_python2
    def tzname(self, dt):
        if self._isdst(dt):
            return self._dst_abbr
        else:
            return self._std_abbr

    def fromutc(self, dt):
        """ Given a datetime in UTC, return local time """
        if not isinstance(dt, datetime):
            raise TypeError("fromutc() requires a datetime argument")

        if dt.tzinfo is not self:
            raise ValueError("dt.tzinfo is not self")

        # Get transitions - if there are none, fixed offset
        transitions = self.transitions(dt.year)
        if transitions is None:
            return dt + self.utcoffset(dt)

        # Get the transition times in UTC
        dston, dstoff = transitions

        dston -= self._std_offset
        dstoff -= self._std_offset

        utc_transitions = (dston, dstoff)
        dt_utc = dt.replace(tzinfo=None)

        isdst = self._naive_isdst(dt_utc, utc_transitions)

        if isdst:
            dt_wall = dt + self._dst_offset
        else:
            dt_wall = dt + self._std_offset

        _fold = int(not isdst and self.is_ambiguous(dt_wall))

        return enfold(dt_wall, fold=_fold)

    def is_ambiguous(self, dt):
        """
        Whether or not the "wall time" of a given datetime is ambiguous in this
        zone.

        :param dt:
            A :py:class:`datetime.datetime`, naive or time zone aware.


        :return:
            Returns ``True`` if ambiguous, ``False`` otherwise.

        .. versionadded:: 2.6.0
        """
        if not self.hasdst:
            return False

        start, end = self.transitions(dt.year)

        dt = dt.replace(tzinfo=None)
        return (end <= dt < end + self._dst_base_offset)

    def _isdst(self, dt):
        if not self.hasdst:
            return False
        elif dt is None:
            return None

        transitions = self.transitions(dt.year)

        if transitions is None:
            return False

        dt = dt.replace(tzinfo=None)

        isdst = self._naive_isdst(dt, transitions)

        # Handle ambiguous dates
        if not isdst and self.is_ambiguous(dt):
            return not self._fold(dt)
        else:
            return isdst

    def _naive_isdst(self, dt, transitions):
        dston, dstoff = transitions

        dt = dt.replace(tzinfo=None)

        if dston < dstoff:
            isdst = dston <= dt < dstoff
        else:
            isdst = not dstoff <= dt < dston

        return isdst

    @property
    def _dst_base_offset(self):
        return self._dst_offset - self._std_offset

    __hash__ = None

    def __ne__(self, other):
        return not (self == other)

    def __repr__(self):
        return "%s(...)" % self.__class__.__name__

    __reduce__ = object.__reduce__
<|endoftext|># -*- coding: utf-8 -*-
from ._parser import parse, parser, parserinfo, ParserError
from ._parser import DEFAULTPARSER, DEFAULTTZPARSER
from ._parser import UnknownTimezoneWarning

from ._parser import __doc__

from .isoparser import isoparser, isoparse

__all__ = ['parse', 'parser', 'parserinfo',
           'isoparse', 'isoparser',
           'ParserError',
           'UnknownTimezoneWarning']


###
# Deprecate portions of the private interface so that downstream code that
# is improperly relying on it is given *some* notice.


def __deprecated_private_func(f):
    from functools import wraps
    import warnings

    msg = ('{name} is a private function and may break without warning, '
           'it will be moved and or renamed in future versions.')
    msg = msg.format(name=f.__name__)

    @wraps(f)
    def deprecated_func(*args, **kwargs):
        warnings.warn(msg, DeprecationWarning)
        return f(*args, **kwargs)

    return deprecated_func

def __deprecate_private_class(c):
    import warnings

    msg = ('{name} is a private class and may break without warning, '
           'it will be moved and or renamed in future versions.')
    msg = msg.format(name=c.__name__)

    class private_class(c):
        __doc__ = c.__doc__

        def __init__(self, *args, **kwargs):
            warnings.warn(msg, DeprecationWarning)
            super(private_class, self).__init__(*args, **kwargs)

    private_class.__name__ = c.__name__

    return private_class


from ._parser import _timelex, _resultbase
from ._parser import _tzparser, _parsetz

_timelex = __deprecate_private_class(_timelex)
_tzparser = __deprecate_private_class(_tzparser)
_resultbase = __deprecate_private_class(_resultbase)
_parsetz = __deprecated_private_func(_parsetz)
<|endoftext|># -*- coding: utf-8 -*-
"""
This module offers a generic date/time string parser which is able to parse
most known formats to represent a date and/or time.

This module attempts to be forgiving with regards to unlikely input formats,
returning a datetime object even for dates which are ambiguous. If an element
of a date/time stamp is omitted, the following rules are applied:

- If AM or PM is left unspecified, a 24-hour clock is assumed, however, an hour
  on a 12-hour clock (``0 <= hour <= 12``) *must* be specified if AM or PM is
  specified.
- If a time zone is omitted, a timezone-naive datetime is returned.

If any other elements are missing, they are taken from the
:class:`datetime.datetime` object passed to the parameter ``default``. If this
results in a day number exceeding the valid number of days per month, the
value falls back to the end of the month.

Additional resources about date/time string formats can be found below:

- `A summary of the international standard date and time notation
  <https://www.cl.cam.ac.uk/~mgk25/iso-time.html>`_
- `W3C Date and Time Formats <https://www.w3.org/TR/NOTE-datetime>`_
- `Time Formats (Planetary Rings Node) <https://pds-rings.seti.org:443/tools/time_formats.html>`_
- `CPAN ParseDate module
  <https://metacpan.org/pod/release/MUIR/Time-modules-2013.0912/lib/Time/ParseDate.pm>`_
- `Java SimpleDateFormat Class
  <https://docs.oracle.com/javase/6/docs/api/java/text/SimpleDateFormat.html>`_
"""
from __future__ import unicode_literals

import datetime
import re
import string
import time
import warnings

from calendar import monthrange
from io import StringIO

import six
from six import integer_types, text_type

from decimal import Decimal

from warnings import warn

from .. import relativedelta
from .. import tz

__all__ = ["parse", "parserinfo", "ParserError"]


# TODO: pandas.core.tools.datetimes imports this explicitly.  Might be worth
# making public and/or figuring out if there is something we can
# take off their plate.
class _timelex(object):
    # Fractional seconds are sometimes split by a comma
    _split_decimal = re.compile("([.,])")

    def __init__(self, instream):
        if isinstance(instream, (bytes, bytearray)):
            instream = instream.decode()

        if isinstance(instream, text_type):
            instream = StringIO(instream)
        elif getattr(instream, 'read', None) is None:
            raise TypeError('Parser must be a string or character stream, not '
                            '{itype}'.format(itype=instream.__class__.__name__))

        self.instream = instream
        self.charstack = []
        self.tokenstack = []
        self.eof = False

    def get_token(self):
        """
        This function breaks the time string into lexical units (tokens), which
        can be parsed by the parser. Lexical units are demarcated by changes in
        the character set, so any continuous string of letters is considered
        one unit, any continuous string of numbers is considered one unit.

        The main complication arises from the fact that dots ('.') can be used
        both as separators (e.g. "Sep.20.2009") or decimal points (e.g.
        "4:30:21.447"). As such, it is necessary to read the full context of
        any dot-separated strings before breaking it into tokens; as such, this
        function maintains a "token stack", for when the ambiguous context
        demands that multiple tokens be parsed at once.
        """
        if self.tokenstack:
            return self.tokenstack.pop(0)

        seenletters = False
        token = None
        state = None

        while not self.eof:
            # We only realize that we've reached the end of a token when we
            # find a character that's not part of the current token - since
            # that character may be part of the next token, it's stored in the
            # charstack.
            if self.charstack:
                nextchar = self.charstack.pop(0)
            else:
                nextchar = self.instream.read(1)
                while nextchar == '\x00':
                    nextchar = self.instream.read(1)

            if not nextchar:
                self.eof = True
                break
            elif not state:
                # First character of the token - determines if we're starting
                # to parse a word, a number or something else.
                token = nextchar
                if self.isword(nextchar):
                    state = 'a'
                elif self.isnum(nextchar):
                    state = '0'
                elif self.isspace(nextchar):
                    token = ' '
                    break  # emit token
                else:
                    break  # emit token
            elif state == 'a':
                # If we've already started reading a word, we keep reading
                # letters until we find something that's not part of a word.
                seenletters = True
                if self.isword(nextchar):
                    token += nextchar
                elif nextchar == '.':
                    token += nextchar
                    state = 'a.'
                else:
                    self.charstack.append(nextchar)
                    break  # emit token
            elif state == '0':
                # If we've already started reading a number, we keep reading
                # numbers until we find something that doesn't fit.
                if self.isnum(nextchar):
                    token += nextchar
                elif nextchar == '.' or (nextchar == ',' and len(token) >= 2):
                    token += nextchar
                    state = '0.'
                else:
                    self.charstack.append(nextchar)
                    break  # emit token
            elif state == 'a.':
                # If we've seen some letters and a dot separator, continue
                # parsing, and the tokens will be broken up later.
                seenletters = True
                if nextchar == '.' or self.isword(nextchar):
                    token += nextchar
                elif self.isnum(nextchar) and token[-1] == '.':
                    token += nextchar
                    state = '0.'
                else:
                    self.charstack.append(nextchar)
                    break  # emit token
            elif state == '0.':
                # If we've seen at least one dot separator, keep going, we'll
                # break up the tokens later.
                if nextchar == '.' or self.isnum(nextchar):
                    token += nextchar
                elif self.isword(nextchar) and token[-1] == '.':
                    token += nextchar
                    state = 'a.'
                else:
                    self.charstack.append(nextchar)
                    break  # emit token

        if (state in ('a.', '0.') and (seenletters or token.count('.') > 1 or
                                       token[-1] in '.,')):
            l = self._split_decimal.split(token)
            token = l[0]
            for tok in l[1:]:
                if tok:
                    self.tokenstack.append(tok)

        if state == '0.' and token.count('.') == 0:
            token = token.replace(',', '.')

        return token

    def __iter__(self):
        return self

    def __next__(self):
        token = self.get_token()
        if token is None:
            raise StopIteration

        return token

    def next(self):
        return self.__next__()  # Python 2.x support

    @classmethod
    def split(cls, s):
        return list(cls(s))

    @classmethod
    def isword(cls, nextchar):
        """ Whether or not the next character is part of a word """
        return nextchar.isalpha()

    @classmethod
    def isnum(cls, nextchar):
        """ Whether the next character is part of a number """
        return nextchar.isdigit()

    @classmethod
    def isspace(cls, nextchar):
        """ Whether the next character is whitespace """
        return nextchar.isspace()


class _resultbase(object):

    def __init__(self):
        for attr in self.__slots__:
            setattr(self, attr, None)

    def _repr(self, classname):
        l = []
        for attr in self.__slots__:
            value = getattr(self, attr)
            if value is not None:
                l.append("%s=%s" % (attr, repr(value)))
        return "%s(%s)" % (classname, ", ".join(l))

    def __len__(self):
        return (sum(getattr(self, attr) is not None
                    for attr in self.__slots__))

    def __repr__(self):
        return self._repr(self.__class__.__name__)


class parserinfo(object):
    """
    Class which handles what inputs are accepted. Subclass this to customize
    the language and acceptable values for each parameter.

    :param dayfirst:
        Whether to interpret the first value in an ambiguous 3-integer date
        (e.g. 01/05/09) as the day (``True``) or month (``False``). If
        ``yearfirst`` is set to ``True``, this distinguishes between YDM
        and YMD. Default is ``False``.

    :param yearfirst:
        Whether to interpret the first value in an ambiguous 3-integer date
        (e.g. 01/05/09) as the year. If ``True``, the first number is taken
        to be the year, otherwise the last number is taken to be the year.
        Default is ``False``.
    """

    # m from a.m/p.m, t from ISO T separator
    JUMP = [" ", ".", ",", ";", "-", "/", "'",
            "at", "on", "and", "ad", "m", "t", "of",
            "st", "nd", "rd", "th"]

    WEEKDAYS = [("Mon", "Monday"),
                ("Tue", "Tuesday"),     # TODO: "Tues"
                ("Wed", "Wednesday"),
                ("Thu", "Thursday"),    # TODO: "Thurs"
                ("Fri", "Friday"),
                ("Sat", "Saturday"),
                ("Sun", "Sunday")]
    MONTHS = [("Jan", "January"),
              ("Feb", "February"),      # TODO: "Febr"
              ("Mar", "March"),
              ("Apr", "April"),
              ("May", "May"),
              ("Jun", "June"),
              ("Jul", "July"),
              ("Aug", "August"),
              ("Sep", "Sept", "September"),
              ("Oct", "October"),
              ("Nov", "November"),
              ("Dec", "December")]
    HMS = [("h", "hour", "hours"),
           ("m", "minute", "minutes"),
           ("s", "second", "seconds")]
    AMPM = [("am", "a"),
            ("pm", "p")]
    UTCZONE = ["UTC", "GMT", "Z", "z"]
    PERTAIN = ["of"]
    TZOFFSET = {}
    # TODO: ERA = ["AD", "BC", "CE", "BCE", "Stardate",
    #              "Anno Domini", "Year of Our Lord"]

    def __init__(self, dayfirst=False, yearfirst=False):
        self._jump = self._convert(self.JUMP)
        self._weekdays = self._convert(self.WEEKDAYS)
        self._months = self._convert(self.MONTHS)
        self._hms = self._convert(self.HMS)
        self._ampm = self._convert(self.AMPM)
        self._utczone = self._convert(self.UTCZONE)
        self._pertain = self._convert(self.PERTAIN)

        self.dayfirst = dayfirst
        self.yearfirst = yearfirst

        self._year = time.localtime().tm_year
        self._century = self._year // 100 * 100

    def _convert(self, lst):
        dct = {}
        for i, v in enumerate(lst):
            if isinstance(v, tuple):
                for v in v:
                    dct[v.lower()] = i
            else:
                dct[v.lower()] = i
        return dct

    def jump(self, name):
        return name.lower() in self._jump

    def weekday(self, name):
        try:
            return self._weekdays[name.lower()]
        except KeyError:
            pass
        return None

    def month(self, name):
        try:
            return self._months[name.lower()] + 1
        except KeyError:
            pass
        return None

    def hms(self, name):
        try:
            return self._hms[name.lower()]
        except KeyError:
            return None

    def ampm(self, name):
        try:
            return self._ampm[name.lower()]
        except KeyError:
            return None

    def pertain(self, name):
        return name.lower() in self._pertain

    def utczone(self, name):
        return name.lower() in self._utczone

    def tzoffset(self, name):
        if name in self._utczone:
            return 0

        return self.TZOFFSET.get(name)

    def convertyear(self, year, century_specified=False):
        """
        Converts two-digit years to year within [-50, 49]
        range of self._year (current local time)
        """

        # Function contract is that the year is always positive
        assert year >= 0

        if year < 100 and not century_specified:
            # assume current century to start
            year += self._century

            if year >= self._year + 50:  # if too far in future
                year -= 100
            elif year < self._year - 50:  # if too far in past
                year += 100

        return year

    def validate(self, res):
        # move to info
        if res.year is not None:
            res.year = self.convertyear(res.year, res.century_specified)

        if ((res.tzoffset == 0 and not res.tzname) or
             (res.tzname == 'Z' or res.tzname == 'z')):
            res.tzname = "UTC"
            res.tzoffset = 0
        elif res.tzoffset != 0 and res.tzname and self.utczone(res.tzname):
            res.tzoffset = 0
        return True


class _ymd(list):
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        self.century_specified = False
        self.dstridx = None
        self.mstridx = None
        self.ystridx = None

    @property
    def has_year(self):
        return self.ystridx is not None

    @property
    def has_month(self):
        return self.mstridx is not None

    @property
    def has_day(self):
        return self.dstridx is not None

    def could_be_day(self, value):
        if self.has_day:
            return False
        elif not self.has_month:
            return 1 <= value <= 31
        elif not self.has_year:
            # Be permissive, assume leap year
            month = self[self.mstridx]
            return 1 <= value <= monthrange(2000, month)[1]
        else:
            month = self[self.mstridx]
            year = self[self.ystridx]
            return 1 <= value <= monthrange(year, month)[1]

    def append(self, val, label=None):
        if hasattr(val, '__len__'):
            if val.isdigit() and len(val) > 2:
                self.century_specified = True
                if label not in [None, 'Y']:  # pragma: no cover
                    raise ValueError(label)
                label = 'Y'
        elif val > 100:
            self.century_specified = True
            if label not in [None, 'Y']:  # pragma: no cover
                raise ValueError(label)
            label = 'Y'

        super(self.__class__, self).append(int(val))

        if label == 'M':
            if self.has_month:
                raise ValueError('Month is already set')
            self.mstridx = len(self) - 1
        elif label == 'D':
            if self.has_day:
                raise ValueError('Day is already set')
            self.dstridx = len(self) - 1
        elif label == 'Y':
            if self.has_year:
                raise ValueError('Year is already set')
            self.ystridx = len(self) - 1

    def _resolve_from_stridxs(self, strids):
        """
        Try to resolve the identities of year/month/day elements using
        ystridx, mstridx, and dstridx, if enough of these are specified.
        """
        if len(self) == 3 and len(strids) == 2:
            # we can back out the remaining stridx value
            missing = [x for x in range(3) if x not in strids.values()]
            key = [x for x in ['y', 'm', 'd'] if x not in strids]
            assert len(missing) == len(key) == 1
            key = key[0]
            val = missing[0]
            strids[key] = val

        assert len(self) == len(strids)  # otherwise this should not be called
        out = {key: self[strids[key]] for key in strids}
        return (out.get('y'), out.get('m'), out.get('d'))

    def resolve_ymd(self, yearfirst, dayfirst):
        len_ymd = len(self)
        year, month, day = (None, None, None)

        strids = (('y', self.ystridx),
                  ('m', self.mstridx),
                  ('d', self.dstridx))

        strids = {key: val for key, val in strids if val is not None}
        if (len(self) == len(strids) > 0 or
                (len(self) == 3 and len(strids) == 2)):
            return self._resolve_from_stridxs(strids)

        mstridx = self.mstridx

        if len_ymd > 3:
            raise ValueError("More than three YMD values")
        elif len_ymd == 1 or (mstridx is not None and len_ymd == 2):
            # One member, or two members with a month string
            if mstridx is not None:
                month = self[mstridx]
                # since mstridx is 0 or 1, self[mstridx-1] always
                # looks up the other element
                other = self[mstridx - 1]
            else:
                other = self[0]

            if len_ymd > 1 or mstridx is None:
                if other > 31:
                    year = other
                else:
                    day = other

        elif len_ymd == 2:
            # Two members with numbers
            if self[0] > 31:
                # 99-01
                year, month = self
            elif self[1] > 31:
                # 01-99
                month, year = self
            elif dayfirst and self[1] <= 12:
                # 13-01
                day, month = self
            else:
                # 01-13
                month, day = self

        elif len_ymd == 3:
            # Three members
            if mstridx == 0:
                if self[1] > 31:
                    # Apr-2003-25
                    month, year, day = self
                else:
                    month, day, year = self
            elif mstridx == 1:
                if self[0] > 31 or (yearfirst and self[2] <= 31):
                    # 99-Jan-01
                    year, month, day = self
                else:
                    # 01-Jan-01
                    # Give precedence to day-first, since
                    # two-digit years is usually hand-written.
                    day, month, year = self

            elif mstridx == 2:
                # WTF!?
                if self[1] > 31:
                    # 01-99-Jan
                    day, year, month = self
                else:
                    # 99-01-Jan
                    year, day, month = self

            else:
                if (self[0] > 31 or
                    self.ystridx == 0 or
                        (yearfirst and self[1] <= 12 and self[2] <= 31)):
                    # 99-01-01
                    if dayfirst and self[2] <= 12:
                        year, day, month = self
                    else:
                        year, month, day = self
                elif self[0] > 12 or (dayfirst and self[1] <= 12):
                    # 13-01-01
                    day, month, year = self
                else:
                    # 01-13-01
                    month, day, year = self

        return year, month, day


class parser(object):
    def __init__(self, info=None):
        self.info = info or parserinfo()

    def parse(self, timestr, default=None,
              ignoretz=False, tzinfos=None, **kwargs):
        """
        Parse the date/time string into a :class:`datetime.datetime` object.

        :param timestr:
            Any date/time string using the supported formats.

        :param default:
            The default datetime object, if this is a datetime object and not
            ``None``, elements specified in ``timestr`` replace elements in the
            default object.

        :param ignoretz:
            If set ``True``, time zones in parsed strings are ignored and a
            naive :class:`datetime.datetime` object is returned.

        :param tzinfos:
            Additional time zone names / aliases which may be present in the
            string. This argument maps time zone names (and optionally offsets
            from those time zones) to time zones. This parameter can be a
            dictionary with timezone aliases mapping time zone names to time
            zones or a function taking two parameters (``tzname`` and
            ``tzoffset``) and returning a time zone.

            The timezones to which the names are mapped can be an integer
            offset from UTC in seconds or a :class:`tzinfo` object.

            .. doctest::
               :options: +NORMALIZE_WHITESPACE

                >>> from dateutil.parser import parse
                >>> from dateutil.tz import gettz
                >>> tzinfos = {"BRST": -7200, "CST": gettz("America/Chicago")}
                >>> parse("2012-01-19 17:21:00 BRST", tzinfos=tzinfos)
                datetime.datetime(2012, 1, 19, 17, 21, tzinfo=tzoffset(u'BRST', -7200))
                >>> parse("2012-01-19 17:21:00 CST", tzinfos=tzinfos)
                datetime.datetime(2012, 1, 19, 17, 21,
                                  tzinfo=tzfile('/usr/share/zoneinfo/America/Chicago'))

            This parameter is ignored if ``ignoretz`` is set.

        :param \\*\\*kwargs:
            Keyword arguments as passed to ``_parse()``.

        :return:
            Returns a :class:`datetime.datetime` object or, if the
            ``fuzzy_with_tokens`` option is ``True``, returns a tuple, the
            first element being a :class:`datetime.datetime` object, the second
            a tuple containing the fuzzy tokens.

        :raises ParserError:
            Raised for invalid or unknown string format, if the provided
            :class:`tzinfo` is not in a valid format, or if an invalid date
            would be created.

        :raises TypeError:
            Raised for non-string or character stream input.

        :raises OverflowError:
            Raised if the parsed date exceeds the largest valid C integer on
            your system.
        """

        if default is None:
            default = datetime.datetime.now().replace(hour=0, minute=0,
                                                      second=0, microsecond=0)

        res, skipped_tokens = self._parse(timestr, **kwargs)

        if res is None:
            raise ParserError("Unknown string format: %s", timestr)

        if len(res) == 0:
            raise ParserError("String does not contain a date: %s", timestr)

        try:
            ret = self._build_naive(res, default)
        except ValueError as e:
            six.raise_from(ParserError(str(e) + ": %s", timestr), e)

        if not ignoretz:
            ret = self._build_tzaware(ret, res, tzinfos)

        if kwargs.get('fuzzy_with_tokens', False):
            return ret, skipped_tokens
        else:
            return ret

    class _result(_resultbase):
        __slots__ = ["year", "month", "day", "weekday",
                     "hour", "minute", "second", "microsecond",
                     "tzname", "tzoffset", "ampm","any_unused_tokens"]

    def _parse(self, timestr, dayfirst=None, yearfirst=None, fuzzy=False,
               fuzzy_with_tokens=False):
        """
        Private method which performs the heavy lifting of parsing, called from
        ``parse()``, which passes on its ``kwargs`` to this function.

        :param timestr:
            The string to parse.

        :param dayfirst:
            Whether to interpret the first value in an ambiguous 3-integer date
            (e.g. 01/05/09) as the day (``True``) or month (``False``). If
            ``yearfirst`` is set to ``True``, this distinguishes between YDM
            and YMD. If set to ``None``, this value is retrieved from the
            current :class:`parserinfo` object (which itself defaults to
            ``False``).

        :param yearfirst:
            Whether to interpret the first value in an ambiguous 3-integer date
            (e.g. 01/05/09) as the year. If ``True``, the first number is taken
            to be the year, otherwise the last number is taken to be the year.
            If this is set to ``None``, the value is retrieved from the current
            :class:`parserinfo` object (which itself defaults to ``False``).

        :param fuzzy:
            Whether to allow fuzzy parsing, allowing for string like "Today is
            January 1, 2047 at 8:21:00AM".

        :param fuzzy_with_tokens:
            If ``True``, ``fuzzy`` is automatically set to True, and the parser
            will return a tuple where the first element is the parsed
            :class:`datetime.datetime` datetimestamp and the second element is
            a tuple containing the portions of the string which were ignored:

            .. doctest::

                >>> from dateutil.parser import parse
                >>> parse("Today is January 1, 2047 at 8:21:00AM", fuzzy_with_tokens=True)
                (datetime.datetime(2047, 1, 1, 8, 21), (u'Today is ', u' ', u'at '))

        """
        if fuzzy_with_tokens:
            fuzzy = True

        info = self.info

        if dayfirst is None:
            dayfirst = info.dayfirst

        if yearfirst is None:
            yearfirst = info.yearfirst

        res = self._result()
        l = _timelex.split(timestr)         # Splits the timestr into tokens

        skipped_idxs = []

        # year/month/day list
        ymd = _ymd()

        len_l = len(l)
        i = 0
        try:
            while i < len_l:

                # Check if it's a number
                value_repr = l[i]
                try:
                    value = float(value_repr)
                except ValueError:
                    value = None

                if value is not None:
                    # Numeric token
                    i = self._parse_numeric_token(l, i, info, ymd, res, fuzzy)

                # Check weekday
                elif info.weekday(l[i]) is not None:
                    value = info.weekday(l[i])
                    res.weekday = value

                # Check month name
                elif info.month(l[i]) is not None:
                    value = info.month(l[i])
                    ymd.append(value, 'M')

                    if i + 1 < len_l:
                        if l[i + 1] in ('-', '/'):
                            # Jan-01[-99]
                            sep = l[i + 1]
                            ymd.append(l[i + 2])

                            if i + 3 < len_l and l[i + 3] == sep:
                                # Jan-01-99
                                ymd.append(l[i + 4])
                                i += 2

                            i += 2

                        elif (i + 4 < len_l and l[i + 1] == l[i + 3] == ' ' and
                              info.pertain(l[i + 2])):
                            # Jan of 01
                            # In this case, 01 is clearly year
                            if l[i + 4].isdigit():
                                # Convert it here to become unambiguous
                                value = int(l[i + 4])
                                year = str(info.convertyear(value))
                                ymd.append(year, 'Y')
                            else:
                                # Wrong guess
                                pass
                                # TODO: not hit in tests
                            i += 4

                # Check am/pm
                elif info.ampm(l[i]) is not None:
                    value = info.ampm(l[i])
                    val_is_ampm = self._ampm_valid(res.hour, res.ampm, fuzzy)

                    if val_is_ampm:
                        res.hour = self._adjust_ampm(res.hour, value)
                        res.ampm = value

                    elif fuzzy:
                        skipped_idxs.append(i)

                # Check for a timezone name
                elif self._could_be_tzname(res.hour, res.tzname, res.tzoffset, l[i]):
                    res.tzname = l[i]
                    res.tzoffset = info.tzoffset(res.tzname)

                    # Check for something like GMT+3, or BRST+3. Notice
                    # that it doesn't mean "I am 3 hours after GMT", but
                    # "my time +3 is GMT". If found, we reverse the
                    # logic so that timezone parsing code will get it
                    # right.
                    if i + 1 < len_l and l[i + 1] in ('+', '-'):
                        l[i + 1] = ('+', '-')[l[i + 1] == '+']
                        res.tzoffset = None
                        if info.utczone(res.tzname):
                            # With something like GMT+3, the timezone
                            # is *not* GMT.
                            res.tzname = None

                # Check for a numbered timezone
                elif res.hour is not None and l[i] in ('+', '-'):
                    signal = (-1, 1)[l[i] == '+']
                    len_li = len(l[i + 1])

                    # TODO: check that l[i + 1] is integer?
                    if len_li == 4:
                        # -0300
                        hour_offset = int(l[i + 1][:2])
                        min_offset = int(l[i + 1][2:])
                    elif i + 2 < len_l and l[i + 2] == ':':
                        # -03:00
                        hour_offset = int(l[i + 1])
                        min_offset = int(l[i + 3])  # TODO: Check that l[i+3] is minute-like?
                        i += 2
                    elif len_li <= 2:
                        # -[0]3
                        hour_offset = int(l[i + 1][:2])
                        min_offset = 0
                    else:
                        raise ValueError(timestr)

                    res.tzoffset = signal * (hour_offset * 3600 + min_offset * 60)

                    # Look for a timezone name between parenthesis
                    if (i + 5 < len_l and
                            info.jump(l[i + 2]) and l[i + 3] == '(' and
                            l[i + 5] == ')' and
                            3 <= len(l[i + 4]) and
                            self._could_be_tzname(res.hour, res.tzname,
                                                  None, l[i + 4])):
                        # -0300 (BRST)
                        res.tzname = l[i + 4]
                        i += 4

                    i += 1

                # Check jumps
                elif not (info.jump(l[i]) or fuzzy):
                    raise ValueError(timestr)

                else:
                    skipped_idxs.append(i)
                i += 1

            # Process year/month/day
            year, month, day = ymd.resolve_ymd(yearfirst, dayfirst)

            res.century_specified = ymd.century_specified
            res.year = year
            res.month = month
            res.day = day

        except (IndexError, ValueError):
            return None, None

        if not info.validate(res):
            return None, None

        if fuzzy_with_tokens:
            skipped_tokens = self._recombine_skipped(l, skipped_idxs)
            return res, tuple(skipped_tokens)
        else:
            return res, None

    def _parse_numeric_token(self, tokens, idx, info, ymd, res, fuzzy):
        # Token is a number
        value_repr = tokens[idx]
        try:
            value = self._to_decimal(value_repr)
        except Exception as e:
            six.raise_from(ValueError('Unknown numeric token'), e)

        len_li = len(value_repr)

        len_l = len(tokens)

        if (len(ymd) == 3 and len_li in (2, 4) and
            res.hour is None and
            (idx + 1 >= len_l or
             (tokens[idx + 1] != ':' and
              info.hms(tokens[idx + 1]) is None))):
            # 19990101T23[59]
            s = tokens[idx]
            res.hour = int(s[:2])

            if len_li == 4:
                res.minute = int(s[2:])

        elif len_li == 6 or (len_li > 6 and tokens[idx].find('.') == 6):
            # YYMMDD or HHMMSS[.ss]
            s = tokens[idx]

            if not ymd and '.' not in tokens[idx]:
                ymd.append(s[:2])
                ymd.append(s[2:4])
                ymd.append(s[4:])
            else:
                # 19990101T235959[.59]

                # TODO: Check if res attributes already set.
                res.hour = int(s[:2])
                res.minute = int(s[2:4])
                res.second, res.microsecond = self._parsems(s[4:])

        elif len_li in (8, 12, 14):
            # YYYYMMDD
            s = tokens[idx]
            ymd.append(s[:4], 'Y')
            ymd.append(s[4:6])
            ymd.append(s[6:8])

            if len_li > 8:
                res.hour = int(s[8:10])
                res.minute = int(s[10:12])

                if len_li > 12:
                    res.second = int(s[12:])

        elif self._find_hms_idx(idx, tokens, info, allow_jump=True) is not None:
            # HH[ ]h or MM[ ]m or SS[.ss][ ]s
            hms_idx = self._find_hms_idx(idx, tokens, info, allow_jump=True)
            (idx, hms) = self._parse_hms(idx, tokens, info, hms_idx)
            if hms is not None:
                # TODO: checking that hour/minute/second are not
                # already set?
                self._assign_hms(res, value_repr, hms)

        elif idx + 2 < len_l and tokens[idx + 1] == ':':
            # HH:MM[:SS[.ss]]
            res.hour = int(value)
            value = self._to_decimal(tokens[idx + 2])  # TODO: try/except for this?
            (res.minute, res.second) = self._parse_min_sec(value)

            if idx + 4 < len_l and tokens[idx + 3] == ':':
                res.second, res.microsecond = self._parsems(tokens[idx + 4])

                idx += 2

            idx += 2

        elif idx + 1 < len_l and tokens[idx + 1] in ('-', '/', '.'):
            sep = tokens[idx + 1]
            ymd.append(value_repr)

            if idx + 2 < len_l and not info.jump(tokens[idx + 2]):
                if tokens[idx + 2].isdigit():
                    # 01-01[-01]
                    ymd.append(tokens[idx + 2])
                else:
                    # 01-Jan[-01]
                    value = info.month(tokens[idx + 2])

                    if value is not None:
                        ymd.append(value, 'M')
                    else:
                        raise ValueError()

                if idx + 3 < len_l and tokens[idx + 3] == sep:
                    # We have three members
                    value = info.month(tokens[idx + 4])

                    if value is not None:
                        ymd.append(value, 'M')
                    else:
                        ymd.append(tokens[idx + 4])
                    idx += 2

                idx += 1
            idx += 1

        elif idx + 1 >= len_l or info.jump(tokens[idx + 1]):
            if idx + 2 < len_l and info.ampm(tokens[idx + 2]) is not None:
                # 12 am
                hour = int(value)
                res.hour = self._adjust_ampm(hour, info.ampm(tokens[idx + 2]))
                idx += 1
            else:
                # Year, month or day
                ymd.append(value)
            idx += 1

        elif info.ampm(tokens[idx + 1]) is not None and (0 <= value < 24):
            # 12am
            hour = int(value)
            res.hour = self._adjust_ampm(hour, info.ampm(tokens[idx + 1]))
            idx += 1

        elif ymd.could_be_day(value):
            ymd.append(value)

        elif not fuzzy:
            raise ValueError()

        return idx

    def _find_hms_idx(self, idx, tokens, info, allow_jump):
        len_l = len(tokens)

        if idx+1 < len_l and info.hms(tokens[idx+1]) is not None:
            # There is an "h", "m", or "s" label following this token.  We take
            # assign the upcoming label to the current token.
            # e.g. the "12" in 12h"
            hms_idx = idx + 1

        elif (allow_jump and idx+2 < len_l and tokens[idx+1] == ' ' and
              info.hms(tokens[idx+2]) is not None):
            # There is a space and then an "h", "m", or "s" label.
            # e.g. the "12" in "12 h"
            hms_idx = idx + 2

        elif idx > 0 and info.hms(tokens[idx-1]) is not None:
            # There is a "h", "m", or "s" preceding this token.  Since neither
            # of the previous cases was hit, there is no label following this
            # token, so we use the previous label.
            # e.g. the "04" in "12h04"
            hms_idx = idx-1

        elif (1 < idx == len_l-1 and tokens[idx-1] == ' ' and
              info.hms(tokens[idx-2]) is not None):
            # If we are looking at the final token, we allow for a
            # backward-looking check to skip over a space.
            # TODO: Are we sure this is the right condition here?
            hms_idx = idx - 2

        else:
            hms_idx = None

        return hms_idx

    def _assign_hms(self, res, value_repr, hms):
        # See GH issue #427, fixing float rounding
        value = self._to_decimal(value_repr)

        if hms == 0:
            # Hour
            res.hour = int(value)
            if value % 1:
                res.minute = int(60*(value % 1))

        elif hms == 1:
            (res.minute, res.second) = self._parse_min_sec(value)

        elif hms == 2:
            (res.second, res.microsecond) = self._parsems(value_repr)

    def _could_be_tzname(self, hour, tzname, tzoffset, token):
        return (hour is not None and
                tzname is None and
                tzoffset is None and
                len(token) <= 5 and
                (all(x in string.ascii_uppercase for x in token)
                 or token in self.info.UTCZONE))

    def _ampm_valid(self, hour, ampm, fuzzy):
        """
        For fuzzy parsing, 'a' or 'am' (both valid English words)
        may erroneously trigger the AM/PM flag. Deal with that
        here.
        """
        val_is_ampm = True

        # If there's already an AM/PM flag, this one isn't one.
        if fuzzy and ampm is not None:
            val_is_ampm = False

        # If AM/PM is found and hour is not, raise a ValueError
        if hour is None:
            if fuzzy:
                val_is_ampm = False
            else:
                raise ValueError('No hour specified with AM or PM flag.')
        elif not 0 <= hour <= 12:
            # If AM/PM is found, it's a 12 hour clock, so raise
            # an error for invalid range
            if fuzzy:
                val_is_ampm = False
            else:
                raise ValueError('Invalid hour specified for 12-hour clock.')

        return val_is_ampm

    def _adjust_ampm(self, hour, ampm):
        if hour < 12 and ampm == 1:
            hour += 12
        elif hour == 12 and ampm == 0:
            hour = 0
        return hour

    def _parse_min_sec(self, value):
        # TODO: Every usage of this function sets res.second to the return
        # value. Are there any cases where second will be returned as None and
        # we *don't* want to set res.second = None?
        minute = int(value)
        second = None

        sec_remainder = value % 1
        if sec_remainder:
            second = int(60 * sec_remainder)
        return (minute, second)

    def _parse_hms(self, idx, tokens, info, hms_idx):
        # TODO: Is this going to admit a lot of false-positives for when we
        # just happen to have digits and "h", "m" or "s" characters in non-date
        # text?  I guess hex hashes won't have that problem, but there's plenty
        # of random junk out there.
        if hms_idx is None:
            hms = None
            new_idx = idx
        elif hms_idx > idx:
            hms = info.hms(tokens[hms_idx])
            new_idx = hms_idx
        else:
            # Looking backwards, increment one.
            hms = info.hms(tokens[hms_idx]) + 1
            new_idx = idx

        return (new_idx, hms)

    # ------------------------------------------------------------------
    # Handling for individual tokens.  These are kept as methods instead
    #  of functions for the sake of customizability via subclassing.

    def _parsems(self, value):
