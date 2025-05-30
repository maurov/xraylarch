#!/usr/bin/env python
"""
Code to write and read CVS files

"""
import sys
import os
import time
import json
import csv

import numpy as np
from pyshortcuts import bytes2str, fix_varname, gformat

from dateutil.parser import parse as dateparse
from larch import Group
from larch.math import interp, remove_dups

maketrans = str.maketrans

def groups2csv(grouplist, filename, delim=',',
               x='energy', y='norm', individual=False):
    """save data from a list of groups to a CSV file

    Arguments
    ---------
    grouplist  list of groups to save arrays from
    filname    name of output file
    x          name of group member to use for `x`
    y          name of group member to use for `y`
    individual toggle saving individual x/y in separate files

    """
    delim = delim.strip() + ' '
    def get_label(grp):
        'get label for group'
        for attr in ('filename', 'label', 'name', 'file', '__name__'):
            o = getattr(grp, attr, None)
            if o is not None:
                return o
        return repr(o)

    def save_group(g, delim=', ', x='energy', y='norm'):
        label = get_label(g)
        _x = getattr(g, x)
        _y = getattr(g, y)
        _n = len(_x)
        labels = [x, label]
        outarr = np.array([_x, _y])
        buff = [f"#saved {time.ctime()}",
                f"#saving x array={x}, y array={y}",
                f"#{label}: {g.filename}",
                "#------------------------------------------",
                "# %s" % delim.join(labels)]
        for i in range(_n):
            buff.append(delim.join([gformat(_x[i]), gformat(_y[i])]))
        buff.append('')
        fnout = f"{label}.csv"
        with open(fnout, 'w', encoding=sys.getdefaultencoding()) as fh:
            fh.write("\n".join(buff))
        print(f"Wrote group to {fnout}")

    if individual is True:
        for g in grouplist:
            save_group(g, delim=delim, x=x, y=y)
        return

    ngroups = len(grouplist)
    x0 = getattr(grouplist[0], x)
    npts = len(x0)
    columns = [x0, getattr(grouplist[0], y)]
    labels = [x, get_label(grouplist[0]) ]

    buff = ["# %d files saved %s" % (len(grouplist), time.ctime()),
            "# saving x array='%s', y array='%s'" % (x, y),
            "# %s: %s" % (labels[1], grouplist[0].filename)]

    for g in grouplist[1:]:
        label = get_label(g)
        buff.append("# %s: %s" % (label, g.filename))
        labels.append(label)
        _x = remove_dups(getattr(g, x))
        _y = getattr(g, y)
        if ((len(_x) != npts) or (abs(_x -x0)).sum() > 1.0):
            columns.append(interp(_x, _y, x0, kind='linear'))
        else:
            columns.append(_y)

    buff.append("#------------------------------------------")
    buff.append("# %s" % delim.join(labels))
    for i in range(npts):
        buff.append(delim.join([gformat(s[i]) for s in columns]))

    buff.append('')
    with open(filename, 'w', encoding=sys.getdefaultencoding()) as fh:
        fh.write("\n".join(buff))

    print("Wrote %i groups to %s" % (len(columns)-1, filename))


def str2float(word, allow_times=True):
    """convert a work to a float

    Arguments
    ---------
      word          str, word to be converted
      allow_times   bool, whether to support time stamps [True]

    Returns
    -------
      either a float or text

    Notes
    -----
      The `allow_times` will try to support common date-time strings
      using the dateutil module, returning a numerical value as the
      Unix timestamp, using
          time.mktime(dateutil.parser.parse(word).timetuple())
    """
    mktime = time.mktime
    val = word
    try:
        val = float(word)
    except ValueError:
        try:
            val = mktime(dateparse(word).timetuple())
        except ValueError:
            pass
    return val

def read_csv(filename):
    """read CSV file, return group with data as columns"""
    csvfile = open(filename, 'r')
    dialect = csv.Sniffer().sniff(csvfile.read(),  [',',';', '\t'])
    csvfile.seek(0)

    data = None
    isfloat = None
    for row in csv.reader(csvfile, dialect):
        if data is None:
            ncols = len(row)
            data = [[] for i in range(ncols)]
            isfloat =[None]*ncols
        for i, word in enumerate(row):
            data[i].append(str2float(word))
            if isfloat[i] is None:
                try:
                    _ = float(word)
                    isfloat[i] = True
                except ValueError:
                    isfloat[i] = False

    out = Group(filename=filename, data=data)
    for icol in range(ncols):
        cname = 'col_%2.2d' % (icol+1)
        val = data[icol]
        if isfloat[icol]:
            val = np.array(val)
        setattr(out, cname, val)

    return out
