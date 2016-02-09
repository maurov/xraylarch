#!/usr/bin/env python
from __future__ import print_function

MODDOC = """
=== Epics Scanning Functions for Larch ===


This does not used the Epics SScan Record, and the scan is intended to run
as a python application, but many concepts from the Epics SScan Record are
borrowed.  Where appropriate, the difference will be noted here.

A Step Scan consists of the following objects:
   a list of Positioners
   a list of Triggers
   a list of Counters

Each Positioner will have a list (or numpy array) of position values
corresponding to the steps in the scan.  As there is a fixed number of
steps in the scan, the position list for each positioners must have the
same length -- the number of points in the scan.  Note that, unlike the
SScan Record, the list of points (not start, stop, step, npts) must be
given.  Also note that the number of positioners or number of points is not
limited.

A Trigger is simply an Epics PV that will start a particular detector,
usually by having 1 written to its field.  It is assumed that when the
Epics ca.put() to the trigger completes, the Counters associated with the
triggered detector will be ready to read.

A Counter is simple a PV whose value should be recorded at every step in
the scan.  Any PV can be a Counter, including waveform records.  For many
detector types, it is possible to build a specialized class that creates
many counters.

Because Triggers and Counters are closely associated with detectors, a
Detector is also defined, which simply contains a single Trigger and a list
of Counters, and will cover most real use cases.

In addition to the core components (Positioners, Triggers, Counters, Detectors),
a Step Scan contains the following objects:

   breakpoints   a list of scan indices at which to pause and write data
                 collected so far to disk.
   extra_pvs     a list of (description, PV) tuples that are recorded at
                 the beginning of scan, and at each breakpoint, to be
                 recorded to disk file as metadata.
   pre_scan()    method to run prior to scan.
   post_scan()   method to run after scan.
   at_break()    method to run at each breakpoint.

Note that Postioners and Detectors may add their own pieces into extra_pvs,
pre_scan(), post_scan(), and at_break().

With these concepts, a Step Scan ends up being a fairly simple loop, going
roughly (that is, skipping error checking) as:

   pos = <DEFINE POSITIONER LIST>
   det = <DEFINE DETECTOR LIST>
   run_pre_scan(pos, det)
   [p.move_to_start() for p in pos]
   record_extra_pvs(pos, det)
   for i in range(len(pos[0].array)):
       [p.move_to_pos(i) for p in pos]
       while not all([p.done for p in pos]):
           time.sleep(0.001)
       [trig.start() for trig in det.triggers]
       while not all([trig.done for trig in det.triggers]):
           time.sleep(0.001)
       [det.read() for det in det.counters]

       if i in breakpoints:
           write_data(pos, det)
           record_exrta_pvs(pos, det)
           run_at_break(pos, det)
   write_data(pos, det)
   run_post_scan(pos, det)

Note that multi-dimensional mesh scans over a rectangular grid is not
explicitly supported, but these can be easily emulated with the more
flexible mechanism of unlimited list of positions and breakpoints.
Non-mesh scans are also possible.

A step scan can have an Epics SScan Record or StepScan database associated
with it.  It will use these for PVs to post data at each point of the scan.
"""
import os, shutil
import time
import threading
import json
import numpy as np

from datetime import timedelta

from larch import Group, ValidateLarchPlugin
from larch.utils import debugtime
from larch_plugins.io import fix_varname

try:
    from epics import PV, caget, caput, get_pv, poll
    HAS_EPICS = True
except ImportError:
    HAS_EPICS = False

try:
    import epicsscan
    from epicsscan import (Counter, Trigger, AreaDetector, get_detector,
                           ASCIIScanFile, Positioner, StepScan, XAFS_Scan)
    from epicsscan.scandb import ScanDBException, ScanDBAbort

    HAS_EPICSSCAN = True
except ImportError:
    HAS_EPICSSCAN = False

from epics.devices.struck import Struck
from epics.devices.xspress3 import Xspress3

from larch_plugins.epics.xps_trajectory import XPSTrajectory

MODNAME = '_scan'

MIN_POLL_TIME = 1.e-3

XAFS_K2E = 3.809980849311092
HC       = 12398.4193
RAD2DEG  = 180.0/np.pi
MAXPTS   = 8192

class PVSlaveThread(threading.Thread):
    """
    Sets up a Thread to allow a Master Index PV (say, an advancing channel)
    to send a Slave PV to a value from a pre-defined array.

    undulator = PVSlaveThread(master_pvname='13IDE:SIS1:CurrentChannel',
                              slave_pvname='ID13us:ScanEnergy')
    undulator.set_array(array_of_values)
    undulator.enable()
    # start thread
    undulator.start()

    # other code that may trigger the master PV

    # to finish, set 'running' to False, and join() thread
    undulator.running = False
    undulator.join()

    """
    def __init__(self, master_pvname=None,  slave_pvname=None,
                 values=None, maxpts=8192, wait_time=0.05, offset=3):
        threading.Thread.__init__(self)
        self.maxpts = maxpts
        self.offset = offset
        self.wait_time = wait_time
        self.pulse = -1
        self.last  = None
        self.running = False
        self.vals = values
        if self.vals is None:
            self.vals  = np.zeros(self.maxpts)
        self.master = None
        self.slave = None
        if master_pvname is not None: self.set_master(master_pvname)
        if slave_pvname is not None: self.set_slave(slave_pvname)

    def set_master(self, pvname):
        self.master = PV(pvname, callback=self.onPulse)

    def set_slave(self, pvname):
        self.slave = PV(pvname)

    def onPulse(self, pvname, value=1, **kws):
        self.pulse  = max(0, min(self.maxpts, value + self.offset))

    def set_array(self, vals):
        "set array values for slave PV"
        n = len(vals)
        if n > self.maxpts:
            vals = vals[:self.maxpts]
        self.vals  = np.ones(self.maxpts) * vals[-1]
        self.vals[:n] = vals

    def enable(self):
        self.last = self.pulse = -1
        self.running = True

    def run(self):
        while self.running:
            time.sleep(self.wait_time)
            if self.pulse > self.last and self.last is not None:
                val = self.vals[self.pulse]
                try:
                    self.slave.put(val)
                except:
                    print("Cannot Caput: ", self.slave.pvname , val)
                self.last = self.pulse

def ktoe(k):
    return k*k*XAFS_K2E

def energy2angle(energy, dspace=3.13555):
    omega   = HC/(2.0 * dspace)
    return RAD2DEG * np.arcsin(omega/energy)

class QXAFS_Scan(XAFS_Scan): # (LarchStepScan):
    """QuickXAFS Scan"""

    def __init__(self, label=None, energy_pv=None, read_pv=None,
                 extra_pvs=None, e0=0, _larch=None, **kws):

        self.label = label
        self.e0 = e0
        self.energies = []
        self.regions = []
        XAFS_Scan.__init__(self, label=label, energy_pv=energy_pv,
                           read_pv=read_pv, e0=e0, extra_pvs=extra_pvs,
                           _larch=_larch, **kws)

        self.is_qxafs = True
        self.scantype = 'xafs'
        qconf = self.scandb.get_config('QXAFS')
        qconf = self.qconf = json.loads(qconf.notes)

        self.xps = XPSTrajectory(qconf['host'],
                                 user=qconf['user'],
                                 password=qconf['passwd'],
                                 group=qconf['group'],
                                 positioners=qconf['positioners'],
                                 outputs=qconf['outputs'])


        self.set_energy_pv(energy_pv, read_pv=read_pv, extra_pvs=extra_pvs)

    def make_XPS_trajectory(self, reverse=False,
                            theta_accel=0.25, width_accel=0.25, **kws):
        """this method builds the text of a Trajectory script for
        a Newport XPS Controller based on the energies and dwelltimes"""

        qconf = self.qconf

        dspace = caget(qconf['dspace_pv'])
        height = caget(qconf['height_pv'])
        th_off = caget(qconf['theta_motor'] + '.OFF')
        wd_off = caget(qconf['width_motor'] + '.OFF')

        energy = np.array(self.energies)
        times  = np.array(self.dwelltime)
        if reverse:
            energy = energy[::-1]
            times  = times[::-1]

        traw    = energy2angle(energy, dspace=dspace)
        theta  = 1.0*traw
        theta[1:-1] = traw[1:-1]/2.0 + traw[:-2]/4.0 + traw[2:]/4.0
        width  = height / (2.0 * np.cos(theta/RAD2DEG))

        width -= wd_off
        theta -= th_off

        tvelo = np.gradient(theta)/times
        wvelo = np.gradient(width)/times
        tim0  = abs(tvelo[0] / theta_accel)
        the0  = 0.5 * tvelo[ 0] * tim0
        wid0  = 0.5 * wvelo[ 0] * tim0
        the1  = 0.5 * tvelo[-1] * tim0
        wid1  = 0.5 * wvelo[-1] * tim0

        dtheta = np.diff(theta)
        dwidth = np.diff(width)
        dtime  = times[1:]
        fmt = '%.8f, %.8f, %.8f, %.8f, %.8f'
        efirst = fmt % (tim0, the0, tvelo[0], wid0, wvelo[0])
        elast  = fmt % (tim0, the1, 0.00,     wid1, 0.00)

        buff  = ['', efirst]
        for i in range(len(dtheta)):
            buff.append(fmt % (dtime[i], dtheta[i], tvelo[i],
                               dwidth[i], wvelo[i]))
        buff.append(elast)

        return  Group(buffer='\n'.join(buff),
                      start_theta=theta[0]-the0,
                      start_width=width[0]-wid0,
                      theta=theta, tvelo=tvelo,   times=times,
                      energy=energy, width=width, wvelo=wvelo)


    def init_qscan(self, traj):
        """initialize a QXAFS scan"""

        qconf = self.qconf

        caput(qconf['id_track_pv'],  1)
        caput(qconf['y2_track_pv'],  1)

        time.sleep(0.1)
        caput(qconf['width_motor'] + '.DVAL', traj.start_width)
        caput(qconf['theta_motor'] + '.DVAL', traj.start_theta)
        # caput(qconf['id_track_pv'], 0)
        # caput(qconf['id_wait_pv'], 0)
        caput(qconf['y2_track_pv'], 0)


    def finish_qscan(self):
        """initialize a QXAFS scan"""
        qconf = self.qconf

        caput(qconf['id_track_pv'],  1)
        caput(qconf['y2_track_pv'],  1)
        time.sleep(0.1)

    def run(self, filename=None, comments=None, debug=False, reverse=False):
        """
        run the actual QXAFS scan
        """
        print(" QXAFS run!!!! ")
        self.dtimer = dtimer = debugtime(verbose=debug)

        self.complete = False
        if filename is not None:
            self.filename  = filename
        if comments is not None:
            self.comments = comments

        ts_start = time.time()
        if not self.verify_scan():
            self.write('Cannot execute scan: %s' % self._scangroup.error_message)
            self.set_info('scan_message', 'cannot execute scan')
            return

        qconf = self.qconf
        energy_orig = caget(qconf['energy_pv'])

        traj = self.make_XPS_trajectory(reverse=reverse)

        self.init_qscan(traj)

        idarray = 0.001*traj.energy + caget(qconf['id_offset_pv'])

        # caput(qconf['id_drive_pv'], idarray[0], wait=False)
        caput(qconf['energy_pv'],  traj.energy[0], wait=False)

        self.xps.upload_trajectoryFile(qconf['traj_name'], traj.buffer)

        self.clear_interrupts()
        print("QXAFS Positioners ", self.positioners)
        orig_positions = [p.current() for p in self.positioners]
        dtimer.add('PRE: cleared interrupts')
        #  move to start here?

        sis_prefix = qconf['mcs_prefix']

        und_thread = PVSlaveThread(master_pvname=sis_prefix+'CurrentChannel',
                                   slave_pvname=qconf['id_drive_pv'])

        und_thread.set_array(idarray)
        und_thread.running = False

        npulses = len(traj.energy) + 1

        caput(qconf['energy_pv'], traj.energy[0], wait=True)
        # caput(qconf['id_drive_pv'], idarray[0], wait=True, timeout=5.0)


        npts = len(self.positioners[0].array)
        self.dwelltime_varys = False
        dtime = self.dwelltime[0]
        time_est = npts*dtime

        self.set_info('scan_progress', 'preparing scan')
        extra_vals = []
        for desc, pv in self.extra_pvs:
            extra_vals.append((desc, pv.get(as_string=True), pv.pvname))

        sis_opts = {}
        xsp3_prefix = None

        for d in self.detectors:
            if 'scaler' in d.label.lower():
                sis_opts['scaler'] = d.prefix
            elif 'xspress3' in d.label.lower():
                xsp3_prefix = d.prefix

        qxsp3 = Xspress3(xsp3_prefix)
        sis  = Struck(sis_prefix, **sis_opts)

        caput(qconf['energy_pv'], traj.energy[0])

        dtimer.add('PRE: cleared data')
        out = self.pre_scan()
        self.check_outputs(out, msg='pre scan')
        dtimer.add('PRE: pre_scan done')

        self.counters = []
        qxafs_counters = []
        for i, mca in enumerate(sis.mcas):
            scalername = getattr(sis.scaler, 'NM%i' % (i+1), '')
            if len(scalername) > 1:
                qxafs_counters.append(mca._pvs['VAL'])

        for roi in range(1, 5):
            desc = caget('%sC1_ROI%i:ValueSum_RBV.DESC' % (xsp3_prefix, roi))
            if len(desc) > 0:
                for card in range(1, 5):
                    pvname = '%sC%i_ROI%i:ArrayData_RBV' % (xsp3_prefix, card, roi)
                    qxafs_counters.append(PV(pvname))

        dtimer.add('PRE: got counters')
        sis.stop()
        sis.ExternalMode()
        sis.NuseAll = MAXPTS
        sis.put('PresetReal', 0.0)
        sis.put('Prescale',   1.0)

        qxsp3.NumImages = MAXPTS
        qxsp3.useExternalTrigger()
        qxsp3.setFileWriteMode(2)
        qxsp3.setFileTemplate('%s%s.%4.4d')
        qxsp3.setFileName('xsp3')
        qxsp3.setFileNumber(1)

        ## Start Scan
        qxsp3.setFileNumber(1)
        qxsp3.Acquire = 0
        time.sleep(0.1)
        qxsp3.ERASE  = 1
        dtimer.add('PRE: setup detectors')

        self.datafile = self.open_output_file(filename=self.filename,
                                              comments=self.comments)

        self.filename =  self.datafile.filename

        dtimer.add('PRE: open datafile')
        self.clear_data()
        dtimer.add('PRE: clear data')
        self.datafile.write_data(breakpoint=0)
        dtimer.add('PRE: start data')
        self.init_scandata()

        dtimer.add('PRE: init data')

        self.set_info('request_abort', 0)
        self.set_info('scan_time_estimate', time_est)
        self.set_info('scan_total_points', npts)

        caput(qconf['energy_pv'], traj.energy[0], wait=True)

        time.sleep(0.05)
        # caput(qconf['id_drive_pv'], idarray[0], wait=True, timeout=5.0)

        dtimer.add('PRE: caputs done')

        self.set_info('scan_progress', 'starting scan')
        #self.msg_thread = ScanMessenger(func=self._messenger, npts=npts, cpt=0)
        # self.msg_thread.start()
        self.cpt = 0
        self.npts = npts

        t0 = time.time()
        out = [p.move_to_start(wait=True) for p in self.positioners]
        self.check_outputs(out, msg='move to start, wait=True')
        [p.current() for p in self.positioners]
        [d.pv.get() for d in self.counters]
        i = -1
        caput(qconf['width_motor'] + '.DVAL', traj.start_width, wait=True)
        caput(qconf['theta_motor'] + '.DVAL', traj.start_theta, wait=True)

        und_thread.enable()
        und_thread.start()
        time.sleep(0.2)

        ts_init = time.time()
        self.inittime = ts_init - ts_start
        dtimer.add('Start scan:')
        start_time = time.strftime('%Y-%m-%d %H:%M:%S')

        dtimer.show()
        self.xps.SetupTrajectory(npts+1, dtime, traj_file=qconf['traj_name'])

        sis.start()
        qxsp3.FileCaptureOn()
        qxsp3.Acquire = 1
        time.sleep(0.1)
        print("Trajectory Start")
        self.xps.RunTrajectory()
        print("Trajectory Done")

        self.xps.EndTrajectory()

        sis.stop()
        qxsp3.Acquire = 0

        self.finish_qscan()

        print(" FILENAME ", self.filename)
        GatherFile = '%s.gat' % self.filename
        DataFile   = '%s.dat' % self.filename

        self.xps.SaveResults(GatherFile)
        gather_text = open(GatherFile, 'r').readlines()
        ngather = len(gather_text)
        print( 'Gathering: ', GatherFile, '  %i lines ' % ngather)
        print( 'DataFile:  ', DataFile,   '  %i lines ' % sis.CurrentChannel)

        und_thread.running = False
        und_thread.join()

        time.sleep(1.00)
        caput(qconf['energy_pv'], energy_orig-2.0)

        nout = sis.CurrentChannel
        narr = 0
        t0  = time.time()
        print("QXAFS Counters: %i" % (len(qxafs_counters)))
        for qc in  qxafs_counters: print("   " , qc.pvname)
        print("---")
        while narr < (nout-1) and (time.time()-t0 )<30.0:
            time.sleep(0.25)
            dat =  [p.get() for p in qxafs_counters]
            narr = min([len(d) for d in dat])
            print( " Wait for data ", narr)

        print( 'Read Data from SIS: ', narr)
        fout = open(DataFile, 'w')
        obuff =['# Gathered XRF and IO data',
                '# Scan.start_time: %s' % (start_time),
                '# Scan.end_time: %s' % (time.strftime('%Y-%m-%d %H:%M:%S'))]

        for desc, val, addr in extra_vals:
            obuff.append("# %s: %s || %s" % (desc, val, addr))

        obuff.append('#--------------------------------------------')
        obuff.append('# Time  I0  I1 I2  MCA1R1 MCA2R1 MCA3R1 MCA4R1 MCA1R2 MCA2R2 MCA3R3 MCA4R4 OCR1 OCR2 OCR3 OCR4')

        for i in obuff:
            fout.write("%s\n"% ( i))

        for i in range(narr):
            # fout.write('  %13.4f  ' % (scan.energies[i])
            s = ' '.join(["%13.4f" % d[i] for d in dat])
            fout.write("%s\n" % s)

        fout.write('\n')
        fout.close()

        caput(qconf['energy_pv'], energy_orig, wait=True)
        print( 'Wrote %s' % DataFile)

        ##
        dtimer.add('Post scan start')
        print(" --> Publish Scandata ")
        self.publish_scandata()
        ts_loop = time.time()
        self.looptime = ts_loop - ts_init
        print(" --> Publish Scandata Done")
        for val, pos in zip(orig_positions, self.positioners):
            pos.move_to(val, wait=False)
        dtimer.add('Post: return move issued')
        print(" --> write datafile ", self.datafile)
        self.datafile.write_data(breakpoint=-1, close_file=True, clear=False)
        dtimer.add('Post: file written')
        if self.look_for_interrupts():
            self.write("scan aborted at point %i of %i." % (self.cpt, self.npts))
            raise ScanDBAbort("scan aborted")

        # run post_scan methods
        self.set_info('scan_progress', 'finishing')

        out = self.post_scan()
        self.check_outputs(out, msg='post scan')
        dtimer.add('Post: post_scan done')
        self.complete = True

        self.set_info('scan_progress',
                      'scan complete. Wrote %s' % self.datafile.filename)
        ts_exit = time.time()
        self.exittime = ts_exit - ts_loop
        self.runtime  = ts_exit - ts_start
        dtimer.add('Post: fully done')

        return self.datafile.filename
        ##

def get_prescan_function(_larch=None):
    """find and return pre_scan_command()  script"""
    prescan = None
    try:
        prescan = _larch.symtable.get_symbol('pre_scan_command')
    except NameError:
        pass
    if prescan is not None:
        if (isinstance(prescan, Group) and
            hasattr(prescan, 'pre_scan_command')):
            prescan = prescan.pre_scan_command
    return prescan

@ValidateLarchPlugin
def scan_from_json(text, filename='scan.001', current_rois=None,
                   scandb=None, prescan_func=None, is_qxafs=False,
                   _larch=None, **kws):
    """(PRIVATE)

    creates and returns a StepScan object from a json-text
    representation.

    """
    sdict = json.loads(text)
    print(" -- SCAN from JSON ", sdict, kws)
    scanopts = dict(filename=filename,
                    scandb=scandb,
                    prescan_func=prescan_func, _larch=_larch)
    print("  -- scanoopts ", scanopts)
    #
    # create positioners
    if sdict['type'] == 'xafs':
        min_dtime = sdict['dwelltime']
        if isinstance(min_dtime, np.ndarray):
            min_dtime = min(dtime)
        is_qxafs = (is_qxafs or
                    sdict.get('is_qxafs', False) or
                    (min_dtime < 0.45))
        scanopts['energy_pv'] = sdict['energy_drive']
        scanopts['read_pv'] = sdict['energy_read']
        scanopts['e0v'] = sdict['e0']

        _ScanCreator = XAFS_Scan
        if is_qxafs: _ScanCreator = QXAFS_Scan
        scan = _ScanCreator(**scanopts)

        t_kw  = sdict['time_kw']
        t_max = sdict['max_time']
        nreg  = len(sdict['regions'])
        kws  = {'relative': sdict['is_relative']}
        for i, det in enumerate(sdict['regions']):
            start, stop, npts, dt, units = det
            kws['dtime'] =  dt
            kws['use_k'] =  units.lower() !='ev'
            if i == nreg-1: # final reg
                if t_max > dt and t_kw>0 and kws['use_k']:
                    kws['dtime_final'] = t_max
                    kws['dtime_wt'] = t_kw
            scan.add_region(start, stop, npts=npts, **kws)
    else:
        scan = StepScan(**scanopts)
        if sdict['type'] == 'linear':
            for pos in sdict['positioners']:
                label, pvs, start, stop, npts = pos
                p = Positioner(pvs[0], label=label)
                p.array = np.linspace(start, stop, npts)
                scan.add_positioner(p)
                if len(pvs) > 0:
                    scan.add_counter(pvs[1], label="%s_read" % label)

        elif sdict['type'] == 'mesh':
            label1, pvs1, start1, stop1, npts1 = sdict['inner']
            label2, pvs2, start2, stop2, npts2 = sdict['outer']
            p1 = Positioner(pvs1[0], label=label1)
            p2 = Positioner(pvs2[0], label=label2)

            inner = npts2* [np.linspace(start1, stop1, npts1)]
            outer = [[i]*npts1 for i in np.linspace(start2, stop2, npts2)]

            p1.array = np.array(inner).flatten()
            p2.array = np.array(outer).flatten()
            scan.add_positioner(p1)
            scan.add_positioner(p2)
            if len(pvs1) > 0:
                scan.add_counter(pvs1[1], label="%s_read" % label1)
            if len(pvs2) > 0:
                scan.add_counter(pvs2[1], label="%s_read" % label2)

        elif sdict['type'] == 'slew':
            label1, pvs1, start1, stop1, npts1 = sdict['inner']
            p1 = Positioner(pvs1[0], label=label1)
            p1.array = np.linspace(start1, stop1, npts1)
            scan.add_positioner(p1)
            if len(pvs1) > 0:
                scan.add_counter(pvs1[1], label="%s_read" % label1)
            if sdict['dimension'] >=2:
                label2, pvs2, start2, stop2, npts2 = sdict['outer']
                p2 = Positioner(pvs2[0], label=label2)
                p2.array = np.linspace(start2, stop2, npts2)
                scan.add_positioner(p2)
                if len(pvs2) > 0:
                    scan.add_counter(pvs2[1], label="%s_read" % label2)
    # detectors
    rois = sdict.get('rois', current_rois)
    scan.rois = rois

    for dpars in sdict['detectors']:
        dpars['rois'] = rois
        print("ADD DET ", dpars)
        scan.add_detector(get_detector(**dpars))

    # extra counters (not-triggered things to count
    if 'counters' in sdict:
        for label, pvname  in sdict['counters']:
            scan.add_counter(pvname, label=label)

    # other bits
    scan.add_extra_pvs(sdict['extra_pvs'])
    scan.scantype = sdict.get('type', 'linear')
    scan.scantime = sdict.get('scantime', -1)
    scan.filename = sdict.get('filename', 'scan.dat')
    if filename is not None:
        scan.filename  = filename
    scan.pos_settle_time = sdict.get('pos_settle_time', 0.01)
    scan.det_settle_time = sdict.get('det_settle_time', 0.01)
    scan.nscans          = sdict.get('nscans', 1)
    if scan.dwelltime is None:
        scan.set_dwelltime(sdict.get('dwelltime', 1))
    return scan

@ValidateLarchPlugin
def scan_from_db(name, filename='scan.001', timeout=5.0, is_qxafs=False,
                 _larch=None):
    """(PRIVATE)

    get scan definition from ScanDB

    timeout is for db lookup
    """
    if _larch.symtable._scan._scandb is None:
        return
    sdb = _larch.symtable._scan._scandb
    current_rois = json.loads(sdb.get_info('rois'))
    t0 = time.time()
    while time.time()-t0 < timeout:
        scandef = sdb.get_scandef(name)
        if scandef is not None:
            break
        time.sleep(0.25)

    if scandef is None:
        raise ScanDBException("no scan definition '%s' found" % name)

    prescan_func = get_prescan_function(_larch=_larch)
    return scan_from_json(scandef.text,
                          filename=filename,
                          is_qxafs=is_qxafs,
                          scandb=sdb,
                          prescan_func=prescan_func,
                          current_rois=current_rois,
                          _larch=_larch)


@ValidateLarchPlugin
def do_scan(scanname, filename='scan.001', nscans=1, comments='', _larch=None):
    """do_scan(scanname, filename='scan.001', nscans=1, comments='')

    execute a step scan as defined in Scan database

    Parameters
    ----------
    scanname:     string, name of scan
    filename:     string, name of output data file
    comments:     string, user comments for file
    nscans:       integer (default 1) number of repeats to make.

    Examples
    --------
      do_scan('cu_xafs', 'cu_sample1.001', nscans=3)

    Notes
    ------
      1. The filename will be incremented so that each scan uses a new filename.
    """

    if _larch.symtable._scan._scandb is None:
        print('need to connect to scandb!')
        return
    scandb =  _larch.symtable._scan._scandb
    if nscans is not None:
        scandb.set_info('nscans', nscans)

    scan = scan_from_db(scanname, filename=filename,
                        _larch=_larch)
    scan.comments = comments
    if scan.scantype == 'slew':
        return do_slewscan(scanname, filename=filename, nscans=nscans,
                           comments=comments, _larch=_larch)
    else:
        scans_completed = 0
        nscans = int(scandb.get_info('nscans'))
        abort  = scandb.get_info('request_abort', as_bool=True)
        while (scans_completed  < nscans) and not abort:
            scan.run()
            scans_completed += 1
            nscans = int(scandb.get_info('nscans'))
            abort  = scandb.get_info('request_abort', as_bool=True)
        return scan

@ValidateLarchPlugin
def do_slewscan(scanname, filename='scan.001', comments='',
                nscans=1, _larch=None):
    """do_slewscan(scanname, filename='scan.001', nscans=1, comments='')

    execute a slewscan as defined in Scan database

    Parameters
    ----------
    scanname:     string, name of scan
    filename:     string, name of output data file
    comments:     string, user comments for file
    nscans:       integer (default 1) number of repeats to make.

    Examples
    --------
      do_slewscan('small_map', 'map.001')

    Notes
    ------
      1. The filename will be incremented so that each scan uses a new filename.
    """

    if _larch.symtable._scan._scandb is None:
        print('need to connect to scandb!')
        return
    scan = scan_from_db(scanname, _larch=_larch)
    if scan.scantype != 'slew':
        return do_scan(scanname, comments=comments, nscans=1,
                       filename=filename, _larch=_larch)
    else:
        scan.epics_slewscan(filename=filename)
    return scan

@ValidateLarchPlugin
def make_xafs_scan(label=None, e0=0, _larch=None, **kws):
    return XAFS_Scan(label=label, e0=e0, _larch=_larch, **kws)

@ValidateLarchPlugin
def get_dbinfo(key, default=None, as_int=False, as_bool=False,
               full_row=False, _larch=None, **kws):
    """get a value for a keyword in the scan info table,
    where most status information is kept.

    Arguments
    ---------
     key        name of data to look up
     default    (default None) value to return if key is not found
     as_int     (default False) convert to integer
     as_bool    (default False) convert to bool
     full_row   (default False) return full row, not just value

    Notes
    -----
     1.  if this key doesn't exist, it will be added with the default
         value and the default value will be returned.
     2.  the full row will include notes, create_time, modify_time

    """
    if _larch.symtable._scan._scandb is None:
        print('need to connect to scandb!')
        return
    get_info = _larch.symtable._scan._scandb.get_info
    return get_info(key, default=default, full_row=full_row,
                    as_int=as_int, as_bool=as_bool, **kws)

def initializeLarchPlugin(_larch=None):
    """initialize _scan"""
    if not _larch.symtable.has_group(MODNAME):
        g = Group()
        g.__doc__ = MODDOC
        _larch.symtable.set_symbol(MODNAME, g)

def registerLarchPlugin():
    symbols = {}
    if HAS_EPICSSCAN:
        symbols = {'scan_from_json': scan_from_json,
                   'scan_from_db':   scan_from_db,
                   'make_xafs_scan': make_xafs_scan,
                   'do_scan': do_scan,
                   'do_slewscan': do_slewscan,
                   'do_fastmap':  do_slewscan,
                   'get_dbinfo': get_dbinfo}

    return (MODNAME, symbols)
