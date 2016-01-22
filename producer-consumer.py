from threading import Thread, Event
import thread, random, time, sys, signal
import Queue


class Log():
  def __init__(self):
    self.plock = thread.allocate_lock()
  def log(self, msg):
    self.plock.acquire()
    print msg
    self.plock.release()


class Producer(Thread):
  def __init__(self, q, stopSignal, log):
    Thread.__init__(self)
    self.q = q
    self.stopSignal = stopSignal
    self.l=log
  def run(self):
    print self.stopSignal
    while (not self.stopSignal.is_set()):
      r = random.randint(1,100)
      self.l.log("Producer writes " + str(r) + " from " + self.getName())
      self.q.put(r);
      time.sleep(1)
    print "terminating producer"

class Consumer(Thread):
  def __init__(self, q, stopSignal, log):
    Thread.__init__(self)
    self.q = q
    self.stopSignal = stopSignal
    self.l=log
  def run(self):
    print self.stopSignal
    while (not self.stopSignal.is_set()):
      try:
        r = self.q.get(2);
        self.l.log( "Consumer consumes " + str(r) + " from " + self.getName())
      except Queue.Empty:
        pass # do nothing
    print "terminating consumer"


def mk_signal_handler(stop_signal):
  def signal_handler(signal, frame):
    print stop_signal
    stop_signal.set()
    print "Terminating threads:"
    time.sleep(1)
    print "Exiting."
    sys.exit(0)
  return signal_handler

l=Log()
q = Queue.Queue()
stopSignal = Event()
signal.signal(signal.SIGINT, mk_signal_handler(stopSignal))
p=Producer(q, stopSignal, l)
p.start()
time.sleep(1)
c=Consumer(q, stopSignal, l)
c.start()

# wait until a signal is received
signal.pause()
