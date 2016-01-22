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
  def __init__(self, q, stop_signal, log):
    Thread.__init__(self)
    self.q = q
    self.stop_signal = stop_signal
    self.l=log
  def run(self):
    while (not self.stop_signal.is_set()):
      r = random.randint(1,100)
      self.l.log("Producer writes " + str(r))
      self.q.put(r);
      time.sleep(1)
    self.l.log("terminating producer")

class Consumer(Thread):
  def __init__(self, q, stop_signal, log):
    Thread.__init__(self)
    self.q = q
    self.stop_signal = stop_signal
    self.l=log
  def run(self):
    while (not self.stop_signal.is_set()):
      try:
        # block for 0.5 seconds. If no message arrives,
        # unblock. This will raise a Queue.Empty exception.
        # This keeps the loop responsive so that the thread 
        # can be canceled. See http://stackoverflow.com/a/19206305
        r = self.q.get(True, 0.5);
        self.l.log( "Consumer consumes " + str(r))
      except Queue.Empty:
        pass # do nothing
    self.l.log("terminating consumer")


def mk_signal_handler(stop_signal):
  def signal_handler(signal, frame):
    stop_signal.set()
    print "Terminating threads:"
    time.sleep(1)
    print "Exiting."
    sys.exit(0)
  return signal_handler

l=Log()
q = Queue.Queue()
stop_signal = Event()
signal.signal(signal.SIGINT, mk_signal_handler(stop_signal))
p=Producer(q, stop_signal, l)
p.start()
time.sleep(1)
c=Consumer(q, stop_signal, l)
c.start()

# wait until a signal is received
signal.pause()
