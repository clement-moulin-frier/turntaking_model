import threading
from copy import copy
from numpy import hstack, array
from collections import defaultdict

from explauto.utils.observer import Observer


class Experiment(Observer):
    def __init__(self, agents):
        """ This class is used to setup, run and log the results of an experiment.

            :param environment: an environment
            :type environment: :py:class:`~explauto.environment.environment.Environment`

            :param agent: an agent
            :type agent: :py:class:`~explauto.environment.agent.Agent`

            """
        Observer.__init__(self)

        for ag in agents:
            for topic in ["motor", "presence", "td_error", "activation", "weights", "reward"]:
                ag.subscribe(topic, self)
        self.ags = agents
        self.n_ag = len(agents)

        self.log = [defaultdict(list) for _ in range(self.n_ag)]

        self.eval_at = []


        self._running = threading.Event()

    def run(self, n_iter=-1, bg=False):
        """ Run the experiment.

            :param int n_iter: Number of run iterations, by default will run until the last evaluation step.
            :param bool bg: whether to run in background (using a Thread)

        """
        if n_iter == -1:
            if not self.eval_at:
                raise ValueError('Set n_iter or define evaluate_at.')

            n_iter = self.eval_at[-1] + 1

        self._running.set()

        if bg:
            self._t = threading.Thread(target=lambda: self._run(n_iter))
            self._t.start()
        else:
            self._run(n_iter)

    def wait(self):
        """ Wait for the end of the run of the experiment. """
        self._t.join()

    def stop(self):
        """ Stop the experiment. """
        self._running.clear()

    def fast_forward(self, log):
        self.log = copy(log)
        for x, y, s in zip(*[log.logs[topic] for topic in ['choice', 'inference', 'perception']]):
            m, s_ag = self.ag.extract_ms(x, y)
            self.ag.sensorimotor_model.update(m, s)
            self.ag.interest_model.update(hstack((m, s_ag)), hstack((m, s)))

    def _run(self, n_iter):
        for _ in range(n_iter):
            s = array([ag.produce() for ag in self.ags])
            for ag in self.ags:
                ag.perceive(s)

            self._update_logs()

            if not self._running.is_set():
                break

        self._running.clear()

    def _update_logs(self):
        while not self.notifications.empty():
            topic, (id, msg) = self.notifications.get()
            self.log[id][topic].append(msg)

    def evaluate_at(self, eval_at, testcases):
        """ Sets the evaluation interation indices.

            :param list eval_at: iteration indices where an evaluation should be performed
            :param numpy.array testcases: testcases used for evaluation

        """
        self.eval_at = eval_at
        self.log.eval_at = eval_at

        self.evaluation = Evaluation(self.ag, self.env, testcases)
        for test in testcases:
            self.log.add('testcases', test)



