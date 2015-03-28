from matplotlib import rcParams
from numpy import product, array, convolve, ones, hstack, tile
from matplotlib.pyplot import subplot, plot, xlabel, ylabel, axis, legend, savefig, tight_layout, gcf, figure

import seaborn as sns
sns.set_context("paper")

scale = 2.
params = {'lines.markersize': 10,
          'axes.labelsize': 10 * scale,
          'text.fontsize': 10 * scale,
          'legend.fontsize': 10 * scale,
          'xtick.labelsize': 8 * scale,
          'ytick.labelsize': 8 * scale,
          'text.usetex': False,
          'figure.figsize': [15, 10]}


def runningMeanFast(x, n):
    return convolve(x, ones((n,)) / n)[(n - 1):]


class FigureGenerator(object):
    def __init__(self, expe, scale=3.):
        self.scale = scale
        self.params = {'lines.markersize': 10,
                       'axes.labelsize': 10 * scale,
                       'text.fontsize': 10 * scale,
                       'legend.fontsize': 10 * scale,
                       'xtick.labelsize': 8 * scale,
                       'ytick.labelsize': 8 * scale,
                       'text.usetex': False,
                       'figure.figsize': [15, 10]}
        self.expe = expe

    def body(self):
        raise NotImplementedError

    def generate(self, filename='foo.pdf'):
        self._pre()
        self.body()
        self._post(filename)

    def _pre(self):
        rcParams.update(self.params)
        self.n_ag = self.expe.n_ag
        self.colors = ['r', 'g', 'b', 'm', 'c', 'k'] * 3
        self.n_runs = len(self.expe.log[0]["motor"])
        self.ag_legend = []
        self.end = self.n_runs  # 10040
        self.start = self.end - 100  # 10000
        self.time = range(self.start, self.end)
        for i in range(self.n_ag):
            self.ag_legend.append("Agent " + str(i + 1))  

    def _post(self, filename):
        #tight_layout()
        savefig(filename)


class Reactive(FigureGenerator):
    def __init__(self, expe):
        FigureGenerator.__init__(self, expe)
        sns.set_style("whitegrid")

    def body(self):
        figure(figsize=(10, 8))
        subplot(211)
        print(tile(self.time, (self.expe.n_ag, 1)).T.shape, self.expe.log_array('activation')[:, self.start:self.end].shape)
        plot(tile(self.time, (self.expe.n_ag, 1)).T, self.expe.log_array('activation')[:, self.start:self.end].T, 'o')
        ylabel("Motor activation")
        axis([self.start, self.end, -0.1, 1.1])
        legend(self.ag_legend, loc=1)

        subplot(212)
        plot(tile(self.time, (self.expe.n_ag, 1)).T, self.expe.log_array('motor')[:, self.start:self.end].T, 'o')
        xlabel("Time")
        ylabel("Auditory feature")
                

class Adaptive(FigureGenerator):
    def __init__(self, expe):
        FigureGenerator.__init__(self, expe, scale=2.)

    def body(self):
        figure(figsize=(8, 12))
        subplot(411)
        plot(tile(self.time, (self.expe.n_ag, 1)).T, self.expe.log_array('motor')[:,self.start:self.end].T, 'o')
        ylabel("Acoustic feature")
        legend(self.ag_legend, bbox_to_anchor=(.6, 1.02, 1., .102), loc=3)  #, bbox_to_anchor=(0, 0, 1, 1), bbox_transform=gcf().transFigure)
        # axis([self.start, self.end, -3, 2])

        subplot(412)
        plot(tile(self.time, (self.expe.n_ag, 1)).T, self.expe.log_array('presence', i_ag=0)[self.start:self.end, :], 'o')
        ylabel("Presence estimation")
        axis([self.start, self.end, 0, 1])

        subplot(413)
        plot(tile(self.time, (self.expe.n_ag, 1)).T, self.expe.log_array('activation')[:,self.start:self.end].T, 'o')

        # xlabel("Time")
        ylabel("Motor activation")
        axis([self.start, self.end, -0.1, 1.1])

        subplot(414)
        bounds = [0, self.n_runs]
        win = 1000
        plot(runningMeanFast(product(array(self.expe.log[0]['presence'][bounds[0]:bounds[1]]), axis=1), win) [:-win])
        axis(hstack((bounds, [0, 1])))  
        xlabel("Time")
        ylabel("Reward\n(overall presence)")
    


if __name__ == "__main__":
    pass