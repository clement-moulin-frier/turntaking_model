from matplotlib import rcParams
from numpy import product, array, convolve, ones, hstack, tile, linspace, zeros
from matplotlib.pyplot import subplot, plot, xlabel, ylabel, axis, legend, savefig, figure, tight_layout, imshow, colorbar, xticks, yticks
from matplotlib.gridspec import GridSpec

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
        #rcParams.update(self.params)
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
        # sns.set_style("whitegrid")

    def body(self):
        self.start = 0
        self.end = 100
        figure(figsize=(10, 8))
        subplot(211)
        #print(tile(self.time, (self.expe.n_ag, 1)).T.shape, self.expe.log_array('activation')[:, self.start:self.end].shape)
        plot(tile(self.time, (self.expe.n_ag, 1)).T, self.expe.log_array('activation')[:, self.start:self.end].T, 'o')
        ylabel("Vocalization probability")
        axis([self.start, self.end, -0.1, 1.1])
        legend(self.ag_legend, bbox_to_anchor=(0.65, 1.1), loc='upper left')

        subplot(212)
        plot(tile(self.time, (self.expe.n_ag, 1)).T, self.expe.log_array('motor')[:, self.start:self.end].T, 'o')
        xlabel("Time")
        ylabel("Auditory feature")


class Adaptive(FigureGenerator):
    def __init__(self, expe):
        FigureGenerator.__init__(self, expe, scale=2.)
        self.time_ranges = ((0, 100), (1000, 1100), (10000, 10100))

    # def plot_on_ax(self, ax, data_x, data*plo)

    def body(self):
        n_col = 3
        n_row = 4
        n_col_sub = n_col / 3
        gs = GridSpec(n_row, n_col) #, left=1, right=1)
        figure(figsize=(24, 10))

        for i, (t_min, t_max) in enumerate(self.time_ranges):

            subplot(gs[0, (i * n_col_sub):(i + 1) * n_col_sub])
            plot(tile(range(t_min, t_max), (self.expe.n_ag, 1)).T, self.expe.log_array('presence', i_ag=0)[t_min:t_max, :], 'o')
            if i == 0:
                ylabel("Presence\nestimation")
            axis([t_min, t_max, 0, 1])
            if i == 1:
                legend(self.ag_legend, bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)  # bbox_to_anchor=(.6, 1.02, 1., .102), loc=3)

            subplot(gs[1, (i * n_col_sub):(i + 1) * n_col_sub])
            plot(tile(range(t_min, t_max), (self.expe.n_ag, 1)).T, self.expe.log_array('activation')[:, t_min:t_max].T, 'o')
            if i == 0:
                ylabel("Vocalization\nprobability")
            axis([t_min, t_max, -0.1, 1.1])

            subplot(gs[2, (i * n_col_sub):(i + 1) * n_col_sub])
            plot(tile(range(t_min, t_max), (self.expe.n_ag, 1)).T, self.expe.log_array('motor')[:,t_min:t_max].T, 'o')
            xlabel("Time")
            if i == 0:
                ylabel("Acoustic\nfeature")
            axis([t_min, t_max, -1.2, 0.2])

        subplot(gs[3, :])
        win = 1000
        bounds = [0, self.time_ranges[-1][-1] + win]
        plot(runningMeanFast(product(array(self.expe.log[0]['presence'][bounds[0]:bounds[1]]), axis=1), win) [:-win])
        axis([bounds[0], bounds[1] - win, 0, 1])
        xlabel("Time")
        ylabel("Reward\n(overall presence)")

        tight_layout(pad=3, h_pad=1, w_pad=5)

        return


class ActionPolicy(FigureGenerator):
    def __init__(self, expe):
        FigureGenerator.__init__(self, expe, scale=2.)
        self.time_steps = [0, 1000, 9000]

    def body(self):
        # sns.set(font_scale=1.5)
        #sns.set_context("paper", font_scale=1.3)
        figure(figsize=(5.5, 10./3.))
        n_col_sub = 3 #int(float(n_col - 1) / len(self.time_steps))
        n_col = n_col_sub * len(self.time_steps) + 1
        n_row = 2
        gs = GridSpec(n_row, n_col)
        for i_t, t in enumerate(self.time_steps):
            v0 = zeros((100, 100))
            v1 = zeros((100, 100))
            for i0, p0 in enumerate(linspace(0, 1, 100)):
                for i1, p1 in enumerate(linspace(0, 1, 100)):
                    v0[i0, i1] = self.expe.log[0]['weights'][t].dot(hstack((1., p0, p1)).T)
                    v1[i0, i1] = self.expe.log[1]['weights'][t].dot(hstack((1., p0, p1)).T)

            # pres = array([self.expe.log[0]['presence'][t] for t in range(self.n_runs)])

            subplot(gs[0, (i_t * n_col_sub):((i_t + 1) * n_col_sub)])
            #imshow(v0.T[::-1, :], extent=[0, 1, 0, 1], vmin=-5, vmax=5)
            imshow(self.expe.ags[0].motor.activation_fun(v0.T[::-1, :]), extent=[0, 1, 0, 1], vmin=0, vmax=1)

            #xlabel('P(a1)')
            if i_t == 0:
                ylabel('P(a2)')
            xticks([0., 0.5, 1.])
            yticks([0., 0.5, 1.])
            #if i_t == len(self.time_steps) - 1:
            #    colorbar()
            subplot(gs[1, (i_t * n_col_sub):((i_t + 1) * n_col_sub)])
            #imshow(v1.T[::-1, :], extent=[0, 1, 0, 1], vmin=-5, vmax=5)
            imshow(self.expe.ags[1].motor.activation_fun(v1.T[::-1, :]), extent=[0, 1, 0, 1], vmin=0, vmax=1)
            #plot(pres[-40:, 0], pres[-40:, 1])
            xlabel('P(a1)')
            if i_t == 0:
                ylabel('P(a2)')
            xticks([0., 0.5, 1.])
            yticks([0., 0.5, 1.])
            #if i_t == len(self.time_steps) - 1:
        colorbar(cax=subplot(gs[:, -1]))

        tight_layout(h_pad=0.8, w_pad=0.8)


class RewardStat(object):
    def __init__(self, expe_logs_2ag, expe_logs_3ag):
        self.expe_logs_2ag = expe_logs_2ag
        self.expe_logs_3ag = expe_logs_3ag
        self.n_runs_2ag = len(expe_logs_2ag[0][0]["motor"])
        self.n_runs_3ag = len(expe_logs_3ag[0][0]["motor"])
        self.n_expe = len(expe_logs_2ag)
        win = 1000
        self.run_means_2ag = [runningMeanFast(product(array(expe_logs_2ag[i_e][0]['presence']), axis=1), win)[:-win] for i_e in range(self.n_expe)]
        self.run_means_2ag = array(self.run_means_2ag)
        self.run_means_3ag = [runningMeanFast(product(array(expe_logs_3ag[i_e][0]['presence']), axis=1), win)[:-win] for i_e in range(self.n_expe)]
        self.run_means_3ag = array(self.run_means_3ag)

    def generate(self, filename):
        #figure(figsize=(10, 5./3.))
        sns.set_context("paper", font_scale=2.)
        subplot(211)
        sns.tsplot(self.run_means_2ag, err_style='unit_traces')
        axis([0, self.n_runs_2ag, 0., 1])
        xlabel('Time')
        ylabel("Reward\n(overall presence)")

        subplot(212)
        sns.tsplot(self.run_means_3ag, err_style='unit_traces')
        axis([0, self.n_runs_3ag, 0., 1])
        xlabel('Time')
        ylabel("Reward\n(overall presence)")

        tight_layout()

        savefig(filename)

if __name__ == "__main__":
    pass