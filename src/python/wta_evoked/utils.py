import numpy as np
from scipy import optimize
from brian import second, ms, hertz
from matplotlib.patches import Rectangle
from matplotlib.pyplot import figure, subplot, ylim, legend, ylabel, title, xlabel

class _baseFunctionFit:
    """
    From psychopy
    Not needed by most users except as a superclass for developping your own functions

    You must overide the eval and inverse methods and a good idea to overide the _initialGuess
    method aswell.
    """

    def __init__(self, xx, yy, sems=1.0, guess=None, display=1,
                 expectedMin=0.5):
        self.xx = np.asarray(xx)
        self.yy = np.asarray(yy)
        self.sems = np.asarray(sems)
        self.expectedMin = expectedMin
        self.display=display
        # for holding error calculations:
        self.ssq=0
        self.rms=0
        self.chi=0
        #initialise parameters
        if guess is None:
            self.params = self._initialGuess()
        else:
            self.params = guess

        #do the calculations:
        self._doFit()

    def _doFit(self):
        #get some useful variables to help choose starting fit vals
        self.params = optimize.fmin_powell(self._getErr, self.params, (self.xx,self.yy,self.sems),disp=self.display)
        #        self.params = optimize.fmin_bfgs(self._getErr, self.params, None, (self.xx,self.yy,self.sems),disp=self.display)
        self.ssq = self._getErr(self.params, self.xx, self.yy, 1.0)
        self.chi = self._getErr(self.params, self.xx, self.yy, self.sems)
        self.rms = self.ssq/len(self.xx)

    def _initialGuess(self):
        xMin = min(self.xx); xMax = max(self.xx)
        xRange=xMax-xMin; xMean= (xMax+xMin)/2.0
        guess=[xMean, xRange/5.0]
        return guess

    def _getErr(self, params, xx,yy,sems):
        mod = self.eval(xx, params)
        err = sum((yy-mod)**2/sems)
        return err

    def eval(self, xx=None, params=None):
        """Returns fitted yy for any given xx value(s).
        Uses the original xx values (from which fit was calculated)
        if none given.

        If params is specified this will override the current model params."""
        yy=xx
        return yy

    def inverse(self, yy, params=None):
        """Returns fitted xx for any given yy value(s).

        If params is specified this will override the current model params.
        """
        #define the inverse for your function here
        xx=yy
        return xx


class FitWeibull(_baseFunctionFit):
    """Fit a Weibull function (either 2AFC or YN)
    of the form::

        y = chance + (1.0-chance)*(1-exp( -(xx/alpha)**(beta) ))

    and with inverse::

        x = alpha * (-log((1.0-y)/(1-chance)))**(1.0/beta)

    After fitting the function you can evaluate an array of x-values
    with ``fit.eval(x)``, retrieve the inverse of the function with
    ``fit.inverse(y)`` or retrieve the parameters from ``fit.params``
    (a list with ``[alpha, beta]``)"""
    def eval(self, xx=None, params=None):
        if params is None:  params=self.params #so the user can set params for this particular eval
        alpha = params[0];
        if alpha<=0: alpha=0.001
        beta = params[1]
        xx = np.asarray(xx)
        yy =  self.expectedMin + (1.0-self.expectedMin)*(1-np.exp( -(xx/alpha)**(beta) ))
        return yy
    def inverse(self, yy, params=None):
        if params is None: params=self.params #so the user can set params for this particular inv
        alpha = params[0]
        beta = params[1]
        xx = alpha * (-np.log((1.0-yy)/(1-self.expectedMin))) **(1.0/beta)
        return xx

class FitRT(_baseFunctionFit):
    """Fit a Weibull function (either 2AFC or YN)
    of the form::

        y = chance + (1.0-chance)*(1-exp( -(xx/alpha)**(beta) ))

    and with inverse::

        x = alpha * (-log((1.0-y)/(1-chance)))**(1.0/beta)

    After fitting the function you can evaluate an array of x-values
    with ``fit.eval(x)``, retrieve the inverse of the function with
    ``fit.inverse(y)`` or retrieve the parameters from ``fit.params``
    (a list with ``[alpha, beta]``)"""
    def eval(self, xx=None, params=None):
        if params is None:  params=self.params #so the user can set params for this particular eval
        a = params[0]
        k = params[1]
        tr = params[2]
        xx = np.asarray(xx)
        yy = a*np.tanh(k*xx)+tr
        return yy
    def inverse(self, yy, params=None):
        if params is None: params=self.params #so the user can set params for this particular inv
        a = params[0]
        k = params[1]
        tr = params[2]
        xx = np.arctanh((yy-tr)/a)/k
        return xx

class FitSigmoid(_baseFunctionFit):
    def eval(self, xx=None, params=None):
        if params is None:  params=self.params #so the user can set params for this particular eval
        x0 = params[0]
        k=params[1]
        xx = np.asarray(xx)
        #yy = a+1.0/(1.0+np.exp(-k*xx))
        yy =1.0/(1.0+np.exp(-k*(xx-x0)))
        return yy

    def inverse(self, yy, params=None):
        if params is None:  params=self.params #so the user can set params for this particular eval
        x0 = params[0]
        k=params[1]
        #xx = -np.log((1/(yy-a))-1)/k
        xx = -np.log((1.0/yy)-1.0)/k+x0
        return xx

def get_response_time(e_firing_rates, stim_start_time, stim_end_time, upper_threshold=60, threshold_diff=None, dt=.1*ms):
    rate_1=e_firing_rates[0]
    rate_2=e_firing_rates[1]
    times=np.array(range(len(rate_1)))*(dt/second)
    rt=None
    decision_idx=-1
    for idx,time in enumerate(times):
        time=time*second
        if stim_start_time < time < stim_end_time:
            if rt is None:
                if rate_1[idx]>=upper_threshold and (threshold_diff is None or rate_1[idx]-rate_2[idx]>=threshold_diff):
                    decision_idx=0
                    rt=(time-stim_start_time)/ms
                    break
                elif rate_2[idx]>=upper_threshold and (threshold_diff is None or rate_2[idx]-rate_1[idx]>=threshold_diff):
                    decision_idx=1
                    rt=(time-stim_start_time)/ms
                    break
    return rt,decision_idx

def plot_network_firing_rates(e_rates, sim_params, network_params, std_e_rates=None, i_rate=None, std_i_rate=None,
                              plt_title=None, labels=None, ax=None):
    rt, choice = get_response_time(e_rates, sim_params.stim_start_time, sim_params.stim_end_time,
        upper_threshold = network_params.resp_threshold, dt = sim_params.dt)

    if ax is None:
        figure()
    max_rates=[network_params.resp_threshold]
    if i_rate is not None:
        max_rates.append(np.max(i_rate[500:]))
    for i in range(network_params.num_groups):
        max_rates.append(np.max(e_rates[i,500:]))
    max_rate=np.max(max_rates)

    if i_rate is not None:
        ax=subplot(211)
    elif ax is None:
        ax=subplot(111)
    rect=Rectangle((0,0),(sim_params.stim_end_time-sim_params.stim_start_time)/ms, max_rate+5,
        alpha=0.25, facecolor='yellow', edgecolor='none')
    ax.add_patch(rect)

    for idx in range(network_params.num_groups):
        label='e %d' % idx
        if labels is not None:
            label=labels[idx]
        time_ticks=(np.array(range(e_rates.shape[1]))*sim_params.dt)/ms-sim_params.stim_start_time/ms
        baseline,=ax.plot(time_ticks, e_rates[idx,:], label=label)
        if std_e_rates is not None:
            ax.fill_between(time_ticks, e_rates[idx,:]-std_e_rates[idx,:], e_rates[idx,:]+std_e_rates[idx,:], alpha=0.5,
                facecolor=baseline.get_color())
    ylim(0,max_rate+5)
    ax.plot([0-sim_params.stim_start_time/ms, (sim_params.trial_duration-sim_params.stim_start_time)/ms],
        [network_params.resp_threshold/hertz, network_params.resp_threshold/hertz], 'k--')
    ax.plot([rt,rt],[0, max_rate+5],'k--')
    legend(loc='best')
    ylabel('Firing rate (Hz)')
    if plt_title is not None:
        title(plt_title)

    if i_rate is not None:
        ax=subplot(212)
        rect=Rectangle((0,0),(sim_params.stim_end_time-sim_params.stim_start_time)/ms, max_rate+5,
            alpha=0.25, facecolor='yellow', edgecolor='none')
        ax.add_patch(rect)
        label='i'
        if labels is not None:
            label=labels[network_params.num_groups]
        time_ticks=(np.array(range(len(i_rate)))*sim_params.dt)/ms-sim_params.stim_start_time/ms
        baseline,=ax.plot(time_ticks, i_rate, label=label)
        if std_i_rate is not None:
            ax.fill_between(time_ticks, i_rate-std_i_rate, i_rate+std_i_rate, alpha=0.5, facecolor=baseline.get_color())
        ylim(0,max_rate+5)
        ax.plot([rt,rt],[0, max_rate],'k--')
        ylabel('Firing rate (Hz)')
    xlabel('Time (ms)')


