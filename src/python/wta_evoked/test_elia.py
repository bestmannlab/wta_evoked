import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from brian import second, nS, Clock, Hz, PoissonGroup, Network, network_operation, NeuronGroup, Equations, linked_var, StateMonitor, defaultclock, ms, mA
from wta_evoked.network import simulation_params, default_params, pyr_params, inh_params, WTANetworkGroup
from wta_evoked.monitor import WTAMonitor, SessionMonitor

class LFPSource(NeuronGroup):

    def __init__(self, neuron_group, clock=defaultclock):
        """
        Sum the I_abs value of each neuron in the neuron group
        """
        eqs=Equations('''
         LFP : amp
        ''')
        NeuronGroup.__init__(self, 1, model=eqs, compile=True, freeze=True, clock=clock)
        self.LFP=linked_var(neuron_group, 'I_abs', func=sum, clock=clock)

def run():

    # Create simulation parameters - specify number of trials, trial duration, start and end times of task-related
    # stimulus
    sim_params=simulation_params(ntrials=3, trial_duration=3 * second, stim_start_time=1 * second,
        stim_end_time=2 * second)
    # Create WTA network parameters - specifiy background firing rate and response threshold (firing rate required to
    # reach for a response)
    wta_params=default_params()
    wta_params.background_freq=875
    wta_params.resp_threshold=20
    # Initialize pyramidal and interneuron parameters
    pyramidal_params=pyr_params(w_nmda=0.145*nS, w_ampa_ext_correct=1.6*nS, w_ampa_ext_incorrect=0.9*nS)
    inhibitory_params=inh_params()

    # Create simulation clock and input update clock (inputs mean rate is updated according to the refresh rate of the
    # monitor (60Hz)
    simulation_clock = Clock(dt=sim_params.dt)
    input_update_clock = Clock(dt=1 / (wta_params.refresh_rate / Hz) * second)

    # Create Poisson background (noise) inputs
    background_input = PoissonGroup(wta_params.background_input_size, rates=wta_params.background_freq,
        clock=simulation_clock)
    # Create Poisson task-related inputs
    task_inputs = []
    for i in range(wta_params.num_groups):
        task_inputs.append(PoissonGroup(wta_params.task_input_size, rates=wta_params.task_input_resting_rate,
            clock=simulation_clock))

    # Create WTA network
    wta_network = WTANetworkGroup(params=wta_params, background_input=background_input, task_inputs=task_inputs,
        pyr_params=pyramidal_params, inh_params=inhibitory_params, clock=simulation_clock)

    # Set up an LFP source that will sum the I_abs variable from each neuron in the group
    lfp_source=LFPSource(wta_network.group_e, clock=simulation_clock)

    # Create network monitor to record and plot network activity - only recording firing rate here
    wta_monitor = WTAMonitor(wta_network, sim_params, lfp_source=lfp_source, record_neuron_state=False, record_spikes=False,
        record_firing_rate=True, record_inputs=True, save_summary_only=False, clock=simulation_clock)

    # Create session monitor
    session_monitor=SessionMonitor(wta_network, sim_params, conv_window=40, record_firing_rates=True, record_lfp=True)

    # Run on three coherence levels
    coherence_levels=[0.032, .128, .512]
    # Trials per coherence level
    trials_per_level=1
    # Create inputs for each trial
    trial_task_input_rates=np.zeros((trials_per_level*len(coherence_levels),2))
    # Create only left direction for each coherence level
    for i in range(len(coherence_levels)):
        coherence = coherence_levels[i]
        # Left

        trial_task_input_rates[i, 0] = wta_params.mu_0 + wta_params.p_a * coherence * 100.0
        trial_task_input_rates[i, 1] = wta_params.mu_0 - wta_params.p_b * coherence * 100.0

        # Set firing rate of task-related inputs given a level of coherence (from Wang, 2002)
        #coherence_levels=[0.032, .064, .128, .256, .512]
        #for i in range(len(coherence_levels)):
        #  coherence = coherence_levels[i]
    """
    coherence_level=0.512
    task_input_rates=[wta_params.mu_0+wta_params.p_a*coherence_level*100.0,
                      wta_params.mu_0-wta_params.p_b*coherence_level*100.0]
    """
    task_input_rates=[]
    # Function to update the task-related inputs, runs in the simulation update step according to the input_update_clock
    @network_operation(when='start', clock=input_update_clock)
    def set_task_inputs():
        # Iterate over task-related inputs (two of them)
        for idx in range(len(task_inputs)):
            # Initialize to the resting task rate
            rate = wta_params.task_input_resting_rate
            # If the current simulation time is in between the stimulus start and end times
            if sim_params.stim_start_time <= simulation_clock.t < sim_params.stim_end_time:
                # Set the current rate to be either the resting rate or a random rate around the mean (set above),
                # whichvever is greater
                rate = np.max([wta_params.task_input_resting_rate,
                               task_input_rates[idx] * Hz + np.random.randn() * wta_params.input_var])
                # Set the rate of the Poisson group
            task_inputs[idx]._S[0, :] = rate

    # Create Brian network
    net = Network(background_input, task_inputs, wta_network, wta_network.connections.values(),
        wta_monitor.monitors.values(), lfp_source, set_task_inputs)
    net.clock=simulation_clock

    # Simulate each trial
    for t in range(sim_params.ntrials):
        # Run simulation
        task_input_rates=trial_task_input_rates[t,:]
        net.reinit(states=True)
        net.run(sim_params.trial_duration, report='text')
        print('Trial %d' % t)

        # Record trial
        session_monitor.record_trial(t, task_input_rates, np.where(task_input_rates==np.max(task_input_rates))[0], wta_monitor)

        # Plot firing rate
        #wta_monitor.plot()
        #plt.show()

    session_monitor.plot()
    plt.show()



if __name__=='__main__':
    run()
