import math
import multiprocessing
import os
import random

import numpy as np
import torch

from models.ensemble_net import EnsembleNet
from models.mlp import MLP_1HL
from utils import get_optim


class ptReplica(multiprocessing.Process):
    def __init__(
        self,
        exp,
        config,
        num_nets,
        minlim_param,
        maxlim_param,
        samples_chains,
        path,
        temperature,
        parameter_queue,
        main_process,
        event,
    ):

        self.exp = exp
        self.config = config
        self.num_nets = num_nets

        # MULTIPROCESSING VARIABLES
        multiprocessing.Process.__init__(self)
        self.processID = temperature
        self.parameter_queue = parameter_queue
        self.signal_main = main_process
        self.event = event

        self.temperature = temperature
        self.adapttemp = temperature
        self.swap_interval = self.config.params.swap_interval
        self.path = path
        self.burn_in = int(self.config.params.burn_in / self.config.params.samples)
        # FNN CHAIN VARIABLES (MCMC)
        self.samples = samples_chains
        self.topology = (self.config.feat_d, self.config.hidden_d, self.config.out_d)
        self.train = self.exp.train
        self.test = self.exp.test

        self.minY = np.zeros((1, 1))
        self.maxY = np.zeros((1, 1))

        self.minlim_param = minlim_param
        self.maxlim_param = maxlim_param

        self.use_langevin_gradients = self.config.params.langevin_gradients

        self.sgd_depth = 1  # always should be 1

        self.learn_rate = self.config.params.lr

        self.l_prob = self.config.params.lg_rate
        self.w_size = 0

    def run(self):

        model_type = MLP_1HL

        samples = self.samples

        self.net_ensemble = EnsembleNet(self.exp.c0, self.config.params.lr)
        # self.train.shuffle()

        x, y = self.train.feat, self.train.label
        x_test, y_test = self.test.feat, self.test.label
        if self.config.cuda:
            x = torch.as_tensor(x, dtype=torch.float32).cuda()
            y = torch.as_tensor(y, dtype=torch.float32).cuda()

            x_test = torch.as_tensor(x_test, dtype=torch.float32).cuda()
            y_test = torch.as_tensor(y_test, dtype=torch.float32).cuda()

        out = self.net_ensemble.forward(x)
        grad_direction = -1 * self.exp.g_func(y, out)

        w_size = (
            self.config.feat_d * self.config.hidden_d
            + self.config.hidden_d * self.config.out_d
            + self.config.hidden_d
            + self.config.out_d
        )

        # Randomwalk Steps
        if self.config.classification:
            step_w = self.config.params.step_w["classification"]
        else:
            step_w = self.config.params.step_w["regression"]

        sigma_squared = step_w * 10
        nu_1 = 0
        nu_2 = 0
        # step_eta = self.config.step_eta

        # Initialize weights for all models
        model_weights = np.zeros((self.num_nets, w_size))
        models = []
        eta_list = np.zeros((self.num_nets))
        prior_list = np.zeros((self.num_nets))
        self.likelihoods = np.zeros((self.num_nets, samples + 1))
        fx_train_list = torch.zeros((self.num_nets, len(y), self.config.out_d)).cuda()
        fx_test_list = torch.zeros((self.num_nets, len(y_test), self.config.out_d)).cuda()
        self.Fx_test_l = torch.zeros(
            (self.num_nets, samples, len(y_test), self.config.out_d)
        ).cuda()

        self.tr_accs = []
        self.te_accs = []
        for i in range(self.num_nets):
            model_weights[i] = np.random.normal(0, sigma_squared, w_size)
            model = model_type.get_model(self.config)
            if self.config.cuda:
                model.cuda()
            model.decode(model_weights[i])
            models.append(model)

            pred_train = model.evaluate_proposal(x, model_weights[i])
            eta_list[i] = np.log(np.var((pred_train - grad_direction).cpu().numpy()))
            tau_pro = np.exp(eta_list[i])

            prior_list[i] = self.exp.prior_func(
                sigma_squared, nu_1, nu_2, model_weights[i], tau_pro, self.config
            )  # takes care of the gradients

            self.likelihoods[i][0], fx_train_list[i] = self.exp.log_likelihood_func(
                model, x, grad_direction, model_weights[i], tau_sq=tau_pro
            )
            _, fx_test_list[i] = self.exp.log_likelihood_func(
                model, x_test, y_test, model_weights[i], tau_sq=tau_pro
            )

        pt_samples = samples * 0.6
        init_count = 0

        accepted = 0
        self.event.clear()
        for i in range(samples):

            self.net_ensemble.reset()
            with torch.no_grad():
                Fx_train = torch.as_tensor(
                    self.net_ensemble.forward(x), dtype=torch.float32
                ).cuda()
                Fx_test = torch.as_tensor(
                    self.net_ensemble.forward(x_test), dtype=torch.float32
                ).cuda()

            for j in range(self.num_nets):

                out = self.net_ensemble.forward(x)
                grad_direction = -1 * self.exp.g_func(y, out)

                model = models[j]
                w = model_weights[j]
                optimizer = get_optim(model.parameters(), self.config)

                # Parallel tempering
                if i < pt_samples:
                    self.adapttemp = self.temperature

                if i == pt_samples and init_count == 0:
                    self.adapttemp = 1
                    self.likelihoods[j][i], fx_train_list[j] = self.exp.log_likelihood_func(
                        model, x, grad_direction, model_weights[j], tau_sq=tau_pro
                    )
                    _, fx_test_list[j] = self.exp.log_likelihood_func(
                        model, x_test, y_test, model_weights[j], tau_sq=tau_pro
                    )
                    init_count += 1

                lx = random.uniform(0, 1)
                if self.config.params.langevin_gradients and lx < self.config.params.lg_rate:
                    w_gd = self.exp.langevin_gradient(model, x, grad_direction, w, optimizer)
                    w_proposal = np.random.normal(w_gd, step_w, w_size)
                    w_prop_gd = self.exp.langevin_gradient(
                        model, x, grad_direction, w_proposal, optimizer
                    )

                    wc_delta = w - w_prop_gd
                    wp_delta = w_proposal - w_gd

                    sigma_sq = step_w

                    first = -0.5 * np.sum(wc_delta ** 2) / sigma_sq
                    second = -0.5 * np.sum(wp_delta ** 2) / sigma_sq

                    diff_prop = (first - second) / self.adapttemp
                else:
                    diff_prop = 0
                    w_proposal = np.random.normal(w, step_w, w_size)

                # Fix eta
                eta_pro = eta_list[j]
                # eta_pro = eta + np.random.normal(0, step_eta, 1)
                tau_pro = np.exp(eta_pro)

                with torch.no_grad():
                    log_lik_proposal, fx_train_prop = self.exp.log_likelihood_func(
                        model, x, grad_direction, w_proposal, tau_pro
                    )

                    _, fx_test_prop = self.exp.log_likelihood_func(
                        model, x_test, y_test, w_proposal, tau_pro
                    )

                prior_prop = self.exp.prior_func(
                    sigma_squared, nu_1, nu_2, w_proposal, tau_pro, self.config
                )  # takes care of the gradients

                diff_prior = prior_prop - prior_list[j]
                diff_likelihood = log_lik_proposal - self.likelihoods[j][i]

                temp = diff_likelihood + diff_prior + diff_prop
                if temp > 0:
                    mh_prob = 1
                else:
                    mh_prob = math.exp(temp)

                u = random.uniform(0, 1)
                if u < mh_prob:  # Accept
                    accepted += 1
                    self.likelihoods[j][i + 1] = log_lik_proposal
                    prior_list[j] = prior_prop
                    model_weights[j] = w_proposal
                    eta_list[j] = eta_pro
                    fx_train_list[j] = fx_train_prop
                    fx_test_list[j] = fx_test_prop
                else:
                    self.likelihoods[j][i + 1] = self.likelihoods[j][i]

                acc_tr, _ = self.exp.compute_accuracy(fx_train_list[j], Fx_train, self.train)
                acc_te, pred_te = self.exp.compute_accuracy(fx_test_list[j], Fx_test, self.test)

                if j == self.num_nets - 1:
                    self.tr_accs.append(acc_tr)
                    self.te_accs.append(acc_te)

                # Add model to the ensemble
                gamma = self.exp.lambda_func(y, fx_train_list[j], Fx_train)

                self.net_ensemble.add(model, gamma)

                Fx_train += gamma * fx_train_list[j]
                Fx_test += gamma * fx_test_list[j]

                self.Fx_test_l[j][i] = Fx_test

            if (i + 1) % self.swap_interval == 0:
                param = [model_weights, eta_list, self.likelihoods[:, i], self.temperature, i]
                self.parameter_queue.put(param)
                self.signal_main.set()
                self.event.clear()
                self.event.wait()

                # Retrieve parameters
                # (
                #     model_weights,
                #     eta_list,
                #     self.likelihoods,
                #     self.temperature,
                #     i,
                # ) = self.parameter_queue.get()
                (model_weights, eta_list, temp, _, _) = self.parameter_queue.get()
                self.likelihoods[:, i] = temp

        self.parameter_queue.put(param)

        self.signal_main.set()
        # print((num_accepted * 100 / (samples * 1.0)), "% was accepted")
        accept_ratio = accepted / (samples * 1.0) * 100

        # print((langevin_count * 100 / (samples * 1.0)), "% was Lsnngrevin ")
        # langevin_ratio = langevin_count / (samples * 1.0) * 100

        # file_name = self.path + "/posterior/pos_w/" + "chain_" + str(self.temperature) + ".txt"
        # np.savetxt(file_name, pos_w)

        # file_name = self.path+'/predictions/fxtrain_samples_chain_'+ str(self.temperature)+ '.txt'
        # np.savetxt(file_name, fxtrain_samples, fmt='%1.2f')
        # file_name = self.path+'/predictions/fxtest_samples_chain_'+ str(self.temperature)+ '.txt'
        # np.savetxt(file_name, fxtest_samples, fmt='%1.2f')
        # file_name = self.path + "/predictions/rmse_test_chain_" + str(self.temperature) + ".txt"
        # np.savetxt(file_name, rmse_test, fmt="%1.2f")
        # file_name = self.path + "/predictions/rmse_train_chain_" + str(self.temperature) + ".txt"
        # np.savetxt(file_name, rmse_train, fmt="%1.2f")

        file_name = self.path + "/predictions/acc_test_chain_" + str(self.temperature) + ".txt"
        np.savetxt(file_name, np.array(self.te_accs))

        file_name = self.path + "/predictions/acc_train_chain_" + str(self.temperature) + ".txt"
        np.savetxt(file_name, np.array(self.tr_accs))

        file_name = self.path + "/posterior/pos_likelihood/chain_" + str(self.temperature) + ".txt"
        np.savetxt(file_name, self.likelihoods)

        file_name = (
            self.path + "/posterior/accept_list/chain_" + str(self.temperature) + "_accept.txt"
        )
        np.savetxt(file_name, [accept_ratio])


class ParallelTempering:
    def __init__(self, exp, config, num_nets):
        self.exp = exp
        self.config = config
        self.num_nets = num_nets

        self.num_param = self.config.feat_d * self.config.hidden_d
        +self.config.hidden_d * self.config.out_d
        +self.config.hidden_d
        +self.config.out_d

        # PT variables
        self.path = "pt"
        self.num_swap = 0
        self.total_swap_proposals = 0
        self.chains = []
        self.temperatures = []
        self.samples_chains = int(self.config.params.samples / self.config.params.num_chains)
        self.sub_samples_size = max(1, int(0.05 * self.samples_chains))

        self.manager = multiprocessing.Manager()

        # Create queues for transfer of parameters between process chain
        self.parameter_queue = [self.manager.Queue() for i in range(self.config.params.num_chains)]

        self.chain_queue = self.manager.Queue()
        self.wait_chain = [multiprocessing.Event() for i in range(self.config.params.num_chains)]
        self.event = [multiprocessing.Event() for i in range(self.config.params.num_chains)]

        self.all_param = None

        self.minlim_param = 0.0
        self.maxlim_param = 0.0
        self.minY = np.zeros((1, 1))
        self.maxY = np.ones((1, 1))

        self.model_signature = 0.0

    def default_beta_ladder(
        self, ndim, ntemps, Tmax
    ):  # https://github.com/konqr/ptemcee/blob/master/ptemcee/sampler.py
        """
        Returns a ladder of :math:`\beta \equiv 1/T` under a geometric spacing that is determined by the
        arguments ``ntemps`` and ``Tmax``.  The temperature selection algorithm works as follows:
        Ideally, ``Tmax`` should be specified such that the tempered posterior looks like the prior at
        this temperature.  If using adaptive parallel tempering, per `arXiv:1501.05823
        <http://arxiv.org/abs/1501.05823>`_, choosing ``Tmax = inf`` is a safe bet, so long as
        ``ntemps`` is also specified.
         
        """

        if type(ndim) != int or ndim < 1:
            raise ValueError("Invalid number of dimensions specified.")
        if ntemps is None and Tmax is None:
            raise ValueError("Must specify one of ``ntemps`` and ``Tmax``.")
        if Tmax is not None and Tmax <= 1:
            raise ValueError("``Tmax`` must be greater than 1.")
        if ntemps is not None and (type(ntemps) != int or ntemps < 1):
            raise ValueError("Invalid number of temperatures specified.")

        maxtemp = Tmax
        numchain = ntemps
        b = []
        b.append(maxtemp)
        last = maxtemp
        for i in range(maxtemp):
            last = last * (numchain ** (-1 / (numchain - 1)))
            b.append(last)
        tstep = np.array(b)

        if ndim > tstep.shape[0]:
            # An approximation to the temperature step at large
            # dimension
            tstep = 1.0 + 2.0 * np.sqrt(np.log(4.0)) / np.sqrt(ndim)
        else:
            tstep = tstep[ndim - 1]

        appendInf = False
        if Tmax == np.inf:
            appendInf = True
            Tmax = None
            ntemps = ntemps - 1

        if ntemps is not None:
            if Tmax is None:
                # Determine Tmax from ntemps.
                Tmax = tstep ** (ntemps - 1)
        else:
            if Tmax is None:
                raise ValueError(
                    "Must specify at least one of ``ntemps" " and " "finite ``Tmax``."
                )

            # Determine ntemps from Tmax.
            ntemps = int(np.log(Tmax) / np.log(tstep) + 2)

        betas = np.logspace(0, -np.log10(Tmax), ntemps)
        if appendInf:
            # Use a geometric spacing, but replace the top-most temperature with
            # infinity.
            betas = np.concatenate((betas, [0]))

        return betas

    def assign_temperatures(self):

        if self.config.params.geometric == True:
            betas = self.default_beta_ladder(
                2, ntemps=self.config.params.num_chains, Tmax=self.config.params.maxtemp
            )
            for i in range(0, self.config.params.num_chains):
                self.temperatures.append(np.inf if betas[i] is 0 else 1.0 / betas[i])
                # print(self.temperatures[i])
        else:

            tmpr_rate = self.config.params.maxtemp / self.config.params.num_chains
            temp = 1
            for i in range(0, self.config.params.num_chains):
                self.temperatures.append(temp)
                temp += tmpr_rate
                # print(self.temperatures[i])

    def init_chains(self):

        self.assign_temperatures()
        self.minlim_param = np.repeat([-100], self.num_param)  # priors for nn weights
        self.maxlim_param = np.repeat([100], self.num_param)

        for i in range(0, self.config.params.num_chains):

            self.chains.append(
                ptReplica(
                    self.exp,
                    self.config,
                    self.num_nets,
                    self.minlim_param,
                    self.maxlim_param,
                    self.samples_chains,
                    self.path,
                    self.temperatures[i],
                    self.parameter_queue[i],
                    self.wait_chain[i],
                    self.event[i],
                )
            )

    def surr_procedure(self, queue):

        if queue.empty() is False:
            return queue.get()
        else:
            return

    def swap_procedure(self, parameter_queue_1, parameter_queue_2):
        # if parameter_queue_2.empty() is False and parameter_queue_1.empty() is False:
        param1 = parameter_queue_1.get()
        param2 = parameter_queue_2.get()

        (_, _, lhood1, _, _) = param1
        (_, _, lhood2, _, _) = param1

        try:
            swap_proposal = min(1, 0.5 * np.exp(min(709, np.sum(lhood2) - np.sum(lhood1))))
        except OverflowError:
            swap_proposal = 1
        u = np.random.uniform(0, 1)
        swapped = False
        if u < swap_proposal:
            self.total_swap_proposals += 1
            self.num_swap += 1
            param_temp = param1
            param1 = param2
            param2 = param_temp
            swapped = True
        else:
            swapped = False
            self.total_swap_proposals += 1
        return param1, param2, swapped

    def run_chains(self):
        # only adjacent chains can be swapped therefore, the number of proposals is ONE less num_chains
        swap_proposal = np.ones(self.config.params.num_chains - 1)
        # create parameter holders for paramaters that will be swapped
        replica_param = np.zeros((self.config.params.num_chains, self.num_param))
        lhood = np.zeros(self.config.params.num_chains)
        # Define the starting and ending of MCMC Chains
        start = 0
        end = self.samples_chains - 1
        number_exchange = np.zeros(self.config.params.num_chains)
        filen = open(self.path + "/num_exchange.txt", "a")
        # RUN MCMC CHAINS
        for l in range(0, self.config.params.num_chains):
            self.chains[l].start_chain = start
            self.chains[l].end = end
        for j in range(0, self.config.params.num_chains):
            self.wait_chain[j].clear()
            self.event[j].clear()
            self.chains[j].start()
        # SWAP PROCEDURE

        swaps_appected_main = 0
        total_swaps_main = 0
        for i in range(int(self.samples_chains / self.config.params.swap_interval)):
            count = 0
            for index in range(self.config.params.num_chains):
                if not self.chains[index].is_alive():
                    count += 1
                    self.wait_chain[index].set()
                    print(str(self.chains[index].temperature) + " Dead")

            if count == self.config.params.num_chains:
                break
            # print("Waiting")
            timeout_count = 0
            for index in range(0, self.config.params.num_chains):
                # print("Waiting for chain: {}".format(index + 1))
                flag = self.wait_chain[index].wait()
                if flag:
                    # print("Signal from chain: {}".format(index + 1))
                    timeout_count += 1

            if timeout_count != self.config.params.num_chains:
                print("Skipping the swap!")
                continue
            # print("Event occured")
            for index in range(0, self.config.params.num_chains - 1):
                # print("starting swap")
                param_1, param_2, swapped = self.swap_procedure(
                    self.parameter_queue[index], self.parameter_queue[index + 1]
                )
                self.parameter_queue[index].put(param_1)
                self.parameter_queue[index + 1].put(param_2)
                if index == 0:
                    if swapped:
                        swaps_appected_main += 1
                    total_swaps_main += 1
            for index in range(self.config.params.num_chains):
                self.event[index].set()
                self.wait_chain[index].clear()

        print("Joining processes")

        # JOIN THEM TO MAIN PROCESS
        for index in range(0, self.config.params.num_chains):
            self.chains[index].join()
        self.chain_queue.join()

        (acc_train, acc_test, accept, chains) = self.show_results()

        print("NUMBER OF SWAPS =", self.num_swap)
        swap_perc = self.num_swap * 100 / self.total_swap_proposals

        return (
            acc_train,
            acc_test,
            accept,
            chains,
        )

    def show_results(self):

        burnin = int(self.samples_chains * self.config.params.burn_in / self.config.params.samples)

        mcmc_samples = int(self.samples_chains * 0.25)

        likelihood_rep = np.zeros(
            (self.config.params.num_chains, self.num_nets, self.samples_chains - burnin + 1,)
        )
        accept_percent = np.zeros((self.config.params.num_chains, 1))
        accept_list = np.zeros((self.config.params.num_chains, self.samples_chains))

        pos_w = np.zeros(
            (self.config.params.num_chains, self.samples_chains - burnin, self.num_param)
        )

        fx_train_all = np.zeros(
            (self.config.params.num_chains, self.samples_chains - burnin, len(self.exp.train))
        )
        rmse_train = np.zeros((self.config.params.num_chains, self.samples_chains - burnin))
        acc_train = np.zeros((self.config.params.num_chains, self.samples_chains - burnin))
        fx_test_all = np.zeros(
            (self.config.params.num_chains, self.samples_chains - burnin, len(self.exp.test))
        )
        rmse_test = np.zeros((self.config.params.num_chains, self.samples_chains - burnin))
        acc_test = np.zeros((self.config.params.num_chains, self.samples_chains - burnin))

        for i in range(self.config.params.num_chains):

            file_name = (
                self.path
                + "/posterior/pos_likelihood/"
                + "chain_"
                + str(self.temperatures[i])
                + ".txt"
            )
            dat = np.loadtxt(file_name).reshape((self.num_nets, self.samples_chains + 1))
            likelihood_rep[i, :] = dat[:, burnin:]

            file_name = (
                self.path + "/predictions/acc_test_chain_" + str(self.temperatures[i]) + ".txt"
            )
            dat = np.loadtxt(file_name)
            acc_test[i, :] = dat[burnin:]

            file_name = (
                self.path + "/predictions/acc_train_chain_" + str(self.temperatures[i]) + ".txt"
            )
            dat = np.loadtxt(file_name)
            acc_train[i, :] = dat[burnin:]

        chain1_acctest = acc_test[0, :]
        chain1_acctrain = acc_train[0, :]

        posterior = pos_w.transpose(2, 0, 1).reshape(self.num_param, -1)

        fx_train = fx_train_all.transpose(2, 0, 1).reshape(
            len(self.exp.train), -1
        )  # need to comment this if need to save memory
        fx_test = fx_test_all.transpose(2, 0, 1).reshape(len(self.exp.test), -1)

        # fx_test = fxtest_samples.reshape(self.num_chains*(self.NumSamples - burnin), self.testdata.shape[0]) # konarks version

        likelihood_vec = likelihood_rep.flatten()

        rmse_train = rmse_train.reshape(
            self.config.params.num_chains * (self.samples_chains - burnin), 1
        )
        acc_train = acc_train.reshape(
            self.config.params.num_chains * (self.samples_chains - burnin), 1
        )
        rmse_test = rmse_test.reshape(
            self.config.params.num_chains * (self.samples_chains - burnin), 1
        )
        acc_test = acc_test.reshape(
            self.config.params.num_chains * (self.samples_chains - burnin), 1
        )

        accept_vec = accept_list

        accept = np.sum(accept_percent) / self.config.params.num_chains

        # np.savetxt(self.path + '/pos_param.txt', posterior.T)  # tcoment to save space

        np.savetxt(self.path + "/likelihood.txt", likelihood_vec.T, fmt="%1.5f")

        np.savetxt(self.path + "/accept_list.txt", accept_list, fmt="%1.2f")

        np.savetxt(self.path + "/acceptpercent.txt", [accept], fmt="%1.2f")

        return (
            acc_train,
            acc_test,
            accept,
            None,
        )

    def init_directories(self):

        run_nb = 0
        path = f"{self.path}/{run_nb}/"
        while os.path.exists(path):
            run_nb += 1
            path = f"{self.path}/{run_nb}/"

        os.makedirs(path)

        self.path = path
        directories = [
            path + "/predictions/",
            path + "/posterior",
            path + "/posterior/pos_likelihood",
            path + "/posterior/accept_list",
        ]

        for d in directories:
            self.make_directory(d)

    def make_directory(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

