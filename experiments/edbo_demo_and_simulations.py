import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    #  Bayesian Reaction Optimization as A Tool for Chemical Synthesis

    In this notebook we provide example code for Bayesian reaction optimization which can be used to explore some of the software’s features and reproduce simulation results presented in the manuscript. We ran experiments on Windows 10 laptops (i5 or i7 processors), Windows 7 and Linux workstations (Xeon processors), or Princeton's Linux HPC cluster (Della). According to our benchmarking, running a single iteration of BO will take seconds to minutes on a laptop depending on the optimizer configuration and size of the encoded reaction space. However, running simulations of optimizer convergence takes hours to complete due to averaging.
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Software

    Experimental Design via Bayesian Optimization: *edbo* is a practical python implementation of Bayesian optimization. The current build has 16 modules with essential and experimental features. Documentation for the main modules can be found on the documentation page or by using help(edbo.module).

    **Documentation:** https://b-shields.github.io/edbo/index.html

    The main Bayesian optimization program is accessed through the edbo.bro module. The main BO class, edbo.bro.BO, is a full featured program for choosing initial experiments, running BO on human-in-the-loop or computational objectives, modeling, and analyzing results. We used this class to run experiments for the paper. Note: edbo.bro.BO parameters are preset to those optimized for DFT encodings in the paper. See supplementary information for parameters optimized for one-hot-encoded and Mordred encoded reaction spaces.

    It can be fun to play around with the software in optimizing some 1D objectives. This is a nice way to get to know some of the features and be able to visualize the results. Below is a brief demonstration of an arbitrary 1D objective with a discretized domain. While edbo is designed to work with human-in-the-loop experimentation you can also use computational objectives (we also demonstrate this feature below). For example, this is how we accomplished self-optimization of the GP hyperparameters using *edbo*.

    **1D Bayesian Optimization Example:**
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from gpytorch.priors import GammaPrior
    from edbo.bro import BO

    # Define a computational objective

    def f(x):
        """Noise free objective."""

        return np.sin(10 * x[0]) * x[0] * 100

    # BO uses a user defined domain

    X = np.linspace(0,1,1000).reshape(1000, 1)
    domain = pd.DataFrame(X, columns=['x'])                        # Search space

    # Instantiate BO object

    bo = BO(domain=domain,                                         # Search space
            target='f(x)',                                         # Name of target (not required but nice)
            acquisition_function='EI',                             # Acquisition function
            init_method='rand',                                    # Initialization method
            lengthscale_prior=[GammaPrior(1.2,1.1), 0.2],          # GP length scale prior and initial value
            noise_prior=None,                                      # No noise prior
            batch_size=2,                                          # Number of experiments to choose in parallel
            fast_comp=True,                                        # Use gpytorch fast computation features
            computational_objective=f)                             # The objective is defined as a function

    # Select initial experiments

    bo.init_sample(append=True, seed=4)

    # Run a single iteration of BO

    bo.run()

    # Plot model and some posterior predictive samples (we need to unstandardize because BO standardizes internally)

    def plot_results():
        """Plot summary of 1D BO simulations"""

        mean = bo.obj.scaler.unstandardize(bo.model.predict(bo.obj.domain))                             # GP posterior mean
        std = np.sqrt(bo.model.variance(bo.obj.domain)) * bo.obj.scaler.std * 2                         # GP posterior standard deviation
        samples = bo.obj.scaler.unstandardize(bo.model.sample_posterior(bo.obj.domain, batch_size=3))   # GP samples
        next_points = bo.obj.get_results(bo.proposed_experiments)                                       # Next points proposed by BO
        results = bo.obj.results_input()                                                                # Results for known data

        plt.figure(1, figsize=(8,8))

        # Model mean and standard deviation
        plt.subplot(211)
        plt.plot(X.flatten(), [f(x) for x in X], color='black')
        plt.plot(X.flatten(), mean, label='GP')
        plt.fill_between(X.flatten(), mean-std, mean+std, alpha=0.4)

        # Known results and next selected point
        plt.scatter(results['x'], results['f(x)'], color='black', label='known')
        plt.scatter(next_points['x'], next_points['f(x)'], color='red', label='next_experiments')
        plt.ylabel('f(x)')

        # Samples
        plt.subplot(212)
        for sample in samples:
            plt.plot(X.flatten(), sample.numpy())
        plt.xlabel('x')
        plt.ylabel('Posterior Samples')
        plt.show()

    print('\n\nAfter random initializaiton:\n')
    plot_results()

    # Run a simulation and plot results

    print('\n\nAfter 5 rounds of optimization:\n')
    bo.simulate(iterations=5, seed=4)
    plot_results()
    return BO, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The BO class requires users to define their own encoded reaction spaces and choose model parameters. We imagined that a general, fully automated, approach to featurization would render BO more straightforward to apply by non-experts or on the fly. Thus, we also include a class, BO_express, which automates featurization, preprocessing, and selection of model parameters. Reaction components and encodings are passed to BO_express using dictionaries. Then BO_express attempts to encode each component based on the specified encoding type. When chemical names or nicknames are given, *edbo* will search the internet for the corresponding SMILES string. If there is an error in a SMILES string or the name of a molecule could not be found in the NIH database an edbo bot is spawned to help resolve the issue. Once instantiated, BO_express.help() will also spawn an edbo bot to help with tasks.

    **BO_express Example code:**
    """
    )
    return


@app.cell
def _():
    from edbo.bro import BO_express
    reaction_components = {'aryl_halide': ['chlorobenzene', 'iodobenzene', 'bromobenzene'], 'base': ['DBU', 'MTBD', 'potassium carbonate', 'potassium phosphate'], 'solvent': ['THF', 'Toluene', 'DMSO', 'DMAc'], 'ligand': ['c1ccc(cc1)P(c2ccccc2)c3ccccc3', 'C1CCC(CC1)P(C2CCCCC2)C3CCCCC3', 'CC(C)c1cc(C(C)C)c(c(c1)C(C)C)c2ccccc2P(C3CCCCC3)C4CCCCC4'], 'concentration': [0.1, 0.2, 0.3], 'temperature': [20, 30, 40]}
    encoding = {'aryl_halide': 'resolve', 'base': 'ohe', 'solvent': 'resolve', 'ligand': 'smiles', 'concentration': 'numeric', 'temperature': 'numeric'}
    bo_1 = BO_express(reaction_components=reaction_components, encoding=encoding, batch_size=7, init_method='kmeans', acquisition_function='TS', target='yield')
    print('\n')
    bo_1.init_seq.visualize = False
    bo_1.init_sample()
    print('\n')
    bo_1.get_experiments(structures=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Then to add experimental results you can use bo.add_results(). Once there is experimental data you can use bo.run() to run a single iteration of BO. Try help(edbo.bro.BO_express) to see a full lists of methods.""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Data

    All data sets used to develop and test *edbo* can be accessed via the data_loader functions. Data loader functions return pandas.DataFrames of the encoded reactions. To run simulations, these data sets were then preprocessed using the edbo.utils.Data class. For DFT encodings all Gaussian output files, featurization source code, and descriptor computation/extraction scripts can be found on the *auto-qchem* GitHub page: https://github.com/b-shields/auto-QChem. After loading DFT data (*vide infra*), the quality of the DFT stationary points for each component can be assessed via the "<name>_convergence_criteria" columns. When "met" these columns indicate that the optimized geometries have no negative Hessian eigenvalues, maximum force < 0.000450, root-mean-square force < 0.000300, maximum displacement < 0.001800, and root-mean-square displacement < 0.001200 (or maximum force < 0.00001 and root-mean-square force < 0.00001).

    Each descriptor set was treated by (1) removing or one-hot-encoding non-numeric features, (2) decorrelation by removing highly correlated features (Pearson correlation coefficient > 0.95) with any other selected descriptor, and (3) normalizing each feature on the unit hypercube. The standardization of responses (zero mean and unit variance) is handled internally by the software. Normalization of descriptors is important for GP model performance, for example because the priors were optimized for the [0,1] interval.

    Note: "loading" can take a little while because the objectives are being reconstructed rather than actually loaded to save disk space.

    **Example code:**
    """
    )
    return


@app.cell
def _():
    from data_loader import suzuki, aryl_amination, direct_arylation 

    # Suzuki data

    reaction1 = suzuki(electrophile='dft',                 # dft, mordred, or ohe
                       nucleophile='dft',                  # dft, mordred, or ohe
                       base='dft',                         # dft, mordred, or ohe
                       ligand='boltzmann-dft',             # random-dft, boltzmann-dft, mordred, or ohe
                       solvent='dft')                      # dft, mordred, or ohe

    reaction1.head()
    return aryl_amination, direct_arylation, suzuki


@app.cell
def _(aryl_amination):
    # Buchwald-Hartwig data

    reaction2b = aryl_amination(aryl_halide='dft',         # dft, mordred, or ohe
                                additive='dft',            # dft, mordred, or ohe
                                base='dft',                # dft, mordred, or ohe
                                ligand='Pd(0)-dft',        # Pd(0)-dft, mordred, or ohe
                                subset=2)                  # (1) 2a, (2) 2b, (3) 2c, (4) 2d, (4) 2e

    reaction2b.head()
    return


@app.cell
def _(direct_arylation):
    # Direct arylation data

    reaction3 = direct_arylation(base='dft',               # dft, mordred, or ohe
                                 ligand='boltzmann-dft',   # random-dft, boltzmann-dft, mordred, or ohe
                                 solvent='dft')            # dft, mordred, or ohe

    reaction3.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Bayesian Reaction Optimization Experiments

    In this section we define a simulation function for evaluating the performance of *edbo* in the optimization of reactions 1 and 2. These functions can be used to reproduce optimization results with the development objectives and to test other optimizer components.

    **Load development data:**
    """
    )
    return


@app.cell
def _(aryl_amination, suzuki):
    from edbo.utils import Data
    reaction1_1 = Data(suzuki(electrophile='dft', nucleophile='dft', base='dft', ligand='boltzmann-dft', solvent='dft'))
    reaction2a = Data(aryl_amination(aryl_halide='dft', additive='dft', base='dft', ligand='Pd(0)-dft', subset=1))
    reaction2b_1 = Data(aryl_amination(aryl_halide='dft', additive='dft', base='dft', ligand='Pd(0)-dft', subset=2))
    reaction2c = Data(aryl_amination(aryl_halide='dft', additive='dft', base='dft', ligand='Pd(0)-dft', subset=3))
    reaction2d = Data(aryl_amination(aryl_halide='dft', additive='dft', base='dft', ligand='Pd(0)-dft', subset=4))
    reaction2e = Data(aryl_amination(aryl_halide='dft', additive='dft', base='dft', ligand='Pd(0)-dft', subset=5))
    for reaction in [reaction1_1, reaction2a, reaction2b_1, reaction2c, reaction2d, reaction2e]:
        reaction.clean()
        reaction.drop(['entry', 'vibration', 'correlation', 'Rydberg', 'correction', 'atom_number', 'E-M_angle', 'MEAN', 'STDEV'])
        reaction.standardize(scaler='minmax')
        reaction.uncorrelated(threshold=0.95)
        reaction.data.head()
    return Data, reaction2b_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    The classes edbo.bro.BO and edbo.bro.BO_express have built in functions for running simulations if the results in the domain are available as a tabulated list or there is an automated method for getting the results (see the use of the computational_objective option in the 1D example above).

    **Example of a single optimization simulation using edbo:**
    """
    )
    return


@app.cell
def _(BO, reaction2b_1):
    bo_2 = BO(exindex=reaction2b_1.data, domain=reaction2b_1.data.drop('yield', axis=1), batch_size=3, acquisition_function='EI', fast_comp=True)
    bo_2.simulate(iterations=9, seed=0)
    bo_2.plot_convergence()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Simulations used for the paper required averaging. Thus, we used *edbo* to build simulation functions which allowed us to run larger data experiments and get performance statistics. Let's define a simulation function which takes options for data, acquisition function, surrogate model, initialization method, number of experiments to choose in parallel, number of iterations to run the simulation, number of simulations to average, and analysis of the results. Then this function can be used to investigate the results presented in the paper.

    **Define a simulation function:**
    """
    )
    return


@app.cell
def _(BO, pd):
    from edbo.models import GP_Model, RF_Model, Random
    from edbo.plot_utils import average_convergence, plot_avg_convergence

    # Function for simulating different BO parameters

    def simulate(data,                 # Data container (vide supra)
                 acq_func,             # Acquisition function: 'EI', 'PI', 'UCB', 'TS', ...
                 model=GP_Model,       # Surrogate model: GP_Model or RF_Model
                 init_method='rand',   # Initialization method: 'rand', 'pam', 'kmeans'
                 batch_size=5,         # Parallel acquisition: int
                 iterations=9,         # Number of iterations to run each simulation for
                 average_of=10,        # Number of simulations to average
                 export_path=None,     # Export the simulation results to a CSV file
                 plot=False):          # Plot the average and standard deviation averaged convergence
        """
        Simulation function which averages BO runs with given optimization paramters.
        """

        # Average N optimizations with different random initializations
        results = []
        for i in range(average_of):

            # Use random function if the acquisition function is random selection
            if acq_func == 'rand':
                bo = BO(exindex=data.data,
                        domain=data.data.drop('yield', axis=1),
                        init_method=init_method,
                        model=Random,
                        batch_size=batch_size,
                        acquisition_function=acq_func)
            # Otherwise use specified model
            else:
                bo = BO(exindex=data.data,
                        domain=data.data.drop('yield', axis=1),
                        model=model,
                        init_method=init_method,
                        batch_size=batch_size,
                        acquisition_function=acq_func,
                        fast_comp=True)                            # Use gpytorch fast computation features

            # Simulate
            bo.init_seq.visualize = False
            bo.simulate(iterations=iterations, seed=i)

            # Append results to record
            results.append(bo.obj.results_input()['yield'].values)

        # Save the results to a CSV file
        results = pd.DataFrame(results)
        if export_path != None:
            results.to_csv(export_path)

        # Average performance
        index, mean, std = average_convergence(results, batch_size)

        # Plot
        if plot:
            plot_avg_convergence(results, batch_size)

        return results, mean, std
    return GP_Model, simulate


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now we can run an example simulation with averaging. In this case, let's use reaction 2b as the objective, a GP surrogate model, Probability of Improvement as an acquisition function, random selection for initialization, a batch size of 10, and average 10 different random initializations.

    **Simulation Example:**
    """
    )
    return


@app.cell
def _(GP_Model, reaction2b_1, simulate):
    results, mean, std = simulate(reaction2b_1, 'PI', model=GP_Model, init_method='rand', batch_size=10, iterations=4, average_of=20, export_path=None, plot=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In the paper we evaluated the performance of different acquisition functions in the optimization of reactions 1 and 2. These general results can be reproduced using the above simulation function. Let's reproduce the results for reaction 2b.

    **Acquisition function performance simulations:**
    """
    )
    return


@app.cell
def _(GP_Model, reaction2b_1, simulate):
    from edbo.plot_utils import compare_convergence
    reaction_1 = reaction2b_1
    acq_funcs = ['EI', 'TS', 'rand', 'eps-greedy', 'MeanMax', 'VarMax']
    acq_func_results = []
    for acq_func in acq_funcs:
        print('Acquisition Function:', acq_func)
        results_1, mean_1, std_1 = simulate(reaction_1, acq_func, model=GP_Model, init_method='rand', batch_size=5, iterations=9, average_of=30, export_path=None, plot=True)
        acq_func_results.append(results_1)
    print('Average performance for all acquisition functions:')
    compare_convergence(acq_func_results, 5, acq_funcs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Here we see that the software has given slightly different result from figure 3E. The primary difference is in the path taken by the Thompson sampling algorithm. This is likely because of some internal randomnization - e.g., the method used to sample the posterior predictive distribution of the gaussian process model (edbo.models.GP_Model.sample_posterior) does not have a random seed. *Importantly, while the exact result simulation result is slightly different, the conclusions are the same: (1) EI did the best followed by TS and then pure exploration, (2) pure exploitation gave poor results (on average furnishing only marginally better decisions than random selection), and (3) $\epsilon$-greedy gave similar results to pure exploitation.*

    ## Exploration and Exploitation

    As an illustration, in the paper we simulated the decisions made by acquisition functions which only exploit (mean maximization) and only explore (variance maximization). We visualized the decision in a 2D t-SNE embedding with domain points clustered using the k-means algorithm and tracked the fit to the overall space and yields of the selected experiments. We initialized the optimizers at the point closest to the center of one of the clusters. In this section we provide the code required to reproduce this example with any of the cluster centers. We will do this by clustering the data using the *k*-means algorithm, generating a 2D embedding, and then running BO with the explorer and exploiter while tracking the selected experiments, $R^2$ for the fit to the reaction space, and observed yield.

    **Load development data:**
    """
    )
    return


@app.cell
def _(Data, suzuki):
    reaction1_2 = Data(suzuki(electrophile='dft', nucleophile='dft', base='dft', ligand='boltzmann-dft', solvent='dft'))
    reaction1_2.clean()
    reaction1_2.drop(['entry', 'vibration', 'correlation', 'Rydberg', 'correction', 'atom_number', 'E-M_angle', 'MEAN', 'STDEV'])
    reaction1_2.standardize(scaler='minmax')
    reaction1_2.uncorrelated(threshold=0.95)
    reaction1_2.data.head()
    return (reaction1_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To start we will cluster the data using the *k*-means algorithm and generate a *t*-SNE embedding for visualization. Here we will cluster the reaction space into five clusters, as in the paper. Note: we have found that sklearn's TSNE class can give slightly different embeddings even with a random seed.

    **Cluster and embed:**
    """
    )
    return


@app.cell
def _(np, pd, reaction1_2):
    from sklearn.cluster import KMeans
    from sklearn.metrics import pairwise_distances_argmin_min
    from sklearn.manifold import TSNE
    DATA = reaction1_2.data.drop('yield', axis=1)
    clusterer = KMeans(n_clusters=5, random_state=0)
    cluster_labels = clusterer.fit_predict(DATA)
    closest = pd.DataFrame(columns=DATA.columns)
    for i in range(5):
        cluster_i = DATA.iloc[np.where(clusterer.labels_ == i)]
        closest_i, _ = pairwise_distances_argmin_min(clusterer.cluster_centers_[[i]], cluster_i)
        closest = pd.concat([closest, cluster_i.iloc[closest_i]], sort=False)
    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=10)
    tsne_results = tsne.fit_transform(DATA)
    return closest, cluster_labels, tsne_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    You can choose which of the five clusters to start in by specifying the START value below. Let's plot the embedded clusters, points closest to their centers, and selected point. In the below plot the clusters are color coded, the points closest to the cluster centers are marked with *X*'s, and the selected starting point for Bayesian optimization is highlighted in blue.

    **Plot clustering results:**
    """
    )
    return


@app.cell
def _(closest, cluster_labels, np, plt, tsne_results):
    START = 1
    import matplotlib
    import seaborn as sns
    sns.set(rc={'figure.figsize': (8, 8)})
    sns.set(rc={'font.size': 14})
    sns.set(style='darkgrid')
    sns.set(style='white')
    matplotlib.rc('xtick', labelsize=14)
    matplotlib.rc('ytick', labelsize=14)
    for i_1 in range(5):
        cluster_i_1 = np.argwhere(cluster_labels == i_1).flatten()
        plt.scatter(tsne_results[:, 0][cluster_i_1], tsne_results[:, 1][cluster_i_1], s=50, lw=1, alpha=0.3)
    plt.scatter(tsne_results[closest.index.values][:, 0], tsne_results[closest.index.values][:, 1], s=200, lw=4, marker='x', color='black')
    plt.scatter([tsne_results[closest.index.values][START][0]], [tsne_results[closest.index.values][START][1]], s=2000, lw=2, marker='o', color='blue', alpha=0.3)
    plt.title('Stochastic Neighbor Embedding', fontsize=14)
    plt.xlabel('t-SNE1', fontsize=14)
    plt.ylabel('t-SNE2', fontsize=14)
    plt.show()
    return (START,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    To run these simulations, we will need to write a new simulation function which keeps track of the path, $R^2$ for the fit to the whole space and yields for each experiment. In this case *edbo* does not have a built-in method compatible with this type of simulation. Therefore, we will write a custom function using the BO.run method which runs a single iteration of BO at a time. You can also use this function to investigate the decisions made by different acquisition functions using the acq_func argument.

    **Define a simulation function and run the explorer and exploiter:**
    """
    )
    return


@app.cell
def _(BO, START, closest, reaction1_2):
    from edbo.math_utils import model_performance

    def simulation(acq_func, NSTEPS=50):
        """
        Run BO to track the optimization path of a given acquisition function.
        The arguement NSTEPS denotes the number of rounds of optimization to run.
        """
        start = closest.index.values[[START]]
        bo = BO(exindex=reaction1_2.data, domain=reaction1_2.data.drop('yield', axis=1), results=reaction1_2.data.iloc[start], init_method='external', batch_size=1, acquisition_function=acq_func, fast_comp=True)
        r2 = []
        for i in range(NSTEPS):
            bo.run()
            pred = bo.obj.scaler.unstandardize(bo.model.predict(bo.obj.domain))
            obs = bo.obj.exindex['yield'].values
            RMSE, R2 = model_performance(pred, obs)
            r2.append(R2)
            bo.obj.get_results(bo.proposed_experiments, append=True)
        index = list(start) + list(bo.obj.results.index.values[1:])
        yield_ = list(bo.obj.results_input()['yield'].values)
        return (index, yield_, r2)
    exploit, exploit_yield, exploit_r2 = simulation('MeanMax')
    explore, explore_yield, explore_r2 = simulation('VarMax')
    return (
        exploit,
        exploit_r2,
        exploit_yield,
        explore,
        explore_r2,
        explore_yield,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now with the simulation results in hand we can plot the paths in the *t*-SNE embedding and $R^2$/yield against optimizer step. For the path we will use matplotlib's ConnectionPatch to draw arrows between the points selected by each algorithm. You can plot the first *N* choices of both algorithms by specifying keep = *N* below. For now, let's see what the first 10 choices look like when initialized from this particular cluster. Just as in the paper, we will give the explorer a dashed line arrow and the exploiter a solid lined arrow.

    **Plot paths through reaction space:**
    """
    )
    return


@app.cell
def _(cluster_labels, exploit, explore, np, plt, tsne_results):
    from matplotlib.patches import ConnectionPatch
    keep = 10
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
    for i_2 in range(5):
        cluster_i_2 = np.argwhere(cluster_labels == i_2).flatten()
        ax1.scatter(tsne_results[:, 0][cluster_i_2], tsne_results[:, 1][cluster_i_2], s=50, lw=1, alpha=0.2)
    for i_2 in range(len(exploit[:keep]) - 1):
        arrow = np.array(exploit).flatten()
        xyA = tuple(tsne_results[arrow[i_2]])
        xyB = tuple(tsne_results[arrow[i_2 + 1]])
        coordsA = 'data'
        coordsB = 'data'
        con = ConnectionPatch(xyA, xyB, coordsA, coordsB, linestyle='-', arrowstyle='-|>', shrinkA=0, shrinkB=0, connectionstyle='arc3,rad=2', mutation_scale=20, linewidth=2, color='black')
        ax1.add_artist(con)
    for i_2 in range(len(explore[:keep]) - 1):
        arrow = np.array(explore).flatten()
        xyA = tuple(tsne_results[arrow[i_2]])
        xyB = tuple(tsne_results[arrow[i_2 + 1]])
        coordsA = 'data'
        coordsB = 'data'
        con = ConnectionPatch(xyA, xyB, coordsA, coordsB, linestyle='--', arrowstyle='-|>', shrinkA=0, shrinkB=0, connectionstyle='arc3,rad=-0.1', mutation_scale=20, linewidth=2, color='black')
        ax1.add_artist(con)
    plt.title('Stochastic Neighbor Embedding', fontsize=14)
    plt.xlabel('t-SNE1', fontsize=14)
    plt.ylabel('t-SNE2', fontsize=14)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Plot model fit at each optimization step:**""")
    return


@app.cell
def _(exploit_r2, explore_r2, plt):
    index = range(len(exploit_r2))
    fig_1, ax1_1 = plt.subplots(1, 1, figsize=(8, 8))
    ax1_1.plot(index, exploit_r2, '-', color='black', lw=2, label='Exploit')
    ax1_1.plot(index, explore_r2, '--', color='black', lw=2, label='Explore')
    ax1_1.set_ylabel('Full Space Fit $R^{2}$', fontsize=14)
    ax1_1.set_xlabel('Optimization Step', fontsize=14)
    ax1_1.legend(loc='lower right', fontsize=14)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Plot observed yield at each optimization step:**""")
    return


@app.cell
def _(exploit_yield, explore_yield, plt):
    index_1 = range(len(exploit_yield))
    fig_2, ax1_2 = plt.subplots(1, 1, figsize=(8, 8))
    ax1_2.plot(index_1, exploit_yield, '-', color='black', lw=2, label='Exploit')
    ax1_2.plot(index_1, explore_yield, '--', color='black', lw=2, label='Explore')
    ax1_2.set_ylabel('Observed Yield', fontsize=14)
    ax1_2.set_xlabel('Optimization Step', fontsize=14)
    ax1_2.legend(loc='lower right', fontsize=14)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In this case we can see that the results presented in figure 3A and B can be reproduced when initialized from the different cluster centers. Namely (1) the exploiter remained in a single cluster while the explorer traversed the entire space, (2)  over the course of 50 experiments the $R^2$ scores of the explorer and exploiter diverge, with the explorer offering a significantly better fit to the reaction surface, and (3) most points selected by the explorer and exploiter gave low and high yields respectively.

    ## Test Set

    The direct arylation test set collected for this study was not used at all in optimizer development. On the basis of data for rection 1 and 2a–e we established that BO performance was most promising using DFT reaction representations, a GP surrogate model, and expected improvement with batches selected via the Kriging believer algorithm as an acquisition function. Thus, we evaluated only this optimizer configuration on the test set. In this section we provide code to reproduce this experiment. In addition, we provide the results of our initial simulations as a CSV file for comparison. Finally, we provide data and code to compare with human performance.

    Our experiments with reactions 1 and 2a–e suggested that optimization was feasible with only a small fraction of the search space (typically well below 100 experiments). Importantly, in a real application, it is not typically feasible to screen parameters like temperature using a single 96 well plates for HTE. Thus, to fully explore temperature for this search space using fully populated HTE plates would require a minimum of 288 reactions. Given the importance of parallel acquisition functions and the number of candidate experiments in the reaction space, we elected to carry out tests with a batch size of 5. This gives a reasonable model for optimization at the bench. In addition, we imagined selecting 5 parallel experiments would work well in human benchmarking experiments (vide infra).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Load experimental data:**""")
    return


@app.cell
def _(Data, direct_arylation):
    reaction3_1 = Data(direct_arylation(base='dft', ligand='boltzmann-dft', solvent='dft'))
    reaction3_1.clean()
    reaction3_1.drop(['entry', 'vibration', 'correlation', 'Rydberg', 'correction', 'atom_number', 'E-M_angle', 'MEAN', 'STDEV'])
    reaction3_1.standardize(scaler='minmax')
    reaction3_1.uncorrelated(threshold=0.95)
    reaction3_1.data.head()
    return (reaction3_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now we can define a simulation loop, using *edbo*'s built in simulation function, to evaluate the performance of BO on the direct arylation test set. In human benchmarking studies we had 50 participants play the game. Accordingly, we ran the simulation with 50 random initializations for statistical comparison.

    **Testing BO performance:**
    """
    )
    return


@app.cell
def _(BO, reaction3_1):
    NSIM = 50
    NBATCH = 19
    results_2 = []
    for i_3 in range(NSIM):
        bo_3 = BO(exindex=reaction3_1.data, domain=reaction3_1.data.drop('yield', axis=1), batch_size=5, init_method='rand', acquisition_function='EI', fast_comp=True)
        bo_3.simulate(iterations=NBATCH, seed=i_3)
        results_2.append(bo_3.obj.results_input()['yield'].values)
    return (results_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    In the paper we compare the optimizers decisions to those of the human participants. The convergence behavior of all human participants can be seen below and further analyzed using the available data. We found that 28/50 and 32/50 human participants achieved a yield that was within 1% and 5% of the maximum yield, respectively. As in the paper, we found in the reproduction run that 100% of randomly initialized BO runs achieved the optimal yield. This is in part due to some human participants not utilizing all 20 of their possible experiment batches (we deal with this issue below).

    **Load human results and initial simulation results:**
    """
    )
    return


@app.cell
def _(pd, results_2):
    humans = pd.read_csv('arylation_game_summary.csv')
    bo_paper = pd.read_csv('arylation_bo_results_GP-EI_bs=5.csv', index_col=0)
    bo_4 = pd.DataFrame(results_2)
    return bo_4, bo_paper, humans


@app.cell
def _(humans):
    humans.head()
    return


@app.cell
def _(bo_paper):
    bo_paper.head()
    return


@app.cell
def _(bo_4):
    bo_4.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    We will need some analysis functions to compute optimization paths from the data, average the results, and plot convergence. These are defined below.

    **Define helpful analysis functions:**
    """
    )
    return


@app.cell
def _(humans, np, pd, plt):
    # Plotting functions

    yield_columns = ['yield_' + str(i) for i in range(1,101)]    # Columns in human participant data which correspond to yield

    def max_observed(points, batch_size):
        """
        Compute max observed yield for each batch of experiments.
        """

        index = []
        max_obs = []
        for i in range(round(len(points)/batch_size)):
            current_max = points[:batch_size*(i+1)].max()
            max_obs.append(current_max)
            index.append(i+1)

        return index, max_obs

    def average(df, batch_size):
        """
        Compute the point wise average of the max observed yield
        for each batch of experiments. Only average points that 
        are present (i.e. don't pad with zeros).
        """

        # Get max observed for each row
        data = []
        for i in range(len(df)):
            # Drop NAs
            entry = df.iloc[i].dropna().values
            # Get max observed values
            index, max_obs = max_observed(entry, batch_size)
            data.append(max_obs)

        # Average
        max_obs_table = pd.DataFrame(data)
        index = []
        avg = []
        stdev = []
        for i in range(len(max_obs_table.columns.values)):
            # Drop NAs
            entry = max_obs_table.iloc[:,i].dropna().values
            # Get average and stdev
            index.append(i + 1)
            avg.append(entry.mean())
            stdev.append(entry.std())

        return index, avg, stdev

    def avg_subsets(column):
        """Get average convergence data for unique entries of a column."""

        # Break up by entry
        entries = list(humans[column].drop_duplicates().values)

        # Get subset averages
        avgs = []
        for entry in entries:
            subset = humans[humans[column] == entry][yield_columns]
            index, avg, stdev = average(subset, 5)
            avgs.append(avg)

        # Return lists
        return entries, avgs

    def convergence(data_list, legend_list=None, xlabel='Experiment Batch', 
                    ylabel='Max Observed Yield', export_path=None, alpha=1,
                    colormap=plt.cm.viridis, figsize=(5,5), legendloc='lower right'):
        """Plot convergence of points in data list."""

        fig, ax = plt.subplots(1, figsize=figsize)

        # Legend
        legend = True
        if legend_list == None:
            legend = False
            legend_list = [i for i in range(len(data_list))]

        # Plot options
        if colormap != None:  
            ax.set_prop_cycle('color', [colormap(i) for i in np.linspace(0, 1, len(data_list))])
        ax.set(xlabel=xlabel, ylabel=ylabel)
        fig.tight_layout()

        # Plot
        for data, name in zip(data_list, legend_list):
            ax.plot(range(1,len(data)+1), data, lw=3, label=name, alpha=alpha)

        if legend:
            ax.legend(loc=legendloc)

        # Export & return
        if export_path != None:
            plt.savefig(export_path + '.svg', format='svg', dpi=1200, bbox_inches='tight')
            plt.show()
        else:
            return plt.show()
    return average, convergence, max_observed, yield_columns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Human performance:**""")
    return


@app.cell
def _(convergence, humans, max_observed, yield_columns):
    opt_paths_h = [humans[yield_columns].iloc[i].dropna().values for i in range(len(humans))]
    max_obs_paths_h = []
    for path in opt_paths_h:
        index_2, max_obs = max_observed(path, 5)
        max_obs_paths_h.append(max_obs)
    convergence(max_obs_paths_h, alpha=0.8)
    return max_obs_paths_h, opt_paths_h


@app.cell
def _(opt_paths_h, pd):
    # Compute the number of humans within 1% and 5% of the max observed yield

    print('Humans within 1% of max yield:', len(pd.DataFrame(opt_paths_h).max(axis=1)[pd.DataFrame(opt_paths_h).max(axis=1) > 99]))
    print('Humans within 5% of max yield:', len(pd.DataFrame(opt_paths_h).max(axis=1)[pd.DataFrame(opt_paths_h).max(axis=1) > 95]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Reproduced Bayesian optimizer performance:**""")
    return


@app.cell
def _(bo_4, convergence, max_observed):
    opt_paths_bo = bo_4.values
    max_obs_paths_bo = []
    for path_1 in opt_paths_bo:
        index_3, max_obs_1 = max_observed(path_1, 5)
        max_obs_paths_bo.append(max_obs_1)
    convergence(max_obs_paths_bo, alpha=0.8)
    return max_obs_paths_bo, opt_paths_bo


@app.cell
def _(opt_paths_bo, pd):
    # Compute the number of BO runs within 1% and 5% of the max observed yield

    print('BO runs within 1% of max yield:', len(pd.DataFrame(opt_paths_bo).max(axis=1)[pd.DataFrame(opt_paths_bo).max(axis=1) > 99]))
    print('BO runs within 5% of max yield:', len(pd.DataFrame(opt_paths_bo).max(axis=1)[pd.DataFrame(opt_paths_bo).max(axis=1) > 95]))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    As in the paper experiments, we found that the optimizer typically converged to conditions which maximize yield of test reaction 3 withing 10 batches of 5 experiments (< 3% of experiments in the search space), irrespective of the 5 randomly selected reactions used for initialization. This trend is also reflected in the reproduction run.

    The objective of this experiment was to statistically compare the optimization performance of BO and human experts. Each player had up to 20 batches of reactions for a total of 100 experiments to find the best experimental conditions. However, most participants played fewer than 20 rounds of experiments, for example because they believed they had achieved a globally optimal solution. Thus, in addition to comparing the raw optimization paths we sought to put hard bounds on average human performance in the data set. If we assume that players who stopped early would not have achieved any higher yielding conditions had they continued playing, we can impute a lower bound (worst case) on human performance. Conversely, if we assume optimistically that had players continued, they would have achieved 100% yield in their next batch of experiments, we can impute an upper bound (best case).

    **Bounding human performance:**
    """
    )
    return


@app.cell
def _(
    convergence,
    humans,
    max_obs_paths_h,
    max_observed,
    opt_paths_h,
    yield_columns,
):
    opt_paths_lower = humans[yield_columns].fillna(0.0).values
    max_obs_paths_lower = []
    for path_2 in opt_paths_lower:
        index_4, max_obs_2 = max_observed(path_2, 5)
        max_obs_paths_lower.append(max_obs_2)
    opt_paths_upper = humans[yield_columns].fillna(100.0).values
    max_obs_paths_upper = []
    for path_2 in opt_paths_upper:
        index_4, max_obs_2 = max_observed(path_2, 5)
        max_obs_paths_upper.append(max_obs_2)
    i_4 = 0
    convergence([opt_paths_lower[i_4], opt_paths_upper[i_4], opt_paths_h[i_4]], alpha=1, legend_list=['Lower Bound', 'Upper Bound', 'Raw Data'], colormap=None, legendloc='center right', xlabel='Experiment', ylabel='% Yield', export_path=None)
    convergence([max_obs_paths_lower[i_4], max_obs_paths_upper[i_4], max_obs_paths_h[i_4]], alpha=1, legend_list=['Lower Bound', 'Upper Bound', 'Raw Data'], colormap=None, ylabel='Max Observed Yield', legendloc='lower right', export_path=None)
    return max_obs_paths_lower, max_obs_paths_upper


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now we can simply average the raw data, upper, and lower bound for comparison with BO.

    **Bounding average human performance:**
    """
    )
    return


@app.cell
def _(average, convergence, humans, yield_columns):
    # Get averages
    humans_i, humans_avg, humans_std = average(humans[yield_columns], 5)
    humans_lower_i, humans_lower_avg, humans_lower_std = average(humans[yield_columns].fillna(0.0), 5)
    humans_upper_i, humans_upper_avg, humans_upper_std = average(humans[yield_columns].fillna(100.0), 5)

    convergence([humans_lower_avg, humans_upper_avg, humans_avg],
                alpha=1,
                legend_list=['Lower Bound', 'Upper Bound', 'Raw Data'],
                colormap=None,
                ylabel='Average Max Observed Yield',
                legendloc='lower right')
    return humans_avg, humans_lower_avg, humans_upper_avg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""**Compare raw data and bounds to reproduced BO results:**""")
    return


@app.cell
def _(
    average,
    bo_4,
    convergence,
    humans_avg,
    humans_lower_avg,
    humans_upper_avg,
):
    bo_i, bo_avg, bo_std = average(bo_4, 5)
    convergence([humans_lower_avg, humans_upper_avg, humans_avg, bo_avg], alpha=0.8, legend_list=['Humans Lower Bound', 'Humans Upper Bound', 'Humans Raw Data', 'Bayesian Optimization'], colormap=None, ylabel='Average Max Observed Yield', legendloc='lower right')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Each of these bounds is unrealistic and the actual average performance of course would fall somewhere in between. However, notice that the raw averaged data (where we only average the available data for each batch of experiments and the number of points in batch i is not necessarily the same as batch i + 1) closely follows the lower bound up until batch 11. In addition, the upper bound closely follows the average path of the optimizer.

    Comparing the bounds on average human convergence to average BO convergence suggests that BO offers improved performance in optimization. However, we wanted to assess their performance statistically. To do this, at each step in the optimization we conducted a hypothesis test under the null hypothesis that the central tendency of human and machine performance was identical. Here, we remove the assumption of equal variance between the two population samples and utilize an unpaired Welch’s t-test with Satterthwait degrees of freedom. To do this you may need to install statsmodels. When installing edbo it is not installed because it is not a dependency. The below code will try to install statsmodels via the IPython shell when defining the statistics functions.

    **Hypothesis testing functions:**
    """
    )
    return


@app.cell
def _(pd):
    # try: 
    import statsmodels.stats.api as sms
    # except:
    #     # !conda install -y statsmodels

    def t_test(X1, X2, sample_vars='equal', print_out=True):
        """
        t-test for the null hypothesis of identical means. The unpaired 
        t-test should not be used if there is a significant difference 
        between the variances of the two samples. Here, if the variance
        of the samples is not assumed to be the same, then Welsh t-test 
        with Satterthwait degrees of freedom is used.
        """

        if sample_vars == 'equal':
            usevar = 'pooled'
        else:
            usevar = 'unequal'

        cm = sms.CompareMeans(sms.DescrStatsW(X1), sms.DescrStatsW(X2))
        ci = cm.tconfint_diff(usevar=usevar)
        t_value = cm.ttest_ind(usevar=usevar)

        if print_out == True:
            print('95% CI:', ci)
            print('DOF:', t_value[2])
            print('t-stat:', t_value[0])
            print('p-value:', t_value[1])
        else:
            return [t_value[2], t_value[0], t_value[1], ci[0], ci[1]]

    # t-test for optimization paths

    def path_stats(pop1_paths, pop2_paths, steps=range(21)):
        """
        Conduct a Welsh's t-test with the null hypothesis of identical
        means at each step of optimization.
        """

        summary = []
        for i in steps:

            # Data for optimization step i
            pop1 = pop1_paths.iloc[:,i].dropna().values
            pop2 = pop2_paths.iloc[:,i].dropna().values

            # Hypothesis test
            test = t_test(pop1, 
                          pop2, 
                          sample_vars='unequal', 
                          print_out=False)

            summary.append([i] + test)

        columns = ['Batch', 'DOF', 't-value', 'p-value', '95% CI (l)', '95% CI (u)']
        summary = pd.DataFrame(summary, columns=columns)

        return summary
    return path_stats, t_test


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    For example, consider the distribution of max observed yields for batch 1 of human and BO decisions. Using the above functions, a Welch’s t-test with $H0: μ_{Human} = μ_{BO}$ gives t = 2.99 and p = 0.0035. Thus, we reject the null hypothesis and conclude the initial choice distributions do not have an identical average. Further, we infer from their averages that humans tended to make better initial decisions than random selection. 

    **Example t-test for first step:**
    """
    )
    return


@app.cell
def _(max_obs_paths_bo, max_obs_paths_h, pd, plt, t_test):
    step = 0
    human_step = pd.DataFrame(max_obs_paths_h).iloc[:, step].dropna().values
    bo_step = pd.DataFrame(max_obs_paths_bo).iloc[:, step].dropna().values
    fig_3, ax = plt.subplots(1, figsize=(5, 5))
    ax.hist(human_step, alpha=0.5, label='Humans')
    ax.hist(bo_step, alpha=0.5, label='BO')
    ax.legend(loc='upper left')
    ax.set_xlabel('Max Observed Yield')
    ax.set_ylabel('Count')
    plt.show()
    print('\nStudents t-test:\n')
    t_test(human_step, bo_step, sample_vars='equal')
    print('\nWelchs t-test:\n')
    t_test(human_step, bo_step, sample_vars='unequal')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now, by carrying out this analysis at each step in the optimization for the null hypotheses $μ_{Human} = μ_{BO}$, $μ_{Lower} = μ_{BO}$, and $μ_{Upper} = μ_{BO}$ we can statistically compare convergence behavior of each case.

    **Compute statistics for the entire path:**
    """
    )
    return


@app.cell
def _(
    max_obs_paths_bo,
    max_obs_paths_h,
    max_obs_paths_lower,
    max_obs_paths_upper,
    path_stats,
    pd,
):
    humans_vs_bo = path_stats(pd.DataFrame(max_obs_paths_h), pd.DataFrame(max_obs_paths_bo), steps=range(14))
    upper_vs_bo = path_stats(pd.DataFrame(max_obs_paths_upper), pd.DataFrame(max_obs_paths_bo), steps=range(20))
    lower_vs_bo = path_stats(pd.DataFrame(max_obs_paths_lower), pd.DataFrame(max_obs_paths_bo), steps=range(20))

    print('\n', humans_vs_bo.head())
    print('\n', upper_vs_bo.head())
    print('\n', lower_vs_bo.head())
    return humans_vs_bo, lower_vs_bo, upper_vs_bo


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Now we can plot the p-values for each case. A p-value < 0.05 indicates that we can reject the null hypothesis. That is, the performance of humans and machines are statistically different. For the raw data and lower bound, we found that on average after the 5th batch of experiments the optimizers performance is better than that of the humans. In contrast, for the upper bound we found that there is not a statistically significant difference between the two average curves. Thus, we conclude that in the optimization of reaction 3, Bayesian reaction optimization on average outperforms human experts, tracing the unrealistic absolute upper bound of the games recorded data.

    **Plot a summary figure:**
    """
    )
    return


@app.cell
def _(humans_vs_bo, lower_vs_bo, plt, upper_vs_bo):
    fig_4, ax3 = plt.subplots(1, figsize=(5, 5))
    ax3.fill_between([0, 20], [0.05, 0.05], color='black', alpha=0.2, label='p-value < 0.05')
    ax3.plot(range(1, 15), humans_vs_bo['p-value'], linewidth=3, label='Raw Data')
    ax3.plot(range(1, 21), lower_vs_bo['p-value'], '--', linewidth=3, label='Lower Bound')
    ax3.plot(range(1, 21), upper_vs_bo['p-value'], '-.', linewidth=3, label='Upper Bound')
    ax3.set_ylim(-0.05, 1)
    ax3.set_ylabel('p-value ($\\mu_{human}=\\mu_{BO}$)')
    ax3.set_xlabel('Experiment Batch')
    ax3.set_xlim(0, 21)
    ax3.legend(loc='upper right')
    plt.show()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
