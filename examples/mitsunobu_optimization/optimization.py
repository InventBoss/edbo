import marimo

__generated_with = "0.14.10"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Bayesian Optimization of a Mitsunobu Reaction

    Jason Stevens & Ben Shields
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Mitsunobu Reaction""")
    return


@app.cell
def _():
    from rdkit.Chem import AllChem, Draw

    rxn = AllChem.ReactionFromSmarts('BrC1=CNC2=CC(C(OC)=O)=CC=C21.OCC3=CC=CC=C3>>BrC4=CN(CC5=CC=CC=C5)C6=CC(C(OC)=O)=CC=C64')
    Draw.ReactionToImage(rxn, subImgSize=(250, 250))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Defining the Reaction Space

    We included 7 optimization parameters for this reaction:
    1. azadicarboxylate (6)
    2. phosphine (12)
    3. solvent (5)
    4. substrate concentration (4)
    5. azadicarboxylate equiv. (5)
    6. phosphine equiv. (5)
    7. temperature (5)

    ## Reaction Encoding

    In keeping with our paper, we are going to use DFT encoding for all categorical chemical parameters and numerical encoding for continuous parameters. 

    For the continuous parameters we will define our own discrete grid. On the technical side this is because we are going to explicitly take the $argmax$ of our acquisition function over the search space rather than run an optimization algorithm. I will add support for optimization algorithms down the line though this is practical up to millions of data points. However, it does require some care in the selection of grid points ($10^9$ points in the search space will take quite a while to compute). So, take some time to think about what is reasonable (i.e. there is no point in including 30 and 30.5 as temperatures if your stir plate varies by (+/-) 1 degree. Here we use a course but reasonable grid which gives a search space of $6 * 12 * 5 * 4 * 5 * 5 * 5 = 180,000$ possible points. 

    Let's start by loading the precomputed DFT chemical descriptor matrices. We will use an EDBO data utility (```edbo.utils.Data```) to clean up the descriptor sets a bit by removing some unwanted descriptors.
    """
    )
    return


@app.cell
def _():
    # Imports

    import pandas as pd
    from edbo.utils import Data

    # Load DFT descriptor CSV files computed with auto-qchem using pandas
    # Instantiate a Data object

    azadicarbs = Data(pd.read_csv('./examples//mitsunobu_optimization/descriptors/azadicarboxylate_boltzmann_dft.csv'))
    phosphines = Data(pd.read_csv('./examples//mitsunobu_optimization/descriptors/phosphine_boltzmann_dft.csv'))
    solvents = Data(pd.read_csv('./examples//mitsunobu_optimization/descriptors/solvent_dft.csv'))

    # Use Data.drop method to drop descriptors containing some unwanted keywords

    for data in [azadicarbs, phosphines, solvents]:
        data.drop(['file_name', 'entry', 'vibration', 'correlation', 'Rydberg', 
                   'correction', 'atom_number', 'E-M_angle', 'MEAN', 'MAXG', 
                   'STDEV'])
    return azadicarbs, pd, phosphines, solvents


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Here are what the descriptor sets look like before preprocessing.""")
    return


@app.cell
def _(azadicarbs):
    azadicarbs.data
    return


@app.cell
def _(phosphines):
    phosphines.data
    return


@app.cell
def _(solvents):
    solvents.data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Next, we define dictionaries for the reaction parameters, type of encoding, and external descriptors. Note that the corresponding keys need to match across the three dictionaries. We are going to pass these objects to the main BO class which will automatically build a search space.

    The reaction space generation method can also be used to: 
    -	One-hot-encode descriptors: ```encoding={…, ‘parameter’:‘ohe’, …}```
    -	Compute Mordred fingerprints from SMILES strings: ```encoding={…, ‘parameter’:‘mordred’, …}```
    -	Look up chemical names from the NIH database: ```encoding={…, ‘parameter’:’resolve’, …}```
    """
    )
    return


@app.cell
def _(azadicarbs, phosphines, solvents):
    # Parameters in reaction space

    components = {'azadicarboxylate':'DFT',                             # DFT descriptors
                  'phosphine':'DFT',                                    # DFT descriptors
                  'solvent':'DFT',                                      # DFT descriptors
                  'substrate_concentration':[0.05, 0.10, 0.15, 0.20],   # Discrete grid of concentrations
                  'azadicarb_equiv':[1.1, 1.3, 1.5, 1.7, 1.9],          # Discrete grid of equiv.
                  'phos_equiv':[1.1, 1.3, 1.5, 1.7, 1.9],               # Discrete grid of equiv.
                  'temperature':[5, 15, 25, 35, 45]}                    # Discrete grid of temperatures

    # Encodings - if not specified EDBO will automatically use OHE

    encoding = {'substrate_concentration':'numeric',                    # Numerical encoding
                'azadicarb_equiv':'numeric',                            # Numerical encoding
                'phos_equiv':'numeric',                                 # Numerical encoding
                'temperature':'numeric'}                                # Numerical encoding

    # External descriptor matrices override specified encoding

    dft = {'azadicarboxylate':azadicarbs.data,                          # Unprocessed descriptor DataFrame
           'phosphine':phosphines.data,                                 # Unprocessed descriptor DataFrame
           'solvent':solvents.data}                                     # Unprocessed descriptor DataFrame
    return components, dft, encoding


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Instantiating EDBO

    EDBO has methods for automatically generating the search space from the components above. For example, when you instantiate a new ```edbo.bro.BO_express``` object (which is a more user friendly super class of ```edbo.bro.BO```) it will automatically parse parameters (e.g., look up chemical names from NIH database), compute descriptors (e.g., OHE or Mordred), and generate an encoded combinatorial set of all experiments. After generating the encoded search space it will also take care of any necessary pre-processing steps (e.g., normalization, de-correlation, etc.).

    The main BO class has a lot of keyword arguments (some of which are shown below). The preset parameters should work pretty well for most problems which means that all you really need to do in order to start an optimization is:
    -	Pass the search space
    -	Choose a batch size

    In this example, ```edbo.bro.BO_express``` actually attempts to automatically choose model priors to match the data type. In order to be consistent with the paper, here we will reset the priors directly after instantiation.
    """
    )
    return


@app.cell
def _(components, dft, encoding):
    from edbo.bro import BO_express

    # BO object

    bo = BO_express(components,                                 # Reaction parameters
                    encoding=encoding,                          # Encoding specification
                    descriptor_matrices=dft,                    # DFT descriptors
                    acquisition_function='EI',                  # Use expectation value of improvement
                    init_method='rand',                         # Use random initialization
                    batch_size=10,                              # 10 experiments per round
                    target='yield')                             # Optimize yield

    # BO_express actually automatically chooses priors
    # We can reset them manually to make sure they match the ones from our paper

    from gpytorch.priors import GammaPrior

    bo.lengthscale_prior = [GammaPrior(2.0, 0.2), 5.0]
    bo.outputscale_prior = [GammaPrior(5.0, 0.5), 8.0]
    bo.noise_prior = [GammaPrior(1.5, 0.5), 1.0]
    return BO_express, bo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The parameterized domain is held as a ```pandas.DataFrame``` in the objective module.""")
    return


@app.cell
def _(bo):
    bo.obj.domain.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""For ```edbo.bro.BO_express```, information about the reaction not directly relevant to the BO algorithm is held in a reaction data container. For example, the corresponding unencoded domain points.""")
    return


@app.cell
def _(bo):
    bo.reaction.get_experiments([0,1,2,3,4])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""The reaction object also has other useful methods. For example, visualize the chemical structures corresponding to a given experiment.""")
    return


@app.cell
def _(bo):
    bo.reaction.visualize(0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Initialization

    In keeping with our paper, we will start the optimization campaign by randomly selecting experiments. We made this choice when we instantiated the optimizer when we specified ```init_method='rand'```. We can then export the proposed experiments to a CSV file and fill in the results once we run the experiments.
    """
    )
    return


@app.cell
def _(bo):
    bo.init_sample(seed=0)             # Initialize
    bo.export_proposed('init.csv')     # Export design to a CSV file
    bo.get_experiments()               # Print selected experiments
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Since we ran this optimization over a few days it was advantageous to save the BO instance for later. That way we didn't have to rebuild the reaction space and reinstantiate the BO object if we close the kernel. The ```edbo.bro.BO.save()``` method will pickle the optimizer for later.""")
    return


@app.cell
def _(bo):
    bo.save()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Once we have saved the optimizer workspace we can load it by calling ```edbo.bro.BO.load()```.""")
    return


@app.cell
def _(BO_express):
    bo_1 = BO_express()
    bo_1.load()
    return (bo_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Bayesian Optimization Loop

    Now that we have defined the reaction space and optimizer, we can iteratively run the optimizer to select experiments, evaluate the experiments in the lab, and read in the results as CSV files. In this case, we ran the experiments selected at random above. We can load the data using the ```edbo.bro.BO_express.add_results``` method.
    """
    )
    return


@app.cell
def _(bo_1):
    bo_1.add_results('./examples//mitsunobu_optimization/results/init.csv')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Next in the BO loop, we fit the surrogate model (by default ```edbo.models.GP_Model```) and optimize the acquisition function (we selected expected improvement at instantiation with ```acquisition_function='EI'```) to select the next experiments to run. This is accomplished with the ```edbo.bro.BO.run``` method. This method returns the encoded domain points for the selected experiments and stores them in ```edbo.bro.BO.proposed_experiments```.""")
    return


@app.cell
def _(bo_1):
    bo_1.run()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Once we have used the run method, we can carry out miscellaneous analysis using some built in methods. For example, we can track the convergence of the optimizer (obviously not interesting here).""")
    return


@app.cell
def _(bo_1):
    bo_1.plot_convergence()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can see how well the surrogate model fits the available data.""")
    return


@app.cell
def _(bo_1):
    bo_1.model.regression()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""We can also crack open the optimizer and check out what is happening under the hood by writing some custom analysis functions. For example, we can plot a 1D projection of the parallel acquisition function's choices (For EI batching is accplished with ```edbo.acq_func.Kriging_believer```).""")
    return


@app.cell
def _(bo_1):
    import numpy as np
    import matplotlib.pyplot as plt

    def plot_kb_projections(n=2):
        """
        Plot 1D projection of Kriging believer parallel batch selection algorithm.
        """
        fig, ax = plt.subplots(len(bo_1.acq.function.projections[:n]), 1, figsize=(12, n * 12 / 5))
        for i, p in enumerate(bo_1.acq.function.projections[:n]):
            ax[i].plot(range(len(p)), p, color='C' + str(i))
            ax[i].plot([np.argmax(p)], p[np.argmax(p)], 'X', markersize=10, color='black')
            ax[i].set_xlabel('X')
            ax[i].set_ylabel('EI')
            ax[i].set_title('Kriging Believer Projection:' + str(i))
        plt.tight_layout()
        plt.show()
    plot_kb_projections()
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Back to the main task – optimizing this Mitsunobu reaction. Now we can export the next round of experiments as a CSV file, run the experiments, fill in the CSV, and iterate.""")
    return


@app.cell
def _(bo_1):
    bo_1.export_proposed('round0.csv')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Let's say we are happy with this workflow and want to carry out the same analysis in each round. We can wrap each of these steps up into a simple function.""")
    return


@app.cell
def _(bo_1):
    def workflow(export_path):
        """
        Function for our BO pipeline.
        """
        bo_1.run()
        bo_1.plot_convergence()
        bo_1.model.regression()
        bo_1.export_proposed(export_path)
    return (workflow,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""And run the same analysis for each round.""")
    return


@app.cell
def _(bo_1, workflow):
    bo_1.add_results('./examples/mitsunobu_optimization/results/round0.csv')
    workflow('./examples/mitsunobu_optimization/round1.csv')
    return


@app.cell
def _(bo_1, workflow):
    bo_1.add_results('./examples/mitsunobu_optimization/results/round1.csv')
    workflow('./examples/mitsunobu_optimization/round2.csv')
    return


@app.cell
def _(bo_1, workflow):
    bo_1.add_results('./examples/mitsunobu_optimization/results/round2.csv')
    workflow('./examples/mitsunobu_optimization/round3.csv')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Awesome, so in 4 rounds of 10 experiments we have hit essentially quantitative yield for this reaction. Once you have the data you can do whatever analysis your heart desires. Let’s just check out the conditions for the best yielding reactions.""")
    return


@app.cell
def _(bo_1, pd):
    results = pd.DataFrame(columns=bo_1.reaction.index_headers + ['yield'])
    for path in ['init', 'round0', 'round1', 'round2']:
        results = pd.concat([results, pd.read_csv('./examples/mitsunobu_optimization/results/' + path + '.csv', index_col=0)], sort=False)
    results = results.sort_values('yield', ascending=False)
    results.head()
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Plot a bar chart for parameter frequency in the top 5 yielding reactions.""")
    return


@app.cell
def _(plt, results):
    fig, ax = plt.subplots(1, len(results.columns.values[:-1]), figsize=(30,5))

    for i, feature  in enumerate(results.columns.values[:-1]):
        results[feature].iloc[:5].value_counts().plot(kind="bar", ax=ax[i]).set_title(feature)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Visualize the reagents in the top 5 yielding reactions using ```edbo.chem_utils.ChemDraw```.""")
    return


@app.cell
def _(results):
    from edbo.chem_utils import ChemDraw

    for col in results.iloc[:,:3].columns.values:
        print('\nComponent:', col, '\n')
        cdx = ChemDraw(results[col].iloc[:5].drop_duplicates())
        cdx.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Total compute time on my laptop (i7-6500U): 8 min 38 s""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
