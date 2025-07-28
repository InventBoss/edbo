import marimo

__generated_with = "0.14.13"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(
        r"""
    # Bayesian Optimization with Suzuki Reactions 

    This notebook demonstrates Bayesian Optimization with a simpler dataset example. The notebook is designed so that inputting your own dataset is made simpler.

    Dataset came from:

    *Suzuki–Miyaura cross-coupling optimization
    enabled by automated feedback*
    """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import io
    import matplotlib.pyplot as plt
    from edbo.bro import BO
    from edbo.feature_utils import one_hot_encode, build_experiment_index
    return BO, build_experiment_index, io, one_hot_encode, pd


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Inputing Data

    To input your own data, please choose the `.cvs` file that contains your dataset. Otherwise, the example suzuki reaction dataset will be used.

    **Requirements for User Data:**

    - Must have an entry column
    """
    )
    return


@app.cell
def _(mo):
    # Ask for optional user-specified database
    user_file = mo.ui.file(filetypes=[".csv"], multiple=False)
    user_file
    return (user_file,)


@app.cell
def _(io, pd, user_file):
    # Data is either from the user, or our own suzuki_data.csv file

    suzuki_data = ""
    if user_file.value:
        file_like = io.BytesIO(user_file.value[0].contents)
        suzuki_data = pd.read_csv(file_like)
    else:
        suzuki_data = pd.read_csv("data/suzuki_data.csv")

    suzuki_data.dropna(subset=suzuki_data.columns)
    suzuki_data
    return (suzuki_data,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Specifying Columns to One Hot Encode

    We can't put numbers as values into Bayesian Optimization. So, we use [One Hot Encoding](https://www.geeksforgeeks.org/machine-learning/ml-one-hot-encoding/) (OHE) to fix this problem. By default, we specify the one column that needs encoding for our default database, but you can input your own columns that need OHE **(comma separate your values please)**.
    """
    )
    return


@app.cell
def _(mo):
    # Get user input
    column_input = mo.ui.text_area(value="catalyst")
    column_input
    return (column_input,)


@app.cell
def _(column_input):
    # Convert user input to list of strings
    raw_column_list = column_input.value
    column_list = [x.strip() for x in raw_column_list.split(",") if x.strip()]
    return (column_list,)


@app.cell
def _(column_list, one_hot_encode, suzuki_data):
    # One hot encode all of the specified values
    column_data_list = []

    for column_i in column_list:
        column_ohe_data = one_hot_encode(suzuki_data[f"{column_i}"], name=f"{column_i}")
        column_data_list.append(column_ohe_data)
    return (column_data_list,)


@app.cell
def _(column_list, suzuki_data):
    # First requirement for building the experiment index (technically entry list is first, but this is the first computed one) is the data from the columns that are being one hot encoded.
    index_list = []

    for column_j in column_list:
        index_list.append(suzuki_data[f"{column_j}"])
    return (index_list,)


@app.cell
def _(column_data_list):
    # Next, we get the reference tables that will swap out the values in that column with new, ohe columns.
    lookup_table_list = []

    for column_data in column_data_list:
        lookup_table_list.append(column_data)
    return (lookup_table_list,)


@app.cell
def _(
    build_experiment_index,
    column_list,
    index_list,
    lookup_table_list,
    suzuki_data,
):
    # Build the matrix that will be used as the input values for BO and the table to reference to
    experiment_index = build_experiment_index(
        suzuki_data["entry"],
        index_list,
        lookup_table_list,
        column_list,
    )

    for column_k in column_list:
        experiment_index = experiment_index.drop(f"{column_k}", axis=1)

    # Potentially helpful input data for the model (can worsen BO)
    # experiment_index["catalyst_mol_percent"] = suzuki_data["catalyst_mol_percent"]
    # experiment_index["temperature"] = suzuki_data["temperature"]
    # experiment_index["time"] = suzuki_data["time"]

    # # Output data/objectives from the model
    # experiment_index["yield"] = suzuki_data["yield"]
    # experiment_index["TON"] = suzuki_data["TON"]

    # # Drop the OHE catalyst data from the table
    # experiment_index = experiment_index.drop("catalyst", axis=1)
    return (experiment_index,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Adding Numerical Columns

    For all the columns we want as input data for the model that **doesn't** require one hot encoding, we can just add them to the experiment index.
    """
    )
    return


@app.cell
def _(mo):
    # Get user input
    extra_column_input = mo.ui.text_area()
    extra_column_input
    return (extra_column_input,)


@app.cell
def _(extra_column_input):
    # Convert user input to list of strings
    raw_extra_column_list = extra_column_input.value
    extra_column_list = [x.strip() for x in raw_extra_column_list.split(",") if x.strip()]
    return (extra_column_list,)


@app.cell
def _(experiment_index, extra_column_list, suzuki_data):
    for extra_column in extra_column_list:
        experiment_index[f"{extra_column}"] = suzuki_data[f"{extra_column}"]
    experiment_index
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Specifying Model Objective

    Here you specify which column you want the Bayesian Optimization to optimize for.
    """
    )
    return


@app.cell
def _(mo):
    # Get user input
    objective_input = mo.ui.text_area(value="TON")
    objective_input
    return (objective_input,)


@app.cell
def _(objective_input):
    # Convert user input to string
    raw_objective_list = objective_input.value
    objective = [x.strip() for x in raw_objective_list.split(",") if x.strip()]
    objective = objective[0]
    return (objective,)


@app.cell
def _(experiment_index, objective, suzuki_data):
    # Create a domain with the objective for the model
    final_index = experiment_index.copy()
    final_index[f"{objective}"] = suzuki_data[f"{objective}"].values
    final_index = final_index.dropna(subset=experiment_index.columns)
    final_index
    return (final_index,)


@app.cell
def _(BO, final_index, objective):
    from gpytorch.priors import GammaPrior

    bo = BO(
        exindex=final_index,
        domain=final_index.drop(f"{objective}", axis=1),
        batch_size=5,
        acquisition_function="EI",
        fast_comp=True,
        target=f"{objective}",
        init_method="kmeans",
        noise_constraint=1e+1,
        matern_nu=2.5,
        lengthscale_prior=[
            GammaPrior(2.0, 1.0),
            5.0,
        ],
        outputscale_prior=[
            GammaPrior(5.0, 0.5),
            8.0,
        ],
        noise_prior=[
            GammaPrior(1.0, 1.0),
            1.0,
        ],
    )
    bo.simulate(iterations=15, seed=0)
    return GammaPrior, bo


@app.cell
def _(bo):
    bo.plot_convergence()
    return


@app.cell
def _(bo, suzuki_data):
    results = []
    results.append(bo.obj.results_input()["entry"].values)

    suzuki_data[suzuki_data["entry"].isin(results[0])]
    return


@app.cell
def _(BO, GammaPrior, final_index, objective, pd):
    from edbo.plot_utils import average_convergence, plot_avg_convergence

    simulation_results = []
    index_results = []

    for simulation_loop in range(30):
        bo_simulation = BO(
            exindex=final_index,
            domain=final_index.drop(f"{objective}", axis=1),
            batch_size=5,
            acquisition_function="EI",
            fast_comp=True,
            target=f"{objective}",
            init_method="kmeans",
            noise_constraint=1e-2,
            matern_nu=2.5,
            lengthscale_prior=[
                GammaPrior(2.0, 1.0),
                5.0,
            ],
            outputscale_prior=[
                GammaPrior(5.0, 0.5),
                8.0,
            ],
            noise_prior=[
                GammaPrior(1.0, 1.0),
                1.0,
            ],
        )

        bo_simulation.init_seq.visualize = False
        bo_simulation.simulate(iterations=15, seed=simulation_loop)

        simulation_results.append(bo_simulation.obj.results_input()[f'{objective}'].values)
        index_results.append(bo_simulation.obj.results_input()["entry"].values)

    simulation_results = pd.DataFrame(simulation_results)
    index, mean, std = average_convergence(simulation_results, 5)
    return index_results, plot_avg_convergence, simulation_results


@app.cell
def _(index_results, plot_avg_convergence, simulation_results, suzuki_data):
    plot_avg_convergence(simulation_results, 5)
    ran_experiments = suzuki_data[suzuki_data["entry"].isin([item for sublist in index_results for item in sublist])]
    ran_experiments
    return (ran_experiments,)


@app.cell
def _(column_list, ran_experiments, simulation_results):
    batch_size = 5

    # Make the file name "column1_column2_..._column_n.cvs"
    file_name = "_".join(column_list) 

    ran_experiments.to_csv(path_or_buf=f"./{file_name}_experiments.csv", index=False)
    simulation_results.to_csv(path_or_buf=f"./{file_name}_results_batch_size={batch_size}.csv")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
