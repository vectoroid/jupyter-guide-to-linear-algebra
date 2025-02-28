import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Common Python Error Messages

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        One typical error is something called a **Name Error**.  This error gets displayed when Python doesn't recognize a command or variable in the code.  Run the following cell to see an example.
        """
    )
    return


@app.cell
def _(resutl):
    _result = 128 * 17
    print(resutl)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We made a typo in the name of the variable name and the Python interpreter doesn't reconize the variable it is instructed to print.  This same error will arise if you try to use the $\texttt{result}$ in another calculation.

        """
    )
    return


@app.cell
def _(resutl):
    _result = 128 * 17
    next_result = resutl / 10
    return (next_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        There are other common ways to receive the Name Error.

        1. If we begin working on a notebook somewhere in the middle, then that means all of the code cells at the start did not get executed.  If one of the cells we are working on references a variable from an earlier cell, we will get the Name Error.
        2. If we try to use function from a module, but forget to actually import the module, we will get the Name Error.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
