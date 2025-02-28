import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Introduction to Jupyter
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        [Jupyter](https://jupyter.org/) is a web application that allows users to create and share documents called notebooks.  Jupyter notebooks can contain text, equations, and graphics, as well as live Python code with which the user can interact.  [Python](https://www.python.org/) is a very popular computer language for a wide range of applications, including scientific computing, and is known for being a language that is easy to learn for novice programmers.    

        The purpose of this introduction is to provide enough instruction on the use of Jupyter and Python for the reader to successfully navigate and engage with the Jupyter Guide to Linear Algebra.  There are a vast number of 
        resources available online for those that wish to learn more about these topics.  Some references are supplied at the end of the section. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Cells 

        The Jupyter Guide to Linear Algebra is made up of a number of Jupyter notebooks.  Each notebook consists of a number of cells.  Every cell is either a **markdown cell**, or a **code cell**.  The markdown cells mostly contain text and [LaTex] instructions.  When executed, these cells generate all of the formatted text and formulas that you see as you read through the guide.  The code cells contain instructions, written in Python, to be executed by the Notebook.  Code cells will not produce much visual output to the Notebook unless they contain instructions to do so.    

        All cells in the Notebooks, markdown and code, must be executed in order to produce output and results.  To execute a cell, select the cell and press 'Shift + Enter'.  You can select a cell by clicking it, or by moving up/down with the arrow keys.  In order to fully engage with the notebooks, you will also want to edit cells to change the outputs.  To change a cell you will have to enter **edit mode**.  When not in edit mode, the notebook is said to be in **command mode**.  

        **Important: Your keyboard will do different things depending on the mode.**  

        Let's see how to change modes right now, in this very cell.  When a cell is selected, there will be a colored bar on the left border of the cell.  The bar will be green in edit mode and blue in command mode.  To enter command mode we press 'Enter'.  To exit command mode without executing the cell we press 'Esc'.  To execute the cell press 'Shift + Enter'.  

        Try it!  Edit the line below to include your name.

        This copy of the Jupyter Guide to Linear Algebra belongs to ...


        Although all cells can be edited and executed, there are some distinctions in the two types of cells.

        - **If there is anything amiss in the code cell, a Python error will be displayed**.  These errors might be difficult to decipher at first, but with a little practice in reading the messages you will be able to understand the common errors that arise.  **If there is anything amiss in a markdown cell, there will be no error message**, although the displayed text or formula may not look correct.  The markdown cells are not executed in the same sense as the code cells.  


        - **All markdown cells are executed when you open the notebook, but code cells are not**.  This means that when you first open a notebook, the formatted text and formulas produced by the markdown cells will be displayed.  Clicking on the markdown cells will not do anything other than select the cell, and pressing 'Shift + Enter' will not display anything new since the output is already displayed.  By contrast, the output from the code cells might be displayed, but the code in the cell does not get executed by the Notebook until the cell has been run explicity by selecting the cell and pressing 'Shift + Enter'.  It will be quite common for the code in one cell to depend on code in earlier cell.  **If the code cells are executed in the wrong order, we may get error messages or other unexpected results.** 

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Menus

        At the top of the Notebook there are several dropdown menus, some of which are quite common and easy to understand.  

        - The *File* menu contains familiar options that create or open other Notebooks, save or rename the current Notebook, download the current Notebook, or close the Notebook.  


        - The *Edit* menu contains several options for manipulating cells within the Notebook.  The common cell operations include cutting, copying, adding, deleting, merging, etc.  


        - The *View* menu contains some display options.


        - The *Cell* menu offers different options to run collections of cells, rather than individual cells.  It also contains the *Cell Type* option, which toggles cells between code and markdown.


        - The *Kernel* menu interacts with the Python interpreter.  It is occasionally necessary to restart the kernal to reload a module, or interrupt the kernal if something goes wrong, but most of the time interaction with the kernal is not needed for routine operations.


        - The *Widget* menu interacts with widgets, which are dynamic elements that can be added to Notebooks.  We will not make use of widgets in our Notebooks.


        -  The *Help* menu offers documentation about the Notebook environment as well as links to external references.

        Located just below the menus is a tool bar which contains buttons for the most common operations.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Resources

        - [A Gallery of Interesting Jupyter Notebooks](https://github.com/jupyter/jupyter/wiki) 

        - [The Programming Historian](https://programminghistorian.org/en/lessons/jupyter-notebooks)

        - [Jupyter Notebook Documentation](https://jupyter-notebook.readthedocs.io/en/stable/notebook.html)
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
