import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Preface

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The Jupyter Guide to Linear Algebra covers many of the core topics that would appear in an introductory course on linear algebra, together with several applications.  The guide also provides a brief introduction to the Python programming langauge, with focus on the portions that are relevant to linear algebra computations, as well as some general guidance on programming.  In its current form, the Jupyter Guide to Linear Algebra is not intended to be a replacement for a textbook in a traditional university course, although it should prove useful to students in such courses.  The guide may also be useful to those who have some knowledge of linear algebra, and wish to learn how to carry out computations in Python, or experienced Jupyter users that would like to learn a bit about Linear Algebra.

        Features:

        - Development of a module to be used along side of the Jupyter Guide to Linear Algebra, or independently.
        - Exercises aimed at exploring linear algebra concepts, as well as exercises to practice writing Python code.
        - Instruction on the basic use of NumPy, SciPy, and Matplotlib.

        The code supplied in Jupyter Guide to Linear Algebra performs all operations numerically.  We do not make use of SymPy to perform symbolic computations, which hide the practical challenges of accuracy and stability.  We will discover and acknowledge roundoff error very early in the guide, but we will not provide a detailed error analysis, nor advanced algorithms to minimize its impact.  We will also not be overly concerned with computational efficiency.  This resource is meant for a first course in linear algebra, and we will leave the larger challenges of numerical linear algebra for a second course.   


        The Jupyter Guide to Linear Algebra presents the mathematics in a relatively informal way.  We offer explanations in place of proofs and do not follow a traditional model of definitions and theorems.  Instead our main objective is to present methods to solve problems, demonstrate how to carry out calculationsm, provide the basic terminology of linear algebra, and examine ways in which the abstract ideas can be used in practical ways.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### How to use this Guide

        This work is currently distributed in two forms, either a Jupyter book, or a collection of Jupyter notebooks.  The real purpose of this material is for the reader to engage with the material by experimenting and trying out computations for themselves.  In the case that you are working with the notebooks and have access to a JupyterHub, you are all set to begin.  The Jupyter book form of the LA Guide may found as a pdf or static html site.  If you are reading this on a website, you can download the source notebooks individually by clicking the icon in the upper right.  Alternatively, you can download all source notebooks at URL here.  If you are reading a pdf version, please find the digital version online.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Prerequisites or Corequisites

        An introductory programming course, or an equivalent programming experience would be useful to the reader.  While we do introduce all of the syntax needed to write Python scripts, we do not delve into the details of traditional topics in an introductory programming course (data types, logic, iteration, and complex data structures).  We address the features of Python that we use most frequetly, and provide only brief mentions of those that are tangential to our goals.

        Some applications make use of Calculus and will be noted.  Full appreciation of these sections will require an introductory Calculus course.

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
