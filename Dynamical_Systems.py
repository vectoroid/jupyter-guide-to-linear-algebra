import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Discrete Dynamical Systems
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As we saw in the previous chapter, it is often useful to describe a structure that has multiple components with a single vector.  If that structure is changing in time due to some process, it is typical to refer to the vector as a **state vector** since it describes the *state* of the structure at some particular time.  It is quite common to model such dynamic processes at discrete times and use linear transformations to model the evolution of the state vector from one time to the next.

        Let's suppose that we aim to describe sequence of vectors at times $t=1, 2, 3,$... with state vectors $X_1$, $X_2$, $X_3$.... at those times.  We propose to calculate the state vector $X_t$ based only on the previous state vector $X_{t-1}$.  If we model the transition from $X_{t-1}$ to $X_t$ with a linear transformation, then there is a matrix such that $X_t = AX_{t-1}$.  This sort of model is known as a **discrete dynamical system** and is used in many areas from economics to biology.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Infectious Disease Model

        In this example we consider a basic model of an infectious disease that is spreading within a population.  A well known family of models for this scenario is known as the $SIR$ models.  The acronym comes from a basic modeling assumption that the population is divided into three categories: Susceptible, Infectious, and Recovered.  As the disease, spreads a portion of the Susceptible individuals become Infectious, and a portion of Infectious individuals become Recovered.  We will consider a small variation in the model which assumes that a portion of Recovered individuals return to the Susceptible category.  This variation would be a more accurate model for a disease which can be contracted multiple times.

        We suppose that the population is completely homogeneous in all regards, so that all individuals in a given category have the same probabilities to move to the next category.

        To model real-world epidemics, it is necessary to estimate some parameters that specify how quickly individuals move among the categories.  These parameters will be important in making any predictions with the model.  For our demonstration, we will create an example.  Let us suppose that our state vectors describe the population at time intervals of 1 week, and that every week, 5% of the Susceptible population becomes Infectious, and 20% of the Infectious population becomes Recovered.  We also suppose that 15% of the Recovered population again becomes Susceptible every week.

        If we let $s_t$, $i_t$, and $r_t$ represent the percentage of the three categories of the population at time $t$, we can write equations based on the modeling assumptions that allows us to calculate the values based on $s_{t-1}$, $i_{t-1}$, and $r_{t-1}$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $$
        \begin{eqnarray}
        s_t & = & 0.95s_{t-1} \hspace{2cm} + 0.15r_{t-1} \\
        i_t & = & 0.05s_{t-1}  + 0.80i_{t-1} \\
        r_t & = & \hspace{2cm} 0.20i_{t-1} + 0.85r_{t-1} 
        \end{eqnarray}
        $$

        Now we can define $X_t$ as the vector with components $s_t$, $i_t$, and $r_t$, so that the equations can be written using matrix multiplication.

        $$
        \begin{equation}
        X_t = \left[ \begin{array}{r} s_t \\ i_t \\ r_t  \end{array}\right] =
        \left[ \begin{array}{rrr} 0.95 & 0 & 0.15 \\ 0.05 & 0.80 & 0 \\ 0 & 0.20 & 0.85 \end{array}\right]
        \left[ \begin{array}{r} s_{t-1} \\ i_{t-1} \\ r_{t-1}  \end{array}\right]=
        AX_{t-1}
        \end{equation}
        $$

        The linear transformation $L:\mathbb{R}^3 \to \mathbb{R}^3$ defined by this matrix multiplication maps the state of the 
        population at time $t-1$, to the state of the population at time $t$.  For an example let's label the initial state vector $X_0$, and consider what happens if initially 5% of the population is infective, and the other 95% of the population is susceptible. 

        $$
        \begin{equation}
        X_0 = \left[ \begin{array}{r} 0.95 \\ 0.05 \\ 0  \end{array}\right] 
        \end{equation}
        $$

        We compute $X_1 = AX_0$.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    A = np.array([[0.95, 0, 0.15], [0.05, 0.8, 0], [0, 0.2, 0.85]])
    _X = np.array([[0.95], [0.05], [0]])
    _X = A @ _X
    print(_X)
    return A, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Applying the transformation again gives $X_2 = AX_1 = A^2X_0$, the state of the population at time $t=2$.  In general, $n$ repeated applications of the transformation yield $X_n = A^nX_0$, the state of the population $n$ weeks into the future.  Let's compute $X_{50}$ as an example.
        """
    )
    return


@app.cell
def _(A, np):
    _X = np.array([[0.95], [0.05], [0]])
    for _t in range(50):
        _X = A @ _X
    print(_X)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In such models attention is typically focused on the ultimate behaviour of the state vector.  We want to know if the composition of the population reaches an equilibrium, or continues to change.  If it reaches an equilibrium, can we calculate it directly, instead of applying the matrix multiplication a large number of times?  We will address these questions in Chapter 5, when we learn more about computing $A^n$ efficiently. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Navigating Webpages

        Browsing webpages can also be modeled using a discrete dynamical system.  In this model, the entries of the state vector $X_t$, represent the probability that the browser is on a particular page at time $t$.  Advancing in time represents the browser moving from one page to another.  Many pages contain links to other pages.  The model assumes that the browser is more likely to follow a link to a new page rather than navigating to a new unlinked page.  The goal is to build a linear transformation $L$, defined by a matrix multiplication, which maps the vector of probabilities at time $t-1$ to the vector of probabilities at time $t$ based on the link structure of the pages.  Such a model was the foundation of the PageRank algorithm, which is the basis of Google's very successful search engine.

        To build the required matrix $A$, the entries $a_{ij}$ are set to the probability that the browser navigates to page $i$ from page $j$.  For fixed column $j$, the entries represent a  probability distribution that describes location of the browser at the next step.  The entries in each column therefore must add to one.  Let's make some additional assumptions to complete the model.   

        - The browser follows a link with probability 0.8.
        - All links on a page are equally likely to be followed.
        - A browser not following a link is equally likely to reach any new page at the next step.
        - The browser always changes pages at each step

        These assumptions completely define the matrix.  If we let $n$ be the number of webpages in the model, $A$ will be an $n\times n$ matrix with zeros on its main diagonal due to the last assumption.  If we focus on a particular page $j$, There are a total of $n-1$ nonzero entries since $a_{jj} = 0$.  If page $j$ has links to $k$ different pages, then $a_{ij} = 0.8/k + 0.2/(n-1)$ if $i$ is the index of one of the link linked pages, and $a_{ij} = 0.2/(n-1)$ if $i$ is the index of one of the unlinked pages.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Here is an example of what such a matrix would look like with $n=5$, page 1 linking to pages 2 and 5, page 2 linking to page 5, page 3 linking to page 2, page 4 linking to pages 1 and 2, and page 5 linking to all other pages.

        $$
        \begin{equation}
        \left[ \begin{array}{ccccc} 
        0 & 0.05 & 0.05 & 0.45 & 0.25 \\ 
        0.45 & 0 & 0.85 & 0.45 & 0.25  \\
        0.05 & 0.05 & 0 & 0.05 & 0.25  \\
        0.05 & 0.05 & 0.05 & 0 & 0.25  \\
        0.45 & 0.85 & 0.05 & 0.05 & 0  \\
        \end{array}\right]
        \end{equation}
        $$

        To complete the model example, we specify $X_0$.  In this case the interpretation is easiest if we take $X_0$ to be a vector with a one entry equal to one, and all other entries equal to zero.  For example, the following choice of $X_0$ means that the browser starts the navigation at page 4.

        $$
        \begin{equation}
        X_0 = \left[ \begin{array}{ccccc} 0 \\ 0 \\ 0 \\ 1 \\ 0
        \end{array}\right]
        \end{equation}
        $$

        Applying the transformation once gives $X_1 = AX_0$ which gives a vector that contains the probabilities that the browser is at each of the pages.
        """
    )
    return


@app.cell
def _(np):
    A_1 = np.array([[0, 0.05, 0.05, 0.45, 0.25], [0.45, 0, 0.85, 0.45, 0.25], [0.05, 0.05, 0, 0.05, 0.25], [0.05, 0.05, 0.05, 0, 0.25], [0.45, 0.85, 0.05, 0.05, 0]])
    X_0 = np.array([[0], [0], [0], [1], [0]])
    X_1 = A_1 @ X_0
    print(X_1)
    return A_1, X_0, X_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Applying the transformation $n$ times gives $X_n$, the vector that contains the probabilities of the browser being at each of the pages after $n$ steps of navigation.

        """
    )
    return


@app.cell
def _(A_1, np):
    _X = np.array([[0], [0], [0], [1], [0]])
    for _t in range(20):
        _X = A_1 @ _X
    print(_X)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As with the population model, the interest lies in predicting the probabilities in $X_n$ when $n$ is large, and thus determining what pages a browser is more likely to visit after this sort of random navigation.
        """
    )
    return


@app.cell
def _(np):
    A_2 = np.array([[0.2, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.2, 0.075, 0.075], [0.075, 0.2, 0.075, 0.2, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075], [0.2, 0.2, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075], [0.2, 0.2, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075], [0.2, 0.2, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075], [0.2, 0.2, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075], [0.2, 0.2, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075], [0.2, 0.2, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075], [0.2, 0.2, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075, 0.075], [0.075, 0.075, 0.075, 0.075, 0.075, 0.2, 0.2, 0.075, 0.075, 0.075]])
    print(A_2)
    return (A_2,)


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
