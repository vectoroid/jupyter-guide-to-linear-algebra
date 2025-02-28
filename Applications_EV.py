import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Applications of Eigenvalues and Eigenvectors
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With the concepts related to eigenvalues and eigenvectors in place, we return to examine [Discrete Dynamical Systems](Applications_LT.ipynb).

        ### Infectious disease model

        For the $SIRS$ model of infectious disease, we had the following discrete dynamical system.


        $$
        \begin{equation}
        X_t = \left[ \begin{array}{r} s_t \\ i_t \\ r_t  \end{array}\right] =
        \left[ \begin{array}{rrr} 0.95 & 0 & 0.15 \\ 0.05 & 0.80 & 0 \\ 0 & 0.20 & 0.85 \end{array}\right]
        \left[ \begin{array}{r} s_{t-1} \\ i_{t-1} \\ r_{t-1}  \end{array}\right]=
        AX_{t-1}
        \end{equation}
        $$

        Given an initial condition $X_0$, we know that $X_n = A^nX_0$.  We are interested in determining the behavior of the system for large $n$.  We might now recognize that this calculation is exactly the same as the Power Method from the previous section, and therefore expect that the sequence of vectors produced should tend toward the eigenvector corresponding to the largest eigenvalue.

        In this case the components of the vector have individual meaning, so let's calculate the first $20$ iterations and plot $s_t$, $i_t$, and $r_t$ to get a sense of how they are changing.  For this calculation we store each vector $X_t$ as a column in an array named $\texttt{results}$. 
        """
    )
    return


app._unparsable_cell(
    r"""
    import matplotlib.pyplot as plt
    import numpy as np
    %matplotlib inline

    A = np.array([[0.95, 0, 0.15],[0.05,0.8,0],[0,0.2,0.85]])

    ## T is final time
    T = 20
    ## X at time 0
    X = np.array([[0.95],[0.05],[0]])

    ## The first column of results contains the initial values 
    results = np.copy(X)

    for i in range(T):
        X = A@X
        results = np .hstack((results,X))

    ## t contains the time indices 0, 1, 2, ..., T
    t = np.linspace(0,T,T+1)
    ## s, i, r values are the rows of the results array
    s = results[0,:]
    i = results[1,:]
    r = results[2,:]

    fig,ax = plt.subplots()

    ## The optional label keyword argument provides text that is used to create a legend
    ax.plot(t,s,'b+',label=\"Susceptible\");
    ax.plot(t,i,'rx',label=\"Infectious\");
    ax.plot(t,r,'g+',label=\"Recovered\");

    ax.set_ylim(0,1.1)
    ax.grid(True)
    ax.legend();
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Based on the calculation it appears that the state of the population has reached an *equilibrium* after 20 weeks.  In the equilibrium state, each category of the population, $S$, $I$, and $R$, have as many individuals entering the category as leaving it.  In terms of the matrix equation, if $X$ is the vector that contains the equilibrium values of $s_t$, $i_t$, and $r_t$, then $X$ must be a solution to the equation $AX=X$, since $X_{t-1}=X_t$ when the components are not changing.  The equation $AX=X$ implies that $X$ is an eigenvector of $A$ corresponding to an eigenvalue of one.  

        The equation $AX=X$ represents a linear system, even though it is not written in the typical form.  We can rearrange it to look more familiar by using the tools of matrix algebra.  The key is to recognize that $X$ can be written as $IX$, where $I$ is the identity matrix.  This gives $AX=IX$, which can be rewritten as $(A-I)X = 0$.  This matrix equation is the type with which we have been working throughout the previous chapters.

        Before attempting to solve the system $(A-I)X = 0$, we first recognize it as a [Homogeneous System](Linear_Independence.ipynb) and apply some reasoning based on our knowledge of linear systems.  We know that the solution set of a homogeneous system certainly contains the trivial solution $X=0$.  Therefore if there is also a nonzero solution to $(A-I)X=0$, the system must have an infinite number of solutions since linear systems always have zero, exactly one, or an infinite number of solutions.  If the system $(A-I)X=0$ does not have a unique solution there must be a free variable, which means that one of the columns of the matrix $A-I$ will not have a pivot. In this case the direct methods of solution based on $LU$ or $QR$ factorizations will fail.  We must go back to the basic row operations and calculate the RREF of the augmented matrix $[(A-I)|0]$.     
        """
    )
    return


@app.cell
def _(A, np):
    import laguide as lag

    I = np.eye(3)
    ZERO = np.zeros((3,1))
    augmented_matrix = np.hstack((A-I,ZERO))
    reduced_matrix = lag.FullRowReduction(augmented_matrix)
    print(reduced_matrix)
    return I, ZERO, augmented_matrix, lag, reduced_matrix


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In the reduced system for the equilibrium values $s$, $i$, and $r$, we can take $r$ as the free variable and write $s=3r$ and $i=0.75r$.  For any value of $r$, a vector of the following form is an eigenvector for $A-I$, corresponding to the eigenvalue one.

        $$
        \begin{equation}
        r\left[ \begin{array}{r} 3 \\ 0.75 \\ 1  \end{array}\right]
        \end{equation}
        $$

        Although this eigenvector is not unique, there is one more constraint on our equilibrium values that we have not yet addressed.  The variables $s$, $i$, and $r$ represent percentages of the population in each category, therefore we must also require that $s+i+r = 1$ in order for the results to be meaningful in the application.  This gives the equation $3r + 0.75r + r = 1$, which gives the unique equilibrium values of $r = 4/19$, $s=12/19$, and $i=3/19$.  If we carry out a large number of iterations, we see that the computed values are very close to the theoretical equilibrium values. 
        """
    )
    return


@app.cell
def _(A, np):
    ## T is final time
    T = 100
    ## X at time 0
    X = np.array([[0.95],[0.05],[0]])

    for i in range(T):
        X = A@X

    print("Expected values of s, i, r:")
    print("r =",12/19,"i =", 3/19,"i =", 4/19,'\n')
    print("Computed values of s, i, r at time ",T,":")
    print(X)
    return T, X, i


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The results agree nicely with our analysis, but there are two remaining questions:

        1. Why is it that $s_t+i_t+r_t = 1$ for every $t$ in the calculation, even though this constraint does not appear to be enforced?
        2. What are the results if we change the initial population composition given by $X_0$?

        The answer to the first question is simple, but not immediately obvious.  If we add the three equations that define $s_t$, $i_t$, and $r_t$, we find that $s_t+i_t+r_t = s_{t-1} + i_{t-1} + r_{t-1}$.  This means that as long as the three equations are true, the sum $s_t+i_t+r_t$ does not change as we carry the iteration forward in time.  Since we start with the components of $X_0$ summing to one, the components of $X_t$ will sum to one for all other $t$.

        In order to answer the second question, we assume that we start with an arbitrary $X_0$, although we will require that the components of $X_0$ to be nonnegative and add to one so that the results are meaningful in the application.  As we did when considering the Power Method, we write $X_0$ as a linear combination of $\{V_1, V_2, V_3\}$, the eigenvectors of $A$.  This makes  it easy to express $X_t = A^tX_0$ in terms of the eigenvectors as well, just as did before.

        $$
        \begin{eqnarray*}
        X_0 & = & c_1V_1 + c_2V_2 + c_3V_3 \\
        X_t & = & c_1\lambda_1^tV_1 + c_2\lambda_2^tV_2 + c_3\lambda_3^tV_3 
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Based on the results of our calculations, we expect that $\lambda_1 = 1$ is the largest eigenvalue.

        $$
        \begin{equation}
        X_t  =  c_1V_1 + c_2\lambda_2^tV_2 + c_3\lambda_3^tV_3 
        \end{equation}
        $$

        If $|\lambda_2|< 1$ and $|\lambda_3|< 1$ then the second and the third terms will approach zero as $t$ gets large and $X_t$ will approach the direction of $V_1$.  We use SciPy to verify that $\lambda_2$ and $\lambda_3$ are in fact a pair of complex numbers with magnitude less than one.
        """
    )
    return


@app.cell
def _(A, np):
    import scipy.linalg as sla
    EVAL, EVEC = sla.eig(A)

    print(EVAL,'\n')
    ## The absolute function finds the absolute value of real numbers, or magnitude of complex numbers.
    print(np.absolute(EVAL[1]))
    print(np.absolute(EVAL[2]))
    return EVAL, EVEC, sla


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Therefore unless $c_1 = 0$, the vectors $X_t$ approach the direction of $V_1$.  The case where $c_1 = 0$ corresponds to initial conditions with negative entries, which are not relevant in the application.  The conclusion is that $X_t$ will approach the equilibrium state for all meaningful initial configurations of the population. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercises

        **Exercise 1:** Experiment with a range of initial conditions in the infectious disease model to provide evidence that an equilibrium state is reached for all meaningful initial states. 
        """
    )
    return


@app.cell
def _():
    ## Code solution here.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** Perform an analysis similar to that in the example for the following infectious disease model.  In this model the rate at which individuals move from the Recovered category to the Susceptible category is less than that in the example.  Make a plot similar to that in the example and also calculate the theoretical equilibrium values for $s$, $i$, and $r$.

        $$
        \begin{equation}
        X_t = \left[ \begin{array}{r} s_t \\ i_t \\ r_t  \end{array}\right] =
        \left[ \begin{array}{rrr} 0.95 & 0 & 0.05 \\ 0.05 & 0.80 & 0 \\ 0 & 0.20 & 0.95 \end{array}\right]
        \left[ \begin{array}{r} s_{t-1} \\ i_{t-1} \\ r_{t-1}  \end{array}\right]=
        AX_{t-1}
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _():
    ## Code solution here.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Navigating webpages

        In the dynamical system $X_t = AX_{t-1}$, the entries in the state vector $X_t$ can represent probabilities that a web browser is on a particular page after $t$ steps of random browsing defined by the matrix $A$, which is built using the link structure among the web pages.  The matrix and initial condition previously considered are given below.

        $$
        \begin{equation}
        A = \left[ \begin{array}{ccccc} 
        0 & 0.05 & 0.05 & 0.45 & 0.25 \\ 
        0.45 & 0 & 0.85 & 0.45 & 0.25  \\
        0.05 & 0.05 & 0 & 0.05 & 0.25  \\
        0.05 & 0.05 & 0.05 & 0 & 0.25  \\
        0.45 & 0.85 & 0.05 & 0.05 & 0  \\
        \end{array}\right]
        \hspace{2cm}
        X_0 = \left[ \begin{array}{ccccc} 0 \\ 0 \\ 0 \\ 1 \\ 0
        \end{array}\right]
        \end{equation}
        $$

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Just as with the infectious disease model, we can determine an equilibrium probability distribution by finding the vector $X$ that solves $AX = X$.  
        """
    )
    return


@app.cell
def _(lag, np):
    A = np.array([[0, 0.05, 0.05, 0.45, 0.25], [0.45, 0, 0.85, 0.45, 0.25], [0.05, 0.05, 0, 0.05, 0.25], [0.05, 0.05, 0.05, 0, 0.25], [0.45, 0.85, 0.05, 0.05, 0]])
    I_1 = np.eye(5)
    A_I = A - I_1
    reduced_matrix_1 = lag.FullRowReduction(A_I)
    print(reduced_matrix_1)
    return A, A_I, I_1, reduced_matrix_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this case $x_5$ is a free variable, and we have $x_4 = x_3 = 0.33444562x_5$, $x_2 = 0.89261338x_5$, and $x_1 = 0.46185348x_5$.  Since these values represent a collection of probabilities, we require that their sum is one.  We can calculate the approximate value of $x_5$ and the corresponding values of the other entries.
        """
    )
    return


@app.cell
def _(reduced_matrix_1):
    x_5 = 1 / (1 - reduced_matrix_1[0][4] - reduced_matrix_1[1][4] - reduced_matrix_1[2][4] - reduced_matrix_1[3][4])
    x_4 = -x_5 * reduced_matrix_1[3][4]
    x_3 = -x_5 * reduced_matrix_1[2][4]
    x_2 = -x_5 * reduced_matrix_1[1][4]
    x_1 = -x_5 * reduced_matrix_1[0][4]
    equilibrium_probabilities = (x_1, x_2, x_3, x_4, x_5)
    print(equilibrium_probabilities)
    return equilibrium_probabilities, x_1, x_2, x_3, x_4, x_5


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To verify the calculations we can again compare these values to those we get if we compute $X_t$ for large $t$.
        """
    )
    return


@app.cell
def _(A, np):
    T_1 = 100
    X_1 = np.array([[0], [0], [0], [1], [0]])
    for i_1 in range(T_1):
        X_1 = A @ X_1
    print(X_1)
    return T_1, X_1, i_1


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
