import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Diagonalization
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this section, we explain the effect of matrix multiplication in terms of eigenvalues and eigenvectors.  This will allow us to write a new matrix factorization, known as diagonalization, which will help us to further understand matrix multiplication.  We also introduce a SciPy method to find the eigenvalues and eigenvectors of arbitrary square matrices.

        Since the action of an $n\times n$ matrix $A$ on its eigenvectors is easy to understand, we might try to understand the action of a matrix on an arbitrary vector $X$ by writing $X$ as a linear combination of eigenvectors.  This will always be possible *if the eigenvectors form a basis for* $\mathbb{R}^n$.  Suppose $\{V_1, V_2, ..., V_n\}$ are the eigenvectors of $A$ and they do form a basis for $\mathbb{R}^n$.  Then for any vector $X$, we can write 
        $X = c_1V_1 + c_2V_2 + ... c_nV_n$.  The product $AX$ can easily be computed then by multiplying each eigenvector component by the corresponding eigenvalue.

        $$
        \begin{eqnarray*}
        AX & = & A(c_1V_1 + c_2V_2 + ... c_nV_n) \\
           & = & c_1AV_1 + c_2AV_2 + ... c_nAV_n \\
           & = & c_1\lambda_1V_1 + c_2\lambda_2V_2 + ... c_n\lambda_nV_n
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 1:  Matrix representation of shear

        We first consider a matrix that represents a shear along the line $x_1=-x_2$.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rr} 2 & -1  \\ -1 & 2 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import laguide as lag
    _S = np.array([[1, 1], [1, -1]])
    _D = np.array([[1, 0], [0, 3]])
    S_inverse = lag.Inverse(_S)
    _A = _S @ _D @ S_inverse
    print(_A, '\n')
    X = np.array([[3], [1]])
    print(_A @ X)
    return S_inverse, X, lag, np


@app.cell
def _():
    # Cell tags: hide-input
    # '%matplotlib inline\nimport matplotlib.pyplot as plt\n\nfig, ax = plt.subplots()\nx=np.linspace(-6,6,100)\n\noptions = {"head_width":0.1, "head_length":0.2, "length_includes_head":True}\n\nax.arrow(0,0,3,1,fc=\'b\',ec=\'b\',**options)\nax.arrow(0,0,5,-1,fc=\'r\',ec=\'r\',**options)\nax.plot(x,-x,ls=\':\')\n\nax.set_xlim(-1,6)\nax.set_ylim(-3,4)\nax.set_aspect(\'equal\')\nax.set_xticks(np.arange(-1,7,step = 1))\nax.set_yticks(np.arange(-3,5,step = 1))\n\nax.text(2,1,\'$X$\')\nax.text(3,-1.2,\'$AX$\')\nax.text(0.8,-2.5,\'$x_1=-x_2$\')\n\nax.axvline(color=\'k\',linewidth = 1)\nax.axhline(color=\'k\',linewidth = 1)\n\nax.grid(True,ls=\':\')' command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The effect of multiplication by $A$ can be understood through its eigenvalues and eigenvectors.  The first eigenvector of $A$ lies in the direction of the line $x_1=-x_2$.  We label it $V_1$ and observe that it gets scaled by a factor of 3 when multiplied by $A$.  The corresponding eigenvalue is $\lambda_1 = 3$.

        $$
        \begin{equation}
        AV_1 = \left[ \begin{array}{rr} 2 & -1  \\ -1 & 2 \end{array}\right]
        \left[ \begin{array}{r} 1 \\ -1 \end{array}\right] =
        \left[ \begin{array}{r} 3 \\ -3 \end{array}\right] = 3V_1
        \end{equation}
        $$

        The other eigenvector, which we label $V_2$, is orthogonal to the line $x_1=-x_2$.  This vector is left unchanged when multiplied by $A$, which implies that $\lambda_2 = 1$.


        $$
        \begin{equation}
        AV_2 = \left[ \begin{array}{rr} 2 & -1  \\ -1 & 2 \end{array}\right]
        \left[ \begin{array}{r} 1 \\ 1 \end{array}\right] = V_2
        \end{equation}
        $$

        Since $\{V_1, V_2\}$ form a basis for $\mathbb{R}^2$, any other vector in $\mathbb{R}^2$ can be written as a linear combination of these vectors.  Let's take the following vector $X$ as an example.


        $$
        \begin{equation}
        X = \left[ \begin{array}{r} 3 \\ 1 \end{array}\right]
        \end{equation}
        $$

        To express $X$ in terms of the eigenvectors, we have to solve the vector equation $c_1V_1 + c_2V_2 = X$.
        """
    )
    return


@app.cell
def _(X, lag, np):
    # Solve the matrix equation BC = X.  C is the vector of coefficients.
    B = np.array([[1, 1],[-1,1]])
    C = lag.SolveSystem(B,X)
    print(C)
    return B, C


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's look at a visual representation of the linear combination $X = V_1 + 2V_2$.
        """
    )
    return


@app.cell
def _(np, options, plt, x):
    # Cell tags: hide-input
    _fig, _ax = plt.subplots()
    _ax.arrow(0, 0, 3, 1, fc='k', ec='k', **options)
    _ax.arrow(0, 0, 1, -1, fc='b', ec='b', **options)
    _ax.arrow(0, 0, 2, 2, fc='b', ec='b', **options)
    _ax.plot(x, -x, ls=':')
    _ax.set_xlim(-1, 6)
    _ax.set_ylim(-3, 4)
    _ax.set_aspect('equal')
    _ax.set_xticks(np.arange(-1, 7, step=1))
    _ax.set_yticks(np.arange(-3, 5, step=1))
    _ax.text(1, 2, '$2V_2$')
    _ax.text(2, 1, '$X$')
    _ax.text(0.1, -1, '$V_2$')
    _ax.axvline(color='k', linewidth=1)
    _ax.axhline(color='k', linewidth=1)
    _ax.grid(True, ls=':')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can now understand, and compute, the product $AX$ by evaluating the products of $A$ with its eigenvectors. 

        $$
        \begin{equation}
        AX = A(V_1 + 2V_2) = AV_1 + 2AV_2 = \lambda_1V_1 + 2\lambda_2V_2 = 3V_1 + 2V_2
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(np, options, plt, x):
    # Cell tags: hide-input
    _fig, _ax = plt.subplots()
    _ax.arrow(0, 0, 3, 1, fc='k', ec='k', **options)
    _ax.arrow(0, 0, 3, -3, fc='r', ec='r', **options)
    _ax.arrow(0, 0, 2, 2, fc='r', ec='r', **options)
    _ax.arrow(0, 0, 5, -1, fc='r', ec='r', **options)
    _ax.plot(x, -x, ls=':')
    _ax.set_xlim(-1, 6)
    _ax.set_ylim(-3, 4)
    _ax.set_aspect('equal')
    _ax.set_xticks(np.arange(-1, 7, step=1))
    _ax.set_yticks(np.arange(-3, 5, step=1))
    _ax.text(1, 2, '$2V_2$')
    _ax.text(2, 1, '$X$')
    _ax.text(3, -1.2, '$AX$')
    _ax.text(1, -2, '$3V_1$')
    _ax.axvline(color='k', linewidth=1)
    _ax.axhline(color='k', linewidth=1)
    _ax.grid(True, ls=':')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Computation of eigenvectors with SciPy

        We now demonstrate how to compute eigenvalues and eigenvectors for any square matrix using the function $\texttt{eig}$ from the SciPy $\texttt{linalg}$ module.  This function accepts an $n\times n$ array representing a matrix and returns two arrays, one containing the eigenvalues, the other the eigenvectors.  We examine the usage by supplying our projection matrix as the argument. 
        """
    )
    return


@app.cell
def _(np):
    import scipy.linalg as sla
    _A = np.array([[0.2, -0.4], [-0.4, 0.8]])
    print(_A)
    print('\n')
    evalues, evectors = sla.eig(_A)
    print(evalues)
    return evalues, evectors, sla


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The array of eigenvalues contains two entries that are in the format of $\alpha + \beta j$, which represents a complex number.  The symbol $j$ is used for the imaginary unit.  The value of $\beta$ is zero for both of the eigenvalues, which means that they are both real numbers.  The results confirm our conclusions that the eigenvalues are 0 and 1.

        Next let's look at the array of eigenvectors.  We can slice the array into columns to give us convenient access to the vectors.
        """
    )
    return


@app.cell
def _(evectors):
    print(evectors)
    V_1 = evectors[:, 0:1]
    _V_2 = evectors[:, 1:2]
    return (V_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We may not recognize the eigenvectors as those we found previously, but recall that eigenvectors are not unique.  The $\texttt{eig}$ function scales all the eigenvectors to unit length, and we arrive at the same result if we scale our choice of eigenvector.
        """
    )
    return


@app.cell
def _(lag, np):
    V = np.array([[-2],[-1]])
    print(V)
    print('\n')
    print(V/lag.Magnitude(V))
    return (V,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2:  Verifying eigenvectors using matrix multiplication 

        Let's try another example with a $3\times 3$ matrix.

        $$
        \begin{equation}
        B = \left[ \begin{array}{rrr} 1 & 2 & 0  \\ 2 & -1 & 4 \\  0 & 3 & 1\end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(np, sla):
    B_1 = np.array([[1, 2, 0], [2, -1, 4], [0, 3, 1]])
    evalues_1, evectors_1 = sla.eig(B_1)
    print(evalues_1)
    print('\n')
    print(evectors_1)
    V_1_1 = evectors_1[:, 0:1]
    _V_2 = evectors_1[:, 1:2]
    V_3 = evectors_1[:, 2:3]
    E_1 = evalues_1[0]
    E_2 = evalues_1[1]
    E_3 = evalues_1[2]
    return B_1, E_1, E_2, E_3, V_1_1, V_3, evalues_1, evectors_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We don't have the exact eigenvalues, but we can check that $BV_i - \lambda_iV_i = 0$ for $i=1, 2, 3$, allowing for the usual roundoff error.  
        """
    )
    return


@app.cell
def _(B_1, E_1, V_1_1):
    print(B_1 @ V_1_1 - E_1 * V_1_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Instead of $BV_i - \lambda_iV_i = 0$ for each $i$, we can package the calculations into a single matrix multiplication.  If $S$ is the matrix with columns $V_i$, then $BS$ is the matrix with columns $BV_i$.  This matrix should be compared to the matrix that has $\lambda_iV_i$ as its columns.  To construct this matrix we use a diagonal matrix $D$, that has the $\lambda_i$ as its diagonal entries.  The matrix product $SD$ will then have columns $\lambda_iV_i$.

        $$
        \begin{equation}
        SD = \left[ \begin{array}{c|c|c} & & \\ V_1 & V_2 & V_3 \\ & & \end{array} \right]
        \left[ \begin{array}{rrr} \lambda_1 & 0 & 0  \\ 0 & \lambda_2 & 0 \\  0 & 0 & \lambda_3 \end{array}\right]= 
        \left[ \begin{array}{c|c|c} & & \\ \lambda_1V_1 & \lambda_2V_2 & \lambda_3V_3 \\ & & \end{array}\right]
        \end{equation}
        $$

        We can now simply check that $BS-SD = 0$.
        """
    )
    return


@app.cell
def _(B_1, evalues_1, evectors_1, np):
    _S = evectors_1
    _D = np.zeros((3, 3), dtype='complex128')
    for i in range(3):
        _D[i, i] = evalues_1[i]
    print(_D)
    print('\n')
    print(B_1 @ _S - _S @ _D)
    return (i,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Diagonal factorization

        The calculation that we used to verify the eigenvalues and eigenvectors is also very useful to construct another important matrix factorization.  Suppose that $A$ is an $n\times n$ matrix, $D$ is a diagonal matrix with the eigenvalues of $A$ along its diagonal, and $S$ is the $n\times n$ matrix with the eigenvectors of $A$ as its columns.  We have just seen that $AS=SD$.  If $S$ is invertible, we may also write $A=SDS^{-1}$, which is known as the **diagonalization** of $A$.

        The diagonalization of $A$ is important because it provides us with a complete description of the action of $A$ in terms of its eigenvectors.  Consider an arbitrary vector $X$, and the product $AX$, computed by using the three factors in the diagonalization.

        - $S^{-1}X$ computes the coordinates of $X$ in terms of the eigenvectors of $A$.
        - Multiplication by $D$ then simply scales each coordinate by the corresponding eigenvalue.
        - Multiplication by $S$ gives the results with respect to the standard basis.

        This understanding does not provide a more efficient way of computing the product $AX$, but it does provide a much more general way of understanding the result of a matrix-vector multiplication.  As we will see in the next section, the diagonalization also provides a significant shortcut in computing powers of $A$.  

        It should be noted that diagonalization of an $n\times n$ matrix $A$ is not possible when the eigenvectors of $A$ form a linearly dependent set.  In that case the eigenvectors do not span $\mathbb{R}^n$, which means that not every $X$ in $\mathbb{R}^n$ can be written as a linear combination of the eigenvectors.  In terms of the computation above, $S$ will not be invertible exactly in this case.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercises

        **Exercise 1:** Find the diagonalization of the matrix from **Example 1**.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rr} 2 & -1  \\ -1 & 2 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(np):
    _A = np.array([[2, -1], [-1, 2]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** Find the diagonalization of the following matrix.

        $$
        \begin{equation}
        B = \left[ \begin{array}{rrr} 2 & 0 & 0  \\ 3 & -2 & 1 \\  1 & 0 & 1\end{array}\right]
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
        **Exercise 3:** Write a function that accepts an $n\times n$ matrix $A$ as an argument, and returns the three matrices $S$, $D$, and $S^{-1}$ such that $A=SDS^{-1}$.  Make use of the $\texttt{eig}$ function in SciPy.
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
        **Exercise 4:** Construct a $3\times 3$ matrix that is not diagonal and has eigenvalues 2, 4, and 10. 
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
        **Exercise 5:** Suppose that $C = QDQ^{-1}$ where

        $$
        \begin{equation}
        Q = \left[ \begin{array}{c|c|c} & & \\ U & V & W \\ & & \end{array} \right] \hspace{2cm} D = \left[ \begin{array}{rrr} 0 & 0 & 0  \\ 0 & 3 & 0 \\  0 & 0 & 5 \end{array}\right]
        \end{equation}
        $$

        $(a)$ Give a basis for $\mathcal{N}(C)$

        $(b)$ Find all the solutions to $CX = V + W$
        """
    )
    return


@app.cell
def _():
    ## Code solutions here.
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
