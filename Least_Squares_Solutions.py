import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Least Squares Solutions
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this section we address the problem of inconsistent systems, and the common resolution known as the least squares solution.

        In the case that $AX=B$ is inconsistent, there is no vector $X$ such that the two vectors $AX$ and $B$ are the same.  A natural idea then is to choose a vector $X$ such that $AX$ and $B$ are as close as possible.  To do this we define the error vector $E=AX-B$, and choose the $X$ that minimizes $||E||$.  This choice of $X$ is known as the **least squares solution** to the system $AX=B$, and we will assign it the symbol $\hat{X}$.  The direct approach to this formulation of the problem requires the use of calculus to minimize $||E||$, but we will take a different approach that only requires inner products and projections.  We will first give the solution to the least squares problem and provide only a visual explanation.  After some examples, we will give a more rigorous explanation.  

        Recall that if the system $AX=B$ is inconsistent, the vector $B$ is not in $\mathcal{C}(A)$, the column space of $A$.  **The error vector $E=AX-B$ has minimum magnitude exactly when it is orthogonal to $\mathcal{C}(A)$**.  We can easily visualize this in the case that $\mathcal{C}(A)$ as a one dimensional subspace of $\mathbb{R}^2$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 1:  Two inconsistent equations

        Let's consider a simple example of an inconsistent system $AX=B$, where $A$ is a $2\times 2$ matrix.

        $$
        \begin{equation}
        \left[\begin{array}{rr} 3 & 9 \\ 1 & 3 \end{array}\right]X = \left[\begin{array}{r} 2 \\ 4 \end{array}\right]    
        \end{equation}
        $$

        In this example, the second column of $A$ is a multiple of the first, so $\mathcal{C}(A)$ is a one-dimensional space that contains only multiples of these vectors.  Since $B$ is not a multiple of either column, $B$ is not in $\mathcal{C}(A)$.
        """
    )
    return


@app.cell
def _():
    # Cell tags: hide-input
    # '%matplotlib inline\nimport numpy as np\nimport matplotlib.pyplot as plt\n\nx=np.linspace(-6,6,100)\n\nfig, ax = plt.subplots()\n\noptions = {"head_width":0.1, "head_length":0.2, "length_includes_head":True}\n\nax.arrow(0,0,2,4,fc=\'b\',ec=\'b\',**options)\nax.plot(x,x/3,ls=\':\')\n\n\nax.set_xlim(-2,6)\nax.set_ylim(-2,6)\nax.set_aspect(\'equal\')\nax.set_xticks(np.arange(-2,6,step = 1))\nax.set_yticks(np.arange(-2,6,step = 1))\n\nax.text(0.8,2.5,\'$B$\')\nax.text(4.5,2,\'$\\mathcal{C}(A)$\')\n\nax.axvline(color=\'k\',linewidth = 1)\nax.axhline(color=\'k\',linewidth = 1)\n\nax.grid(True,ls=\':\')' command supported automatically in marimo
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's look at the relationship between $AX$, $B$, and $E$ for an arbitrary vector $X$ in $\mathbb{R}^2$.
        """
    )
    return


@app.cell
def _(np, plt, x):
    # Cell tags: hide-input
    _fig, _ax = plt.subplots()
    _options = {'head_width': 0.1, 'head_length': 0.2, 'length_includes_head': True}
    _ax.arrow(0, 0, 2, 4, fc='b', ec='b', **_options)
    _ax.arrow(0, 0, 4.5, 1.5, fc='r', ec='r', **_options)
    _ax.arrow(2, 4, 2.5, -2.5, fc='r', ec='r', **_options)
    _ax.plot(x, x / 3, ls=':')
    _ax.set_xlim(-2, 6)
    _ax.set_ylim(-2, 6)
    _ax.set_aspect('equal')
    _ax.set_xticks(np.arange(-2, 6, step=1))
    _ax.set_yticks(np.arange(-2, 6, step=1))
    _ax.text(0.8, 2.5, '$B$')
    _ax.text(4.5, 2, '$\\mathcal{C}(A)$')
    _ax.text(3, 0.5, '$AX$')
    _ax.text(3.3, 3.1, '$E=AX-B$')
    _ax.axvline(color='k', linewidth=1)
    _ax.axhline(color='k', linewidth=1)
    _ax.grid(True, ls=':')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can see that $||E||$ is a measure of the distance between $B$ and a vector in $\mathcal{C}(A)$, and that $||E||$ will be minimized if we choose $X$ so that $E$ is orthogonal to $\mathcal{C}(A)$.  This is the least squares solution we refer to as $\hat{X}$.  The closest vector in $\mathcal{C}(A)$ to $B$ is the *orthogonal projection* of $B$ onto $\mathcal{C}(A)$.  We will use the notation $\hat{B}$ for this projection so that we now have $A\hat{X}=\hat{B}$. 
        """
    )
    return


@app.cell
def _(np, plt, x):
    # Cell tags: hide-input
    _fig, _ax = plt.subplots()
    _options = {'head_width': 0.1, 'head_length': 0.2, 'length_includes_head': True}
    _ax.arrow(0, 0, 2, 4, fc='b', ec='b', **_options)
    _ax.arrow(0, 0, 3, 1, fc='b', ec='b', **_options)
    _ax.arrow(2, 4, 1, -3, fc='b', ec='b', **_options)
    _ax.plot(x, x / 3, ls=':')
    _ax.set_xlim(-2, 6)
    _ax.set_ylim(-2, 6)
    _ax.set_aspect('equal')
    _ax.set_xticks(np.arange(-2, 6, step=1))
    _ax.set_yticks(np.arange(-2, 6, step=1))
    _ax.text(0.8, 2.5, '$B$')
    _ax.text(4.5, 2, '$\\mathcal{C}(A)$')
    _ax.text(3, 0.5, '$A\\hat{X}=\\hat{B}$')
    _ax.text(2.7, 3.1, '$E=A\\hat{X}-B$')
    _ax.axvline(color='k', linewidth=1)
    _ax.axhline(color='k', linewidth=1)
    _ax.grid(True, ls=':')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this example, we can use the [projection formula](Inner_Products.ipynb) from the beginning of the chapter to calculate $E$ and $\hat{B}$

        $$
        \begin{equation}
        E = \left[ \begin{array}{r} 1 \\ -3 \end{array} \right] \hspace{2cm} 
        \hat{B} = \left[ \begin{array}{r} 3 \\ 1 \end{array} \right]
        \end{equation}
        $$

        To find $\hat{X}$, we solve the system $A\hat{X}=\hat{B}$

        $$
        \begin{equation}
        \left[\begin{array}{rr} 3 & 9 \\ 1 & 3 \end{array}\right]\hat{X} = \left[\begin{array}{r} 3 \\1 \end{array}\right]    
        \end{equation}
        $$

        In this system, the second equation is just a multiple of the first.  This means that $x_1+3x_2 = 1$ is the only constraint on the unknowns, and that we can take $x_2$ to be a free variable.  If we assign $x_2$ a parameter, we can describe all possible solutions as follows.

        $$
        \begin{eqnarray*}
        x_1 & = & 1-3s \\
        x_2 & = & s
        \end{eqnarray*}
        $$

        In this particular example, which is meant to show a clear picture of $E$, there is not a unique least squares solution since many vectors solve the matrix equation $A\hat{X}=\hat{B}$.  Inconsistent systems that arise in applications typically do have a unique least squares solution.   
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Normal equations

        If we assume for now that the error $E=AX-B$ will have minimum magnitude when it is orthogonal to the column space of $A$, we can use our knowledge of the fundamental subspaces to solve the least squares problem when the column space is of higher dimension.  Recall from a previous section that any vector orthogonal to $\mathcal{C}(A)$ must lie in $\mathcal{N}(A^T)$, the null space of $A^T$.  This means that $A^TE=0$ for the least squares solution.  Filling in $E=AX-B$ gives us the system $A^T(AX-B)=0$, which can be written as $A^TAX=A^TB$.  This system of equations is referred to as the **normal equations**.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2:  Four inconsistent equations

        Although the method of least squares can be applied to any inconsistent system, it is usually associated with systems that have more equations than unknowns.  These systems are called overdetermined, and here is one such example.

        $$
        \begin{eqnarray*}
        2x_1 + x_2 & = & 0 \\
        2x_1 - x_2 & = & 2 \\
        3x_1 + 2x_2 & = & 1 \\
        5x_1 + 2x_2 & = & -2
        \end{eqnarray*}
        $$

        Let $A$ be the $4\times 2$ coefficient matrix, and let $B$ be the vector of the right-hand sides of the equations.  To verify that the system is indeed inconsistent, we can compute the RREF of the augmented matrix $\left[A|B\right]$.
        """
    )
    return


@app.cell
def _(np):
    import laguide as lag

    A_augmented = np.array([[2, 1, 0],[2, -1, 2],[3, 2, 1],[5,2, -2]])
    A_reduced = lag.FullRowReduction(A_augmented)
    print(A_reduced)
    return A_augmented, A_reduced, lag


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The pivot in the last column indicates that the system is inconsistent.  The two columns of $A$ are linearly independent and form a basis for $\mathcal{C}(A)$, which is a two-dimensional subspace of $\mathbb{R}^4$, but $B$ does not lie in this subspace.  

        To find the least squares solution, we will construct and solve the normal equations, $A^TAX = A^TB$.
        """
    )
    return


@app.cell
def _(np):
    A = np.array([[2, 1],[2, -1],[3, 2],[5,2]])
    B = np.array([[0],[2],[1],[-2]])

    # Construct A^TA
    N_A = A.transpose()@A
    # Construct A^TA
    N_B = A.transpose()@B
    print(N_A,'\n')
    print(N_B)
    return A, B, N_A, N_B


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The normal equations are a $2\times 2$ system, which can be solved using elimination.


        $$
        \begin{eqnarray*}
        42x_1 + 16x_2 & = & -3 \\
        16x_2 + 10x_2 & = & -4 
        \end{eqnarray*}
        $$

        """
    )
    return


@app.cell
def _(A, B, N_A, N_B, lag):
    X_hat = lag.SolveSystem(N_A, N_B)
    print(X_hat)
    print('\n')
    _E = A @ X_hat - B
    print('Magnitude of minimum error is:', lag.Magnitude(_E))
    return (X_hat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this example, there is a unique vector $\hat{X}$ that minimizes $||E||$.  Remember from the previous example that $\hat{X}$ is the vector such that $A\hat{X}=\hat{B}$, where $\hat{B}$ is the orthogonal projection of $B$ onto $\mathcal{C}(A)$.  Note however that when we find the solution using the normal equations, we do not actually need to compute $\hat{B}$.

        We have not provided proof yet that $\hat{X}$ minimizes $||E||$, but we could provide some numerical evidence by computing $||E||$ for other vectors that are "near" $\hat{X}$.  For example, we might compute $||E||$ for a vector $X$ that has components that are within one of $\hat{X}$.  We will use the $\texttt{random}$ module to generate a typical vector.
        """
    )
    return


@app.cell
def _(A, B, X_hat, lag, np):
    P = np.random.rand(2, 1)
    _X = X_hat + P
    print(_X)
    _E = A @ _X - B
    print('Magnitude of error is:', lag.Magnitude(_E))
    return (P,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Running the code in this cell does not prove that $||E||$ is at minimum for $X=\hat{X}$, even if we were to put it in a loop and execute it a million times.  It does allows us to demonstrate some evidence to augment our reasoning.  Another thing we might do to gather evidence is let $X= \delta P$, and plot $||E||$ as a function of $\delta$. 
        """
    )
    return


@app.cell
def _(A, B, P, X_hat, lag, np, plt):
    N = 50
    delta = np.linspace(-1, 1, N)
    E_delta = np.linspace(-1, 1, N)
    for i in range(N):
        _X = X_hat + delta[i] * P
        E_delta[i] = lag.Magnitude(A @ _X - B)
    _fig, _ax = plt.subplots()
    _ax.scatter(delta, E_delta, color='xkcd:brick red')
    _ax.set_xlabel('$\\delta$')
    _ax.set_ylabel('$|E_{\\delta}|$')
    _ax.grid(True)
    return E_delta, N, delta, i


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Best approximation

        Although numerical evidence and the plot of $B$, $\hat{B}$, and $E$ may convince us that $||E||$ is minimum when $E$ and $\hat{B}$ are orthogonal, we have not yet given proof of the fact.  To accomplish this, we need to show that $||B-Y|| \ge ||B-\hat{B}||$ where $Y$ is an arbitrary vector in $\mathcal{C}(A)$.  This can be explained using properties of the dot product.  

        $$
        \begin{eqnarray*}
        ||B-Y||^2 & = & ||(B-\hat{B}) + (\hat{B}-Y)||^2 \\
         & = & \left((B-\hat{B}) + (\hat{B}-Y)\right) \cdot \left((B-\hat{B}) + (\hat{B}-Y)\right) \\
         & = & (B-\hat{B})\cdot (B-\hat{B}) + 2(\hat{B}-Y) \cdot (B-\hat{B}) + (\hat{B}-Y)\cdot(\hat{B}-Y) \\
         & = & ||B-\hat{B}||^2 + ||\hat{B}-Y||^2 \\
         & \ge & ||B-\hat{B}||^2
        \end{eqnarray*}
        $$

        The first key fact is that $(\hat{B}-Y)$ is in $\mathcal{C}(A)$ so $(\hat{B}-Y) \cdot (B-\hat{B}) = 0$.  We also need to observe that magnitudes of vectors can never be negative since they are sums of squares.  Indeed, the only vector with magnitude zero is the vector with all zero entries.  In particular, this means that $||\hat{B}-Y||^2 \ge 0$. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ###  QR factorization to solve normal equations

        In this final example, we consider making use of the $QR$ factorization to solve the normal equations.  If we insert $A=QR$ into the normal equations $A^TAX=A^TB$, we will see a simplification.

        $$
        \begin{eqnarray*}
        (QR)^T(QR)X & = & (QR)^TB \\
        R^TQ^TQRX & = & R^TQ^TB  \\
        R^TRX & = &  R^TQ^TB \\
        RX & = & Q^TB
        \end{eqnarray*}
        $$

        The system $RX=Q^TB$ is triangular and can be solved with back substitution.  Note that this approach replaces elimination ($LU$ factorization) with $QR$ factorization.
        """
    )
    return


@app.cell
def _(A, B, lag):
    Q, R = lag.QRFactorization(A)
    QTB = Q.transpose() @ B
    X_hat_1 = lag.BackSubstitution(R, QTB)
    print(X_hat_1)
    return Q, QTB, R, X_hat_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercises

        **Exercise 1:** Verify that the following system is inconsistent, then find the least squares solution.

        $$
        \begin{eqnarray*}
            x_2 + x_3 & = & 3 \\
        3x_1 - x_2 - 2x_3 & = & 2 \\
        x_1 - 2x_2 - x_3 & = & 1 \\
        4x_1 + 2x_2 + 4x_3 & = & 0
        \end{eqnarray*}
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
        **Exercise 2:** Another way find the least squares solution to an inconsistent system is to find $\hat{B}$ by projecting $B$ onto $\mathcal{C}(A)$ and then solving $A\hat{X}=\hat{B}$ directly.  (*Review [Orthogonal Subspaces](Orthogonal_Subspaces.ipynb) for how compute this projection.*)  Demonstrate the entire calculation using $A$ and $B$ from **Example 2**.
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
        **Exercise 3:** Explain why an inconsistent system, $AX=B$, does not have a unique least squares solution if the columns of $A$ are linearly dependent.

        **Exercise 4:** Demonstrate that the following inconsistent system does not have a unique least squares solution.


        $$
        \begin{eqnarray*}
            x_2 - x_3 & = & 3 \\
        3x_1 - x_2 + 4x_3 & = & 2 \\
        x_1 - 2x_2 + 3x_3 & = & 1 \\
        4x_1 + 2x_2 + 2x_3 & = & 0
        \end{eqnarray*}
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
        **Exercise 5:** If the system $AX = B$ is inconsistent, find the least squares solution to it and determine whether or not the least squares solution is unique.


        $$
        \begin{equation}
        A = \left[\begin{array}{rr} 1 & 2 & 3 \\ 1 & 1 & 1 \\ 2 & 2 & 0 \\ 1 & 2 & 1 \end{array}\right]
        \quad\quad
        B = \left[\begin{array}{r} 1 \\1 \\ 1 \\ 1 \end{array}\right]  
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _():
    ## Code solution here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 6:** Find the equation of the line that best fits through the three given points:  $(0,2), (0,3)$ and $(1,4)$ in the sense of least squares.
        """
    )
    return


@app.cell
def _():
    ## Code solution here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 7:** Find the equation of the parabola that best fits through the given points: $(-1,2), (1,0), (3,1)$ and $(4,2)$ in the sense of least squares.
        """
    )
    return


@app.cell
def _():
    ## Code solution here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 8:** Find the least squares solution for the given system $AX = B$ without using the Normal equation. Instead, find the orthogonal projection of $B$ onto $C(A)$ to find the least squares solution. Is the solution unique?

        $$
        \begin{equation}
        A = \left[\begin{array}{rr} 1 & 2 & 2 \\ 2 & 1 & 4 \\ 1 & 2 & 2 \end{array}\right]
        \quad\quad
        B= \left[\begin{array}{r} 1 \\1 \\ 2 \end{array}\right] 
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _():
    ## Code solution here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 9:** Can you use $QR$ factorization in **Exercise 7** to solve the normal equation ? Explain.
        """
    )
    return


@app.cell
def _():
    ## Code solution here
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
