import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Vector Spaces
        """
    )
    return


@app.cell
def _():
    import laguide as lag
    import numpy as np
    import scipy.linalg as sla
    return lag, np, sla


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### General Linear Systems
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 1:** Determine the number of solutions to the following system, then find all possible solutions.


        $$
        \begin{eqnarray*}
        5x_1 + 4x_2 - x_3 & = & \hspace{0.3cm} 3\\
        x_1 \hspace{1.2cm} + x_3  & = & \hspace{0.3cm} 2\\
        -2x_1 + 2x_2 + 4x_3 & = & -3 \\
        x_1 + 8x_2 + 7x_3 & = & -3 \\
        3x_1 \hspace{1.2cm} - 3x_3 & = & \hspace{0.3cm} 3 \\
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    _A_augmented = np.array([[5, 4, -1, 3], [1, 0, 1, 2], [-2, 2, 4, -3], [1, 8, 7, -3], [3, 0, -3, 3]])
    A_augmented_reduced = lag.FullRowReduction(_A_augmented)
    print(A_augmented_reduced)
    return (A_augmented_reduced,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Every column except the last one has exactly one pivot, so there exists a unique solution to this system. That solution is

        $$
        \begin{eqnarray*}
        x_1 & = & 1.5\\
        x_2  & = & -1\\
        x_3 & = & 0.5
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** Determine the number of solutions to the following system, then find all possible solutions.


        $$
        \begin{eqnarray*}
        5x_1 + 4x_2 - x_3 & = & \hspace{0.3cm} 0\\
        x_1 \hspace{1.2cm} + x_3  & = & \hspace{0.3cm} 2\\
        -2x_1 + 2x_2 + 4x_3 & = & -3 \\
        x_1 + 8x_2 + 7x_3 & = & -3 \\
        3x_1 \hspace{1.2cm} - 3x_3 & = & \hspace{0.3cm} 3 \\
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    _B_augmented = np.array([[5, 4, -1, 0], [1, 0, 1, 2], [-2, 2, 4, -3], [1, 8, 7, -3], [3, 0, -3, 3]])
    _B_augmented_reduced = lag.FullRowReduction(_B_augmented)
    print(_B_augmented_reduced)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The last column has a pivot in it so this system is inconsistent and therefore has no solution.  The reduced augmented matrix represents the following equations.

        $$
        \begin{eqnarray*}
        x_1 & = & 0\\
        x_2  & = & 0\\
        x_3 & = & 0 \\
        0 & = & 1
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** Determine the number of solutions to the following system, then find all possible solutions.

        $$
        \begin{eqnarray*}
        5x_1 + 4x_2 - x_3 & = & \hspace{0.3cm} 3\\
        x_1  +2x_2 + x_3  & = & \hspace{0.3cm} 0\\
        -2x_1 + 2x_2 + 4x_3 & = & -3 \\
        x_1 + 8x_2 + 7x_3 & = & -3 \\
        3x_1 \hspace{1.2cm} - 3x_3 & = & \hspace{0.3cm} 3 \\
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    _C_augmented = np.array([[5, 4, -1, 3], [1, 2, 1, 0], [-2, 2, 4, -3], [1, 8, 7, -3], [3, 0, -3, 3]])
    _C_augmented_reduced = lag.FullRowReduction(_C_augmented)
    print(_C_augmented_reduced)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Neither the third nor the last column have pivots, so there exist an infinite number of solutions to this system. This RREF matrix corresponds to the following system

        $$
        \begin{eqnarray*}
        x_1 - x_3 & = & 1 \\
        x_2 + x_3 & = & -0.5
        \end{eqnarray*}
        $$

        If we parametrize $x_3 = s$ and solve the previous equations for $x_1$ and $x_2$ respectively then the following is a solution to the system for any $s \in \mathbb{R}$

        $$
        \begin{eqnarray*}
        x_1 & = & 1 + s\\
        x_2  & = & -0.5 - s\\
        x_3 & = & s
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 4:** Construct an example of an inconsistent system with 2 equations and 4 unknowns.  Check your example by using $\texttt{FullRowReduction}$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $$
        \begin{eqnarray*}
        2x_1 + 4x_2 - 3x_3 & = & \hspace{0.3cm} 3\\
        4x_1 + 8x_2 - 6x_3 & = & \hspace{0.3cm} 5\\
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    D_augmented = np.array([[2,4,-3,3],[4,8,-6,5]])
    D_augmented_reduced = lag.FullRowReduction(D_augmented)
    print(D_augmented,'\n')
    print(D_augmented_reduced)
    return D_augmented, D_augmented_reduced


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The scond row in the augmented matrix represents the equation $0x_3 = 1$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 5:** Explain why it is not possible for a system with 2 equations and 4 unknowns to have a unique solution.  Base your argument on pivot positions.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The augmented matrix that represents a system with 2 equations and 4 unknowns must can have at most 2 pivots since it has only 2 rows.  This means that one of the first four columns does not have a pivot and corresponds to a free variable in the system.  The guaranteed existence of a free variable means the system does not have a unique solution.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 6:** Write a function that accepts the augmented matrix for a system and returns the number of free variables in that system.  Make use of $\texttt{FullRowReduction}$ in the $\texttt{laguide}$ module. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since every pivot in the RREF of the augmented matrix corresponds to exactly one non-free variable, we can find the number of free variables by subtracting the number of pivots from the total number of variables. We will find the number of pivots by subtracting the number zero rows from the total number of rows.
        """
    )
    return


@app.cell
def _(lag):
    def FreeVariables(A):
        """
        FreeVariables finds the number of free variables given an augmented matrix A. There is no
        error checking to ensure the system is consistent.
        """
        m = _A.shape[0]
        n = _A.shape[1]
        numOfZeroRows = 0
        _A_reduced = lag.FullRowReduction(_A)
        for i in range(m - 1, -1, -1):
            for j in range(0, n - 1):
                if _A_reduced[i][j] != 0:
                    return n - 1 - (m - numOfZeroRows)
            numOfZeroRows = numOfZeroRows + 1
        return n - 1 - (m - numOfZeroRows)
    return (FreeVariables,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Lets use $\texttt{FullRowReduction}$ to find the number of free variables in a few example augmented matrices, and then check to make sure our function $\texttt{FreeVariables}$ correctly counts them.
        """
    )
    return


@app.cell
def _(FreeVariables, lag, np):
    _A = np.array([[5, 4, -1, 3], [1, 0, 1, 2], [-2, 2, 4, -3], [1, 8, 7, -3], [3, 0, -3, 3]])
    _A_reduced = lag.FullRowReduction(_A)
    print(_A, '\n')
    print(_A_reduced, '\n')
    print('There are', FreeVariables(_A), 'free variables in the augmented matrix A \n')
    _B = np.array([[5, 4, -1, 3], [1, 2, 1, 0], [-2, 2, 4, -3], [1, 8, 7, -3], [3, 0, -3, 3]])
    _B_reduced = lag.FullRowReduction(_B)
    print(_B, '\n')
    print(_B_reduced, '\n')
    print('There is', FreeVariables(_B), 'free variable in the augmented matrix B \n')
    _C = np.array([[0, 3, -6, 6, 4, -5], [3, -7, 8, -5, 8, 9], [3, -9, 12, -9, 6, 15]])
    _C_reduced = lag.FullRowReduction(_C)
    print(_C, '\n')
    print(_C_reduced, '\n')
    print('There are', FreeVariables(_C), 'free variables in the augmented matrix C')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Other than the last column in each reduced matrix, every column of $A$ has a pivot, 1 column of $B$ does not have a pivot, and 2 columns of $C$ do not have a pivot, so they have 0,1, and 2 free variables respectively. As we can see above, our $\texttt{FreeVariable}$ function reported correctly the number of free variables in each matrix.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 7:** Write a function that accepts the augmented matrix for a system and returns whether or not that system is consistent.  The function should return the value 1 if the system is consistent or the value 0 if the system is inconsistent.  Make use of $\texttt{FullRowReduction}$ in the $\texttt{laguide}$ module. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If a system is inconsistent, the RREF of the augmented matrix will have a pivot in the final column. 
        """
    )
    return


@app.cell
def _(lag):
    def CheckConsistent(A):
        """
        CheckConsistent checks to see if the augmented matrix of a system has a pivot in its final column. If
        a pivot is found (inconsistent) it returns a 0, otherwise (consistent) it returns a 1.
        """
        m = _A.shape[0]
        n = _A.shape[1]
        _A_reduced = lag.FullRowReduction(_A)
        for i in range(m - 1, -1, -1):
            currentRowZero = 1
            if _A_reduced[i][n - 1] != 0:
                for j in range(0, n - 2):
                    if _A_reduced[i][j] != 0:
                        currentRowZero = 0
                if currentRowZero == 1:
                    return 0
        return 1
    return (CheckConsistent,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Lets use $\texttt{FullRowReduction}$ to determine whether or not a couple example augmented matrices are consistent, and then check to make sure our function $\texttt{CheckConsistent}$ correctly decides this.
        """
    )
    return


@app.cell
def _(CheckConsistent, lag, np):
    _D = np.array([[1, -2, 2, 0], [2, 2, 2, 1], [0, -1, -1, -2], [-2, -1, -1, 0]])
    _D_reduced = lag.FullRowReduction(_D)
    print(_D, '\n')
    print(_D_reduced, '\n')
    if CheckConsistent(_D) == 1:
        print('D is consistent \n')
    else:
        print('D is inconsistent \n')
    E = np.array([[-2, 2, -2, 2, 0], [1, -2, -2, 0, -1], [1, 0, 2, -2, 1]])
    E_reduced = lag.FullRowReduction(E)
    print(E, '\n')
    print(E_reduced, '\n')
    if CheckConsistent(E) == 1:
        print('E is consistent \n')
    else:
        print('E is inconsistent \n')
    return E, E_reduced


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Augmented matrix D has a pivot in the last column and is thus represents an inconsistent system, while E does not have a pivot in the last column and is thus represents a consistent system.  For these two examples, we see that our $\texttt{CheckConsistent}$ function correctly determined which was consistent.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Linear Combinations
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 1:**  Find a linear combination of the vectors $V_1, V_2$ and $V_3$ which equals to the vector $X$.

        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} 2 \\ 0 \\ 7 \end{array}\right] \hspace{1cm}
        V_2 = \left[ \begin{array}{r} 2 \\ 4 \\ 5   \end{array}\right] \hspace{1cm} 
        V_3 = \left[ \begin{array}{r} 2 \\ -12 \\ 13 \end{array}\right] \hspace{1cm}
        X = \left[ \begin{array}{r}  -1 \\ 5 \\ -6  \end{array}\right] \hspace{1cm}
        \end{equation}
        $$


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        We want to find $c_1$, $c_2$,and $c_3$ such that $c_1V_1 + c_2V_2 + c_3V_3 = X$.

        $$
        \begin{equation}
        c_1\left[ \begin{array}{r} 2 \\ 0 \\ 7  \end{array}\right]+
        c_2\left[ \begin{array}{r} 2 \\ 4 \\ 5   \end{array}\right] +
        c_3\left[ \begin{array}{r} 2 \\ -12 \\ 13 \end{array}\right] =
        \left[ \begin{array}{r}  -1 \\ 5 \\ -6 \end{array}\right]
        \end{equation}
        $$

        The corresponding matrix equation is the following.

        $$
        \begin{equation}
        \left[ \begin{array}{r} 2 & 2 & 2 \\  0 & 4 & -12 \\ 7 & 5 & 13 \end{array}\right]
        \left[ \begin{array}{r} c_1 \\ c_2 \\ c_3 \end{array}\right]=
        \left[ \begin{array}{r} -1 \\ 5 \\ -6 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    _A_aug = np.array([[2, 2, 2, -1], [0, 4, -12, 5], [7, 5, 13, -6]])
    _A_aug_reduced = lag.FullRowReduction(_A_aug)
    print('A_augmented: \n', _A_aug, '\n')
    print('A_augmented_reduced: \n', _A_aug_reduced, '\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since $c_3$ is a free variable, there are infinite linear combinations of $V_1$, $V_2$, and $V_3$ which give the vector $X$.

        We can produce one specific example by setting $c_3 = 0.25$.  Then, $c_2 = 2$ and $c_1 = -2.75$ and one possible linear combination is

        $$
        \begin{equation}
        (-2.75)\left[ \begin{array}{r} 2 \\ 0 \\ 7  \end{array}\right]+
        (2)\left[ \begin{array}{r} 2 \\ 4 \\ 5   \end{array}\right] +
        (0.25)\left[ \begin{array}{r} 2 \\ -12 \\ 13 \end{array}\right] =
        \left[ \begin{array}{r}  -1 \\ 5 \\ -6 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:**  Determine whether or not $X$ lies in the span of $\{ V_1, V_2 ,V_3\}$.

        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} 1 \\ 0 \\ 0 \end{array}\right] \hspace{1cm}
        V_2 = \left[ \begin{array}{r} 2 \\ -2 \\ 1  \end{array}\right] \hspace{1cm} 
        V_3 = \left[ \begin{array}{r} 2 \\ 0 \\ 4 \end{array}\right] \hspace{1cm}
        X = \left[ \begin{array}{r}  1 \\ 3 \\ -1  \end{array}\right] \hspace{1cm}
        \end{equation}
        $$


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        $X$ lies in the span of $\{ V_1, V_2 ,V_3\}$ if we can express $X$ as the linear combination of $V_1$, $V_2$ and $V_3$.  We want to determine if there are scalars $c_1$, $c_2$,and $c_3$ such that $c_1V_1 + c_2V_2 + c_3V_3 = X$. 

        $$
        \begin{equation}
        c_1\left[ \begin{array}{r} 1 \\ 0 \\ 0  \end{array}\right]+
        c_2\left[ \begin{array}{r} 2 \\ -2 \\ 1   \end{array}\right] +
        c_3\left[ \begin{array}{r} 2 \\ 0 \\ 4 \end{array}\right] =
        \left[ \begin{array}{r}  1 \\ 3 \\ -1 \end{array}\right]
        \end{equation}
        $$

        The corresponding matrix equation is the following.

        $$
        \begin{equation}
        \left[ \begin{array}{r} 1 & 2 & 2 \\  0 & -2 & 0 \\ 0 & 1 & 4 \end{array}\right]
        \left[ \begin{array}{r} c_1 \\ c_2 \\ c_3 \end{array}\right]=
        \left[ \begin{array}{r} 1 \\ 3 \\ -1 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    _A_augmented = np.array([[1, 2, 2, 1], [0, -2, 0, 3], [0, 1, 4, -1]])
    _A_aug_reduced = lag.FullRowReduction(_A_augmented)
    print('A_augmented: \n', _A_augmented, '\n')
    print('A_augmented_reduced: \n', _A_aug_reduced, '\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The above system has a unique solution since there is a pivot in each row and column.

        $$
        \begin{equation}
        \left[ \begin{array}{r} c_1 \\ c_2\\ c_3 \end{array}\right] \ = 
        \left[ \begin{array}{r} 3.75 \\ -1.5 \\ 0.125 \end{array}\right] \
        \end{equation}
        $$

        $X$ as a linear combination of $V_1$,$V_2$ and $V_3$ and thereforre lies in the span of $\{V_1, V_2, V_3\}$.


        $$
        \begin{equation}
        (3.75)\left[ \begin{array}{r} 1 \\ 0 \\ 0  \end{array}\right]+
        (-1.5)\left[ \begin{array}{r} 2 \\ -2 \\ 1   \end{array}\right] +
        (0.125\left[ \begin{array}{r} 2 \\ 0 \\ 4 \end{array}\right] =
        \left[ \begin{array}{r}  1 \\ 3 \\ -1 \end{array}\right]
        \end{equation}
        $$



        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:**  Does the set $\{ X_1, X_2 ,X_3, X_4\}$ span $\mathbb{R}^4$? Explain why or why not.

        $$
        \begin{equation}
        X_1 = \left[ \begin{array}{r} 1 \\ 1\\ 1\\1 \end{array}\right] \hspace{1cm}
        X_2 = \left[ \begin{array}{r} 1 \\ 0 \\ 0\\2  \end{array}\right] \hspace{1cm} 
        X_3 = \left[ \begin{array}{r} 2 \\ 0 \\ 1\\1 \end{array}\right] \hspace{1cm}
        X_4 = \left[ \begin{array}{r} 3 \\ 0 \\ 1\\2 \end{array}\right] \hspace{1cm}
        \end{equation}
        $$


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We build a matrx $A$ which contains the given vectors as its columns.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrrr} 
        1 & 1 & 2 & 3 \\ 1 & 0 & 0 & 0  \\ 1 & 0 & 1 & 1 \\ 1 & 2 & 1 & 2   
        \end{array}\right]
        \end{equation}
        $$

        The set of vectors $\{ X_1, X_2 ,X_3, X_4\}$ spans $\mathbb{R}^4$ if there is a pivot in every row of the matrix $A$.
        """
    )
    return


@app.cell
def _(lag, np):
    _A = np.array([[1, 1, 2, 3], [1, 0, 0, 0], [1, 0, 1, 1], [1, 2, 1, 2]])
    _A_reduced = lag.FullRowReduction(_A)
    print('A: \n', _A, '\n')
    print('A_reduced: \n', _A_reduced, '\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The row reduced form of A reveals that there is a pivot in every row of $A$, which means that the set of vectors $\{X_1,X_2,X_3,X_4\}$ spans $\mathbb{R}^4\$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 4:** Answer questions regarding the following vectors.

        $$
        \begin{equation}
        W_1 = \left[ \begin{array}{r} 2 \\ 3 \\ -1 \end{array}\right] \hspace{1cm}
        W_2 = \left[ \begin{array}{r} 3 \\ 0 \\ 1   \end{array}\right] \hspace{1cm} 
        W_3 = \left[ \begin{array}{r} 4 \\ -3 \\ 3 \end{array}\right] \hspace{1cm}
        B = \left[ \begin{array}{r}  1 \\ 1 \\ 1  \end{array}\right] \hspace{1cm}
        C = \left[ \begin{array}{r}  3 \\ 0 \\ 1  \end{array}\right] \hspace{1cm}
        V_1 = \left[ \begin{array}{r} 8 \\ -3 \\ 5 \end{array}\right] \hspace{1cm}
        V_2 = \left[ \begin{array}{r} 4 \\ 6 \\ -2   \end{array}\right]  
        \end{equation}
        $$

        $(a)$ Determine if $B$ is in the span of $\{W_1, W_2, W_3\}$.

        We need to determine if there are scalars $a_1$, $a_2$, $a_3$ such that $a_1W_1 + a_2W_2 + a_3W_3 = B$.  We build an augmented matrix that represents the corresponding linear system and then use $\texttt{FullRowReduction}$.
        """
    )
    return


@app.cell
def _(lag, np):
    _B_augmented = np.array([[2, 3, 4, 1], [3, 0, -3, 1], [-1, 1, 3, 1]])
    _B_augmented_reduced = lag.FullRowReduction(_B_augmented)
    print(_B_augmented, '\n')
    print(_B_augmented_reduced)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This augmented matrix has a pivot in the last column.  Therefore the system is inconsistent and $B$ is not  in the span of $\{W_1, W_2, W_3\}$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $(b)$ Determine if $C$ is in the span of $\{W_1, W_2, W_3\}$.

        We need to determine if there are scalars $a_1$, $a_2$, $a_3$ such that $a_1W_1 + a_2W_2 + a_3W_3 = C$.  We build an augmented matrix that represents the corresponding linear system and then use $\texttt{FullRowReduction}$.
        """
    )
    return


@app.cell
def _(lag, np):
    _C_augmented = np.array([[2, 3, 4, 3], [3, 0, -3, 0], [-1, 1, 3, 1]])
    _C_augmented_reduced = lag.FullRowReduction(_C_augmented)
    print(_C_augmented, '\n')
    print(_C_augmented_reduced)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since there is no pivot in the last column, the system is consistent and $C$ is in the span of $\{W_1, W_2, W_3\}$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $(c)$ Find a nonzero vector in the span of $\{W_1, W_2, W_3\}$ that has zero as its first entry.

        Any linear combination of $W_1$, $W_2$, and $W_3$ is in the span of $\{W_1, W_2, W_3\}$.  There are many that have zero as the first entry, including the following example.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $$
        \begin{equation}
        -2W_1 + W_3 = \left[ \begin{array}{r} -4 \\ -6 \\ 2 \end{array}\right] + \left[ \begin{array}{r} 4 \\ -3 \\ 1 \end{array}\right] = \left[ \begin{array}{r} 0 \\ -9 \\ 3 \end{array}\right] \hspace{1cm}
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $(d)$ How can we determine if the span of $\{W_1, W_2, W_3\}$ equal the span of $\{V_1, V_2\}$?
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If each of the vectors in $\{V_1, V_2\}$ is in the span of $\{W_1, W_2, W_3\}$ and vice versa, then the two sets should have the same span. However, we can quickly show that $V_1$ is not in the span of $\{W_1, W_2, W_3\}$.
        """
    )
    return


@app.cell
def _(lag, np):
    V_1_augmented = np.array([[2,3,4,8],[3,0,-3,-3],[-1,1,3,5]])
    V_1_augmented_reduced = lag.FullRowReduction(V_1_augmented)
    print(V_1_augmented,'\n')
    print(V_1_augmented_reduced)
    return V_1_augmented, V_1_augmented_reduced


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since the last column has a pivot this system is inconsistent. Therefore $V_1$ is not in the span of $\{W_1, W_2, W_3\}$ and thus the two sets have different spans.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 5:** Show that the vector $X$ lies in the span of the columns of $A$. Also find another vector that is in the span of the columns of $A$ and verify your answer.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrrr} 
        1 & 1 & 3  \\ 2 & 0 & 1   \\ 3 & 1 & 1     
        \end{array}\right] \hspace{2 cm}
        x = \left[ \begin{array}{r} 1 \\ 1\\ 1 \end{array}\right] \hspace{1cm}
        \end{equation}
        $$


        """
    )
    return


@app.cell
def _(lag, np):
    _A = np.array([[1, 1, 3], [2, 0, 1], [3, 1, 1]])
    _A_reduced = lag.FullRowReduction(_A)
    print('A: \n', _A, '\n')
    print('A_reduced: \n', _A_reduced, '\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The row reduced form of A shows that there is a pivot in every row of $A$, which implies that the columns of $A$ span $\mathbb{R}^3$. This means that every vector in $\mathbb{R}^3$, including $X$ is in the span of the columns of $A$.

        Let us choose another vector $Y$.

        $$
        \begin{equation}
         Y = \left[ \begin{array}{r} 1 \\ 2 \\ 5 \end{array}\right] \
        \end{equation}
        $$

        If $Y$ lies in the span of columns of $A$, then there is a solution to the following system.

        $$
        \begin{equation}
        \left[ \begin{array}{r} 1 & 1 & 3 \\  2 & 0 & 1 \\ 3 & 1 & 1 \end{array}\right]
        \left[ \begin{array}{r} c_1 \\ c_2 \\ c_3 \end{array}\right]=
        \left[ \begin{array}{r} 1 \\ 2 \\ 5 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    # Building the matrix A
    Aug = np.array([[1,1,3,1],[2,0,1,2],[3,1,1,5]])
    Aug_reduced = lag.FullRowReduction(Aug)

    print("Augmented: \n", Aug, '\n')
    print("Augmented_reduced: \n", Aug_reduced, '\n')
    return Aug, Aug_reduced


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""

        $$
        \begin{equation}
        \left[ \begin{array}{r} c_1 \\ c_2\\ c_3 \end{array}\right] \ = 
        \left[ \begin{array}{r} 1.33 \\ 1.67 \\ -0.67 \end{array}\right] \
        \end{equation}
        $$

        Since we can find a solution for the system, $Y$ is in the span of the columns of $A$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 6:** Consider the matrix $R$ from **Example 3**.  Find one vector in $\mathbb{R}^4$ that is in the span of the columns of $R$, and one vector in $\mathbb{R}^4$ that is not.  Demontrate with an appropriate computation.


        $$
        \begin{equation}
        R = \left[ \begin{array}{rrrr} 
        1 & 1 & 0 & -1 \\ 1 & 1 & 0 & 1  \\ -1 & -1 & 1 & -1 \\ 1 & 1 & -2 & 0   
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
        Consider the following vectors $V$ and $W$. $V$ is not in the span of the columns of $R$, but $W$ is.

        $$
        \begin{equation}
        V = \left[ \begin{array}{r} 0 \\ 0 \\ 0 \\ 1 \end{array}\right] \hspace{1cm}
        W = \left[ \begin{array}{r} 0 \\ 2 \\ -2 \\ 1   \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    V_augmented = np.array([[1,1,0,-1,0],[1,1,0,1,0],[-1,-1,1,-1,0],[1,1,-2,0,1]])
    V_augmented_reduced = lag.FullRowReduction(V_augmented)
    print(V_augmented,'\n')
    print(V_augmented_reduced,'\n')

    W_augmented = np.array([[1,1,0,-1,0],[1,1,0,1,2],[-1,-1,1,-1,-2],[1,1,-2,0,1]])
    W_augmented_reduced = lag.FullRowReduction(W_augmented)
    print(W_augmented,'\n')
    print(W_augmented_reduced)
    return (
        V_augmented,
        V_augmented_reduced,
        W_augmented,
        W_augmented_reduced,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that for any vector $X$ in $\mathbb{R}^4$, the vector $RX$ is a linear combination of the columns of $R$ and therefore in the span of the columns of $R$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 7:** Find a vector $V_3$ that is not in the span of $\{ V_1, V_2\}$. Explain why the set $\{ V_1, V_2, V_3 \}$ spans $\mathbb{R}^3$.

        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} 1 \\ 1\\ 1 \end{array}\right] \hspace{1cm}
        V_2 = \left[ \begin{array}{r} 2 \\ 0 \\ 4  \end{array}\right] \hspace{1cm} 
        \end{equation}
        $$


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 8:** Explain why the system $AX=B$ cannot be consistent for every vector $B$ if $A$ is a $5\times 3$ matrix.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 9:** Find the value of $a$ for which $\{ V_1, V_2 ,V_3\} $ does not span $\mathbb{R}^3$.

        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} 1 \\ 2\\ 3 \end{array}\right] \hspace{1cm}
        V_2 = \left[ \begin{array}{r} 4 \\ 5 \\ 6  \end{array}\right] \hspace{1cm} 
        V_3 = \left[ \begin{array}{r} 1 \\ 0 \\ a \end{array}\right] \hspace{1cm}
        \end{equation}
        $$

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        The matrix $A$ which contains the given vectors as its columns is as follows:

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrrr} 
        1 & 4 & 1 \\ 2& 5&0   \\ 3& 6 & a  \\   
        \end{array}\right]
        \end{equation}
        $$

        The given set of vectors $\{V_1,V_2,V_3\}$ does not span $\mathbb{R}^3$ if there is a pivot missing in at least one of the rows of the matrix $A$.  Applying fow operations gives the following matrix.

        $$
        \begin{equation}
        \left[ \begin{array}{rrrr} 
        1 & 4& 1 \\ 0 &-3& -2  \\ 0 & 0 & a + 1
        \end{array}\right]
        \end{equation}
        $$

        If $a=-1 $, then there is a pivot missing from the third row of the matrix $A$ and $\{V_1,V_2,V_3\}$ does not span $\mathbb{R}^3$. 

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""


        ### Linear Independence
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 1:** Determine if the set of vectors $\{U_1, U_2, U_3\}$ is linearly independent.

        $$
        \begin{equation}
        U_1 = \left[ \begin{array}{r} 0 \\ 5  \\ 2  \\ 2 \end{array}\right] \hspace{1cm} 
        U_2 = \left[ \begin{array}{r} 1 \\ -1 \\ 0  \\ -1 \end{array}\right] \hspace{1cm}
        U_3 = \left[ \begin{array}{r} 3 \\ 2 \\ 2  \\ -1 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We need to determine if there are nonzero scalars $a_1,a_2,a_3$ such that $a_1U_1 + a_2U_2 + a_3U_3 = 0$. We build the augmented matrix that represents the corresponding linear system and then use $\texttt{FullRowReduction}$.
        """
    )
    return


@app.cell
def _(lag, np):
    U = np.array([[0,1,3,0],[5,-1,2,0],[2,0,2,0],[2,-1,-1,0]])
    U_reduced = lag.FullRowReduction(U)
    print(U_reduced)
    return U, U_reduced


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The absence of a pivot in the third column means that $a_3$ is a free variable. Since we can choose any nonzero value for $a_3$ and get a solution to our equation, by definition $\{U_1,U_2,U_3\}$ is linearly dependent.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** Determine whether or not the set of vectors $ \{W_1, W_2, W_3, W_4 \} $ is linearly independent. If the set is linearly dependent, then represent one of the vectors of the set as the linear combination of others.

        $$
        \begin{equation}
        W_1 = \left[ \begin{array}{r} 1 \\ 0  \\ 0 \\1  \end{array}\right] \hspace{1cm} 
        W_2 = \left[ \begin{array}{r} 0 \\ 1 \\ -1 \\0  \end{array}\right] \hspace{1cm}
        W_3 = \left[ \begin{array}{r} -1 \\ 0 \\ -1 \\ 0  \end{array}\right] \hspace{1cm}
        W_4 = \left[ \begin{array}{r} 1 \\1 \\1\\-1 \end{array}\right] 
        \end{equation}
        $$


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        We need to determine if there are nonzero scalars $a_1,a_2,a_3,a_4$ such that $a_1W_1 + a_2W_2 + a_3W_3 + a_4W_4 = 0$. We build the augmented matrix that represents the corresponding linear system and then use $\texttt{FullRowReduction}$.
        """
    )
    return


@app.cell
def _(lag, np):
    _A = np.array([[1, 0, -1, 1, 0], [0, 1, 0, 1, 0], [0, -1, -1, 1, 0], [1, 0, 0, -1, 0]])
    print('A: \n', _A, '\n')
    _A_reduced = lag.FullRowReduction(_A)
    print('A_reduced: \n', _A_reduced, '\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The absence of a pivot in the fourth column of means there is a free variable in the system and the set of vectors $\{W_1, W_2, W_3, W_4 \} $ is linearly dependent.

        In order to represent $W_4$ are a linear combination of $W_1$, $W_2$ and $W_3$, we solve a vector equation. 

        $$
        \begin{equation}
        c_1\left[ \begin{array}{r} 1 \\ 0 \\ 0 \\ 1 \end{array}\right] +
        c_2\left[ \begin{array}{r} 0 \\ 1\\ -1\\0  \end{array}\right] +
        c_3\left[ \begin{array}{r} -1 \\ 0 \\ -1 \\0  \end{array}\right] =
        \left[ \begin{array}{r} 1 \\ 1 \\ 1 \\ -1 \end{array}\right]\end{equation}
        $$

        The corresponding matrix equation is the following.

        $$
        \begin{equation}
        AX = \left[ \begin{array}{rrr} 1 & 0 & -1\\ 0 & 1 & 0  \\ 0 & -1 & -1  \\ 1 & 0 & 0 \end{array}\right]
        \left[ \begin{array}{r} c_1 \\ c_2 \\ c_3 \end{array}\right]=
        \left[ \begin{array}{r} 1 \\ 1 \\ 1 \\ -1  \end{array}\right]= B
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    _A_aug = np.array([[1, 0, -1, 1], [0, 1, 0, 1], [0, -1, -1, 1], [1, 0, 0, -1]])
    print('A_augmented: \n', _A_aug, '\n')
    _A_aug_reduced = lag.FullRowReduction(_A_aug)
    print('A_augmented_reduced: \n', _A_aug_reduced, '\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We find that $c_1 = -1$, $c_2 = 1$, $c_3 = -2$.

        $$
        \begin{equation}
        (-1)\left[ \begin{array}{r} 1 \\ 0 \\ 0 \\ 1 \end{array}\right] +
        1\left[ \begin{array}{r} 0 \\ 1\\ -1\\0  \end{array}\right] +
        (-2)\left[ \begin{array}{r} -1 \\ 0 \\ -1 \\0  \end{array}\right] =
        \left[ \begin{array}{r} 1 \\ 1 \\ 1 \\ -1 \end{array}\right]\end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** Consider the following vectors in $ \mathbb{R}^3$:
        $$
        \begin{equation}
        X_1 = \left[ \begin{array}{r} 1 \\ 0  \\ 1  \end{array}\right] \hspace{1cm} 
        X_2 = \left[ \begin{array}{r} 1 \\ 2\\ 3  \end{array}\right] \hspace{1cm}
        X_3 = \left[ \begin{array}{r} 2 \\ 4 \\ b   \end{array}\right] \
        \end{equation}
        $$

        Find the value of $b$ for which the given vectors are linearly dependent. Then, represent one vector as the linear combination of the other two.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**
        The matrix $A$ contains the given vectors as its columns and is as follows:

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrr} 1 & 1 & 2 \\ 0 & 2 & 4 \\ 1 & 3 & b \end{array}\right]
        \end{equation}
        $$

        On carrying out the row reduction of $A$, we get:
        $$
        \begin{equation}
         \left[ \begin{array}{rrr} 1 & 0 & 0 \\ 0 & 2 & 0 \\ 1 & 0 & b-6 \end{array}\right]
        \end{equation}
        $$

        The columns of $A$ will be linearly dependent when there is no pivot in the third column. This means that $b-6$ needs to be zero. Therfore, the set of vectors $\{ X_1, X_2, X_3\}$ is linearly dependent when $ b = 6$.

        In order to represent $X_3$ are a linear combination of $X_1$ and $X_2$, we solve a vector equation. 

        $$
        \begin{equation}
        X_1 = \left[ \begin{array}{r} 1 \\ 0  \\ 1  \end{array}\right] \hspace{1cm} 
        X_2 = \left[ \begin{array}{r} 1 \\ 2\\ 3  \end{array}\right] \hspace{1cm}
        X_3 = \left[ \begin{array}{r} 2 \\ 4 \\ 6   \end{array}\right] \
        \end{equation}
        $$

        Let us represent $X_3$ as the linear combination of $X_1$ and $X_2$.

        $$
        \begin{equation}
        c_1\left[ \begin{array}{r} 1 \\ 0 \\ 1 \end{array}\right] +
        c_2\left[ \begin{array}{r} 1 \\ 2\\ 3  \end{array}\right] =
        \left[ \begin{array}{r} 2 \\ 4 \\ 6  \end{array}\right]\end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    _A_aug = np.array([[1, 1, 2], [0, 2, 4], [1, 3, 6]])
    print('A_augmented: \n', _A_aug, '\n')
    _A_aug_reduced = lag.FullRowReduction(_A_aug)
    print('A_augmented_reduced: \n', _A_aug_reduced, '\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We find that $ c_1 = 0$ and $c_2 = 2$.  $X_3$ is a multiple of $X_2$.

        $$
        \begin{equation}
        (0)\left[ \begin{array}{r} 1 \\ 0 \\ 1 \end{array}\right] +
        2\left[ \begin{array}{r} 1 \\ 2\\ 3  \end{array}\right] =
        \left[ \begin{array}{r} 2 \\ 4 \\ 6  \end{array}\right]\end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 4:** Find the value($s$) of $a$ such that the set of vectors $ \{V_1, V_2, V_3\} $ is linearly independent.

        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} 1 \\ 1  \\ 1  \end{array}\right] \hspace{1cm} 
        V_2 = \left[ \begin{array}{r} a \\ 1 \\ 1   \end{array}\right] \hspace{1cm}
        V_3 = \left[ \begin{array}{r} 0 \\ 1 \\ a  \end{array}\right]
        \end{equation}
        $$


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        We consider the homogeneous system $AX=0$ where $A$ is the matrix with the given vectors as its columns.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrr} 1 & a & 0 \\ 1 & 1 & 1 \\ 1 & 1 & a \end{array}\right]
        \end{equation}
        $$

        After carrying out row operations on $A$, we get the reduced form of $A$ which is as follows:
        $$
        \begin{equation}
        A_{Red} = \left[ \begin{array}{rrr} 1 & 0 & 0 \\ 0 & 1-a & 0 \\ 0 & 0 & a-1 \end{array}\right]
        \end{equation}
        $$

        The columns of $A$ are linearly independent if there is no free variable in the homogeneous system $AX=0$. This means that there needs to be a pivot in each column of $A$. This condition is true when $a - 1$ and $ 1-a$ are nonzero. 
        Therefore, the set of vectors $\{V_1, V_2, V_3\}$ will be linearly independent for all real values of $a$ except 1.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 5:**  Use the concept of linear independence of vectors to show that the given system does not have infinitely many solutions.

        $$
        \begin{equation}
        AX = \left[ \begin{array}{rrr} 1 & 1 & 2 \\ 2 & 0 & 1 \\ 3 & 1 & 1 \end{array}\right]
        \left[ \begin{array}{r} x_1 \\ x_2 \\ x_3 \end{array}\right]=
        \left[ \begin{array}{r} 1 \\ 1 \\ 1  \end{array}\right]= B
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:** 
        """
    )
    return


@app.cell
def _(lag, np):
    _A = np.array([[1, 1, 2], [2, 0, 1], [3, 1, 1]])
    print('A: \n', _A, '\n')
    _A_reduced = lag.FullRowReduction(_A)
    print('A_reduced: \n', _A_reduced)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can see that the reduced form of $A$ has a pivot in each column. This means that there is no free variable and the columns of $A$ are linearly independent. This also means that the null space of $A$ contains only the zero vector. Hence, the given system can never have infinitely many solutions.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 6:** Can you find a nonzero vector in the null space of the matrix $A$? Use this information to determine the number of solutions for the system $AX = B$, where $B$ is any vector in $\mathbb{R}^3\$.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrr} 1 & 2 & 3 \\ 1 & 0 & 1 \\ 1 & 1 & 2 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        A vector $X$ is in the null space of $A$ if it is a solution to the homogeneous system $AX=0$.
        """
    )
    return


@app.cell
def _(lag, np):
    _A_aug = np.array([[1, 2, 3, 0], [1, 0, 1, 0], [1, 1, 2, 0]])
    print('A augmented: \n', _A_aug, '\n')
    _A_aug_reduced = lag.FullRowReduction(_A_aug)
    print('A augmented reduced: \n', _A_aug_reduced, '\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since there is no pivot in the third column, there is a free variable in the system and the columns of $A$ are linearly dependent. This means that the null space of $A$ contains a non-zero vector. 

        Let $x_3 = t$ where $t$ is any scalar.  For any value of $t$, the following vector is a solution to $AX=0$.

        $$
        \begin{equation}
        X = \left[ \begin{array}{r} -t \\ -t  \\ t  \end{array}\right] \hspace{1cm} 
        \end{equation}
        $$

        Choose $t=1$ for example gives the following specific vector in the null space of $A$.

        $$
        \begin{equation}
        X = \left[ \begin{array}{r} -1 \\ -1  \\ 1  \end{array}\right] \hspace{1cm} 
        \end{equation}
        $$


        Since the null space of $A$ contains nonzero vector, the system $AX = B$ has infinitely many solutions.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 7:** Find two distinct nonzero vectors in the null space of the matrix $D$.

        $$
        \begin{equation}
        D = \left[ \begin{array}{rrr} 4 & 4 & 3 \\ 8 & 8 & 6 \\ 1 & 0 & 1 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Recall that the null space of $D$ is the set of all vectors $X = \left[ \begin{array}{rrr} x_1 \\ x_2 \\ x_3 \end{array}\right] $ such that $DX = 0$.
        """
    )
    return


@app.cell
def _(lag, np):
    _D = np.array([[4, 4, 3, 0], [8, 8, 6, 0], [1, 0, 1, 0]])
    _D_reduced = lag.FullRowReduction(_D)
    print(_D_reduced)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If we parametrize $x_3 = s$ then the following is a solution to the system and therefore in the null space of $D$ for any $s \in \mathbb{R}$

        $$
        \begin{eqnarray*}
        x_1 & = & -s\\
        x_2  & = & 0.25 s\\
        x_3 & = & s
        \end{eqnarray*}
        $$

        For example, if we take $s = 1$ and $s = 2$, then we get that the following two vectors are distinct, nonzero elements of $\mathcal{N}(D)$

        $$
        \begin{equation}
        \left[ \begin{array}{rrr} -1 \\ 0.25 \\ 1 \end{array}\right] \hspace{1cm} 
        \left[ \begin{array}{rrr} -2 \\ 0.5 \\ 2 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 8:** 
        Suppose the vector $X$ given below is a solution for a system $AX = B$ for any value of $t$.

        $$
        \begin{equation}
        X = \left[ \begin{array}{r} -3 + 2t \\ 2 - t  \\ t  \end{array}\right] \hspace{1cm} 
        \end{equation}
        $$


        $(a)$ Following the discussion in this section on homogeneous systems, find $X_h$ and $X_p$ so that $X= X_p+X_h$ so that $X_p$ is the solution to some system $AX=B$ and $X_h$ is the solution to the corresponding homogeneous system $AX=0$. 

        $(b)$ Given the coefficient matrix $A$, find $B$.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrr} 1 & 2 & 0 \\ 0 & 1 & 1  \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:** 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ($a$) The given solution can be split as:

        $$
        \begin{equation}
        X  = 
        \left[ \begin{array}{c} -3 \\ 2  \\ 0 \end{array}\right] + 
        t\left[ \begin{array}{c} 2 \\ -1  \\ 1 \end{array}\right] = X_p + X_h
        \end{equation}
        $$

        $AX_p= B$, $AX_h= 0$ for any $t$, and $A(X_p+X_h) = B$ for any $t$.

        ($b$) $B= AX_p$
        """
    )
    return


@app.cell
def _(np):
    _A = np.array([[1, 2, 0], [0, 1, 1]])
    X_p = np.array([[-3], [2], [0]])
    _B = _A @ X_p
    print('B: \n', _B, '\n')
    return (X_p,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Bases
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 1:** Find a basis for the set of solutions to the system $PX=0$ where $P$ is defined as follows.

        $$
        \begin{equation}
        P = \left[ \begin{array}{rrrr} 1 & 0 & 3 & -2 & 4 \\ -1 & 1 & 6 & -2 & 1 \\ -2 & 1 & 3 & 0 & -3 \end{array}\right] 
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        First we find a general solution to the system above.
        """
    )
    return


@app.cell
def _(lag, np):
    _P = np.array([[1, 0, 3, -2, 4], [-1, 1, 6, -2, 1], [-2, 1, 3, 0, -3]])
    _P_reduced = lag.FullRowReduction(_P)
    print(_P_reduced)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this system, $x_3$, $x_4$, and $x_5$ are free variables. If we parametrize them as $x_3 = r$, $x_4 = s$, and $x_5 = t$ then $x_1 = -3r + 2s - 4t$ and $x_2 = -9r + 4s - 5t.$ Then we can write the components of a general solution vector $X$ in terms of these parameters.

        $$
        \begin{equation}
        X = \left[ \begin{array}{r} x_1 \\ x_ 2 \\ x_ 3 \\ x_4 \\ x_5 \end{array}\right] =  
        r\left[ \begin{array}{r} -3 \\ -9 \\  1 \\ 0 \\ 0 \end{array}\right] +
        s\left[ \begin{array}{r} 2 \\ 4 \\ 0 \\ 1 \\ 0 \end{array}\right] +
        t\left[ \begin{array}{r} -4 \\ -5 \\ 0 \\ 0 \\ 1 \end{array}\right]
        \end{equation}
        $$

        Therefore any solution to the system must be a linear combination of $V_1, V_2,$ and $V_3$ as defined below, and therefore $\{V_1,V_2,V_3\}$ forms a basis for the set of solutons to $PX = 0$.

        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} -3 \\ -9 \\  1 \\ 0 \\ 0 \end{array}\right] \hspace{1cm}
        V_2 = \left[ \begin{array}{r} 2 \\ 4 \\ 0 \\ 1 \\ 0  \end{array}\right] \hspace{1cm}
        V_3 = \left[ \begin{array}{r} -4 \\ -5 \\ 0 \\ 0 \\ 1  \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** Determine if the following set of vectors is a basis for $\mathbb{R}^4$.  Give justification with an appropriate calculation.

        $$
        \begin{equation}
        W_1 = \left[ \begin{array}{r} -1 \\ 0 \\ 1 \\ 2 \end{array}\right] \hspace{0.7cm} 
        W_2 = \left[ \begin{array}{r} 2 \\ 1 \\ 2 \\ 4 \end{array}\right] \hspace{0.7cm}
        W_3 = \left[ \begin{array}{r} 0 \\ 0 \\ 1 \\ 0 \end{array}\right] \hspace{0.7cm}
        W_4 = \left[ \begin{array}{r} -1 \\ 0 \\ -1 \\ 1 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We need to check that these vectors are linearly independent and span $\mathbb{R}^4$. We can do both by forming the matrix whose columns are these vectors and then using $\texttt{FullRowReduction}$.
        """
    )
    return


@app.cell
def _(lag, np):
    W = np.array([[-1,2,0,-1],[0,1,0,0],[1,2,1,-1],[2,4,0,1]])
    W_reduced = lag.FullRowReduction(W)
    print(W_reduced)
    return W, W_reduced


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can see that the RREF of $W$ has a pivot in every row and column, and thus this set of vectors forms a basis for $\mathbb{R}^4$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** Give an example of a set of three vectors that does **not** form a basis for $\mathbb{R}^3$.  Provide a calculation that shows why the example is not a basis.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Consider the set of vectors $\{V_1,V_2,V_3\}$ defined below.

        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} 1 \\ 0 \\ 0 \end{array}\right] \hspace{0.7cm} 
        V_2 = \left[ \begin{array}{r} 0 \\ 1 \\ 0 \end{array}\right] \hspace{0.7cm}
        V_3 = \left[ \begin{array}{r} 1 \\ 1 \\ 0 \end{array}\right] \hspace{0.7cm}
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(np):
    V_1 = np.array([[1],[0],[0]])
    V_2 = np.array([[0],[1],[0]])
    V_3 = np.array([[1],[1],[0]])
    print(V_1 + V_2 - V_3)
    return V_1, V_2, V_3


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since we have found a non-trivial linear combination of the vectors that equals zero, the set is linearly dependent and therefore not a basis.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 4:** Calculate the dimension of the span of $ \{U_1, U_2, U_3, U_4\}$.

        $$
        \begin{equation}
        U_1 = \left[ \begin{array}{r} 1 \\ 2 \\ -1 \\ 3 \end{array}\right] \hspace{0.7cm} 
        U_2 = \left[ \begin{array}{r} 2 \\ -3 \\ 3 \\ -2 \end{array}\right] \hspace{0.7cm}
        U_3 = \left[ \begin{array}{r} 3 \\ -1 \\ 2 \\ 1 \end{array}\right] \hspace{0.7cm}
        U_4 = \left[ \begin{array}{r} 5 \\ -4 \\ 4 \\ -1 \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Recall that the dimension of a vector space is the number of vectors in any basis for that space, so if we can find a basis for the span of $U = \{U_1, U_2, U_3, U_4\}$ then we will know its dimension. A basis for this space is a linearly independent set of vectors that span it. By definition $U$ spans the span of $U$, so we just need to check that the vectors of $U$ are linearly independent, removing any redundant ones.
        """
    )
    return


@app.cell
def _(lag, np):
    U_12 = np.array([[1,2],[2,-3],[-1,3],[3,-2]])
    U_12_reduced = lag.FullRowReduction(U_12)
    print(U_12_reduced)
    return U_12, U_12_reduced


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can see the $U_1$ and $U_2$ are linearly independent, so now we check if adding $U_3$ changes that.
        """
    )
    return


@app.cell
def _(lag, np):
    U_123 = np.array([[1,2,3],[2,-3,-1],[-1,3,2],[3,-2,1]])
    U_123_reduced = lag.FullRowReduction(U_123)
    print(U_123_reduced)
    return U_123, U_123_reduced


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The last column does not have a pivot, so $U_3$ is included in the span of $U_1$ and $U_2$. Therefore the span of $\{U_1, U_2, U_3\}$ is the same as the span of $\{U_1, U_2\}$, and so we can remove $U_3$ from consideration. Next we check $U_4$.
        """
    )
    return


@app.cell
def _(lag, np):
    U_124 = np.array([[1,2,5],[2,-3,-4],[-1,3,4],[3,-2,-1]])
    U_124_reduced = lag.FullRowReduction(U_124)
    print(U_124_reduced)
    return U_124, U_124_reduced


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        $U_1$, $U_2$, and $U_4$ are linearly independent and $U_3$ can be expressed as a linear combination of the other three. Therefore the subspace of $\mathbb{R}^4$ spanned by $\{U_1,U_2,U_4\}$ is three-dimensional.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 5:** Determine whether the set of vectors $ \{V_1, V_2, V_3\}$ is a basis for $\mathbb{R}^4$. If not, find a vector which can be added to the set such that the resulting set of vectors is a basis for $\mathbb{R}^4$.

        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} 1 \\ 2 \\ 1 \\ 1  \end{array}\right] \hspace{0.7cm} 
        V_2 = \left[ \begin{array}{r} 1 \\ 0 \\ 2 \\ 2 \end{array}\right] \hspace{0.7cm}
        V_3 = \left[ \begin{array}{r} 1 \\ 3 \\ 1 \\ 2 \end{array}\right] \hspace{0.7cm}
        \end{equation}
        $$


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        The matrix $A$ which contains the given vectors as its columns is as follows:

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrrr} 1 & 1 & 1  \\ 2 & 0 & 3  \\ 1 & 2 & 1 \\ 1 & 2 & 2 \end{array}\right] 
        \end{equation}
        $$

        Since there are four rows and three columns, there can be a maximum of three pivots. This means that at least one of the rows will not have a pivot, which implies that the set of vectors does not span $\mathbb{R}^4$. The set $ \{V_1, V_2, V_3\}$ is not a basis for $\mathbb{R}^4$.

        Let us consider another vector $V_4$ in $\mathbb{R}^4$ and check if the set of vectors $\{V_1,V_2,V_3,V_4\}$ is a basis for $\mathbb{R}^4$.

        $$
        \begin{equation}
        V_4 = \left[ \begin{array}{r} 1 \\ 0 \\ 1 \\ 2 \end{array}\right] \
        \end{equation}
        $$

        We build a matrix $A$ that has these vectors as its columns and then perform row reduction to find the location of the pivots.
        """
    )
    return


@app.cell
def _(lag, np):
    _A = np.array([[1, 1, 1, 1], [2, 0, 3, 0], [1, 2, 1, 1], [1, 2, 2, 2]])
    _A_red = lag.FullRowReduction(_A)
    print('A: \n', _A, '\n')
    print('A_reduced: \n', _A_red, '\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since there is a pivot in each row, the set of vectors spans $\mathbb{R}^4$. Since there is a pivot in each column, the set of vectors is linearly independent. Hence, $\{V_1, V_2, V_3,V_4\}$ is a basis for $\mathbb{R}^4$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 6:** Find the dimension of the subspace spanned by $\{W_1, W_2\}$. Explain your answer.

        $$
        \begin{equation}
        W_1 = \left[ \begin{array}{r} 1 \\ 2 \\ 0 \\ 0  \end{array}\right] \hspace{0.7cm} 
        W_2 = \left[ \begin{array}{r} 2 \\ 3 \\ 0 \\ 0 \end{array}\right] \hspace{0.7cm}
        \end{equation}
        $$


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        The matrix $P$ which contains $W_1$, $W_2$ as its columns is as follows:

        $$
        \begin{equation}
        P = \left[ \begin{array}{rrrr} 1 & 2  \\ 2 & 3  \\  0 & 0 \\ 0 & 0 \end{array}\right] 
        \end{equation}
        $$

        """
    )
    return


@app.cell
def _(lag, np):
    _P = np.array([[1, 2], [2, 3], [0, 0], [0, 0]])
    _P_reduced = lag.FullRowReduction(_P)
    print('P: \n', _P, '\n')
    print('P_reduced: \n', _P_reduced, '\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since only two rows of the matrix $P$ contain a pivot, the basis for the subspace will contain two vectors. Hence, the dimension of the subspace spanned by $\{W_1, W_2\}$ is 2.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 7:** Find the value(s) of $a$ for which the set of vectors $\{X_1,X_2,X_3\}$ is **not** a basis for $\mathbb{R}^3$.

        $$
        \begin{equation}
        X_1 = \left[ \begin{array}{r} 1 \\ 2 \\ 1   \end{array}\right] \hspace{0.7cm} 
        X_2 = \left[ \begin{array}{r} 2 \\ a \\ 3  \end{array}\right] \hspace{0.7cm}
        X_3 = \left[ \begin{array}{r} 1 \\ 2 \\ a \end{array}\right] \hspace{0.7cm}
        \end{equation}
        $$


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        The matrix $S$ which contains the given vectors as its columns is as follows:

        $$
        \begin{equation}
        S = \left[ \begin{array}{rrrr} 1 & 2 & 1  \\ 2 & a & 2  \\ 1 & 3 & a  \end{array}\right] 
        \end{equation}
        $$

        The row reduced form of $S$ looks like:

        $$
        \begin{equation}
        \left[ \begin{array}{rrrr} 1 & 0 & 0  \\ 0 & a - 4 & 0  \\ 0 & 0 & a - 1  \end{array}\right] 
        \end{equation}
        $$

        The set of vectors $\{X_1, X_2, X_3\}$ is not be a basis for $\mathbb{R}^3$ if there is not a pivot in each row. If $ a = 4$, there is not a pivot in the second row. If $ a= 1$, there is no pivot in the third row.  We can conclude that the set of vectors $\{X_1, X_2, X_3\}$ is not a basis for $\mathbb{R}^3$ if $a = 1$ or $a = 4$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 8:** Let $U$ be the subspace of $\mathbb{R}^5$ which contains vectors with their first and second enteries same and the third entry as zero. What the vectors in the subspace $U$ look like? Use this information to find a basis for $U$ and determine the dimension of $U$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        The vectors in $U$ have the form

        $$
        \begin{equation}
        \left[ \begin{array}{r} a \\ a \\ 0 \\ b \\ c  \end{array}\right] \
        \end{equation}
        $$ 

        where $a$, $b$, and $c$ are arbitrary real numbers.

        $$
        \begin{equation}
        \left[ \begin{array}{r} a \\ a \\ 0 \\ b \\ c \end{array}\right]  =  
        a\left[ \begin{array}{r} 1 \\ 1 \\ 0 \\ 0 \\ 0 \end{array}\right]  +
        b\left[ \begin{array}{r} 0 \\ 0 \\ 0 \\ 1 \\ 0  \end{array}\right]  +
        c\left[ \begin{array}{r} 0 \\ 0 \\ 0\\ 0 \\ 1 \end{array}\right]\end{equation}
        $$

        If 

        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} 1 \\ 1 \\ 0 \\ 0 \\ 0  \end{array}\right] \hspace{0.7cm} 
        V_2 = \left[ \begin{array}{r} 0 \\ 0 \\ 0 \\ 1 \\ 0 \end{array}\right] \hspace{0.7cm}
        V_3 = \left[ \begin{array}{r} 0 \\ 0 \\ 0 \\ 0 \\ 1 \end{array}\right] \hspace{0.7cm}
        \end{equation}
        $$ 

        then $\{V_1, V_2, V_3\}$ is the basis for the subspace $U$.  Since there are three vectors in the basis for the subspace $U$, the dimension of $U$ is 3.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 9:** Let $\beta = \{U_1,U_2,U_3\}$ be a basis for $\mathbb{R}^3$. Find the **coordinates** of $V$ with respect to $\beta$.

        $$
        \begin{equation}
        U_1 = \left[ \begin{array}{r} 1 \\ 2 \\ 3   \end{array}\right] \hspace{0.7cm} 
        U_2 = \left[ \begin{array}{r} 2 \\ 1 \\ 0  \end{array}\right] \hspace{0.7cm}
        U_3 = \left[ \begin{array}{r} 3 \\ 2 \\ 5 \end{array}\right] \hspace{0.7cm}
        V = \left[ \begin{array}{r} 8 \\ 6 \\ 8 \end{array}\right] \hspace{0.7cm}
        \end{equation}
        $$


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:** 

        In order to find the **coordinates** of $V$ with respect to $\beta$ we need to find the linear combination of $U_1$, $U_2$ and $U_3$ that equals $V$.

        $$
        \begin{equation}
        \left[ \begin{array}{r} 8 \\ 6 \\ 8 \end{array}\right]  =  
        c_1\left[ \begin{array}{r} 1 \\ 2 \\ 3  \end{array}\right]  +
        c_2\left[ \begin{array}{r} 2 \\ 1 \\ 0   \end{array}\right]  +
        c_3\left[ \begin{array}{r} 3 \\ 2 \\ 5 \end{array}\right]\end{equation}
        $$

        That is, we need to solve the system $A[V]_{\alpha} = V$ where $A$ is the matrix with $U_1$ , $U_2$ and $U_3$ as its columns, and $[V]_{\alpha}$ is the vector of the unknown coefficients $c_1$, $c_2$, $c_3$.
        """
    )
    return


@app.cell
def _(lag, np):
    _A = np.array([[1, 2, 3], [2, 1, 2], [3, 0, 5]])
    V = np.array([[8], [6], [8]])
    X_alpha = lag.SolveSystem(_A, V)
    print('A: \n', _A, '\n')
    print('V: \n', V, '\n')
    print('X_alpha: \n', X_alpha, '\n')
    return V, X_alpha


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 10:** Can a set of four vectors in $\mathbb{R}^3$ be a basis for $\mathbb{R}^3$? Explain and verify your answer through a computation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        If we have four vectors in $\mathbb{R}^3$ and we use them as the columns of a matrix $A$, then we get a $ 3 \times 4 $ matrix. This means that there can be at most three pivots since there are only three rows. Therefore, there will always be a column without a pivot, which means the set will be linearly dependent. Hence, a set of four vectors cannot be a basis for $\mathbb{R}^3$. 

        We can demonstrate with some arbitrary vectors $V_1$, $V_2$, $V_3$ and $V_4$ in $\mathbb{R}^3$.

        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} 1 \\ 2 \\ 1  \end{array}\right] \hspace{0.7cm} 
        V_2 = \left[ \begin{array}{r} 1 \\ 0 \\ 2  \end{array}\right] \hspace{0.7cm}
        V_3 = \left[ \begin{array}{r} 1 \\ 3 \\ 1  \end{array}\right] \hspace{0.7cm}
        V_4 = \left[ \begin{array}{r} 0 \\ 1 \\ 1  \end{array}\right] \
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    _A = np.array([[1, 1, 1, 0], [2, 0, 3, 1], [1, 2, 1, 1]])
    _A_red = lag.FullRowReduction(_A)
    print('A: \n', _A, '\n')
    print('A_reduced: \n', _A_red, '\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can see from the computation that there is no pivot in the fourth column. Therefore, the set of vectors $\{V_1, V_2, V_3, V_4\}$ is linearly dependent and therefore not a basis for $\mathbb{R}^3$.  Any set of four vectors in $\mathbb{R}^3$ must be linearly dependent.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Vector Space Examples


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 1:** Determine whether or not the set of polynomials $\{p_1, p_2, p_3\}$ is a basis for $\mathbb{P}_2$.

        $$
        \begin{eqnarray*}
        p_1 & = & 3x^2 + 2x + 1 \\
        p_2 & = & 2x^2 + 5x + 3 \\
        p_3 & = & 6x^2 + 4x  +5 
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        A set of polynomials is a basis for $\mathbb{P}_2$ if every polynomial in $\mathbb{P}_2$ can be written as a linear combination of the basis polynomials in a unique way.

        Every polynomial in $\mathbb{P}_2$ is of the form: $ax^2 + bx + c$.

        If $\{p_1,p_2,p_3\}$ is a basis for $\mathbb{P}_2$, then

        $ax^2 + bx + c = c_1(3x^2 + 2x + 1) + c_2(2x^2 + 5x + 3) + c_3(6x^2 + 4x  +5)$.

        The above equation gives us the following system of equations:

        $$
        \begin{eqnarray*}
        3c_1 + 2c_2 + 6c_3  & = & a \\
        2c_1 +5c_2 + 4c_3  & = & b \\
        c_1 + 3c_2 + 5c_3  & = & c  \\
        \end{eqnarray*}
        $$

        The corresponding system is:

        $$
        \begin{equation}
        AX = \left[ \begin{array}{rrr} 3 & 2 & 6 \\ 2 & 5 & 4 \\ 1 & 3 & 5 \end{array}\right]
        \left[ \begin{array}{r} c_1 \\ c_2 \\ c_3 \end{array}\right]=
        \left[ \begin{array}{r} a \\ b \\ c  \end{array}\right]= B
        \end{equation}
        $$

        The system has at least one solution if the coefficient matrix $A$ has a pivot in every row.  The system has at most one solution if $A$ has a pivot in every column.
        """
    )
    return


@app.cell
def _(lag, np):
    _A = np.array([[3, 2, 6], [2, 5, 4], [1, 3, 5]])
    _A_reduced = lag.FullRowReduction(_A)
    print('A: \n', _A, '\n')
    print('A_Reduced: \n', _A_reduced, '\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since there is a pivot in each row and each column of $A$, the system will have a unique solution for any given values of $a$, $b$, and $c$.  This implies that each polynomial in $\mathbb{P}_2$ can be written as a linear combination of the vectors $p_1$, $p_2$, and $p_3$ in a unique way.  We say that $\{p_1,p_2,p_3\}$ spans $\mathbb{P}_2$ and is linearly independent. Therefore, $\{p_1,p_2,p_3\}$ is a basis for $\mathbb{P}_2$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** Find the coordinates of $p_4$ with respect to the basis $\alpha\ = \{p_1, p_2, p_3\}$. 

        $$
        \begin{eqnarray*}
        p_1 & = & x^2 + x + 2 \\
        p_2 & = & 2x^2 + 4x + 0 \\
        p_3 & = & 3x^2  + 2x      +1 \\
        p_4 & = & 11x^2 + 13x + 4
        \end{eqnarray*}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        In order to find the coordinates of $p_4$ with respect to $\alpha$, we need to express $p_4$ as a linear combination of $p_1$, $p_2$ and $p_3$.

        So, $11x^2 + 13x + 4 = c_1(x^2 + x + 2) + c_2(2x^2 + 4x + 0) + c_3(3x^2  + 2x +1)$. We are looking for scalars $c_1$, $c_2$ and $c_3$ which make this equation true.

        The corresponding system of equations is:

        $$
        \begin{eqnarray*}
        c_1 + 2c_2 + 3c_3  & = & 11 \\
        c_1 +4c_2 + 2c_3  & = & 13\\
        2c_1 \,\,  \quad\quad    +  c_3  & = & 4\\
        \end{eqnarray*}
        $$

        We need to solve the following system:

        $$
        \begin{equation}
        AX = \left[ \begin{array}{rrr} 1 & 2 & 3 \\ 1 & 4 & 2 \\ 2 & 0 & 1 \end{array}\right]
        \left[ \begin{array}{r} c_1 \\ c_2 \\ c_3 \end{array}\right]=
        \left[ \begin{array}{r} 11 \\ 13 \\ 4  \end{array}\right]= B
        \end{equation}
        $$


        """
    )
    return


@app.cell
def _(lag, np):
    _A = np.array([[1, 2, 3], [1, 4, 2], [2, 0, 1]])
    _B = np.array([[11], [13], [4]])
    _X = lag.SolveSystem(_A, _B)
    print('A: \n', _A, '\n')
    print('B: \n', _B, '\n')
    print('X: \n', _X, '\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        From the computations we can conclude that the coordinates of $p_4$ with respect to $\alpha$ are $1$,$2$ and $2$.  We can express these coordinates together in a vector.

        $$
        \begin{equation}
        [p_4]_{\alpha} =
        \left[ \begin{array}{r} 1 \\ 2 \\ 2  \end{array}\right]
        \end{equation}
        $$


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** Demonstrate that a set of four polynomials in $\mathbb{P}_4$ cannot span $\mathbb{P}_4$ through a computation.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        Let us consider any four polynomials in $\mathbb{P}_4$.

        $$
        \begin{eqnarray*}
        p_1 & = & x^4 + x^3 + x^2 + x + 2  \\
        p_2 & = & x^4 + 2x^2 + x + 1 \\
        p_3 & = & 2x^4 + x^3 + x^2 + x + 2 \\
        p_4 & = & 3x^4 + x^3 + 2x^2 + x + 2
        \end{eqnarray*}
        $$

        If the four polynomials span $\mathbb{P}_4$, then we should be able to express every polynomial in $\mathbb{P}_4$ as a linear combination of $p_1$, $p_2$, $p_3$ and $p_4$.  An arbitrary polynomial in $\mathbb{P}_4$ has the form $ax^4 + bx^3 + cx^2 + dx  + e $. We are therefore concerned with solving the following equation.

        $$
        \begin{equation}
        ax^4 + bx^3 + cx^2 + dx  + e  = c_1(x^4 + x^3 + x^2 + x + 2 ) + c_2(x^4 + 2x^2 + x + 1) + c_3(2x^4 + x^3 + x^2 + x + 2) + c_4(3x^4 + x^3 + 2x^2 + x + 2)
        \end{equation}
        $$

        This equation is true for all values of $x$ only if the left and right side are the same polynomial.  Gathering the like terms on the right side gives a system of equations.

        $$
        \begin{eqnarray*}
        c_1 + c_2 + 2c_3 + 3c_4 & = & a \\
        c_1 +\,\,\,\,\,\,\,\,\,\,\,\,c_3 \,\,+ \,\,\,c_4 & = & b \\
        c_1 + 2c_2 + c_3 + 2c_4 & = & c  \\
        c_1 + \,c_2 \,+\, c_3\, +\, c_4 & = & d \\
        2c_1 + c_2 + 2c_3 + 2c_4 & = & e \\
        \end{eqnarray*}
        $$

        We write the system as a matrix equation and observe that the coefficient matrix $A$ is $5\times 4$.

        $$
        \begin{equation}
        AX = \left[ \begin{array}{rrr} 1 & 1 & 2 & 3 \\ 1 & 0 & 1 & 1 \\ 1 & 2 & 1 & 2 \\ 1 & 1 & 1 & 1 \\2 & 1 & 2 &2\end{array}\right]
        \left[ \begin{array}{r} c_1 \\ c_2 \\ c_3 \\c_4\end{array}\right]=
        \left[ \begin{array}{r} a \\ b \\ c \\d\\e \end{array}\right]= B
        \end{equation}
        $$

        The system is consistent for any vector $B$ if there is a pivot in each row of $A$.
        """
    )
    return


@app.cell
def _(lag, np):
    _A = np.array([[1, 1, 2, 3], [1, 0, 1, 1], [1, 2, 1, 2], [1, 1, 1, 1], [2, 1, 2, 2]])
    _A_red = lag.FullRowReduction(_A)
    print('A: \n', _A, '\n')
    print('A_red: \n', _A_red, '\n')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The computations show that there is no pivot in the fifth row, which implies that the set $\{ p_1, p_2, p_3, p_4 \}$ does not span $\mathbb{P}_4$.  In general, a $5\times 4$ matrix can have at most $4$ pivots and can thus never have a pivot in each row.  We can therefore conclude that *any* set of four polynomials does not span $\mathbb{P}_4$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 4:** The set of matrices $\{A, B\}$ form a basis for a subspace of $\mathbb{M}_{2\times 2}$. Find a matrix which is in the subspace (but is not $A$ or $B$) and a matrix which is not in the subspace. Verify your answer.


        $$
        \begin{equation}
        A = \left[ \begin{array}{ccc} 1 & 0  \\ 2 & 0  \end{array}\right] \hspace{1cm}
        B = \left[ \begin{array}{ccc} 4 & 0 \\ 5 & 0  \end{array}\right] \hspace{1cm}
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        Since $A$ and $B$ are the basis for a subspace of $\mathbb{M}_{2\times 2}$, any matrix which can be formed from the linear combination of $A$ and $B$ will also be in the subspace. A matrix which cannot be represented as a linear combination of $A$ and $B$ will not be in the subspace.

        For instance, let

        $$
        \begin{equation}
        C = 2A + B = 
        2\left[ \begin{array}{ccc}1 & 0  \\2 & 0  \end{array}\right] +
        \left[ \begin{array}{ccc} 4& 0 \\ 5 &0  \end{array}\right] =
        \left[ \begin{array}{ccc} 6 & 0 \\ 
        9 & 0 \end{array}\right]
        \end{equation}
        $$

        Since $C$ is formed from linear combination of $A$ and $B$, $C$ lies in the subspace.

        Let $c_1$, $c_2$ be two scalars.

        Linear combination of $A$ and $B$ looks like:

        $$
        \begin{equation}
        c_1\left[ \begin{array}{ccc}1 & 0  \\2 & 0  \end{array}\right] +
        c_2\left[ \begin{array}{ccc} 4& 0 \\ 5 &0  \end{array}\right] =
        \left[ \begin{array}{ccc} c_1 + 4c_2 & 0 \\ 
        2c_1+5c_2 & 0 \end{array}\right]
        \end{equation}
        $$

        Therefore, the matrix $D$ does not lie in the subspace because it cannot be represented as a linear combination of $A$ and $B$ for any values of $c_1$ and $c_2$.

        $$
        \begin{equation}
        D = \left[ \begin{array}{ccc} 1 & 1  \\ 2 & 3  \end{array}\right] \hspace{1cm}
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 5:** Find the **coordinate vector** of $F$ with respect to the basis $\beta = \{A,B,C,D\}$ for $\mathbb{M}_{2\times 2}$.

        $$
        \begin{equation}
        A = \left[ \begin{array}{ccc} 1 & 0  \\ 0 & 1  \end{array}\right] \hspace{1cm}
        B = \left[ \begin{array}{ccc} 2 & 1  \\ 2 & 2  \end{array}\right] \hspace{1cm}
        C = \left[ \begin{array}{ccc} 3 & 0 \\ 1 & 4   \end{array}\right] \hspace{1cm}
        D = \left[ \begin{array}{ccc} 3 & 4\\ 1 & 1   \end{array}\right] \hspace{1cm}
        F = \left[ \begin{array}{ccc} 14 & 10\\ 7 & 11   \end{array}\right] \hspace{1cm}
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        The coordinate vector of $F$ with respect to $\beta$ is a vector in $\mathbb{R}^4$ whose components are the coordinates of $F$ with respect to  $\beta$.

        Let us represent $F$ as a linear combination of $A$, $B$, $C$, and $D$.

        $$
        \begin{equation}
        c_1\left[ \begin{array}{ccc}1 & 0  \\0 & 1  \end{array}\right] +
        c_2\left[ \begin{array}{ccc} 2& 1 \\ 2 &2  \end{array}\right] +
        c_3\left[ \begin{array}{ccc} 3 & 0 \\ 1 & 4 \end{array}\right] + 
        c_4\left[ \begin{array}{ccc} 3 & 4 \\ 1 & 1 \end{array}\right]  = 
        \left[ \begin{array}{ccc} 14 & 10 \\ 7 & 11 \end{array}\right]  
        \end{equation}
        $$

        The corresponding linear system is:

        $$
        \begin{equation}
        PX = \left[ \begin{array}{rrr} 1 & 2 & 3 & 3 \\ 0 & 1 & 0 & 4\\ 0 & 2 & 1 & 1 \\ 1 & 2 & 4 & 1\end{array}\right]
        \left[ \begin{array}{r} c_1 \\ c_2 \\ c_3 \\ c_4 \end{array}\right]=
        \left[ \begin{array}{r} 14 \\ 10 \\ 7 \\11  \end{array}\right]= Q
        \end{equation}
        $$

        Here, the columns of $P$ are the coordinate vectors of the given matrices $A$, $B$, $C$, and $D$ with respect to the standard basis of $\mathbb{M}_{2\times 2}$ and $X$ represents the coordinate vector of $F$ with respect to $\beta$.

        Therefore, $X = \left[F\right]_{\beta}$.

        """
    )
    return


@app.cell
def _(lag, np):
    _P = np.array([[1, 2, 3, 3], [0, 1, 0, 4], [0, 2, 1, 1], [1, 2, 4, 1]])
    Q = np.array([[14], [10], [7], [11]])
    _X = lag.SolveSystem(_P, Q)
    print('P: \n', _P, '\n')
    print('Q: \n', Q, '\n')
    print('X: \n', _X, '\n')
    return (Q,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 6:** Let $\mathbb{D}_{2\times 2}$ be the set of $ 2 \times 2 $ diagonal matrices. 

        ($a$)  Explain why $\mathbb{D}_{2\times 2}$ is a subspace of $\mathbb{M}_{2\times 2}$.

        ($b$)  Find a basis for $\mathbb{D}_{2\times 2}$.

        ($c$)  Determine the dimension of $\mathbb{D}_{2\times 2}$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        $(a)$ 

        Let us consider two arbitrary matrices $X$ and $Y$ in $\mathbb{D}_{2\times 2}$:

        $$
        \begin{equation}
        A = \left[ \begin{array}{ccc} a & 0  \\ 0 & b  \end{array}\right] \hspace{1cm}
        B = \left[ \begin{array}{ccc} c & 0 \\ 0 & d  \end{array}\right] \hspace{1cm}
        \end{equation}
        $$

        $$
        \begin{equation}
        A + B = \left[ \begin{array}{ccc} a + c & 0  \\ 0 & b + d \end{array}\right] 
        \end{equation}
        $$ 

        $$
        \begin{equation}
        kA  = \left[ \begin{array}{ccc} ka & 0  \\ 0 & kb  \end{array}\right] 
        \end{equation}
        $$ 

        For any two arbitrary matrices $X$ and $Y$ in $\mathbb{D}_{2\times 2}$, $X+Y$ and $kA$ are also in $\mathbb{D}_{2\times 2}$. Therefore, $\mathbb{D}_{2\times 2}$ is a subspace of $\mathbb{M}_{2\times 2}$.





        $(b)$ 

        Any matrix in $\mathbb{D}_{2\times 2}$ has the form:

        $$
        \begin{equation}
        \left[ \begin{array}{ccc} a & 0  \\ 0 & b  \end{array}\right] \hspace{1cm}
        \end{equation}
        $$ where $a$ and $b$ are some scalars.

        This matrix can also be written as:

        $$
        \begin{equation}
        \left[ \begin{array}{ccc}a & 0  \\0 & b  \end{array}\right] =
        a\left[ \begin{array}{ccc} 1& 0\\ 0 &0  \end{array}\right] +
        b\left[ \begin{array}{ccc} 0 & 0 \\ 0 & 1 \end{array}\right]  
        \end{equation}
        $$

        Let $$
        \begin{equation}
        P = \left[ \begin{array}{ccc} 1 & 0  \\ 0 & 0  \end{array}\right] \hspace{1cm}
        Q = \left[ \begin{array}{ccc} 0 & 0  \\ 0 & 1 \end{array}\right] \hspace{1cm}
        \end{equation}
        $$

        Since every matrix in $\mathbb{D}_{2\times 2}$ can be written as a unique linear combination of $P$ and $Q$, $\{P,Q\}$ is a basis for $\mathbb{D}_{2\times 2}$.





        $(c)$ 

        Since there are two matrices in the basis for $\mathbb{D}_{2\times 2}$, the dimension of $\mathbb{D}_{2\times 2}$ is 2. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Applications
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Set up and solve a linear system to balance each chemical equation.  

        **Exercise 1:** Combustion of methane:

        $$
        \begin{equation}
        C_3H_8 + O_2 \to CO_2 + H_2O
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        We will denote each molecule as a vector whose first entry is the number of carbon atoms, the second the number of oxygen atoms, and the third the number of hydrogen atoms. Then the chemical equation above can be expressed as the following vector equation

        $$
        \begin{equation}
        x_1\left[\begin{array}{c} 3 \\ 0 \\ 8 \end{array}\right]
        + x_2\left[\begin{array}{c} 0 \\ 2 \\ 0 \end{array}\right]
        = x_3\left[\begin{array}{c} 1 \\ 2 \\ 0 \end{array}\right]
        + x_4\left[\begin{array}{c} 0 \\ 1 \\ 2 \end{array}\right]
        \end{equation}
        $$

        If we move everything to the left side then we get the following homogeneous system

        $$
        \begin{equation}
        \left[\begin{array}{rrr} 3 & 0 & -1 & 0 \\ 0 & 2 & -2 & -1 \\ 8 & 0 & 0 & -2 \end{array}\right]
        \left[\begin{array}{r} x_1 \\ x_2 \\ x_3 \\ x_4 \end{array}\right]
        = \left[\begin{array}{r} 0 \\ 0 \\ 0  \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    _A = np.array([[3, 0, -1, 0], [0, 2, -2, -1], [8, 0, 0, -2]])
    _A_reduced = lag.FullRowReduction(_A)
    print(_A_reduced)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since there is no pivot in the column corresponding to $x_4$, we treat $x_4$ as a free variable.  Since we want solutions that are integers, we will take $x_4=4$. This gives $x_1=1$, $x_2=5$, and $x_3=3$.  The correct equation for the chemical reaction is the following.

        $$
        \begin{equation}
        C_3H_8 + 5O_2 \to 3CO_2 + 4H_2O
        \end{equation}
        $$

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 2:** Aluminum reaction with sulfuric acid 

        $$
        \begin{equation}
        Al + H_2SO_4 \to Al_2(SO_4)_3 + H_2
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        We will denote each molecule as a vector whose first entry is the number of aluminum atoms, the second the number of oxygen atoms, the third the number of hydrogen atoms, and the fourth the number of sulfur atoms. Then the chemical equation above can be expressed as the following vector equation

        $$
        \begin{equation}
        x_1\left[\begin{array}{c} 1 \\ 0 \\ 0 \\ 0 \end{array}\right]
        + x_2\left[\begin{array}{c} 0 \\ 4 \\ 2 \\ 1 \end{array}\right]
        = x_3\left[\begin{array}{c} 2 \\ 12 \\ 0 \\ 3 \end{array}\right]
        + x_4\left[\begin{array}{c} 0 \\ 0 \\ 2 \\ 0 \end{array}\right]
        \end{equation}
        $$

        If we move everything to the left side then we get the following homogeneous system


        $$
        \begin{equation}
        \left[\begin{array}{rrr} 1 & 0 & -2 & 0 \\ 0 & 4 & -12 & 0 \\ 0 & 2 & 0 & -2 \\ 0 & 1 & -3 & 0 \end{array}\right]
        \left[\begin{array}{r} x_1 \\ x_2 \\ x_3 \\ x_4 \end{array}\right]
        = \left[\begin{array}{r} 0 \\ 0 \\ 0 \\ 0  \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    _B = np.array([[1, 0, -2, 0], [0, 4, -12, 0], [0, 2, 0, -2], [0, 1, -3, 0]])
    _B_reduced = lag.FullRowReduction(_B)
    print(_B_reduced)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since there is no pivot in the column corresponding to $x_4$, we treat $x_4$ as a free variable. Since we want solutions that are integers, we will take $x_4 = 3$. This gives $x_1=2$, $x_2=3$, and $x_3=1$.  The correct equation for the chemical reaction is the following.

        $$
        \begin{equation}
        2Al + 3H_2SO_4 \to Al_2(SO_4)_3 + 3H_2
        \end{equation}
        $$

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** Silver tarnish

        $$
        \begin{equation}
        Ag + H_2S + O_2 \to Ag_2S + H_2O
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Solution:**

        We will denote each molecule as a vector whose first entry is the number of silver atoms, the second the number of oxygen atoms, the third the number of hydrogen atoms, and the fourth the number of sulfur atoms. Then the chemical equation above can be expressed as the following vector equation

        $$
        \begin{equation}
        x_1\left[\begin{array}{c} 1 \\ 0 \\ 0 \\ 0 \end{array}\right]
        + x_2\left[\begin{array}{c} 0 \\ 0 \\ 2 \\ 0 \end{array}\right]
        + x_3\left[\begin{array}{c} 0 \\ 2 \\ 0 \\ 0 \end{array}\right]
        = x_4\left[\begin{array}{c} 2 \\ 0 \\ 0 \\ 1 \end{array}\right]
        + x_5\left[\begin{array}{c} 0 \\ 1 \\ 2 \\ 0 \end{array}\right]
        \end{equation}
        $$

        If we move everything to the left side then we get the following homogeneous system

        $$
        \begin{equation}
        \left[\begin{array}{rrr} 1 & 0 & 0 & -2 & 0 \\ 0 & 0 & 2 & 0 & -1 \\ 0 & 2 & 0 & 0 & -2 \\ 0 & 1 & 0 & -1 & 0 \end{array}\right]
        \left[\begin{array}{r} x_1 \\ x_2 \\ x_3 \\ x_4 \\ x_5 \end{array}\right]
        = \left[\begin{array}{r} 0 \\ 0 \\ 0 \\ 0  \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    _C = np.array([[1, 0, 0, -2, 0], [0, 0, 2, 0, -1], [0, 2, 0, 0, -2], [0, 1, 0, -1, 0]])
    _C_reduced = lag.FullRowReduction(_C)
    print(_C_reduced)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since there is no pivot in the column corresponding to $x_5$, we treat $x_5$ as a free variable. Since we want soltions that are integers, we will take $x_5 = 2$ This gives $x_1=4$, $x_2=2$, $x_3 = 1$, and $x_4=2$.  The correct equation for the chemical reaction is the following.

        $$
        \begin{equation}
        4Ag + 2H_2S + O_2 \to 2Ag_2S + 2H_2O
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
