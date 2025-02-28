import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Linear Independence
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A set of vectors $\{V_1, V_2, V_3, ... V_n\}$ is said to be **linearly independent** if no linear combination of the vectors is equal to zero, except the combination with all weights equal to zero.  Thus if the set is linearly independent and 

        $$
        \begin{equation}
        c_1V_1 + c_2V_2 + c_3V_3 + .... + c_nV_n = 0
        \end{equation}
        $$

        it must be that $c_1 = c_2 = c_3 = .... = c_n = 0$.  Equivalently we could say that the set of vectors is linearly independent if there is *no vector in the set* that is equal to a linear combination of the others.  If a set of vectors is not linearly independent, then we say that it is **linearly dependent**.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 1:  Vectors in $\mathbb{R}^2$

        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} 2 \\ 1 \end{array}\right] \hspace{1cm} 
        V_2 = \left[ \begin{array}{r} 1 \\ -6  \end{array}\right] \hspace{1cm}
        \end{equation}
        $$

        In order to determine if this set of vectors is linearly independent, we must examine the following vector equation.

        $$
        \begin{equation}
        c_1\left[ \begin{array}{r} 2 \\ 1 \end{array}\right] +
        c_2\left[ \begin{array}{r} 1 \\ -6  \end{array}\right] =
        \left[ \begin{array}{r} 0 \\ 0 \end{array}\right]\end{equation}
        $$


        """
    )
    return


@app.cell
def _():
    import numpy as np
    import laguide as lag
    _A_augmented = np.array([[2, 1, 0], [1, -6, 0]])
    print(lag.FullRowReduction(_A_augmented))
    return lag, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We see from the reduced augmented matrix that the only solution to the equation is $c_1 = c_2 = 0$.  The set $\{V_1, V_2\}$ is linearly independent.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2:  Vectors in $\mathbb{R}^3$

        $$
        \begin{equation}
        W_1 = \left[ \begin{array}{r} 2 \\ -1  \\ 1 \end{array}\right] \hspace{1cm} 
        W_2 = \left[ \begin{array}{r} 1 \\ -4 \\ 0  \end{array}\right] \hspace{1cm}
        W_3 = \left[ \begin{array}{r} 3 \\ 2 \\ 2  \end{array}\right] \hspace{1cm}
        \end{equation}
        $$

        Again, we must examine the solution to a vector equation.

        $$
        \begin{equation}
        c_1\left[ \begin{array}{r} 2 \\ -1 \\ 1 \end{array}\right] +
        c_2\left[ \begin{array}{r} 1 \\ -4 \\ 0  \end{array}\right] +
        c_3\left[ \begin{array}{r} 3 \\ 2 \\ 2  \end{array}\right] =
        \left[ \begin{array}{r} 0 \\ 0 \\ 0\end{array}\right]\end{equation}
        $$

        """
    )
    return


@app.cell
def _(lag, np):
    B_augmented = np.array([[2,1,3,0],[-1,-4,2,0],[1,0,2,0]])
    print(lag.FullRowReduction(B_augmented))
    return (B_augmented,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this case we see that $c_3$ is a free variable.  If we set $c_3 = 1$, then $c_2 = 1$, and $c_1 = -2$.  Since we are able to find a solution other than $c_1 = c_2 = c_3 = 0$, the set of vectors $\{W_1, W_2, W_3\}$ is linearly dependent.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Homogeneous systems
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        A linear system is said to be **homogeneous** if it can be described with the matrix equation $AX = 0$.  The solution to such a system has a connection to the solution of the system $AX=B$.  The homogeneous system also has a connection to the concept of linear independence.  If we link all of these ideas together we will be able to gain information about the solution of the system $AX=B$, based on some information about linear independence.

        In the previous examples we were solving the vector equation $c_1V_1 + c_2V_2 + c_3V_3 + .... + c_nV_n = 0$ in order
        to determine if the set of vectors $\{V_1, V_2, V_3 .... V_n\}$ were linearly independent.  This vector equation represents a homogeneous linear system that could also be described as $AX=0$, where $V_1$, $V_2$, ... $V_n$ are the columns of the matrix $A$, and $X$ is the vector of unknown coefficients.  The set of vectors is linearly dependent if and only if the associated homogeneous system has a solution other than the vector with all entries equal to zero.  The vector of all zeros is called the **trivial solution**.  This zero vector is called a trivial solution because it is a solution to *every homogeneous system* $AX=0$, regardless of the entries of $A$.  For this reason, we are interested only in the existence of *nontrivial solutions* to $AX=0$.

        Let us suppose that a homogeneous system $AX=0$ has a nontrivial solution, which we could label $X_h$.  Let us also suppose that a related nonhomogeneous system $AX=B$ also has some particular solution, which we could label $X_p$.  So we have $AX_h = 0$ and $AX_p = B$.  Now by the properties of matrix multiplication, $X_p + X_h$ is also a solution to $AX=B$ since $A(X_p + X_h) = AX_p + AX_h = B + 0$. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Consider the following system as an example.

        $$
        \begin{equation}
        AX = \left[ \begin{array}{rrr} -3 & 2 & 1 \\ -2 & 1 & -1 \\ 4 & 3 & 3 \end{array}\right]
        \left[ \begin{array}{r} x_1 \\ x_2 \\ x_3 \end{array}\right]=
        \left[ \begin{array}{r} -6 \\ 1 \\ 13  \end{array}\right]= B
        \end{equation}
        $$

        We can look at the associated homogeneous system to determine if the columns of $A$ are linearly independent.

        $$
        \begin{equation}
        \left[ \begin{array}{rrr} -3 & 2 & 1 \\ -2 & 1 & -1 \\ 4 & 3 & 3 \end{array}\right]
        \left[ \begin{array}{r} c_1 \\ c_2 \\ c_3 \end{array}\right]=
        \left[ \begin{array}{r} 0 \\ 0 \\ 0  \end{array}\right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    _A_augmented = np.array([[-3, 2, 1, 0], [-2, 1, -1, 0], [4, -3, -3, 0]])
    _A_augmented_reduced = lag.FullRowReduction(_A_augmented)
    print(_A_augmented_reduced)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The absence of a pivot in the third column indicates that $c_3$ is a free variable, and that there exists a nontrivial solution to the homogeneous system.  One possibility is $c_1 = 3$, $c_2=5$, $c_3 = -1$.  It is worth noting here that it was unnecessary to carry out the row operations on the last column of the augmented matrix since all the entries are zero.  When considering homogeneous systems, finding the RREF of the coefficient matrix is enough.

        The fact that the homogeneous system has a nontrivial solution implies that the columns of $A$, if we think of them as vectors, are linearly dependent.  Based on the discussion then, we expect that if the $AX=B$ system has a solution, it will not be unique. 
        """
    )
    return


@app.cell
def _(lag, np):
    _A_augmented = np.array([[-3, 2, 1, -6], [-2, 1, -1, 1], [4, -3, -3, 13]])
    _A_augmented_reduced = lag.FullRowReduction(_A_augmented)
    print(_A_augmented_reduced)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The RREF represents two equations, and again the third variable is free.

        $$
        \begin{eqnarray*}
        x_1 \quad\quad + 3x_3 & = & -8\\
        x_2 + 5x_3 & = & = -15 
        \end{eqnarray*}
        $$

        To express the possible solutions, we can set $x_3 = t$, which gives $x_2 = -15 -5t$ and $x_1 = -8-3t$.  These components can be assembled into a vector that involves the parameter $t$.

        $$
        \begin{equation}
        X = \left[ \begin{array}{c} -8-3t \\ -15-5t  \\ t \end{array}\right]
        \end{equation}
        $$

        Splitting this vector into two pieces helps us connect this solution to that of the homogeneous system.


        $$
        \begin{equation}
        X  = 
        \left[ \begin{array}{c} -8 \\ -15  \\ 0 \end{array}\right] + 
        t\left[ \begin{array}{c} -3 \\ -5  \\ 1 \end{array}\right] = X_p + X_h
        \end{equation}
        $$

        We can check to see that $AX_p= B$, $AX_h= 0$ for any $t$, and $A(X_p+X_h) = B$ for any $t$.
        """
    )
    return


@app.cell
def _(np):
    A = np.array([[-3,2,1],[-2,1,-1],[4,-3,-3]])

    X_p = np.array([[-8],[-15],[0]])
    X_h = np.array([[-3],[-5],[1]])

    t = np.random.rand()
    X = X_p + t*X_h

    print(X)
    print('\n')
    print(A@X_p)
    print('\n')
    print(A@(t*X_h))
    print('\n')
    print(A@X)
    return A, X, X_h, X_p, t


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It is important to notice that the first three columns of the RREF for the augmented matrix of the homogeneous system are exactly the same as those of the RREF for the system $AX=B$.  Of course that must be the case since the first three columns come from the coefficient matrix $A$, which is the same in both systems.  The point here is that **the system $AX=B$ can only have a unique solution if the columns of $A$ are linearly independent**.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Null space

        With the concept of homogeneous systems in place, we are ready to define the second fundamental subspace.  If $A$ is an $m\times n$ matrix, the **null space** of $A$ is the set of vectors $X$, such that $AX=0$.  In other words, the null space of $A$ is the set of all solutions to the homogeneous system $AX=0$.  The null space of $A$ is a subspace of $\mathbb{R}^n$, and is written with the notation $\mathcal{N}(A)$.  We can now reformulate earlier statements in terms of the null space.  

        - The columns of a matrix $A$ are linearly independent if and only if $\mathcal{N}(A)$ contains only the zero vector.

        - The system $AX=B$ has at most one solution if and only if $\mathcal{N}(A)$ contains only the zero vector.

        Making connections between the fundamental subspaces of $A$ and the solution sets of the system $AX=B$ allows us to make general conclusions that further build our understanding of linear systems, and the methods by which we might solve them. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercises

        **Exercise 1:** Determine if the set of vectors $ \{U_1, U_2, U_3\} $is linearly independent.

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


@app.cell
def _():
    ## Code solution here.
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


@app.cell
def _():
    ## Code solution here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** Find the value of $b$ for which the given vectors are linearly dependent. Then, represent one vector as the linear combination of the other two.

        $$
        \begin{equation}
        X_1 = \left[ \begin{array}{r} 1 \\ 0  \\ 1  \end{array}\right] \hspace{1cm} 
        X_2 = \left[ \begin{array}{r} 1 \\ 2\\ 3  \end{array}\right] \hspace{1cm}
        X_3 = \left[ \begin{array}{r} 2 \\ 4 \\ b   \end{array}\right] \
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


@app.cell
def _():
    ## Code solution here
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


@app.cell
def _():
    ## Code solution here
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


@app.cell
def _():
    ## Code solution here
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


@app.cell
def _():
    ## Code solution here.
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


@app.cell
def _():
    ## Code Solution here
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
