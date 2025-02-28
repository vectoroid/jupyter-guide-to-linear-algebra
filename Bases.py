import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Bases
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        When working with vector spaces, or subspaces, it is often useful to express arbitrary vectors as linear combinations of a set of given vectors known as a basis.  A **basis** for a vector space (or subspace) is a collection of linearly independent vectors that span the space.  This definition pulls together the ideas of linear independence and spanning sets to describe a set of vectors which we can understand as building blocks for a vector space.   

        Given a set of vectors $\{V_1, V_2, V_3, ... V_n\}$ in $\mathbb{R}^n$, and any vector $Y$ in $\mathbb{R}^n$, let us consider the vector equation $c_1V_1 + c_2V_2 + ... c_nV_n = Y$.  This vector equation represents a linear system that could also be represented by the matrix equation $AX=Y$, where $A$ is the $n\times n$ matrix that has the vectors $V_i$ as its columns.  In previous sections we have made two important conclusions regarding this system.

        - $AX=Y$ has at least one solution if $\{V_1, V_2, V_3, ... V_n\}$ span $\mathbb{R}^n$.

        - $AX=Y$ has at most one solution if $\{V_1, V_2, V_3, ... V_n\}$ are linearly independent.

        Combining these statements tells us that $AX=Y$ has exactly one solution for each $Y$ in $\mathbb{R}^n$ if and only if the columns of $A$ are linearly independent and span the space $\mathbb{R}^n$.   In other words, the vector equation $c_1V_1 + c_2V_2 + ... c_nV_n = Y$ has a unique solution for any $Y$ in $\mathbb{R}^n$ if and only if the set of vectors $\{V_1, V_2, V_3, ... V_n\}$ is a basis for $\mathbb{R}^n$. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 1:  Basis for $\mathbb{R}^5$

        The following set of vectors form a basis for $\mathbb{R}^5$.

        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} -3 \\ 3 \\ 2 \\ -3 \\ -2 \end{array}\right] \hspace{0.7cm} 
        V_2 = \left[ \begin{array}{r} 3 \\ 3 \\ 2 \\ -1 \\ 3  \end{array}\right] \hspace{0.7cm}
        V_3 = \left[ \begin{array}{r} 2 \\ 0 \\ -2 \\ 2 \\ 2 \end{array}\right] \hspace{0.7cm}
        V_4 = \left[ \begin{array}{r} -3 \\ -1 \\ 2 \\ -1 \\ 3 \end{array}\right] \hspace{0.7cm}
        V_5 = \left[ \begin{array}{r} -2 \\ 0 \\ -3 \\ 3 \\ -2 \end{array}\right] 
        \end{equation}
        $$

        To verify these vectors form a basis, we need to check that they are linearly independent and that they span $\mathbb{R}^5$.  Following the previous discussion, we can verify these properties by determining the structure of the solution set for the system $AX=Y$, where $A$ is a $5\times 5$ matrix with the $V_i$ as its columns and $Y$ is an arbitrary vector in $\mathbb{R}^5$.  We know that $AX=Y$ will be consistent for arbitrary $Y$ if $A$ has a pivot in each row.  We know that $AX=Y$ will have at most one solution for arbitrary $Y$ if there are no free variables.  This implies that $A$ has a pivot in each column.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import laguide as lag
    _A = np.array([[-3, 3, 2, -3, -2], [3, 3, 0, -1, 0], [2, 2, -2, 2, -3], [-3, -1, 2, -1, 3], [-2, 3, 2, 3, -2]])
    print(lag.FullRowReduction(_A))
    return lag, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We see that the RREF of $A$ is the $5\times 5$ identity matrix, which indicates that $A$ has a pivot in each row and each column.  This verifies that the columns of $A$ form a basis for $\mathbb{R}^5$, and guarantees that the linear system $AX=Y$ has a unique solution for any vector $Y$. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2: Standard bases

        The **standard basis** for $\mathbb{R}^n$ is the set of vectors $\{E_1, E_2, ..., E_n\}$ that correspond to the columns of the $n\times n$ identity matrix $I$.  That is, $E_i$ is the vector with the $i$th entry equal to 1, and all other entries equal to 0.  The standard basis for $\mathbb{R}^4$ is shown here as an example.


        $$
        \begin{equation}
        E_1 = \left[ \begin{array}{r} 1 \\ 0 \\ 0 \\ 0 \end{array}\right] \hspace{0.7cm} 
        E_2 = \left[ \begin{array}{r} 0 \\ 1 \\ 0 \\ 0 \end{array}\right] \hspace{0.7cm}
        E_3 = \left[ \begin{array}{r} 0 \\ 0 \\ 1 \\ 0 \end{array}\right] \hspace{0.7cm}
        E_4 = \left[ \begin{array}{r} 0 \\ 0 \\ 0 \\ 1 \end{array}\right]
        \end{equation}
        $$

        Clearly there is a unique solution to the system $IX=Y$ for every $Y$ in $\mathbb{R}^4$.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 3: Basis for a Subspace of $\mathbb{R}^3$

        The following set of vectors forms a basis for the subspace of $\mathbb{R}^3$ that consists of vectors with middle entry equal to zero.

        $$
        \begin{equation}
        U_1 = \left[ \begin{array}{r} 1 \\ 0 \\ 2 \end{array}\right] \hspace{1cm} 
        U_2 = \left[ \begin{array}{r} 0 \\ 0 \\ 1 \end{array}\right] 
        \end{equation}
        $$

        We need to check that any vector in $\mathbb{R}^3$ with middle entry equal to zero can be expressed as a linear combination of $U_1$ and $U_2$, and we need to check that $\{U_1, U_2 \}$ is a linearly independent set.  We define $A$ as a matrix with $U_1$ and $U_2$ as the columns.


        $$
        \begin{equation}
        A = \left[ \begin{array}{rr} 1 & 0 \\ 0 & 0 \\ 2 & 1 \end{array}\right] 
        \end{equation}
        $$

        To determine if $\{U_1, U_2\}$ is linearly independent, we need to find out if the homogeneous system $AX=0$ has any nontrivial solutions.  If both columns of $A$ contain a pivot, there are no nontrivial solutions and the vectors are linearly independent.  Since there are only two columns, it suffices to check that one column is not a scalar multiple of the other, but we will compute the RREF of $A$ for the sake of completeness.
        """
    )
    return


@app.cell
def _(lag, np):
    _A = np.array([[1, 0], [0, 0], [2, 1]])
    print(lag.FullRowReduction(_A))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        To determine if $U_1$ and $U_2$ span the subspace, we need to check if an arbitrary vector $Y$ with middle entry equal to zero can be expressed as a linear combination of the two vectors.  That is, given any values of $y_1$ and $y_3$, can we always find $c_1$ and $c_2$ to solve the vector equation?  

        $$
        \begin{equation}
        c_1\left[ \begin{array}{r} 1 \\ 0 \\ 2 \end{array}\right] +
        c_2\left[ \begin{array}{r} 0 \\ 0 \\  1 \end{array}\right] =
        \left[ \begin{array}{c} y_1 \\ 0 \\ y_3 \end{array}\right] 
        \end{equation}
        $$

        In this example it is clear that we can take $c_1 = y_1$ and $c_2 = y_3 - 2y_1$ by using substitution.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 4: Solution set of a homogeneous system

        Consider the homogeneous system $AX=0$ where $X$ is a vector in $\mathbb{R}^4$ and $A$ is the following $2\times 4$ matrix.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrrr} 1 & 0 & 3 & -2 \\ -2 & 1 & 3 & 0 \end{array}\right] 
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _(lag, np):
    _A = np.array([[1, 0, 3, -2], [-2, 1, 3, 0]])
    print(lag.FullRowReduction(_A))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this system, $x_3$ and $x_4$ are free variables.  If we set $x_3 = t$, and $x_4=s$, then $x_1 = 2s -3t$ and 
        $x_2 = 4s -9t$.  We can write the components of a general solution vector $X$ in terms of these parameters.

        $$
        \begin{equation}
        X = \left[ \begin{array}{r} x_1 \\ x_ 2 \\ x_ 3 \\ x_4 \end{array}\right] =  
        t\left[ \begin{array}{r} -3 \\ -9 \\  1 \\ 0 \end{array}\right] +
        s\left[ \begin{array}{r} 2 \\ 4 \\ 0 \\ 1  \end{array}\right] 
        \end{equation}
        $$

        In this form we can see that any solution of $AX=B$ must be a linear combination of the vectors $W_1$ and $W_2$ defined as follows:


        $$
        \begin{equation}
        W_1 = \left[ \begin{array}{r} -3 \\ -9 \\  1 \\ 0 \end{array}\right] \hspace{1cm}
        W_2 = \left[ \begin{array}{r} 2 \\ 4 \\ 0 \\ 1  \end{array}\right] 
        \end{equation}
        $$

        The set $\{W_1, W_2\}$ is linearly independent and forms a basis for the set of solutions to $AX=0$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Dimension

        We now examine some sets of vectors that do not form bases.  By looking at these non-examples we will uncover a concept that is related directly to bases.


        The following set of vectors is **not** a basis for $\mathbb{R}^5$.


        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} -3 \\ 3 \\ 2 \\ -3 \\ -2 \end{array}\right] \hspace{0.7cm} 
        V_2 = \left[ \begin{array}{r} 3 \\ 3 \\ 2 \\ -1 \\ 3  \end{array}\right] \hspace{0.7cm}
        V_3 = \left[ \begin{array}{r} 2 \\ 0 \\ -2 \\ 2 \\ 2 \end{array}\right] \hspace{0.7cm}
        V_4 = \left[ \begin{array}{r} -3 \\ -1 \\ 2 \\ -1 \\ 3 \end{array}\right] \hspace{0.7cm}
        V_5 = \left[ \begin{array}{r} -2 \\ 0 \\ -3 \\ 3 \\ -2 \end{array}\right] \hspace{0.7cm}
        V_6 = \left[ \begin{array}{r} -1 \\ 0 \\ 2 \\ 2 \\ 1 \end{array}\right] 
        \end{equation}
        $$

        Although we can assemble a $5\times 6$ matrix with these vectors as the columns, and compute the RREF as before, we should expect to find that the matrix does not have a pivot in each column due to its shape.  Recall that each row can have at most one pivot, which means that the matrix has at most 5 pivots.  Since there are 6 columns, one of the columns does not have a pivot.  This shows that the set of vectors is linearly dependent, and thus not a basis.
        """
    )
    return


@app.cell
def _(lag, np):
    _B = np.array([[-3, 3, 2, -3, -2, -1], [3, 3, 0, -1, 0, 0], [2, 2, -2, 2, -3, 2], [-3, -1, 2, -1, 3, 2], [-2, 3, 2, 3, -2, 1]])
    print(lag.FullRowReduction(_B))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The following set of vectors is **not** a basis for $\mathbb{R}^5$.


        $$
        \begin{equation}
        V_1 = \left[ \begin{array}{r} -3 \\ 3 \\ 2 \\ -3 \\ -2 \end{array}\right] \hspace{0.7cm} 
        V_2 = \left[ \begin{array}{r} 3 \\ 3 \\ 2 \\ -1 \\ 3  \end{array}\right] \hspace{0.7cm}
        V_3 = \left[ \begin{array}{r} 2 \\ 0 \\ -2 \\ 2 \\ 2 \end{array}\right] \hspace{0.7cm}
        V_4 = \left[ \begin{array}{r} -3 \\ -1 \\ 2 \\ -1 \\ 3 \end{array}\right] \hspace{0.7cm}
        \end{equation}
        $$

        The matrix made up of these column vectors has 5 rows and 4 columns.  There are at most 4 pivots since each column can contain no more than 1 pivot.  This means that there is at least one zero row in the RREF, which implies that the system $AX = B$ is not consistent for every $B$ in $\mathbb{R}^5$.
        """
    )
    return


@app.cell
def _(lag, np):
    _B = np.array([[-3, 3, 2, -3], [3, 3, 0, -1], [2, 2, -2, 2], [-3, -1, 2, -1], [-2, 3, 2, 3]])
    print(lag.FullRowReduction(_B))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        These two examples illustrate why any set of vectors that does not contain exactly five vectors cannot be a basis for $\mathbb{R}^5$.  We might suspect that exactly five vectors are needed because the space is five-dimensional, and this is almost correct.  It is more accurate to say that $\mathbb{R}^5$ is five-dimensional *because* a basis for the space must have five vectors.  The **dimension** of a vector space (or subspace) is defined as the number of vectors in any basis for the space.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Coordinates

        We now understand that if we are given a basis $\{V_1, V_2, V_3, ... V_n\}$ for $\mathbb{R}^n$ and a vector $X$ in $\mathbb{R}^n$, there is a unique linear combination of the basis vectors equal to $X$.  The **coordinates** of $X$ with respect to this basis is the unique set of weights $c_1$, $c_2$, ... $c_n$ that satisfy the vector equation $X=c_1V_1 + c_2V_2 + ... c_nV_n$.  It becomes useful at this point to assign labels to bases that are under discussion.  For example, we might say that $\beta = \{V_1, V_2, V_3, ... V_n\}$, and refer to the coordinates of $X$ with respect to $\beta$.  It is only natural to collect these weights into an $n\times 1$ array, to which we will assign the notation $[X]_{\beta}$.  Despite potential confusion, this array is referred to as the "coordinate vector".

        To demonstrate, suppose that we use the basis for $\mathbb{R}^5$ given in **Example 1**, and assign it the label $\beta$.  Consider then the calculation needed to find of the coordinates of a vector $X$ with respect to $\beta$.


        $$
        \begin{equation}
        X = \left[ \begin{array}{r} 3 \\ 5 \\ -3 \\ -2 \\ -7 \end{array}\right]= 
        c_1\left[ \begin{array}{r} -3 \\ 3 \\ 2 \\ -3 \\ -2 \end{array}\right] 
        +c_2\left[ \begin{array}{r} 3 \\ 3 \\ 2 \\ -1 \\ 3  \end{array}\right]
        +c_3\left[ \begin{array}{r} 2 \\ 0 \\ -2 \\ 2 \\ 2 \end{array}\right] 
        +c_4\left[ \begin{array}{r} -3 \\ -1 \\ 2 \\ -1 \\ 3 \end{array}\right] 
        +c_5\left[ \begin{array}{r} -2 \\ 0 \\ -3 \\ 3 \\ -2 \end{array}\right] 
        \end{equation}
        $$

        To find the coordinates we need to solve the linear system $A[X]_{\beta} = X$, where $A$ is the matrix with the basis vectors as its columns, and $[X]_{\beta}$ is the vector of unknowns.  
        """
    )
    return


@app.cell
def _(lag, np):
    _A = np.array([[-3, 3, 2, -3, -2], [3, 3, 0, -1, 0], [2, 2, -2, 2, -3], [-3, -1, 2, -1, 3], [-2, 3, 2, 3, -2]])
    X = np.array([[3], [5], [-3], [-2], [-7]])
    X_beta = lag.SolveSystem(_A, X)
    print(X_beta)
    return X, X_beta


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since calculating coordinates always involves solving a linear system, it can be helpful at times to remember that whenever we solve a linear system $AX=B$, we are really finding the coordinates of $B$ with respect to the columns of $A$. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercises

        **Exercise 1:** Find a basis for the set of solutions to the system $PX=0$ where $P$ is defined as follows.

        $$
        \begin{equation}
        P = \left[ \begin{array}{rrrr} 1 & 0 & 3 & -2 & 4 \\ -1 & 1 & 6 & -2 & 1 \\ -2 & 1 & 3 & 0 & -3 \end{array}\right] 
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


@app.cell
def _():
    ## Code solution here.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** Give an example of a set of three vectors that does **not** form a basis for $\mathbb{R}^3$.  Provide a calculation that shows why the example is not a basis.
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


@app.cell
def _():
    ## Code solution here.
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


@app.cell
def _():
    ## Code solution here
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


@app.cell
def _():
    ## Code solution here
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


@app.cell
def _():
    ## Code solution here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 8:** Let $U$ be the subspace of $\mathbb{R}^5$ which contains vectors with their first and second entries equal and their third entry equal to zero. What the vectors in the subspace $U$ look like? Use this information to find a basis for $U$ and determine the dimension of $U$.
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


@app.cell
def _():
    ## Code solution here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 10:** Can a set of four vectors in $\mathbb{R}^3$ be a basis for $\mathbb{R}^3$? Explain and verify your answer through a computation.
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
