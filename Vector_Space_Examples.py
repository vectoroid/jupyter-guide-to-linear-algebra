import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Vector Space Examples

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        With some of the basic concepts of vector spaces in place, we now return to the abstraction of understanding the vector space as a collection of objects and algebraic operations that satisfy certain properties. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 1: Polynomials

        The collection of all polynomials with degree three or less, together with typical algebraic operations, constitutes a vector space.  This vector space is commonly written with the symbol $\mathbb{P}_3$.  If we take two elements from $\mathbb{P}_3$, $p = 2x^3 - x^2 + 6x -8$ and $q = x^3 - 3x^2 -4x -3$ for example, the linear combination $p+2q = 4x^3 - 7x^2 -2x -14$ is well-defined, and is another element in $\mathbb{P}_3$.  Indeed any linear combination of polynomials in $\mathbb{P}_3$ will be some other polynomial in $\mathbb{P}_3$ by the usual rules of algebra.

        $$
        \begin{equation}
        s(a_3x^3 + a_2x^2 + a_1 + a_0) + t(b_3x^3 + b_2x^2 + b_1x + b_0) = (sa_3+tb_3)x^3 + (sa_2+tb_2)x^2 + (sa_1+tb_1)x
        + (sa_0 + ta_0)
        \end{equation}
        $$

        The required properties of the algebraic operations are all satisfied, though we will not verify them here.

        Specifying a polynomial in $\mathbb{P}_3$ requires four coefficients, so it is somewhat natural to understand that $\mathbb{P}_3$ has dimension 4.  One such basis for $\mathbb{P}_3$ is the set $\{x^3, x^2, x, 1\}$.  The coordinates for any polynomial, with respect to this basis, will just be the coefficients on the $x^3$, $x^2$, $x$ terms, and constant term.  If we use this standard basis, we can proceed to discuss the ideas of spans and linear independence.

        The following collection of polynomials (vectors) in $\mathbb{P}_3$ are linearly dependent.

        $$
        \begin{eqnarray*}
        p_1 & = & 3x^3 + 2x^2 + x  - 1 \\
        p_2 & = & x^2 - 5x + 4 \\
        p_3 & = & 6x^3 + 3x^2 + 7x + -6 \\
        p_4 & = & -2x^3 + 2x^2 + 8x
        \end{eqnarray*}
        $$

        In order to show that the set is linearly dependent, we must find a nontrivial solution to the equation
        $c_1p_1 + c_2p_2 + c_3p_3 + c_4p_4  = 0$.  The polynomial on the left can only be zero if the coefficients on all the terms are equal to zero.  Carrying out the algebra and gives one equation for each of the four terms.

        $$
        \begin{eqnarray*}
        3c_1 \quad\quad + 6c_3 -2c_4 & = & 0 \\
        2c_1 +c_2 + 3c_3 + 2c_4 & = & 0 \\
        c_1 -5c_2 + 7c_3 + 8c_4 & = & 0 \\
        -c_1 +4c_2 - 6c_3 \quad\quad & = & 0 \\
        \end{eqnarray*}
        $$

        Examining the RREF of the coefficient matrix for this system, we see that there is not a pivot in each of the four columns, which implies that the system has a nontrivial solution and that that the set of polynomials is not linearly independent.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import laguide as lag

    A = np.array([[3,0,6,-2],[2,1,3,2],[1,-5,7,8],[-1,4,-6,0]])
    print(lag.FullRowReduction(A))
    return A, lag, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        There is nothing exceptional about polynomials that make up $\mathbb{P}_3$.  In general, the collection of polynomials of degree $n$ or less, known as $\mathbb{P}_n$, is also a vector space for any choice of $n$. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2: Matrices

        In [Matrix Algebra](Matrix_Algebra.ipynb), we pointed out that the matrices with only one column are commonly referred to as vectors, and that the algebra of matrices does not need to be defined in any special way to work on those with a single column.  The algebraic machinery required to compute linear combinations of matrices in general is no different than that required to compute linear combinations of those with a single column.  Perhaps it is then natural to think that matrices with multiple columns are as much a vector as those with a single column.  In one sense the single column matrix, which we have been calling a vector, is just a special type of matrix.  Based on the formal definition of a vector space however, the collection of all matrices of a given shape constitute a vector space, since the operations of matrix addition and scalar multiplication satisfy all the algebraic requirements.  In this sense *the matrices are vectors* since they are objects that make up a vector space.  No doubt this langauge is confusing.   


        As an concrete example, we consider the collection of all $2\times 3$ matrices, to which we will assign the symbol $\mathbb{M}_{2\times 3}$.  If we select two aribtrary matrices from this collection, and form an arbitrary linear combination, we will get another $2\times 3$ matrix. 


        $$
        \begin{equation}
        c_1A + c_2B = 
        c_1\left[ \begin{array}{ccc} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23}  \end{array}\right] +
        c_2\left[ \begin{array}{ccc} b_{11} & b_{12} & b_{13} \\ b_{21} & b_{22} & b_{23}  \end{array}\right] =
        \left[ \begin{array}{ccc} c_1a_{11}+c_2b_{11} & c_1a_{12}+c_2b_{12} & c_1a_{13}+c_2b_{13} \\ 
        c_1a_{21}+c_2b_{21} & c_1a_{22}+c_2b_{22} & c_1a_{23}+c_2b_{23}  \end{array}\right]
        \end{equation}
        $$

        Again these alegbraic operations satisfy all the requirements necessary for a valid vector space and again we will omit the details of verification.

        We can write the following collection of matrices to act as a standard basis, which we will label $\alpha$.

        $$
        \begin{equation}
        E_1 = \left[ \begin{array}{ccc} 1 & 0 & 0 \\ 0 & 0 & 0  \end{array}\right] \hspace{1cm}
        E_3 = \left[ \begin{array}{ccc} 0 & 1 & 0 \\ 0 & 0 & 0  \end{array}\right] \hspace{1cm}
        E_5 = \left[ \begin{array}{ccc} 0 & 0 & 1 \\ 0 & 0 & 0  \end{array}\right] \hspace{1cm}
        \end{equation}
        $$

        $$
        \begin{equation}
        E_2 = \left[ \begin{array}{ccc} 0 & 0 & 0 \\ 1 & 0 & 0  \end{array}\right] \hspace{1cm}
        E_4 = \left[ \begin{array}{ccc} 0 & 0 & 0 \\ 0 & 1 & 0  \end{array}\right] \hspace{1cm}
        E_6 = \left[ \begin{array}{ccc} 0 & 0 & 0 \\ 0 & 0 & 1  \end{array}\right] \hspace{1cm}
        \end{equation}
        $$

        Consider now the distinction between a matrix $B$, and the *coordinate vector* of $B$ with respect to the basis $\alpha$

        $$
        \begin{equation}
        B = \left[ \begin{array}{ccc} 2 & 0 & 5 \\ 1 & -1 & 2  \end{array}\right] \hspace{1cm} 
        \left [B\right]_{\alpha} = \left[ \begin{array}{r} 2\\1\\0\\-1\\5\\2  \end{array}\right] \hspace{1cm}
        \end{equation}
        $$

        The coordinate vector $\left[B\right]_{\alpha}$ is a vector in $\mathbb{R}^6$ that gives us a description of how to assemble the matrix $B$ as a linear combination of the elements in the basis $\alpha$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 3:  Functions

        For a third example, we look at the collection of all continuous functions on the interval $[0,1]$, which is given the symbol $C\left[0,1\right]$.  If $f$ and $g$ are two such functions, and $c$ a scalar, we can define $cf$ and $f+g$ using the algebra of functions.  

        $$
        \begin{eqnarray*}
        (cf)(x) & = & c(f(x)) \\
        (f + g)(x) & = & f(x) + g(x)
        \end{eqnarray*}
        $$

        These algebraic combinations of functions satisfy all the requirements to make $C\left[0,1\right]$ a vector space.  This space is quite different from the other examples in that it does not have a finite dimension.  Any basis for $C\left[0,1\right]$ must contain an infinite number of functions.  For this reason, we cannot easily do calculations such as those in the previous example to determine if a set of functions are linearly independent.  We will however make use of this example in a later application. 

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercises
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


@app.cell
def _():
    ## Code solution here
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


@app.cell
def _():
    ## Code solution here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 3:** Demonstrate that a set of four polynomials in $\mathbb{P}_4$ cannot span $\mathbb{P}_4$ through a computation.
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


@app.cell
def _():
    ## Code solution here
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


@app.cell
def _():
    ## Code solution here
    return


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
