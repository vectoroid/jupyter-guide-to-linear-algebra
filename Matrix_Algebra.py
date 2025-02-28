import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Matrix Algebra
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this section we look at matrix algebra and some of its common properties.  We will also see how operations involving matrices are connected to linear systems of equations.

        A **matrix** a is two-dimensional array of numbers.  When we do computations with matrices using NumPy, we will be using arrays just as we did before.  Let's write down some of examples of matrices and give them names.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rr} 1 & 3 \\ 2 & 1 \end{array}\right] \hspace{1cm} 
        B = \left[ \begin{array}{rrr} 3 & 0 & 4 \\ -1 & -2 & 1 \end{array}\right] \hspace{1cm}
        C = \left[ \begin{array}{rr} -2 & 1 \\ 4 & 1 \end{array}\right] \hspace{1cm}
        D = \left[ \begin{array}{r} 2 \\ 6 \end{array}\right]
        \end{equation}
        $$

        When discussing matrices it is common to talk about their dimensions, or shape, by specifying the number of rows and columns.  The number of rows is usually listed first.  For our examples, $A$ and $C$ are $2\times 2$ matrices, $B$ is a $2 \times 3$ matrix, and $D$ is a $2 \times 1 $ matrix.  Matrices that have only 1 column, such as $D$, are commonly referred to as **vectors**.  We will adhere to this convention as well, but do be aware that when we make statements about matrices, we are also making statements about vectors even if we don't explicitly mention them.  We will also adopt the common convention of using uppercase letters to name matrices.

        It is also necessary to talk about the individual entries of matrices.  The common notation for this is a lowercase letter with subscripts to denote the position of the entry in the matrix.  So $b_{12}$ refers to the 0 in the first row and second column of the matrix $B$.  If we are talking about generic positions, we might use variables in the subscripts, such as $a_{ij}$.

        Let's create these matrices as NumPy arrays before further discussion.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    A = np.array([[1, 3],[2,1]])
    B = np.array([[3, 0, 4],[-1, -2, 1]])
    C = np.array([[-2, 1],[4, 1]])
    D = np.array([[2],[6]])
    print(D)
    return A, B, C, D, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        It will be useful for us to access the dimensions of our arrays.  When the array is created, this information gets stored as part of the array object and can be accessed with a method called $\texttt{shape}$.  If $\texttt{B}$ is an array, the object $\texttt{B.shape}$ is itself an array that has two entries.  The first (*with index 0!*) is the number of rows, and the second (*with index 1!*) is the number of columns.
        """
    )
    return


@app.cell
def _(B):
    print("Array B has",B.shape[0],"rows.")
    print("Array B has",B.shape[1],"columns.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Operations on matrices

        There are three algebraic operations for matrices that we will need to perform.  For our definitions let us suppose that $A$ and $C$ are $m \times n$ matrices, $B$ is an $n \times k$ matrix, and $c$ is a number.  When discussing algebra involving matrices and numbers, the numbers are usually referred to as **scalars**. 

        1. A matrix of any shape can be multiplied by a scalar.  The result is that all entries are multiplied by that scalar.  Using the subscript notation, we would write

        $$
        (cA)_{ij} = ca_{ij}
        $$

        2. Two matrices that *have the same shape* can be added.  The result is that all corresponding entries are added.

        $$
        (A+C)_{ij} = a_{ij} + c_{ij}
        $$

        3. If the number of columns of matrix $A$ is equal to the number of rows of matrix $B$, the matrices can be multiplied in the order $A$, $B$.  The result will be a new matrix $AB$, that has the same number of rows as $A$ and the same number of columns as $B$.  The entries $(AB)_{ij}$ will be the following combination of the entries of row $i$ of $A$ and column $j$ of $B$.

        $$
        (AB)_{ij} = \sum_{k=1}^n a_{ik}b_{kj}
        $$

        The last operation, known as **matrix multiplication**, is the most complex and least intuitive of the three.  No doubt this last formula is a bit intimidating the first time we read it.  Let's give some examples to clarify.

        1.  The multiplication of a number and a matrix:

        $$
        \begin{equation}
        3A = 3\left[ \begin{array}{rr} 1 & 3 \\ 2 & 1 \end{array}\right] 
        = \left[ \begin{array}{rr} 3 & 9 \\ 6 & 3 \end{array}\right]
        \end{equation}
        $$

        2. The sum of two matrices of the same shape:

        $$
        \begin{equation}
        A + C = \left[ \begin{array}{rr} 1 & 3 \\ 2 & 1 \end{array}\right] + 
        \left[ \begin{array}{rr} -2 & 1 \\ 4 & 1 \end{array}\right] 
        = \left[ \begin{array}{rr} -1 & 4 \\ 6 & 2 \end{array}\right]
        \end{equation}
        $$

        3.  The multiplication of two matrices:

        $$
        \begin{equation}
        AB = \left[ \begin{array}{rr} 1 & 3 \\ 2 & 1 \end{array}\right]
        \left[ \begin{array}{rrr} 3 & 0 & 4 \\ -1 & -2 & 1 \end{array}\right]
         = \left[ \begin{array}{rrr} 0 & -6 & 7  \\  5 & -2 & 9  \end{array}\right]
         \end{equation}
        $$
         
        To clarify what happens in the  matrix multiplication, lets calculate two of the entries in detail.

        $$
        \begin{eqnarray*}
        (AB)_{12} & = & 1\times 0 + 3 \times (-2) = -6 \\
        (AB)_{23} & = & 2 \times 4 + 1 \times 1 = 9
        \end{eqnarray*}
        $$

        These matrix operations are all built into NumPy, but we have to use the symbol $\texttt{@}$ instead of $\texttt{*}$ for matrix multiplication.
        """
    )
    return


@app.cell
def _(A, B, C):
    print(3*A,'\n')
    print(A+C,'\n')
    print(A@B)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Properties of matrix operations

        It is useful to observe that some common algebraic properties hold true for matrix multiplication.  Let $A$, $B$, and $C$ be matrices, and $k$ be a scalar.  The associative and distributive properties stated here hold for matrix multiplication.

        $$
        \begin{equation}
        k(A+B) = kA + kB
        \end{equation}
        $$

        $$
        \begin{equation}
        C(A+B) = CA + CB
        \end{equation}
        $$

        $$
        \begin{equation}
        A(BC) = (AB)C
        \end{equation}
        $$

        These statements only make sense of course if the matrices have dimensions that allow for the operations.

        It is also worth noting that the commutative property does not generally hold for matrix multiplication.  Suppose for example that $A$ and $B$ are both $3\times 3$ matrices.  It is **not true in general** that $AB = BA$.  One example with random matrices is enough to prove this point.
        """
    )
    return


@app.cell
def _(np):
    A_1 = np.random.randint(-5, 5, size=(3, 3))
    B_1 = np.random.randint(-5, 5, size=(3, 3))
    print(A_1 @ B_1)
    print('\n')
    print(B_1 @ A_1)
    return A_1, B_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Matrix transposes

        Another common idea that we will find useful is that of the matrix transpose.  The **transpose** of a matrix $A$ is another matrix, $A^T$, defined so that its columns are the rows of $A$.  To build $A^T$, we simple swap the row index with the column index for every entry, $a^T_{ij} = a_{ji}$.  Two examples should be enough to clarify this definition.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrr} 5 & 4 & 0 \\ 1 & 8 & 3 \\ 6 & 7 & 2\end{array}\right] \hspace{1cm}
        A^T = \left[ \begin{array}{rrr} 5 & 1 & 6 \\ 4 & 8 & 7 \\ 0 & 3 & 2\end{array}\right] \hspace{1cm}
        \end{equation}
        $$

        $$
        \begin{equation}
        B = \left[ \begin{array}{rrr} 1 & 2 & 7 & 0 \\ 3 & 1 & 5 & 2 \\ 4 & 9 & 8 & 6\end{array}\right] \hspace{1cm}
        B^T = \left[ \begin{array}{rrr} 1 & 3 & 4 \\ 2 & 1 & 9 \\ 7 & 5 & 8 \\ 0 & 2 & 6\end{array}\right] \hspace{1cm}
        \end{equation}
        $$


        NumPy array objects have a method named $\texttt{transpose}$ for this purpose.
        """
    )
    return


@app.cell
def _(np):
    A_2 = np.array([[5, 4, 0], [1, 8, 3], [6, 7, 2]])
    A_T = A_2.transpose()
    print(A_2)
    print('\n')
    print(A_T)
    return A_2, A_T


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        When a matrix $A$ is equal to its own transpose, it has the property of being symmetric across its main diagonal. For this reason a matrix $A$ is said to be **symmetric** if $A = A^T$. Equivalently, we can say that $A$ is symmetric if $a_{ij} = a_{ji}$ for every entry $a_{ij}$ in the matrix.  The matrix $P$ below is one such example.

        $$
        \begin{equation}
        P = \left[ \begin{array}{rrr} 1 & 0 & 6 \\ 0 & 3 & 5 \\ 6 & 5 & -2\end{array}\right] \hspace{1cm}
        \end{equation}
        $$

        Similarly, we say that a matrix $A$ is **skew-symmetric** if $A^T = -A$ (equivalently $a_{ij} = -a_{ji}$ for every entry $a_{ij}$ in $A$). The matrix $Q$ below is a skew-symmetric matrix.

        $$
        \begin{equation}
        Q = \left[ \begin{array}{rrr} 0 & 1 & -4 \\ -1 & 0 & 5 \\ 4 & -5 & 0\end{array}\right] \hspace{1cm}
        \end{equation}
        $$

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Application to linear systems

        A very important example of matrix multiplication is that where a known matrix multiplies an *unknown* vector to produce a known vector.  If we let $A$ be the known matrix, $B$ be the known vector, and $X$ be the unknown vector, we can write the matrix equation $AX=B$ to describe this scenario.  Let's write a specific example.

        $$
        \begin{equation}
        A= \left[ \begin{array}{rrr} 1 & 3 & -2 \\ 5 & 2 & 0 \\ 4 & 2 & -1 \\ 2 & 2 & 0 \end{array}\right] \hspace{1cm}
        X= \left[ \begin{array}{r} x_1 \\ x_2 \\ x_3 \end{array}\right] \hspace{1cm}
        B= \left[ \begin{array}{r} 0 \\ 10 \\ 7 \\ 4  \end{array}\right] \hspace{1cm}
        \end{equation}
        $$

        $$
        \begin{equation}
        AX = \left[ \begin{array}{rrr} 1 & 3 & -2 \\ 5 & 2 & 0 \\ 4 & 2 & -1 \\ 2 & 2 & 0 \end{array}\right]
        \left[ \begin{array}{r} x_1 \\ x_2 \\ x_3 \end{array}\right]=
        \left[ \begin{array}{r} 0\\ 10 \\ 7 \\ 4  \end{array}\right]= B
        \end{equation}
        $$

        If we apply the definition of matrix multiplication we see that this single matrix equation $AX=B$ in fact represents a system of linear equations.

        $$
        \begin{eqnarray*}
        x_1 + 3x_2 - 2x_3 & = & 0\\
        5x_1 + 2x_2 \quad\quad & = & 10 \\
        4x_1 + 2x_2 - x_3 & = & 7 \\
        2x_1 + 2x_2 \quad\quad & = & 4
        \end{eqnarray*}
        $$

        In this context, the matrix $A$ is known as the **coefficient matrix**.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Identity matrices

        An **identity matrix** is a square matrix that behaves similar to the number 1 with respect to ordinary multiplication.  Identity matrices, labeled with $I$, are made up of ones along the main diagonal, and zeros everywhere else.  Below is the $4 \times 4$ version of $I$.

        $$
        \begin{equation}
        I = \left[ \begin{array}{ccc} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0\\ 0 & 0 & 0 & 1 \end{array}\right]
        \end{equation}
        $$

        If $A$ is any other $4 \times 4$ matrix, multiplication with $I$ will produce $A$.  Furthermore it doesn't matter in this case which order the multiplication is carried out.

        $$
        \begin{equation}
        AI = IA = A
        \end{equation}
        $$

        The NumPy function $\texttt{eye}$ generates an identity matrix of the specified size.  Note we only need to provide $\texttt{eye}$ with one parameter since the identity matrix must be square.  We show here the product of $I$ with a random $5\times 5$ matrix.  
        """
    )
    return


@app.cell
def _(np):
    I5 = np.eye(5)
    print(I5)
    return (I5,)


@app.cell
def _(I5, np):
    R = np.random.randint(-10,10,size=(5,5))
    print(R)
    print('\n')
    print(R@I5)
    print('\n')
    print(I5@R)
    return (R,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notes:

        1. Actual calculation with identity matrices is rather uncommon, but the idea is useful for symbolic calculations and progressing further with the theory.
        2. If we are discussing a non-square matrix, then we must take care to use the correct size identity matrix depending on the order of multiplication.  For example, if $C$ is a $2\times 3$ matrix, $I_2$ is the $2\times 2$ identity, and $I_3$ is the $3\times 3 $ identity, we would have the following result.

        $$
        \begin{equation}
        I_2 C = CI_3 = C
        \end{equation}
        $$
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Matrix-vector multiplication

        One special case of matrix multiplication deserves close attention, the case where one of the matrices is a vector.  This case is so important that it is commonly discussed separately and given a special name, **matrix-vector multiplication**.  Let's suppose that $P$ is our matrix and that its shape is $n \times m$, and $Y$ is our vector which is $m \times 1$.  The product $PY$ then is a $n \times 1$ vector.  It is the relationship between this new vector and the columns of the matrix $P$ that makes this situation important.

        Let's have a look with a specific example.

        $$
        \begin{equation}
        P = \left[ \begin{array}{rrr} 1 & 3 & -2 \\ 5 & 2 & 0 \\ 4 & 2 & -1 \\ 2 & 2 & 0 \end{array}\right]\hspace{1cm}
        Y = \left[ \begin{array}{r} 2 \\ -3 \\ 4 \end{array}\right]
        \end{equation}
        $$

        In this case of matrix-vector multiplication, we can package the calculation a bit differently to better understand what is happening.

        $$
        \begin{equation}
        PY = \left[ \begin{array}{rrr} 1 & 3 & -2 \\ 5 & 2 & 0 \\ 4 & 2 & -1 \\ 2 & 2 & 0 \end{array}\right]
        \left[ \begin{array}{r} 2 \\ -3 \\ 4 \end{array}\right]=
        2\left[ \begin{array}{r} 1 \\ 5 \\ 4 \\ 2 \end{array}\right] -
        3\left[ \begin{array}{r} 3 \\ 2 \\ 2 \\ 2 \end{array}\right] +
        4\left[ \begin{array}{r} -2 \\ 0 \\ -1 \\ 0 \end{array}\right] =
        \left[ \begin{array}{r} -15\\ 4 \\ -2 \\ -2  \end{array}\right]
        \end{equation}
        $$

        This is the same operation that we were doing before, but now we see that this product is a result of adding the columns of $P$ after first multiplying each by the corresponding entry in $Y$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Matrix multiplication by columns

        We can extend the calculation of matrix-vector multiplication to better understand exactly what is produced by matrix-matrix multiplication.  Suppose for example that $Y$ from the earlier calculation was actually the third column of a $3\times 4$ matrix $C$.

        $$
        \begin{equation}
        C = \left[ \begin{array}{rrrr} * & * & 2 & * \\ * & * & -3 & * \\ * & * & 4 & *\end{array}\right]
        \end{equation}
        $$

        The third column of the product $PC$ will be exactly $PY$!  The other columns of $PC$ will be the products of $P$ with the corresponding columns of $C$.

        $$
        \begin{equation}
        PC = \left[ \begin{array}{rrrr} * & * & -15 & * \\ * & * & 4 & * \\* & * & -2 & *  \\ * & * & -2 & *\end{array}\right]
        \end{equation}
        $$

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        This discussion offers a great opportunity to learn how to perform operations on portions of NumPy arrays using a feature called **slicing**.  Let's build the matrices $P$ and $C$, and then define $X$ as a **subarray** of $C$.  To create a subarray of $C$, we use the syntax $\texttt{C[a:b,c:d]}$.  This will create an array object that has shape $(b-a)\times(d-c)$ and contains the entries of rows $a$ to $b-1$ and columns $c$ to $d-1$ of $C$.    

        Specifically, we want $X$ to include all rows of $C$, but only the third column (which has Python index 2!).  
        """
    )
    return


@app.cell
def _(np):
    P = np.array([[1, 3, -2], [5, 2, 0], [4, 2, -1], [2, 2, 0]])
    C_1 = np.array([[0, 6, 2, 6], [-1, 1, -3, 4], [0, 2, 4, 8]])
    X = C_1[0:3, 2:3]
    print(P, '\n')
    print(C_1, '\n')
    print(X, '\n')
    print(P @ X)
    return C_1, P, X


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Notes on array slicing:

        - Another way to select all rows (or all columns) from an array is to use : without any numbers.  In our example above, we could have used the following line of code to produce the same result.  Try it out by editing the cell above!
        ```
        X = C[:,2:3]
        ```
        - If we only want to select a single row or column, it is tempting to try a line of code like the following.
        ```
        X = C[:,2]
        ```
        this is indeed valid code, but the array $X$ is not exactly what we expect.  Instead we get an array with the correct entries, but not the correct shape.  It is possible to make this work, but let's avoid this complication.

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercises

        **Exercise 1:** Calculate $-2E$, $G+F$, $4F-G$, $HG$, and $GE$ using the following matrix definitions.  Do the exercise on paper first, then check by doing the calculation with NumPy arrays.

        $$
        \begin{equation}
        E = \left[ \begin{array}{r} 5 \\ -2 \end{array}\right] \hspace{1cm} 
        F = \left[ \begin{array}{rr} 1 & 6 \\ 2 & 0 \\ -1 & -1 \end{array}\right] \hspace{1cm}
        G = \left[ \begin{array}{rr} 2 & 0\\ -1 & 3 \\ -1 & 6 \end{array}\right] \hspace{1cm}
        H = \left[ \begin{array}{rrr} 3 & 0 & 1 \\ 1 & -2 & 2 \\ 3 & 4 & -1\end{array}\right]
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
        **Exercise 2:** Find the values of $x$ and $y$ so that this equation holds.

        $$
        \begin{equation}
        \left[ \begin{array}{rr} 1 & 3 \\ -4 & 2 \end{array}\right]
        \left[ \begin{array}{rr} 3 & x \\ 2 & y \end{array}\right]=
        \left[ \begin{array}{rr} 9 & 10 \\ -8 & 16 \end{array}\right] 
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
        **Exercise 3:** Define NumPy arrays for the matrices $H$ and $G$ given below.  

        $$
        \begin{equation}
        H = \left[ \begin{array}{rrr} 3 & 3 & -1  \\ -3 & 0 & 8 \\  1 & 6 & 5 \end{array}\right]\hspace{2cm}
        G = \left[ \begin{array}{rrrr} 1 & 5 & 2 & -3 \\ 7 & -2 & -3 & 0 \\ 2 & 2 & 4 & 6\end{array}\right]
        \end{equation}
        $$

        $(a)$ Multiply the second and third column of $H$ with the first and second row of $G$.  Use slicing to make subarrays.  Does the result have any relationship to the full product $HG$?

        $(b)$ Multiply the first and second row of $H$ with the second and third column of $G$.  Does this result have any relationship to the full product $HG$?

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
        **Exercise 4:** Generate a $4\times 4$ matrix $B$ with random integer entries.  Compute matrices $P = \frac12(B+B^T)$ and $Q = \frac12(B-B^T)$.  Rerun your code several times to get different matrices.  What do you notice about $P$ and $Q$?  Explain why it must always be true.       
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
        **Exercise 5:** Just as the elements of a matrix can be either integers or real numbers, we can also have a matrix with elements that are themselves matrices. A **block matrix** is any matrix that we have interpreted as being partitioned into these **submatrices**. Evaluate the product $HG$ treating the $H_i$'s and $G_j$'s as the elements of their respective matrices. If we have two matrices that can be multiplied together normally, does any partition allow us to multiply using the submatrices as the elements?

        $$
        \begin{equation}
        H = \left[ \begin{array}{cc|cc} 1 & 3 & 2 & 0  \\ -1 & 0 & 3 & 3 \\ \hline 2 & 2 & -2 & 1 \\ 0 & 1 & 1 & 4 \end{array}\right] = \left[ \begin{array}{} H_1 & H_2 \\ H_3 & H_4\end{array} \right] \hspace{2cm} 
        G = \left[ \begin{array}{cc|c} 3 & 0 & 5 \\ 1 & 1 & -3 \\ \hline 2 & 0 & 1 \\ 0 & 2 & 1\end{array}\right] = \left[ \begin{array}{} G_1 & G_2 \\ G_3 & G_4\end{array} \right]
        \end{equation}
        $$
        """
    )
    return


@app.cell
def _():
    ## Code solution here.
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
