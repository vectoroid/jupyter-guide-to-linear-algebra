import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Solving Systems using Elimination

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In this section we discuss the code needed to solve the linear system $AX=B$ using elimination.  We will restrict our objective to the case where $A$ is a square $n\times n$ matrix, and the system has exactly one solution. The more general case requires more knowledge of the underlying theory and will be addressed in a later chapter.

        When writing code to perform a complex problem, it is often a good idea to first break up the task, and write code to carry out smaller pieces.  Once we have code to reliably perform the small tasks, we can assemble the pieces to solve the larger problem.  In our case we will break down the solution method into two parts.

        1. Carry out elimination on the associated augmented matrix.
        2. Perform back substitution on the triangular system produced by elimination.

        It is also beneficial to consider how we might write the code now so that we can reuse it for other tasks later. 
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Back substitution function

        We will start with the back substitution step, since that is the easier part.  If the elimination step is successful, we will have an upper triangular system $UX=B$ that has the following form.

        $$
        \begin{equation}
        \left[ \begin{array}{rrrr} * & * & * & * \\ 0 & * & * & * \\ 0 & 0 & * & * \\ 0 & 0 & 0 & * \end{array}\right]
        \left[ \begin{array}{r}  x_1 \\  x_2  \\ x_3 \\ x_4  \end{array}\right]=
        \left[ \begin{array}{r}  * \\  *  \\ * \\ *  \end{array}\right]
        \end{equation}
        $$

        We will put the code in a function so that that it is easy to reuse later.  For this function, let's suppose that we are given the upper triangular matrix $U$ and the known vector $B$ and we want to find the vector $X$ so that $UX=B$.  Note we could make other assumptions, such as the matrix $U$ having diagonal entries equal to 1.  The fewer such assumptions we make, the more useful the code will be later.
        """
    )
    return


@app.cell
def _():
    import numpy as np

    def BackSubstitution(U, B):
        m = U.shape[0]
        _X = np.zeros((m, 1))
        for i in range(m - 1, -1, -1):
            _X[i] = _B[i]
            for j in range(i + 1, m):
                _X[i] -= U[i][j] * _X[j]
            if U[i][i] != 0:
                _X[i] /= U[i][i]
            else:
                print('Zero entry found in U pivot position', i, '.')
        return _X
    return BackSubstitution, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Before moving on, let's test this function.  We can build a matrix with the proper triangular form, *choose a solution*, and then construct a system $UX=B$ so that we know the solution.
        """
    )
    return


@app.cell
def _(BackSubstitution, np):
    U = np.array([[3, 0, 1], [0, 1, -1], [0, 0, -3]])
    _X_true = np.array([[3], [4], [3]])
    _B = U @ _X_true
    _X = BackSubstitution(U, _B)
    print(_X)
    return (U,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Before discussing the elimination step, we should make a note that this $\texttt{BackSubstitution}$ function *will fail* to produce meaningful results if any of the diagonal entries of $U$ are zero.  We should keep this in mind when using this code in the future.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Row reduction function

        Elimination is the larger and more complex part of the solution method.  It is also a common task that will arise in future sections, so we will want some code that we can reuse at a later point.  We want a function that will carry out all the steps of elimination, and just return the end result.  It is not necessary to see all the individual row operations that took place as part of the process.  Ideally we would like the function to carry out the elimination on arrays of any size or shape, and also be able to _make the decision_ to perform row swaps when necessary.

        Let's clarify the goal. The function should accept an arbitrary array and return an array that has the following properties.

        1. The first nonzero entry in each row is a 1.  These entries are the pivots.
        2. Each pivot is located to the right of the pivots in all rows above it.
        3. The entries below each pivot are 0.
        4. Rows that are all zeros are located below rows that contain nonzero entries. 

        Such a matrix is said to be in a **row echelon form**.  Here are three examples of matrices in the form that we seek.

           
        $$
        \begin{equation}
        \left[ \begin{array}{cccc} 1 & * & * & * \\ 0 & 1 & * & * \\ 0 & 0 & 1 & * \end{array}\right]
        \end{equation}
        $$

        $$
        \begin{equation}
        \left[ \begin{array}{ccc} 1 & * & *  \\ 0 & 0 & 1  \\ 0 & 0 & 0 \end{array}\right]
        \end{equation}
        $$


        $$
        \begin{equation}
        \left[ \begin{array}{cccccc} 1 & * & * & * & * & * \\ 0 & 0 & 1 & * & * & * \\ 0 & 0 & 0 & 1 & * & * \end{array}\right]
        \end{equation}
        $$

        It is important to notice that each row can contain at most one pivot and each column can contain at most one pivot.

        Before presenting the code to find the row echelon form of a matrix, we first make a clarification about the matrices we work with when we are solving a system of the form $AX=B$.  We recall from the [Gaussian Elimination](Gaussian_Elimination.ipynb) section that the same row operations needed to bring $A$ to row echelon form must also be applied to $B$.  In practice we can join $A$ and $B$ together to form what we call an **augmented matrix**. We then carry out the row operations on this single matrix.  Here is an example of the augmented matrix, which we will write as $[A|B]$, associated with the system $AX=B$.

        $$
        \begin{equation}
        AX = B \hspace{1cm} \left[ \begin{array}{rrr} 3 & -1 \\ 5 & 2 \end{array}\right]
        \left[ \begin{array}{r} x_1 \\ x_2 \end{array}\right]=
        \left[ \begin{array}{r} 0\\ 7  \end{array}\right] \hspace{1cm} \to \hspace{1cm}
        [A|B] = \left[ \begin{array}{rr|r} 3 & -1 & 0 \\ 5 & 2 & 7 \end{array}\right]
        \end{equation}
        $$


        In our current objective, $A$ is $n\times n$, which means that the augmented matrix we need to process will be $n\times(n+1)$.  We will also assume at this point that the system $AX=B$ has a unique solution.  If this is true the augmented matrix will have a pivot in each of the first $n$ columns, with the pivot positions lying along the diagonal line of entries starting at the top left entry.  If the row echelon form of the augmented matrix has a zero in any of these positions, our solution process breaks down as shown in the [Gaussian Elimination](Gaussian_Elimination.ipynb) examples.

        """
    )
    return


@app.cell
def _(np):
    import laguide as lag

    def RowReduction(A):
        m = _A.shape[0]
        n = _A.shape[1]
        _B = np.copy(_A).astype('float64')
        for k in range(m):
            pivot = _B[k][k]
            pivot_row = k
            while pivot == 0 and pivot_row < m - 1:
                pivot_row += 1
                pivot = _B[pivot_row][k]
            if pivot_row != k:
                _B = lag.RowSwap(_B, k, pivot_row)
            if pivot != 0:
                _B = lag.RowScale(_B, k, 1.0 / _B[k][k])
                for i in range(k + 1, m):
                    _B = lag.RowAdd(_B, k, i, -_B[i][k])
            else:
                print('Pivot could not be found in column', k, '.')
        return _B
    return RowReduction, lag


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that in this routine we make use of the row operations written earlier.  Since those functions are not written in *this notebook*, we need to import them from the $\texttt{laguide}$ module.

        Let's test the routine on a random array.  Run the code on several random matrices of different sizes and shapes.  Does it always work?  Do you notice any unusual results?  Does it depend on the size or shape?  Does it depend on the range of numbers used?
        """
    )
    return


@app.cell
def _():
    ## Try out RowReduction here.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        If you run this test enough times, you are likely to come across an example where the results look a little different.  Here is one such case.
        """
    )
    return


@app.cell
def _(RowReduction, np):
    NumericalReductionExample=np.array([[7,-6,6,-8],[-3,-5,-7,2],[1,-4,-7,-6],[-1,0,-2,-8]])
    reduction_result = RowReduction(NumericalReductionExample)
    print(reduction_result)
    return NumericalReductionExample, reduction_result


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        There are two things that we observe in this example.  First and most obvious is that the entries are all displayed in scientific notation.  The more disturbing observation is that the result is not exactly what we wanted.  The elimination process is supposed to produce zeros for all the entries below the main diagonal, but in this case there is one entry that is not $\texttt{0.000}$.  Instead it is an extremely small number, close to $10^{-17}$.  

        At this point we might question the code and start looking for errors, but the problem here does not lie with the code.  The issue here is something more fundamental, and involves the way numbers are represented in the calculation and precision limitations of the computer.  For example, the number $1/3$, which has a decimal representation that consists of an infinite string of $3$s, cannot be represented *exactly* using a finite decimal representation.  This means that the arrays representing the system may be slightly incorrect even before any calculations are performed.  As the computer carries out the operations of arithmetic, the results must be rounded off to numbers that can be represented exactly.  This inherent limitation is known as **roundoff error** and it is the reason we do not get exactly zero for all of the entries below the diagonal.  

        In this example, we can see that the entries that should be zero are indeed very close to zero.  If we want to display the results in a format that is more readable, we can use the NumPy function $\texttt{round}$ to round all of the entries to a specified number of digits.
        """
    )
    return


@app.cell
def _(np, reduction_result):
    rounded_result = np.round(reduction_result,6)
    print(rounded_result)
    return (rounded_result,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note, we should only use the rounded results for the purpose of displaying results.  If we were to carry on and solve a system, or perform some other calculation, we should use the original results rather than the rounded version.

        Roundoff error can present a significant challenge if we work with large arrays, and the errors are allowed to accumulate and compound.  There are strategies that can be employed to mitigate this error, and ensure that usable results can be obtained.  Any software that must provide reliable results must account for roundoff error.  We will carry on with our simple version of elimination with the awareness that the results we get are not always *exactly correct*.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Solve system function

        Now we can combine the $\texttt{RowReduction}$ and the $\texttt{BackSubstitution}$ functions together to carry out the solution algorithm for the system $AX=B$.  Let us assume that the user of the function will supply $A$ and $B$, and the function will return the solution $X$.  Here are the steps that need to be completed.

        1. Build the associated augmented matrix.
        2. Apply $\texttt{RowReduction}$.
        3. Split the matrix.
        4. Apply $\texttt{BackSubstitution}$ and return the result.

        Note that there are other ways we could build our function.  We could require the user to supply the augmented matrix for example, but then that means the user (which is likely us!) has to do step 1 every time they use this function.  It is better to let the function handle that step.
        """
    )
    return


@app.cell
def _(BackSubstitution, RowReduction, np):
    def SolveSystem(A, B):
        if _A.shape[0] != _A.shape[1]:
            print('SolveSystem accepts only square arrays.')
            return
        n = _A.shape[0]
        A_augmented = np.hstack((_A, _B))
        R = RowReduction(A_augmented)
        B_reduced = R[:, n:n + 1]
        A_reduced = R[:, 0:n]
        _X = BackSubstitution(A_reduced, B_reduced)
        return _X
    return (SolveSystem,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Let's test the routine by building a matrix, choosing a solution, and constructing a system $AX=B$ so that we know the solution.
        """
    )
    return


@app.cell
def _(SolveSystem, np):
    _A = np.array([[1, 2, 3], [0, 1, -2], [3, 3, -2]])
    _X_true = np.array([[1], [1], [1]])
    _B = _A @ _X_true
    print(_B, '\n')
    _X = SolveSystem(_A, _B)
    print(_X)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Next, we modify a couple of lines to produce a completely random system with random solution.  We will use $\texttt{SolveSystem}$ to find the solution and then compute the difference between the result and the actual known solution.
        """
    )
    return


@app.cell
def _(SolveSystem, np):
    _A = np.random.randint(-8, 8, size=(4, 4))
    _X_true = np.random.randint(-8, 8, size=(4, 1))
    _B = _A @ _X_true
    _X = SolveSystem(_A, _B)
    print(_X_true - _X)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercises

        **Exercise 1:** Try out the $\texttt{RowReduction}$ function on two different arrays that require the use of $\texttt{RowSwap}$.  Is it possible to test random arrays that require the use of $\texttt{RowSwap}$?
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
        **Exercise 2:** Use $\texttt{np.random.rand(n,n)}$ to generate a coefficient matrix with entries that are random floats.  Create a  linear system with a known solution using this matrix, and test $\texttt{SolveSystem}$.
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
        **Exercise 3:** Experiment to see what might go wrong using $\texttt{RowReduction}$ on a matrix that is *not* $n\times (n+1)$. 
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
        A matrix is said to be in **reduced row echelon form** if it satisfies the following properties.

           1. The first nonzero entry in each row is a 1.  These entries are the pivots.
           2. Each pivot is located to the right of the pivots in all rows above it.
           3. The entries below **and above** each pivot are 0.
           4. Rows that are all zeros are located below other rows. 

        Here is an example of a matrix in reduced row echelon form.


        $$
        \begin{equation}
        \left[ \begin{array}{cccc} 1 & 0 & 0 & * \\ 0 & 1 & 0 & * \\ 0 & 0 & 1 & * \end{array}\right]
        \end{equation}
        $$


        Note that for the system represented by the augmented matrix in the first example, the solution is given by the entries in the final column.  There is no need for back substitution if the augmented matrix is in reduced row echelon form.
           

        **Exercise 4:** Modify $\texttt{RowReduction}$ to compute the reduced row echelon form of a matrix.  Name the new function $\texttt{RREF}$.  
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
        **Exercise 5:** Test your $\texttt{RREF}$ on random $3\times 4$ matrices, then on random $n\times (n+1)$ matrices. 
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
        **Exercise 6:** Construct a $3 \times 3$ system with a known solution and compare the solutions produced using $\texttt{SolveSystem}$ with those produced using $\texttt{RREF}$.
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
