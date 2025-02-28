import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## LU Factorization

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We saw in the last section that given two matrices, $A$ and $B$, of compatible shapes, we are able to define the product matrix $C=AB$ in a useful way.  In this section we discuss the factorization of a matrix.  One might naturally ask if it is possible to start with matrix $C$ and determine the two matrix factors $A$ and $B$.  As it turns out, a useful course of action is to look for matrix factors that have a particular structure.

        One such factorization, that is closely related to the elimination process, is known as the LU Factorization.  Given a matrix $A$, we will look for matrices $L$ and $U$ such that 

        - $LU = A$
        - $L$ is a lower triangular matrix with main diagonal entries equal to 1.
        - $U$ is an upper triangular matrix.

        Here is a visualization of what we are seeking.


        $$
        \begin{equation}
        A = \left[ \begin{array}{cccc} * & * & * & * \\ * & * & * & * \\ * & * & * & * \\ * & * & * & *  \end{array}\right]\hspace{1cm}
        L = \left[ \begin{array}{cccc} 1 & 0 & 0 & 0 \\ * & 1 & 0 & 0 \\ * & * & 1 & 0 \\ * & * & * & 1 \end{array}\right]\hspace{1cm}
        U = \left[ \begin{array}{cccc} * & * & * & * \\ 0 & * & * & * \\ 0 & 0 & * & * \\ 0 & 0 & 0 & *  \end{array}\right]\hspace{1cm}
        \end{equation}
        $$

        Before we tackle the problem of calculating $L$ and $U$ from a known matrix $A$, let's see why such a factorization is useful.  Suppose that we have found $L$ and $U$ so that $A=LU$ and we wish to solve the system $AX=B$.  Another way to write the problem is $LUX=B$.  We can then define another unknown $Y$ by saying that $UX=Y$, and exchange the a single system $AX=B$ for following two systems.

        $$
        \begin{eqnarray*}
        UX & = & Y\\
        LY & = & B 
        \end{eqnarray*}
        $$

        While it is true that we have in fact doubled the number of equations, the two systems that we have are triangular and can be solved easily with back (or forward) substitution.  The first example shows the details for specific system.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 1:  Soving a system using LU factorization

        We want to solve the system of equations.

        $$
        \left[ \begin{array}{ccc} 3 & -1 & -2 \\ 6 & -1 & 0  \\ -3 & 5 & 20  \end{array}\right]X = 
        \left[ \begin{array}{c} -4 \\ -8 \\ 6  \end{array}\right]\hspace{1cm}
        $$

        where $X$ is an unknown $3\times 1$ vector.  Suppose we also have computed $L$ and $U$.

        $$
        L = \left[ \begin{array}{ccc} 1 & 0 & 0 \\ 2 & 1 & 0  \\ -1 & 4 & 1  \end{array}\right] \hspace{2cm} 
        U = \left[ \begin{array}{ccc} 3 & -1 & -2 \\ 0 & 1 & 4  \\ 0 & 0 & 2  \end{array}\right] 
        $$

        """
    )
    return


@app.cell
def _():
    import numpy as np
    import laguide as lag
    import scipy.linalg as sla

    ## Use Python to check for yourself that LU = A.
    return lag, np, sla


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's write out the systems $UX=Y$ and $LY = B$.  For the sake of clarity, we leave the matrix notation aside for a moment and use the variables $x_1$, $x_2$, and $x_3$ for the entries of $X$ and the variables $y_1$, $y_2$, and $y_3$ for the entries of $Y$.


        $$
        \begin{eqnarray*}
        x_1 \hspace{2.1cm}& = & y_1\\
        2x_1 + x_2 \hspace{1.1cm}& = & y_2\\
        -x_1 + 4x_2 +x_3 & = & y_3 \\
        \\
        3y_1 - y_2 - 2y_3 & = & -4\\
        y_2 + 4y_3 & = & -8\\
        2y_3 & = & 6 
        \end{eqnarray*}
        $$

        Now the solution is a matter of substitution.  The last equation tells us $y_3$.  From there we work backwards to find $y_2$ and $y_1$.  Then we go the first three equations to determine the $x$ values in a similar way, starting this time with the very first equation and working our way down.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Elementary matrices

        In order to understand how we can construct the LU factorization through elimination, it helpful to see that the steps of elimination can be carried out by multiplication with special matrices called **elementary matrices**.  Elementary matrices are the result of applying either a $\texttt{RowScale}$ or $\texttt{RowAdd}$ operation to the identity matrix of compatible shape.  (*Remember that rearranging the rows is only necessary if a 0 arises in a pivot position.  We will address row swaps shortly.*) 

        For an example, let's apply one of these operations to a $4\times 4$ identity matrix.
        """
    )
    return


@app.cell
def _(lag, np):
    I = np.eye(4)
    E = lag.RowAdd(I,1,2,-3)
    print(I,'\n')
    print(E)
    return E, I


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The $E$ that we get is the result of adding -3 times the first row of $I$ to the third row of $I$.  The interesting property of the elementary matrix $E$ is that if we multiply another matrix $A$ by $E$, the result will be a the matrix we would get by applying the same row operation to $A$.
        """
    )
    return


@app.cell
def _(E, np):
    _A = np.array([[1, 2, 0, -1], [-1, 1, -1, 4], [2, 13, -4, 9], [-2, 5, -3, 13]])
    print(_A, '\n')
    print(E @ _A)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 2:  Finding an LU factorization

        We can now carry out the elimination by applying a sequence of elementary matrices $E_1$, $E_2$, $E_3$,...to $A$.  Let's see how it works for the matrix above.
        """
    )
    return


@app.cell
def _(lag, np):
    _A = np.array([[1, 2, 0, -1], [-1, 1, -1, 4], [2, 13, -4, 9], [-2, 5, -3, 13]])
    I_1 = np.eye(4)
    E1 = lag.RowAdd(I_1, 0, 1, 1)
    E2 = lag.RowAdd(I_1, 0, 2, -2)
    E3 = lag.RowAdd(I_1, 0, 3, 2)
    print(E3 @ E2 @ E1 @ _A, '\n')
    E4 = lag.RowAdd(I_1, 1, 2, -3)
    E5 = lag.RowAdd(I_1, 1, 3, -3)
    print(E5 @ E4 @ E3 @ E2 @ E1 @ _A)
    return E1, E2, E3, E4, E5, I_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        After using $\texttt{RowAdd}$ to create zeros in the appropriate spaces, we now have the $U$ factor.  Writing out the matrix multiplication in symbols it looks like this.

        $$
        \begin{equation}
        E_5E_4E_3E_2E_1A = U
        \end{equation}
        $$

        Note that the order of the multiplications cannot be changed.  $E_1$ should be the first to multiply $A$, then $E_2$, and so on.  Now let us manipulate the symbols a bit based on the properties of inverse matrices.

        $$
        \begin{eqnarray}
        A &=& (E_5E_4E_3E_2E_1)^{-1}U  \\
        A &=& E_1^{-1}E_2^{-1}E_3^{-1}E_4^{-1}E_5^{-1}U  
        \end{eqnarray}
        $$

        It must be that $L = E_1^{-1}E_2^{-1}E_3^{-1}E_4^{-1}E_5^{-1}$.  The fact that this product of inverse elementary matrices has the correct form to be $L$ is not at all clear.  Let's make the following two observations.

        - Each of the inverse elementary matrices has a simple lower triangular structure.  In fact, the matrix $E_3^{-1}$ is also an elementary matrix.  It is the elementary matrix that undoes the row operation represented by $E_3$!  Multiplication by $E_3$ adds 2 times the first row to the last row.  Multiplication by $E_3^{-1}$ adds -2 times the first row to the last row.
        """
    )
    return


@app.cell
def _(E3, sla):
    print(E3,'\n')
    print(sla.inv(E3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        - Multiplying two lower triangular matrices together produces a lower triangular matrix.  Look at any example and try to figure out why.
        """
    )
    return


@app.cell
def _(np):
    L1 = np.array([[1,0,0,0],[-1,1,0,0],[2,3,1,0],[-2,3,0,1]])
    L2 = np.array([[1,0,0,0],[2,1,0,0],[-5,4,1,0],[4,4,1,1]])
    print(L1,'\n')
    print(L2,'\n')
    print(L1@L2)
    return L1, L2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        These two facts together tell us that the matrix $E_1^{-1}E_2^{-1}E_3^{-1}E_4^{-1}E_5^{-1}$ has the correct structure to be the $L$ factor.  What is even more convenient is that when we multiply these inverse elementary matrices together, the nonzero  entries in the lower triangular portions do not change. 
        """
    )
    return


@app.cell
def _(E3, E4, E5, sla):
    print(sla.inv(E5),'\n')
    print(sla.inv(E4)@sla.inv(E5),'\n')
    print(sla.inv(E3)@sla.inv(E4)@sla.inv(E5))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The nonzero lower triangular entries in $E_3^{-1}E_4^{-1}E_5^{-1}$ are the same as the corresponding entries of $E_3^{-1}$, $E_4^{-1}$, and $E_5^{-1}$.  This means that the entries in $L$ are just the scale factors used in our application of $\texttt{RowAdd}$, multiplied by -1.  Now that we understand how these elementary matrices combine to produce $L$, we don't actually need to construct them.  We can just compute $L$ as we do the row operations by keeping track of the scale factors.  
        """
    )
    return


@app.cell
def _(np):
    L = np.array([[1,0,0,0],[-1,1,0,0],[2,3,1,-0],[-2,3,0,1]])
    U = np.array([[1,2,0,-1],[0,3,-1,3],[0,0,-1,2],[0,0,0,2]])
    print("L:",'\n',L,'\n',sep='')
    print("U:",'\n',U,'\n',sep='')
    print("LU:",'\n',L@U,sep='')
    return L, U


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Permutation matrices

        As we have seen in the previous section, it is sometimes necessary to rearrange the rows of a matrix when performing elimination.  This row operation can also be done by multiplying the matrix with an elementary matrix.  Let's build a matrix $P$ that performs an exchange of rows 2 and 3 in a $4\times 4$ matrix.  Again, we can do this by performing the same row operation on the identity matrix.  
        """
    )
    return


@app.cell
def _(lag, np):
    C = np.random.randint(-6, 6, size=(4, 4))
    I_2 = np.eye(4)
    P = lag.RowSwap(I_2, 1, 2)
    print('C:', '\n', C, '\n', sep='')
    print('P:', '\n', P, '\n', sep='')
    print('PC:', '\n', P @ C, sep='')
    return C, I_2, P


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        When the row operation is a row swap, it is common to refer to the corresponding elementary matrix as a **permutation matrix**, and use the letter $P$ to represent it.  We will follow this convention.  It is also useful to note that an elementary permutation matrix is its own inverse since the operation of swapping two rows can be reversed by performing the exact same operation.  We can check that $PP=I$, which means that $P=P^{-1}$.
        """
    )
    return


@app.cell
def _(P):
    print(P@P)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Note that if a permutation represents more than a single row exchange, then its inverse must represent those row exhanges applied in the reverse order.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Example 3:  Factoring with row swaps

        In order to understand how the row swaps are incorporated into the factorization, it is most helpful to see an example.  In this $4\times 4$ example, we will use our $\texttt{laguide}$ functions to carry out the elimination and build the corresponding elementary matrices along the way.  For the $\texttt{RowAdd}$ operations, we will label the elementary matrix with an $L$, and for the $\texttt{RowSwap}$ operations we will use the label $P$.
        """
    )
    return


@app.cell
def _(np):
    B = np.array([[1,2,-1,-1],[4,8,-4,2],[1,1,1,2],[3,3,4,4]])
    print(B)
    return (B,)


@app.cell
def _(B, I_2, lag):
    B_1 = lag.RowAdd(B, 0, 1, -4)
    L1_1 = lag.RowAdd(I_2, 0, 1, -4)
    B_1 = lag.RowAdd(B_1, 0, 2, -1)
    L2_1 = lag.RowAdd(I_2, 0, 2, -1)
    B_1 = lag.RowAdd(B_1, 0, 3, -3)
    L3 = lag.RowAdd(I_2, 0, 3, -3)
    print(B_1)
    return B_1, L1_1, L2_1, L3


@app.cell
def _(B_1, I_2, lag):
    B_2 = lag.RowSwap(B_1, 1, 2)
    P1 = lag.RowSwap(I_2, 1, 2)
    print(B_2)
    return B_2, P1


@app.cell
def _(B_2, I_2, lag):
    B_3 = lag.RowAdd(B_2, 1, 3, -3)
    L4 = lag.RowAdd(I_2, 1, 3, -3)
    print(B_3)
    return B_3, L4


@app.cell
def _(B_3, I_2, lag):
    B_4 = lag.RowSwap(B_3, 2, 3)
    P2 = lag.RowSwap(I_2, 2, 3)
    print(B_4)
    return B_4, P2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        In terms of matrix multiplication, we have carried out the matrix product $P_2L_4P_1L_3L_2L_1B = U$, as we can check.
        """
    )
    return


@app.cell
def _(L1_1, L2_1, L3, L4, P1, P2, np):
    B_5 = np.array([[1, 2, -1, -1], [4, 8, -4, 2], [1, 1, 1, 2], [3, 3, 4, 4]])
    U_1 = P2 @ L4 @ P1 @ L3 @ L2_1 @ L1_1 @ B_5
    print(U_1)
    return B_5, U_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        As we see with a calculation in the next cell, the inverse matrix $(P_2L_4P_1L_3L_2L_1)^{-1}$ does not have the correct lower triangular structure to be the $L$ factor.    In fact there are no matrices $L$ and $U$ with the correct triangular structure such that $B=LU$ for this particular matrix $B$.
        """
    )
    return


@app.cell
def _(L1_1, L2_1, L3, L4, P1, P2, sla):
    possible_L = sla.inv(P2 @ L4 @ P1 @ L3 @ L2_1 @ L1_1)
    print(possible_L)
    return (possible_L,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Although this matrix does not have the correct structure, we might notice that it is only a matter of rearranging the rows to produce a lower triangular matrix.  In fact, the row swaps that are needed here are *exactly the same* as those used in the elimination process.  
        """
    )
    return


@app.cell
def _(P1, P2, possible_L):
    L_1 = P2 @ P1 @ possible_L
    print(L_1)
    return (L_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Now let's put all of this together to make the factorization of $B$.  Elimination gives us that $B = (P_2L_4P_1L_3L_2L_1)^{-1}U$, but the matrix $(P_2L_4P_1L_3L_2L_1)^{-1}$ is not lower triangular.  We can produce a lower triangular factor by multiplying by the permutation matrices that produce the required row swaps.

        $$
        P_2P_1B = P_2P_1(P_2L_4P_1L_3L_2L_1)^{-1}U
        $$

        We will label $P_2P_1(P_2L_4P_1L_3L_2L_1)^{-1}$ as $L$, since it now has the correct structure.  The end result is that $B=PLU$ where $P=(P_2P_1)^{-1}$.  To compute the inverse of the permutation matrix, we can simply apply the row swaps in the reverse order, so that $P=P_1P_2$.
        """
    )
    return


@app.cell
def _(L_1, P1, P2, U_1):
    P_1 = P1 @ P2
    print('P\n', P_1, '\n', sep='')
    print('L\n', L_1, '\n', sep='')
    print('U\n', U_1, '\n', sep='')
    print('PLU\n', P_1 @ L_1 @ U_1, sep='')
    return (P_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The important point here is that if row swaps are used during elimination, a permutation matrix will be required in the factorization in order for $L$ to have the desired triangular structure.  Therefore in general we expect that $B=PLU$ where $P$ represents all the row swaps that occur during elimination.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Another way to arrive at this result is to realize that if we performed the required row swaps *before* starting the elimination process, they would not interfere with the structure of $L$.  Let's give it a try!
        """
    )
    return


@app.cell
def _(lag, np):
    B_6 = np.array([[1, 2, -1, -1], [4, 8, -4, 2], [1, 1, 1, 2], [3, 3, 4, 4]])
    B_6 = lag.RowSwap(B_6, 1, 2)
    B_6 = lag.RowSwap(B_6, 2, 3)
    print(B_6)
    return (B_6,)


@app.cell
def _(B_6, I_2, lag):
    B_7 = lag.RowAdd(B_6, 0, 1, -1)
    L1_2 = lag.RowAdd(I_2, 0, 1, -1)
    B_7 = lag.RowAdd(B_7, 0, 2, -3)
    L2_2 = lag.RowAdd(I_2, 0, 2, -3)
    B_7 = lag.RowAdd(B_7, 0, 3, -4)
    L3_1 = lag.RowAdd(I_2, 0, 3, -4)
    print(B_7)
    return B_7, L1_2, L2_2, L3_1


@app.cell
def _(B_7, I_2, lag):
    B_8 = lag.RowAdd(B_7, 1, 2, -3)
    L4_1 = lag.RowAdd(I_2, 1, 2, -3)
    print(B_8)
    return B_8, L4_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The process has given us $L_4L_3L_2L_1P_2P_1B=U$. Now $(L_4L_3L_2L_1)^{-1}$ has the correct structure, and is the same matrix $L$ that we produced in the previous calculation. 
        """
    )
    return


@app.cell
def _(L1_2, L2_2, L3_1, L4_1, sla):
    L_2 = sla.inv(L4_1 @ L3_1 @ L2_2 @ L1_2)
    print(L_2)
    return (L_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### PLU factorization with SciPy

        The SciPy $\texttt{linalg}$ function for finding $PLU$ factorizations is called $\texttt{lu}$.  This function accepts the array to be factored as an argument, and returns three arrays representing $P$, $L$, and $U$, in that order.  To store these arrays for later use, we need to provide three names that will be assigned to the output of $\texttt{lu}$.  
        """
    )
    return


@app.cell
def _(np, sla):
    B_9 = np.array([[1, 2, -1, -1], [4, 8, -4, 2], [1, 1, 1, 2], [3, 3, 4, 4]])
    P_2, L_3, U_2 = sla.lu(B_9)
    print('P\n', P_2, '\n', sep='')
    print('L\n', L_3, '\n', sep='')
    print('U\n', U_2, '\n', sep='')
    print('PLU\n', P_2 @ L_3 @ U_2, sep='')
    return B_9, L_3, P_2, U_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        We can see that the $\texttt{lu}$ function produced matrices $P$, $L$, and $U$ such that $B=PLU$, but the factors are different than those we found using our own row operations.  Both factorizations are correct as we can see.  It is important to realize that the factorization of $B$ into $PLU$ is not unique since there is a choice to be made in which rows get swapped.  Instead of only requiring that pivot elements are non-zero, the SciPy function chooses row swaps using a more advanced method in order to minimize potential roundoff error.

        SciPy can also be used to solve a system $AX=B$ by using the $PLU$ factorization of $A$ together with back and forward substitution.  To do this we use $\texttt{lu_factor}$ to factor and $\texttt{lu_solve}$ to carry out the substitutions. 
        """
    )
    return


@app.cell
def _(np, sla):
    _A = np.array([[1, 2, -1, -1], [4, 8, -4, 2], [1, 1, 1, 2], [3, 3, 4, 4]])
    X_true = np.array([[1], [0], [1], [0]])
    B_10 = _A @ X_true
    factorization = sla.lu_factor(_A)
    X_computed = sla.lu_solve(factorization, B_10)
    print('Computed solution X:\n', X_computed, '\n', sep='')
    return B_10, X_computed, X_true, factorization


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The results returned from $\texttt{lu_factor}$ are not the same as those returned from $\texttt{lu}$.  The underlying factorization is the same, but the information is compressed into a more efficient format.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Exercises

        **Exercise 1:** Solve $ AX = B $ using $ A = LU $ and the $L$, $U$, and $B$ given below.  Compute $LUX$ to verify your answer. 

        $$
        \begin{equation}
        A = \left[ \begin{array}{ccc} 5 & 2 & 1 \\ 5 & 3 & 0 \\ -5 & -2 & -4  \end{array}\right] \hspace{2cm} 
        B = \left[ \begin{array}{c} 4 \\ 7 \\ 8  \end{array}\right] \hspace{2cm} 
        L = \left[ \begin{array}{ccc} 1 & 0 & 0 \\ 1 & 1 & 0  \\ -1 & 0 & 1  \end{array}\right] \hspace{2cm} 
        U = \left[ \begin{array}{ccc} 5 & 2 & 1 \\ 0 & 1 & -1  \\ 0 & 0 & 3  \end{array}\right] 
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
        **Exercise 2:** Solve $ AX = B $ using $ A = LU $ and the $L$, $U$, and $B$ given below.  Compute $LUX$ to verify your answer. 

        $$
        \begin{equation}
        L = \left[ \begin{array}{ccc} 1 & 0 & 0 \\ -1 & 1 & 0 \\ 0 & -1 & 1  \end{array}\right] \hspace{2cm} 
        U = \left[ \begin{array}{ccc} 1 & -1 & 0 \\ 0 & 1 & -1  \\ 0 & 0 & 1  \end{array}\right] \hspace{2cm} 
        B = \left[ \begin{array}{c} 2 \\ -3 \\ 4  \end{array}\right] 
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
        **Exercise 3:** Write a function called $\texttt{ForwardSubstitution}$ that will solve a lower triangular system $LY=B$.  It will be helpful to go back and look at the code for $\texttt{BackSubstitution}$.
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
        **Exercise 4:** Let $A$, $B$, and $C$ be the following matrices:

          $$
        \begin{equation}
        A = \left[ \begin{array}{ccc} 1 & 2 & 4 \\ 2 & 1 & 3 \\ 1 & 0 & 2  \end{array}\right] \hspace{2cm} 
        B = \left[ \begin{array}{ccc} 1 & 2 & 4 \\ 2 & 1 & 3  \\ 2 & 2 & 6  \end{array}\right] \hspace{2cm} 
        C = \left[ \begin{array}{ccc} 1 & 2 & 4 \\ 0 & -1 & -3  \\ 2 & 2 & 6  \end{array}\right] 
        \end{equation}
        $$

        $(a)$ Find an elementary matrix $E$ such that $EA = B$.  Verify with a computation.

        $(b)$ Find an elementary matrix $F$ such that $ FB = C$.  Verify with a computation.
            
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
        **Exercise 5:** Consider the following $3\times 3$  matrix :

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrr}  2 & 1 & 1\\  6 & 4 & 5  \\ 4 & 1 & 3 \end{array} \right] 
        \end{equation}
        $$

        $(a)$ Find **elementary matrices** $E_1$, $E_2$, and $E_3$ such that $ E_3E_2E_1A = U $ where $U$ is an upper triangular matrix.

        $(b)$ Find $L$ using the inverses of $E_1$, $E_2$, $E_3$, and verify that $  A = LU $.
        """
    )
    return


@app.cell
def _():
    ## Code Solution here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 6:** Compute the $LDU$ factorization of the following matrix and verify that $A = LDU$.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrr}  1 & 1 & 1\\  3 & 5 & 6  \\ -2 & 2 & 7 \end{array} \right] 
        \end{equation}
        $$

        """
    )
    return


@app.cell
def _():
    ## Code Solution here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 7:** Find $P, L,$ and $U$ such that $PA = LU$.  Following the discussion in this section, $P$ should be a permutation matrix, $L$ should be a lower triangular matrix with ones long the main diagonal, and $U$ should be an upper triangular matrix.

        $$
        \begin{equation}
        A = \left[ \begin{array}{rrr}  1 & 3 & 2\\  -2 & -6 & 1  \\ 2 & 5 & 7 \end{array} \right] 
        \end{equation}
        $$
            
        """
    )
    return


@app.cell
def _():
    ## Code Solution here
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        **Exercise 8:** Use SciPy to compute the $PLU$ factorization of a $3\times 3$ matrix.  Replicate the results using the row operations functions in $\texttt{laguide}$.
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
