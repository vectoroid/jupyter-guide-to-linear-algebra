import marimo

__generated_with = "0.10.14"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Applications of Vector Spaces
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        When an object is made up of multiple components it is often useful to represent the object as a vector, with one entry per component.  The examples discussed in this section involve molecules, which are made up of atoms, and text documents, which are made up of words.  In some cases equations involving the objects give rise to vector equations.  In other instances there are reasons to perform operations on the vectors using matrix algebra.  
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Chemical equations

        In balancing a chemical equation, we seek to determine a number of molecules of reactant that form a number of molecules of product, while maintaining an equal number of each type of atom on both sides of the equation.  Here is an example for the combustion of ethanol

        $$
        \begin{equation}
        C_2H_5OH + O_2 \to CO_2 + H_2O
        \end{equation}
        $$

        This equation is not correct since it does not contain the same number of carbon atoms ($C$), nor the same number of hydrogen atoms ($H$) on each side.  Let's rewrite the equation with unknown coefficients multiplying each molecule.

        $$
        \begin{equation}
        x_1C_2H_5OH + x_2O_2 \to x_3CO_2 + x_4H_2O
        \end{equation}
        $$

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The problem now is to find values for $x_1$, $x_2$, $x_3$, and $x_4$ such that both sides of the equation contain the same number of atoms each of carbon, oxygen, and hydrogen.  Furthermore, since the molecules are discrete units, the unknown coefficients must be *positive integers*.

        In this scenario it is useful to think of each of the molecules as a vector with three entries, one for each type of atom in the equation.  Let's say the first entry is the number of carbon atoms, the second the number of oxygen atoms, and the third the number of hydrogen atoms.  The following vector thus represents $C_2H_5OH$.

        $$
        \begin{equation}
        \left[\begin{array}{c} 2 \\ 6 \\ 1 \end{array}\right]
        \end{equation}
        $$

        The chemical equation is very naturally expressed as a vector equation.

        $$
        \begin{equation}
        x_1\left[\begin{array}{c} 2 \\ 6 \\ 1 \end{array}\right]
        + x_2\left[\begin{array}{c} 0 \\ 0 \\ 2 \end{array}\right]
        = x_3\left[\begin{array}{c} 1 \\ 0 \\ 2 \end{array}\right]
        + x_4\left[\begin{array}{c} 0 \\ 2 \\ 1 \end{array}\right]
        \end{equation}
        $$

        Rearranging so that all the terms with unknowns are on the left side, we see that this vector equation represents a homogeneous system.

        $$
        \begin{equation}
        \left[\begin{array}{rrr} 2 & 0 & -1 & 0 \\ 6 & 0 & 0 & -2 \\ 1 & 2 & -2 & -1 \end{array}\right]
        \left[\begin{array}{r} x_1 \\ x_2 \\ x_3 \\ x_4 \end{array}\right]
        = \left[\begin{array}{r} 0 \\ 0 \\ 0  \end{array}\right]
        \end{equation}
        $$


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Before trying to find a solution, let's apply some of the results of this chapter.  We know that homogeneous systems are always consistent since the trivial solution ($x_1=x_2=x_3=x_4=0$) is always a possible solution.  We can also look at the shape of the coefficient matrix ($4 \times 3$) to make a further conclusion.  Since there can be only one pivot per row, we know that there are at most 3 pivots.  This means that there cannot be a pivot in each of the 4 columns, which means that there is at least one free variable in the system and that the trivial solution is not the only solution.

        Let's now use the RREF to find solutions.  Remember that since this is a homogeneous system, the last column in the augmented matrix, which represents the righthand side, is all zeros.  We will omit this last column from the calculation since all entries will remain zero regardless of any row operations performed.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import laguide as lag

    A = np.array([[2, 0, -1, 0],[6, 0, 0, -2],[1, 2, -2, -1]])
    A_reduced = lag.FullRowReduction(A)
    print(A_reduced)
    return A, A_reduced, lag, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Since there is no pivot in the column corresponding to $x_4$, we treat $x_4$ as a free variable.  For this application, we are specifically looking for solutions that are integers, so we will take $x_4=3$.  This gives $x_3=2$, $x_2=3$, and $x_1=1$.  The correct equation for the chemical reaction is the following.

        $$
        \begin{equation}
        C_2H_5OH + 3O_2 \to 2CO_2 + 3H_2O
        \end{equation}
        $$

        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        #### Exercises

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


@app.cell
def _():
    ## Code solution here.
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


@app.cell
def _():
    ## Code solution here.
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


@app.cell
def _():
    ## Code solution here.
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Information Retrieval

        A well-known idea in information retrieval is that of the **vector space model**, which represents documents in a database as vectors in $\mathbb{R}^n$.  Algorithms which aim to search the database for documents that are most relevant to keyword searches can then make use of the vector representation.  We provide some simple examples of how such algorithms might work.

        Let's begin by the modeling a document with a vector.  All we really need is a list of $n$ words that will be searchable in our database.  This list could be a list of each distinct word in the whole collection of documents, but there are many common words in English that we might disqualify as keywords since they appear abundantly in all documents (the, as, in, it, etc.).  Each document in the database can now be represented by a vector in $\mathbb{R}^n$, with the $n$th entry set to 1 if the corresponding keyword appears in the document, and 0 otherwise.

        For example, suppose our database is simply a list of webpages with content related to information retrieval.  Our set of keywords might be the following.

        **\{algorithm, engine, information, google, computations, matrix, optimization, retrieval, search, theory \}**  

        Each webpage in the database is a vector in $\mathbb{R}^{10}$, with entries corresponding to these words.  It is most convenient to write these vectors as $1\times 10$ row vectors rather than $10\times 1$ column vectors.  For example, *Search Engine Algorithms* would be represented as $[1, 1, 0, 0, 0, 0, 0, 0, 1, 0]$.  Our database then is represented by a $n\times 10$ matrix that has a row for each title.  Suppose for the sake of our example that we have the following 6 titles.

        - Search Engine Algorithms
        - How Google Search Optimization Works
        - Information Retrieval Theory
        - Matrix Models of Information Retrieval
        - Theory Behind Search Engines
        - Computations in Information Retrieval

        We will build a vector for each title, and then assemble them into a matrix $D$.
        """
    )
    return


@app.cell
def _(np):
    T1 = np.array([[1, 1, 0, 0, 0, 0, 0, 0, 1, 0]])
    T2 = np.array([[0, 0, 0, 1, 0, 0, 1, 0, 1, 0]])
    T3 = np.array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 1]])
    T4 = np.array([[0, 0, 1, 0, 0, 1, 0, 1, 0, 0]])
    T5 = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 1, 1]])
    T6 = np.array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 0]])

    D = np.vstack((T1,T2,T3,T4,T5,T6))
    print(D)
    return D, T1, T2, T3, T4, T5, T6


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        Next, let's consider how we might perform a search of the keywords.  Suppose we want to search for entries matching the words "information", "retrieval", and "theory".  We can build a $10\times 1$ query vector $X$ that has entries of 1 corresponding to these keywords.  In this case, the query vector is $[0, 0, 1, 0, 0, 0, 0, 1, 0, 1]^T$.  The matrix-vector product $DX$ will now contain entries that represent the number of words from the search that match each title in the database.  
        """
    )
    return


@app.cell
def _(D, np):
    X = np.array([[0, 0, 1, 0, 0, 0, 0, 1, 0, 1]])
    results = D@X.transpose()
    print(results)
    return X, results


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        The third entry of $DX$ is the largest, which means that the third webpage in the database best matches the list of keywords in the search.  The last three titles might also be reported as partial search matches.  Note that the database search is nothing more than a single matrix multiplication, followed by a search of the vector $DX$ to find the largest entries.

        There are other possible ways that the documents in the database might be represented.   One approach might be to let the $n$th entry in each document vector equal the frequency of the corresponding word in that document.  For example, if the 7th keyword appears 53 times in a document, the vector representing that document would have 53 as the 7th entry.  The effect would be that a document that contains many instances of a matching keyword would be ranked higher in the search that a document with fewer instances of the word.   

        Another idea is to set the $n$th entry equal to the relative frequency of the associated keyword in that document.  The **relative frequency** is the frequency divided by the total number of all keywords in the document.  Shown below is an example what a keyword count might look like if there were only 10 searchable words.

        |Keyword|Document 1|
        |----|------|
        |algorithm|19|
        |engine|23|
        |information|0|
        |google|2|
        |compuations|0|
        |matrix|11|
        |optimization|0| 
        |retrieval|10| 
        |search|31|
        |theory|4|

        There are a total of 100 keyword matches, so each entry would be divided by 100 and the row vector corresponding to document 1 would be $[0.19, 0.23, 0, 0.02, 0, 0.11, 0, 0.1, 0.31, 0.04]$  In practice the number of searchable words and the number of words in each document might be tens of thousands each.  The query vector is again a column vector with entries of 1 corresponding to the words in the search, and all other entries set to zero.  The search is executed by a single matrix multiplication, followed by a search of the resulting vector for the largest entry.  The effect of this representation would be similar to that using frequency, but now a document with 50 matches in 30000 words would be rated higher than a document with 50 matches in 100000 words.  


        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### References

        - Berry, Michael W. and Murray Browne. *Understanding Serach Engines: Mathematical Modeling and Text Retrieval*. 2nd ed., SIAM, 2005 

        - Lay, David, et al. *Linear Algebra and its Applications*. 5th ed., Pearson., 2016

        - Leon, Steven J. *Linear Algebra with Applications*. 9th ed., Pearson., 2015

        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
