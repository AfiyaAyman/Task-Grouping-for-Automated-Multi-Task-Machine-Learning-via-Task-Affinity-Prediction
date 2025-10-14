'''
combalg.py
==========
Algorithms I didn't categorize.

.. todo::

    the local function 'num_partitions(n)' is not smart at all

.. todo::

    the local functions 'num_partitions(n)' and 'choose_dj(n, rho)' can
    be combined, as in [NW1978]_.  I have had better luck implementing
    algorithms from [NW1978] using the mathematical description and NOT
    using the fortran source.

Members
~~~~~~~
'''

import random as rng
import combalg.subset as subset

class composition(object):
    '''
    Compositions of integer N into K parts.
    '''

    @staticmethod
    def all(n, k):
        '''
        A generator that returns all of the compositions of n into k parts.
        :param n: integer to compose
        :type n: int
        :param k: number of parts to compose
        :type k: int
        :return: a list of k-elements which sum to n

        Compositions are an expression of n as a sum k parts, including zero
        terms, and order is important.  For example, the compositions of 2
        into 2 parts:

        >>> compositions(2,2) = [2,0],[1,1],[0,2].

        NOTE: There are C(n+k-1,n) partitions of n into k parts.  See:
        `Stars and Bars <https://en.wikipedia.org/wiki/Stars_and_bars_(combinatorics)>`_.
        '''
        t = n
        h = 0
        a = [0]*k
        a[0] = n
        yield list(a)
        while a[k-1] != n:
            if t != 1:
                h = 0
            t = a[h]
            a[h] = 0
            a[0] = t-1
            a[h+1] += 1
            h += 1
            yield list(a)


    @staticmethod
    def random(n,k):
        '''
        Returns a random composition of n into k parts.

        :param n: integer to compose
        :type n: int
        :param k: number of parts to compose
        :type k: int
        :return: a list of k-elements which sum to n

        Returns random element of :func:`compositions` selected
        `uniformly at random <https://en.wikipedia.org/wiki/Discrete_uniform_distribution>`_.
        '''
        a = subset.random_k_element(range(1, n + k), k - 1)
        r = [0]*k
        r[0] = a[0] - 1
        for j in range(1,k-1):
            r[j] = a[j] - a [j-1] - 1
        r[k-1] = n + k - 1 - a[k-2]
        return r


class permutation(object):
    '''
    Permutations of letters.
    '''
    @staticmethod
    def all(a):
        '''
        A generator that returns all permutations of the input letters.

        :param a: the input set
        :type a: list
        :return: all permutations of the input set, one at a time
        '''
        if len(a) <= 1:
            yield a
        else:
            for perm in permutation.all(a[1:]):
                for i in range(len(a)):
                    yield list(perm[:i]) + list(a[0:1]) + list(perm[i:])


    @staticmethod
    def random(elements):
        '''
        Returns a random permutation of the input list
        :param a: the input set
        :type a: list
        :return: a shuffled copy of the input set
        '''
        a = list(elements)
        n = len(a)
        for m in range(n-1):
            l = m + int(rng.random() * (n - m))
            tmp = a[l]
            a[l] = a[m]
            a[m] = tmp
        return a


class integer_partition(object):
    '''
    Partitions of integers
    '''

    count = {}

    @staticmethod
    def all(n):
        '''
        A generator that returns all integer partitions of n.

        :param n: integer to partition
        :type n: int
        :return: a list of integers that sum to n, one at a time.

        This implementation is due to [ZS1998]_.  Integer partitions are
        unique ways of writing a sum with each term > 0, and order does not
        matter.  The partitions of 5 are:

        >>> [[5], [4, 1], [3, 2], [3, 1, 1], [2, 2, 1], [2, 1, 1, 1], [1, 1, 1, 1, 1]]

        '''
        x = [1]*n
        x[0] = n
        m = 1
        h = 1
        yield x[:m]

        while x[0] != 1:
            if x[h-1] == 2:
                m += 1
                x[h-1] = 1
                h -= 1
            else:
                r = x[h-1] - 1
                t = m - h + 1
                x[h-1] = r
                while t >= r:
                    x[h] = r
                    t = t - r
                    h += 1
                if t == 0:
                    m = h
                else:
                    m = h + 1
                if t > 1:
                    x[h] = t
                    h += 1
            yield x[:m]


    @staticmethod
    def num_partitions(n):
        '''
        Number of partitions of integer n.

        :param n:  Integer n.
        :return: Number of partitions of n
        '''
        if n <= 1:
            return 1
        if n not in integer_partition.count:
            sum = 1
            for j in range(1,n):
                d = 1
                while n - j*d >= 0:
                    sum += d * integer_partition.num_partitions(n - j*d)
                    d += 1
            integer_partition.count[n] = sum
        sum = integer_partition.count[n]
        return sum/n


    @staticmethod
    def choose_dj(n, rho):
        denom = integer_partition.num_partitions(n)
        if n <= 1:
            return 1,1
        sum = 0
        for j in range(1,n):
            d = 1
            while n - j*d >= 0:
                sum += float(d * integer_partition.num_partitions(n - j*d))/n
                if float(sum)/denom >= rho:
                    return d,j
                d += 1
        return 1,n

    @staticmethod
    def random(n):
        '''
        Returns a random integer partition of n.

        :param n: integer to partition
        :type n: int
        :return: a list of integers that sum to n, one at a time.

        '''

        a = []
        m = n
        while m > 0:
            d,j = integer_partition.choose_dj(m, rng.random())
            a = a + [d]*j
            m -= j*d
        return sorted(a,reverse=True)



def binomial_coefficient(n,k):
    '''
    Binomial coefficient
    '''
    # take care of nonsense
    if k > n or k < 0:
        return 0
    nk = 1    # (n)_k : falling factorial
    kf = 1    # k!    : k-factorial
    for i in range(0,k):
        nk *= n-i
        kf *= k-i
    return nk//kf

g_bell = {}
def bell_number(n):
    '''
    Computes the number of ways to partition a set of labeled items.

    :param n: the number of items
    :return: the number of partitions of the set [n].

    See: `OEIS A000110 <https://oeis.org/A000110>`_.
    '''
    if n == 0:
        return 1
    if n not in g_bell:
        g_bell[n] = sum([binomial_coefficient(n-1,k) * bell_number(k) for k in range(n)])
    return g_bell[n]


class set_partition(object):
    '''
    Partitions of sets.
    '''

    @staticmethod
    def all(n):
        '''
        A generator that returns all set partitions of [n] = {0,1,2,...,n-1}.
        The result is returned as a 3-tuple (p,q,nc):
        p - a list where p[i] is the population of the i-th equivalence class
        q - a list where q[i] is the equivalence class of i
        nc - the number of equivalence classes in the partition

       .. todo::

        The following will map the list q into a list of sets that make up the
        partition.  I'd like to see a more efficient/pythonic implementation:

        .. code-block:: python

            def map_set_partition(a, cls):
                y = []
                for t in range(len(a)):
                    x = reduce(lambda x,y: x+[a[y]] if cls[y] == t else x, range(len(a)), [])
                    if x:
                        y.append(x)
                    return y
        '''
        p = [n] + [0]*(n-1)
        nc = 1
        q = [0]*n
        m = n
        yield p,q,nc

        while nc != n:
            m = n
            while True:
                l = q[m-1]
                if p[l] != 1:
                    break
                q[m-1] = 0
                m -= 1
            # the variable z, and the if block below seem to be needed when the number of equivalence classes decreases
            # and we need to set the populations of the 'previously used' classes back to zero.  It would be nice to find
            # a way around this, since the algorithm in [NW1978] is loop free.
            z = m - n
            if z < 0:
                for i in range(0,z,-1):
                    p[nc+i-1] = 0
            nc += z
            p[0] -= z
            if l == nc - 1:
                p[nc] = 0
                nc += 1
            q[m-1] = l+1
            p[l] -= 1
            p[l+1] += 1
            yield p,q,nc


    @staticmethod
    def random(a):
        '''
        Returns a random partition of the input set
        '''
        # probability that the set partition that contains n, contains n-k other members
        def prob(n,k):
            return float(binomial_coefficient(n-1,k-1))*bell_number(n-k)/bell_number(n)

        # labels a partition of [n] with the members of a (the input set)
        def label_partition(a, q):
            nc = 1 + max(q)
            eq_classes = []
            for t in range(nc):
                eq_classes.append([])
            for t in range(len(q)):
                eq_classes[q[t]].append(a[t])
            return eq_classes

        n = len(a)
        q = [0]*n
        m = n
        l = 0
        while m > 0:
            sum = 0
            rho = rng.random()
            k = 1
            while k <= m:
                sum += prob(m,k)
                if sum >= rho:
                    break
                k += 1
            for i in range(m-k,m):
                q[i] = l
            l += 1
            m -= k
        return label_partition(a, permutation.random(q))
