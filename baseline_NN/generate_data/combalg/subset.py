'''
subset.py
=========
Enumeration and random selection of arbitrary and k-element restricted
subsets.

Members
-------
'''
import random as rng



def powerset(elements):
    '''
    A generator that produces all subsets of the input 'set' of elements.  The input doesn't need
    to be a set, any iterable will do (I think).  The point is that the elements need not be unique,
    but will be treated as unique with respect to the subset generation.
    For example:
    powerset(['a','a']) will give the sequence ['a','a'], ['a'], ['a'], []

    :param elements: input set
    :type elements: list
    :return: one of the subsets of the input set
    :rtype: list
    '''
    if len(elements) == 0:
        yield []
    else:
        for item in powerset(elements[1:]):
            yield [elements[0]]+item
            yield item



def all_k_element(elements, k):
    '''
    A generator that produces all k-element subsets of the input elements in lexicographic order.

    :param elements: the input set
    :type elements: list
    :param k: size of the output set
    :type k: int
    :return: one of the k-element subsets of the input set
    :rtype: list
    '''
    n = len(elements)
    a = list(range(k))
    h = 0
    done = False
    first = True
    while not done:
        if not first:
            h = 0 if a[k-h-1] < n-h-1 else h+1
            a[k-h-1] += 1
            for j in range(k-h,k):
                a[j] = a[j-1] + 1
        done = (k == 0) or (k == n) or (a[0] == n - k)
        first = False
        yield list(map(lambda x: elements[x], a))


def random(elements):
    '''
    Returns a random subset of a set of elements.  Equivalent to randomly selecting an element of
    powerset(elements).  For each element of the universal set, select it to be in the subset with p=0.5

    :param elements: the input set
    :type elements: list
    :return: a random subset of the input set
    :rtype: list
    '''

    return filter(lambda x: rng.random() <= 0.5, elements)


def random_k_element(elements, k):
    '''
    Returns a random k-element subset of a set of elements.  This is a clever algorithm, explained in [NW1978]_
    but not actually implemented.  Essentially starting with the first element of the universal set, include
    it in the subset with p=k/n.  If the first was selected, select the next with p=(k-1)/(n-1), if not
    p = k/(n-1), etc.

    :param elements: the input set
    :type elements: list
    :param k: the size of the output set
    :type k: int
    :return: a random k-element subset of the input set
    :rtype: list
    '''
    a = []
    c1 = k
    c2 = len(elements)
    i = 0
    while c1 > 0:
        if rng.random() <= float(c1)/c2:
            a.append(elements[i])
            c1 -= 1
        c2 -= 1
        i += 1