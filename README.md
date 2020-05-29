# Characterising quantum correlations of fixed dimension (SDP relaxations)

[![Build Status](https://travis-ci.com/carlosparaciari/non_local_games.svg?token=qysu8rvspZL66s8hKeeJ&branch=master)](https://travis-ci.com/carlosparaciari/non_local_games)

Numerical implementation of the SDP relaxations introduced in the pre-print Hyejung H. Jee, et al, Characterising quantum correlations of fixed dimension (2020), available at this [link](https://arxiv.org/abs/2005.08883).

The code showcasts the first levels of the SDP hierarchy. In particular, we give an explicit program for

- CHSH game, levels (1,1), (2,1), (2,2), and (3,1)
- A random game with 2 answers and 3 questions for each player, where the sdp(2,1)+NPA1 gives a better value than NPA1
- I3322, level (3,3) with only PPT conditions.

The code is written in Python (using cvxpy), and the solver we used is Mosek.
