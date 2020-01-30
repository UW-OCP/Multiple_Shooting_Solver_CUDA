#!/bin/bash
printf "Solving %s\n" $1
python OCP2ABVP.py $1 > bvp_problem.py
python multiple_shooting_solver.py
