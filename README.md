# Multiple Shooting Solver Using Python and CUDA

###### tags: `nonlinear optimal control problem`, `boundary value problem`, `multiple shooting method`, `Python`, `CUDA`

> A software to solve nonlinear optimal control problems implemented using Python and CUDA. 

> The paper is under publication. :smile: 

> The instruction of the solver is presented below. :arrow_down: 

## :memo: Set up

The solver needs to run on machine equipped with targeted Nvidia **GPU** and **CUDA** installed. 

The installation of CUDA can be found at https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html.

The solver can be ran on **Python3** with the following necessary packages.

- Python (>=3.5.0)
- Numpy (>=1.11.0)
- Sympy (>=1.1)
- Matplotlib (>=3.00)
- Numba (>=0.48)
- Libgcc
- Cudatoolkit

Set up Python with CUDA with installing needed dependencies by
```bash
sudo apt-get install build-essential freeglut3-dev libxmu-dev libxi-dev git cmake libqt4-dev libphonon-dev libxml2-dev libxslt1-dev libqtwebkit-dev libboost-all-dev python-setuptools libboost-python-dev libboost-thread-dev -y
```

## :incoming_envelope:  Development Guide

The solver can be used with the command in terminal as
```bash
./run.sh ocp_example
```
The input `ocp_example` is a plain text file with the optimal control problem (OCP) to be solved defined with necessary fields for the solver. 
The necessary fields for the OCP are 

- [x] **StateVariables**: symbols for the state variables separated by comma and embraced by square brackets. e.g. 
```typescript==*
StateVariables = [x1, x2, x3, x4, x5, x6];
```
- [x] **ControlVariables**: symbols for the control variables separated by comma and embraced by square brackets. e.g. 
```typescript==*
ControlVariables = [u1, u2, u3, u4];
```
- [ ] **ParameterVariables**: symbols for the parameter variables separated by comma and embraced by square brackets. e.g. 
```typescript==*
ParameterVariables = [w];
```
- [ ] **Constants**: symbols for the constants and the corresponding value with the equal sign separated by comma and embraced by square brackets. e.g. 
```typescript==*
Constants = [m = 10.0, d = 5.0, l = 5.0, I = 12.0, pi = 3.14159265358979, rho = 1.0e3];
```
- [ ] **Variables**: symbols for the variables defined in the problem separated by comma and embraced by square brackets. The definition of each variable must be written one variable per line in the file. e.g. 
```typescript==*
Variables = [c5, s5];
c5 = cos(x5);
s5 = sin(x5);
```
- [x] **CostFunctional**: symbols for the cost functional of the problem. e.g. 
```typescript==*
CostFunctional = w*(rho + u1*u1 + u2*u2 + u3*u3 + u4*u4);
```
- [x] **DifferentialEquations**: symbols for the differential equations of the dynamic system of the problem separated by comma and embraced by square brackets. e.g. 
```typescript==*
DifferentialEquations  = [ w*x2,
        w*((u1+u3)*c5 - (u2+u4)*s5)/m,
        w*x4,
        w*((u1+u3)*s5 + (u2+u4)*c5)/m,
        w*x6,
        w*((u1+u3)*d - (u2+u4)*l)/I];
```
- [ ] **TerminalPenalty**: symbols for the terminal penalty term of the problem. e.g. 
```typescript==*
TerminalPenalty = w;
```
- [ ] **InitialConstraints**: symbols for the initial constraints of the state variables of the problem separated by comma and embraced by square brackets. e.g. 
```typescript==*
InitialConstraints = [x1, x2, x3, x4, x5, x6];
```
- [ ] **TerminalConstraints**: symbols for the terminal constraints of the state variables of the problem separated by comma and embraced by square brackets. e.g. 
```typescript==*
TerminalConstraints = [x1 - 4.0, x2, x3 - 4.0, x4, x5 - pi/4.0, x6];
```
- [ ] **InequalityConstraints**: symbols for the control variable inequality constraints separated by comma and embraced by square brackets. e.g. 
```typescript==*
InequalityConstraints = [u1 - 5, 
                        -5 - u1, 
                        u2 - 5, 
                        -5 - u2, 
                        u3 - 5, 
                        -5 - u3, 
                        u4 - 5, 
                        -5 - u4];
```
- [ ] **StateVariableInequalityConstraints**: symbols for the state variable inequality constraints separated by comma and embraced by square brackets. e.g. 
```typescript==*
StateVariableInequalityConstraints = [-(x2+0.25)];
```
- [ ] **EqualityConstraints**: symbols for the equality constraints separated by comma and embraced by square brackets. e.g. 
```typescript==*
EqualityConstraints = [x1 - x4];
```
- [ ] **InitialTime**: symbol or number for the initial time of the problem. Default value is 0.0. e.g. 
```typescript==*
InitialTime = 0.0;
```
- [ ] **FinalTime**: symbol or number for the final time of the problem. Default value is 1.0. e.g. 
```typescript==*
FinalTime   = 1.0;
```
- [ ] **Nodes**: number of nodes for the initial estimate of the problem. Default value is 101. e.g. 
```typescript==*
Nodes = 101;
```
- [ ] **MaximumNodes**: maximum number of nodes allowed during the solving process. Default value is 2000. e.g. 
```typescript==*
MaximumNodes = 2000;
```
- [ ] **Tolerance**: numerical tolerance for the solver. Default value is $1.0e-6$. e.g. 
```typescript==*
Tolerance = 1.0e-6;
```
- [ ] **MaximumMeshRefinements**: maximum number of mesh refinements allowed during the solving process. Default value is 20. e.g. 
```typescript==*
MaximumMeshRefinements = 20;
```
- [ ] **MaximumNewtonIterations**: maximum number of Newton iterations allowed during the solving process. Default value is 200. e.g. 
```typescript==*
MaximumNewtonIterations = 500;
```
- [ ] **StateEstimate**: symbols related to time `t` or numbers to define the initial estimate of the state variables. Default values are all ones. e.g. 
```typescript==*
StateEstimate = [t, t*(1-t)];
```
- [ ] **ControlEstimate**: symbols related to time `t` or numbers to define the initial estimate of the control variables. Default values are all ones. e.g. 
```typescript==*
ControlEstimate = [-1 + 2*t];
```
- [ ] **ParameterEstimate**: symbols or numbers to define the initial estimate of the parameter variables. Default values are all ones. e.g. 
```typescript==*
ParameterEstimate = [10.0];
```
- [ ] **InputFile**: string of the name of the input file to obtain initial estimates for the problem e.g. 
```typescript==*
InputFile = "ex21.data";
```
- [ ] **OutputFile**: string of the name of the output file to save the results for the problem e.g. 
```typescript==*
OutputFile = "ex22.data";
```

The fileds with the check sign are must-included fields in the file while other fileds are optional fields depending on the problem.
**All the necessary field lines must end up with a semicolon.**
The definition of the fields need not to be in one line as long as they are embraced by brackets.
**Some example OCP files can be found at the repo "ocp_test_problems".**

## About the solver

*The solver is developed by Dynamic Systems Modeling and Controls Laboratory at University of Washington.*

![](https://i.imgur.com/kQSpFjN.png)