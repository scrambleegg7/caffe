# -*- coding: utf-8 -*-

from AlzSolverClass import AlzSolverClass


def proc1():
    
    alzSolverCls = AlzSolverClass(test=True)
    alzSolverCls.stepRunSolver()


def eval_acc():
    

    alzSolverCls = AlzSolverClass(test=True)
    alzSolverCls.eval_alzNet_acc()

    
def main():
    
    
    
    #proc1()
    eval_acc()
    
    
if __name__ == "__main__":
    main()