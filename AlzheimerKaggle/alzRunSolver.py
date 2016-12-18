# -*- coding: utf-8 -*-

from AlzSolverClass import AlzSolverClass


def proc1():
    
    alzSolverCls = AlzSolverClass(test=True)
    alzSolverCls.stepRunSolver()


def eval_acc():
    

    alzSolverCls = AlzSolverClass(test=True)
    alzSolverCls.eval_alzNet_acc()


def train_learn_all():

    alzSolverCls = AlzSolverClass(test=True)
    alzSolverCls.train_learn_all()
    
    
def main():
    
    #proc1()
    #eval_acc()
    
    train_learn_all()
    
    
if __name__ == "__main__":
    main()