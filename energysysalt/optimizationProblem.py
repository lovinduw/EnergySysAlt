import time
import pyomo.environ as pyomo
import fine as fn

def declareOptimalCostConstraint(esM, pyM):

    """ Optimum cost should not be more than the objective value obtained in the original optimization + slack value. 
    """
    fn.utils.output("Declaring cost constraint...", esM.verbose, 0)
    # slack = slack

    def optimalCostConstraint(pyM):
        return (
            sum(
                mdl.getObjectiveFunctionContribution(esM, pyM)
                for mdl in esM.componentModelingDict.values()
            )
            <= esM.objectiveValue*(1+esM.slack)
        )
    pyM.optimalCostConstraint = pyomo.Constraint(rule=optimalCostConstraint) 

def declareMGAObjective(esM, pyM,iteration,sense):

    fn.utils.output("Declaring MGA objective function...", esM.verbose, 0)

    def mgaOperation(
        mdl,
        pyM,
        esM,
        iteration,
        opVarName,
        isOperationCommisYearDepending=False,
        ):
        """
        Declare the objective function by obtaining the opertaion rate variables abd capacity variables of the components. The  objective function is the sum of the operation and capacity variables 
        of the componenets multiplied by the Beta value.

            .. math::
                    \\{min/max}\:  {\beta}_{loc,comp,iteration} * ({op}^{comp,opType}_{loc,ip,p,t} + {cap}^{comp}_{loc})

        """

        abbrvName = mdl.abbrvName
        opVar = getattr(pyM, opVarName + "_" + abbrvName)
        capVar = getattr(pyM, "cap_" + abbrvName)
        opVarSet = getattr(pyM, "operationVarSet_" + abbrvName)
        capVarSet = getattr(pyM, "designDimensionVarSet_" + abbrvName) 

        if isOperationCommisYearDepending:

            opsum = sum(opVar[loc, compName, commis, ip, p, t]  * esM.beta[loc][iteration][compName]
                for loc,compName,commis,ip in opVarSet for p,t in pyM.intraYearTimeSet
            )

        else:

            opsum = sum(opVar[loc, compName, ip, p, t] * esM.beta[loc][iteration][compName]
                for loc,compName,ip in opVarSet for p,t in pyM.intraYearTimeSet 
            )
            
        capsum = sum(capVar[loc, compName, ip]  * esM.beta[loc][iteration][compName]
                            for loc, compName, ip in capVarSet)
        
        return (opsum + capsum)
    def mgaObjective(pyM):
        mgaContribution  = 0
        for key,mdl in esM.componentModelingDict.items():
            if key != 'StorageModel':
                mgaContribution += mgaOperation(mdl,pyM, esM, iteration, "op")
            else:
                vars = ["chargeOp","dischargeOp"]
                storageContribution = sum(mgaOperation(mdl,pyM, esM, iteration, var) for var in vars)
                mgaContribution += storageContribution
        return mgaContribution
    if sense == "minimize":
        pyM.Obj = pyomo.Objective(rule=mgaObjective, sense=pyomo.minimize)
    else:
        pyM.Obj = pyomo.Objective(rule=mgaObjective, sense=pyomo.maximize)

def declareMGAOptimizationProblem(
    esM,
    iteration,
    sense,
    timeSeriesAggregation=False,
    relaxIsBuiltBinary=False,
    relevanceThreshold=None,
):

    """
    Declare the optimization problem belonging to the specified energy system for which a pyomo concrete model
    instance is built and filled with

    * basic time sets,
    * sets, variables and constraints contributed by the component modeling classes,
    * basic, component overreaching constraints, and
    * an objective function.

    **Default arguments:**

    :param timeSeriesAggregation: states if the optimization of the energy system model should be done with

        (a) the full time series (False) or
        (b) clustered time series data (True).

        |br| * the default value is False
    :type timeSeriesAggregation: boolean

    :param relaxIsBuiltBinary: states if the optimization problem should be solved as a relaxed LP to get the lower
        bound of the problem.
        |br| * the default value is False
    :type declaresOptimizationProblem: boolean

    :param relevanceThreshold: Force operation parameters to be 0 if values are below the relevance threshold.
        |br| * the default value is None
    :type relevanceThreshold: float (>=0) or None
    """
    fn.utils.output(f"MGA Iteration {iteration} {sense} .....", esM.verbose, 0 )

    # Get starting time of the optimization to, later on, obtain the total run time of the optimize function call
    timeStart = time.time()

    # Check correctness of inputs
    fn.utils.checkDeclareOptimizationProblemInput(
        timeSeriesAggregation, esM.isTimeSeriesDataClustered
    )

    # Set segmentation value if time series aggregation is True
    if timeSeriesAggregation:
        segmentation = esM.segmentation
    else:
        segmentation = False

    ################################################################################################################
    #                           Initialize mathematical model (ConcreteModel) instance                             #
    ################################################################################################################

    # Initialize a pyomo ConcreteModel which will be used to store the mathematical formulation of the model.
    # The ConcreteModel instance is stored in the EnergySystemModel instance, which makes it available for
    # post-processing or debugging. A pyomo Suffix with the name dual is declared to make dual values associated
    # to the model's constraints available after optimization.

    """
    :param pyM: a pyomo ConcreteModel instance which contains parameters, sets, variables,
        constraints and objective required for the optimization set up and solving.
    :type pyM: pyomo ConcreteModel
    """
    esM.pyM = pyomo.ConcreteModel()
    pyM = esM.pyM
    pyM.dual = pyomo.Suffix(direction=pyomo.Suffix.IMPORT)

    # Set time sets for the model instance
    esM.declareTimeSets(pyM, timeSeriesAggregation, segmentation)

    ################################################################################################################
    #                         Declare component specific sets, variables and constraints                           #
    ################################################################################################################

    for key, mdl in esM.componentModelingDict.items():
        _t = time.time()
        fn.utils.output("Declaring sets, variables and constraints for " + key, esM.verbose, 0)
        fn.utils.output("\tdeclaring sets... ", esM.verbose, 0), mdl.declareSets(esM, pyM)
        fn.utils.output("\tdeclaring variables... ", esM.verbose, 0), mdl.declareVariables(esM, pyM,relaxIsBuiltBinary, relevanceThreshold)
        fn.utils.output("\tdeclaring constraints... ", esM.verbose, 0), mdl.declareComponentConstraints(esM, pyM)
        fn.utils.output("\t\t(%.4f" % (time.time() - _t) + " sec)\n", esM.verbose, 0)

    ################################################################################################################
    #                              Declare cross-componential sets and constraints                                 #
    ################################################################################################################

    # Declare constraints for enforcing shared capacities
    _t = time.time()
    esM.declareSharedPotentialConstraints(pyM)
    fn.utils.output("\t\t(%.4f" % (time.time() - _t) + " sec)\n", esM.verbose, 0)

    # Declare constraints for linked quantities
    _t = time.time()
    esM.declareComponentLinkedQuantityConstraints(pyM)
    fn.utils.output("\t\t(%.4f" % (time.time() - _t) + " sec)\n", esM.verbose, 0)

    # Declare commodity balance constraints (one balance constraint for each commodity, location and time step)
    _t = time.time()
    esM.declareCommodityBalanceConstraints(pyM)
    fn.utils.output("\t\t(%.4f" % (time.time() - _t) + " sec)\n", esM.verbose, 0)

    # Declare constraint for balanceLimit
    _t = time.time()
    esM.declareBalanceLimitConstraint(pyM, timeSeriesAggregation)
    fn.utils.output("\t\t(%.4f" % (time.time() - _t) + " sec)\n", esM.verbose, 0)

    # Declare constraint for optimal cost
    _t = time.time()
    declareOptimalCostConstraint(esM, pyM)
    fn.utils.output("\t\t(%.4f" % (time.time() - _t) + " sec)\n", esM.verbose, 0)

    ###############################################################################################################
    #                                         Declare MGA objective function                                           #
    ################################################################################################################

    # Declare objective function by obtaining the contributions to the objective function from all modeling classes
    _t = time.time()
    declareMGAObjective(esM, pyM,iteration,sense)
    fn.utils.output("\t\t(%.4f" % (time.time() - _t) + " sec)\n", esM.verbose, 0)

    # Store the build time of the optimize function call in the EnergySystemModel instance
    esM.solverSpecs["buildtime"] = time.time() - timeStart  