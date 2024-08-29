from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
#from .maser_q_learner import maserQLearner
#from .q_learner_divide import QDivedeLearner
#from .iper import FormDivedeLearner
from .esip_abs_test import Esip as Esip_test
from .esip import Esip
from .esip_abs import Esip as Esip_abs
from .q_learner_divide import QDivedeLearner
from .superQ import SuperQ

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
#REGISTRY["maser_q_learner"] = maserQLearner
#REGISTRY["q_learner_divide"] = QDivedeLearner

#REGISTRY["iper"] = FormDivedeLearner

#新增esip和测试版本
REGISTRY["esip_test"] = Esip_test
REGISTRY["esip"] = Esip
REGISTRY["esip_abs"] = Esip_abs
REGISTRY["q_learner_divide"] = QDivedeLearner
REGISTRY["superQ"] = SuperQ