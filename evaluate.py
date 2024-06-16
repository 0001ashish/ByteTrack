class EvaluatorArgs:
     track_thresh = 0.3
     track_buffer = 30
     min_box_area = 1

args = EvaluatorArgs()
from yolox.evaluators import MOTEvaluator

evaluator = MOTEvaluator(args=args,)