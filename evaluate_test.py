from evaluation_utils import evaluate
import json

gt_1 = "The Eiffel Tower is in Paris."
out_1 = "Eiffel Tower located in Paris, France."

print(evaluate(gt_1, out_1))

gt = "The photosynthesis process converts light energy into chemical energy."
out = "Plants use sun for energy." 
print(json.dumps(evaluate(gt, out), indent=2))