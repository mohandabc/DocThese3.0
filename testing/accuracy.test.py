from d_segmentation import compute_performance
from skimage import io
from pathlib import Path
import os 
import csv

ground_truths_path = Path('gts')
results_path = Path('res')
file_headers = ['img', 'sensitivity', 'Specificity', 'Accuracy']
results_per_model = [dir for dir in os.listdir(results_path) if (results_path/dir).is_dir()]

for model_results in results_per_model:
    print(f"------------------{model_results}-------------------\n")
    results_dict = []
    results = [img for img in os.listdir(results_path/model_results) if not (results_path/model_results/img).is_dir()]
    print(results)
    gts = [img for img in os.listdir(ground_truths_path)]
    for res, gt in zip(results, gts):
        res_path = results_path / model_results / res
        gt_path = ground_truths_path / gt

        res_img = io.imread(res_path)
        gt_img = io.imread(gt_path)

        accuracy = compute_performance(res_img, gt_img)
        accuracy['img'] = f"{res} -- {gt}"
        results_dict.append(accuracy)
        # print(f"{res} -- {gt}\n  {accuracy}\n\n")
    with open(results_path / model_results /"results.csv", "w") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = file_headers)
        writer.writeheader()
        writer.writerows(results_dict)
    
