import time
from util.util import error, print_timestamped, normalize_with_opt, info
import numpy as np
import os


class ExcelEvaluate:
    def __init__(self, filepath, excel=False):
        self.excel_filename = None
        self.excel = excel
        if self.excel:
            if not os.path.exists(os.path.dirname(filepath)):
                os.makedirs(os.path.dirname(filepath))
            self.excel_filename = filepath
            self.ff = open(self.excel_filename, "w")
            init_rows = [
                "query_filename",
                "filter",
                "MSE",
                "TumourMSE",
                "scaled_MSE",
                "scaled_TumourMSE"
            ]
            for i, n in enumerate(init_rows):
                self.ff.write(n)
                if i < len(init_rows) - 1:
                    self.ff.write(",")
                else:
                    self.ff.write("\n")

    def print_to_excel(self, data):
        for i, d in enumerate(data):
            self.ff.write(str(d))
            if i < len(data) - 1:
                self.ff.write(",")
            else:
                self.ff.write("\n")

    def evaluate(self, mri_dict, query_name, smoothing="median"):
        print("Measures for the predicted images.")
        mse, tumour = evaluate_result(mri_dict['real_B'],
                                      mri_dict['fake_B'],
                                      tumor_mask=mri_dict['truth'],
                                      multiplier=1 / 4)  # To make it comparable to [0, 1] range
        print("Measures for the predicted images after smoothing.")
        smooth_mse, smooth_tumour = evaluate_result(mri_dict['real_B'],
                                                    mri_dict['fake_B_smoothed'],
                                                    tumor_mask=mri_dict['truth'],
                                                    multiplier=1 / 4)  # To make it comparable to [0, 1] range)
        print_timestamped("Computing MSE on the scaled data")
        # Scale data in 0,1 and compute everything again
        s_real = normalize_with_opt(mri_dict['real_B'], 0)
        s_predicted = normalize_with_opt(mri_dict['fake_B'], 0)
        s_predicted_smoothed = normalize_with_opt(mri_dict['fake_B_smoothed'], 0)
        print("Measures for the predicted images.")
        scaled_mse, scaled_tumour = evaluate_result(s_real,
                                                    s_predicted,
                                                    tumor_mask=mri_dict['truth'])
        print("Measures for the predicted images after smoothing.")
        s_smooth_mse, s_smooth_tumour = evaluate_result(s_real,
                                                        s_predicted_smoothed,
                                                        tumor_mask=mri_dict['truth'])

        smoothing = 0 if smoothing == "median" else 1
        if self.excel:
            self.print_to_excel([query_name, -1,
                                 mse, tumour, scaled_mse, scaled_tumour,
                                 ])
            self.print_to_excel([query_name, smoothing,
                                 smooth_mse, smooth_tumour, s_smooth_mse, s_smooth_tumour,
                                 ])

    def close(self):
        if self.excel:
            self.ff.close()
            print_timestamped("Saved in " + str(self.excel_filename))


def evaluate_result(seq, learned_seq, tumor_mask=None, round_fact=6, multiplier=1.0):
    if seq.shape != learned_seq.shape:
        error("The shape of the target and learned sequencing are not the same: " +
              str(seq.shape) + ", " + str(learned_seq.shape))

    tumour = None
    mask = np.logical_or(seq != seq.min(), learned_seq != learned_seq.min())
    ground_truth = seq[mask]
    prediction = learned_seq[mask]

    # MSE: avg((A-B)^2)
    mse = (np.square(np.subtract(ground_truth, prediction))).mean()
    mse = round(mse * multiplier, round_fact)

    print("MSE: " + str(mse), end="")
    if np.sum(tumor_mask) > 0:
        tumour = (np.square(np.subtract(seq[tumor_mask], learned_seq[tumor_mask]))).mean()
        tumour = round(tumour * multiplier, round_fact)
        print(", MSE of the tumor area: " + str(tumour), end="")
    print(".")
    return mse, tumour
