import matplotlib.pyplot as plt
import numpy as np
import argparse
PATH = 'results/'

def compute_area(explanation_type):

	TOTAL_TRAIN_SIZE = 7500
	NUM_BLOCKS = 5
	NUM_TRIALS = 20

	# print data size vs accuracy curves
	baseline_area = 0
	max_area = 0
	baseline_acc_final = 0
	max_acc_final = 0
	for i in {-10,-8,-6,-4,-2,0,2}: #range(-10,3,2)
		lam = 0
		if i > -10:
			lam = 0.01 * (10 ** i)
		acc_curve = np.zeros(NUM_BLOCKS+1)
		for sz in range(1,NUM_BLOCKS+1,1):
			accuracies = np.zeros(NUM_TRIALS)
			size = (TOTAL_TRAIN_SIZE // NUM_BLOCKS) * sz
			for j in range(NUM_TRIALS):
				#trial_num = j+20 if lam == 0.01 else j
				curr_path = PATH + explanation_type + "/" + str(size) + "/run-" + str(j) + "-accuracy-curve-" + str(lam) + ".npy"
				if lam == 0: # for base line, always use middle layer explanation path
					curr_path = PATH + 'gradient' + "/" + str(size) + "/run-" + str(j) + "-accuracy-curve-" + str(lam) + ".npy"
				curr_curve = np.load(curr_path)
				accuracies[j] = curr_curve[-1]				
			acc_curve[sz] = np.median(accuracies)
		if acc_curve[-1] > max_acc_final:
			max_acc_final = acc_curve[-1]
		if lam == 0:
			baseline_acc_final = acc_curve[-1]
		label = ''
		power = i-2
		if lam == 0:
			label = '0 (baseline)'
		elif power == 0:
			label = '1'
		else:
			label = "1e" + str(power)
		area = np.trapz(acc_curve)
		if lam == 0:
			baseline_area = area
		if area > max_area:
			max_area = area
		print(label + ': ' + str(area))
	print('increase over baseline:', max_area / baseline_area)
	print('max final acc:', max_acc_final)




if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Parse explanation type")
	parser.add_argument('explanation_type', type=str, help="A required string argument for the explanation type to graph learning curves for")
	args = parser.parse_args()
	explanation = args.explanation_type
	compute_area(explanation_type=explanation)
