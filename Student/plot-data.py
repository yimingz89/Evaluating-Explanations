import matplotlib.pyplot as plt
import numpy as np
import argparse
PATH = 'results/'
COLOR_MAP = {'0 (baseline)': 'black', '1': 'green', '1e-2': 'red', '1e-4': 'blue', '1e-6': 'magenta', '1e-8': 'yellow', '1e-10': 'cyan'}

def plot_middle_and_edge():
	plt.figure()
	save_path = PATH + "/all_explanations.pdf"

	TOTAL_TRAIN_SIZE = 7500
	NUM_BLOCKS = 5
	NUM_TRIALS = 20

	# print data size vs accuracy curves
	for i in range(5):
		# lam = 0
		# if i > -10:
		# 	lam = 0.01 * (10 ** i)
		acc_curve = np.zeros(NUM_BLOCKS+1)
		for sz in range(1,NUM_BLOCKS+1,1):
			accuracies = np.zeros(NUM_TRIALS)
			size = (TOTAL_TRAIN_SIZE // NUM_BLOCKS) * sz
			for j in range(NUM_TRIALS):
				curr_path = ''
				if i == 0: # for base line, always use middle layer explanation path
					lam = 0
					curr_path = PATH + 'gradient' + "/" + str(size) + "/run-" + str(j) + "-accuracy-curve-" + str(lam) + ".npy"
				elif i == 1:
					lam = 0.01 * (10 ** 0)
					curr_path = PATH + 'middle-layer' + "/" + str(size) + "/run-" + str(j) + "-accuracy-curve-" + str(lam) + ".npy"
				elif i == 2:
					lam = 0.01 * (10 ** 0)
					curr_path = PATH + 'edge-detector' + "/" + str(size) + "/run-" + str(j) + "-accuracy-curve-" + str(lam) + ".npy"
				elif i == 3:
					lam = 0.01 * (10 ** 2)
					curr_path = PATH + 'gradient' + "/" + str(size) + "/run-" + str(j) + "-accuracy-curve-" + str(lam) + ".npy"
				elif i == 4:
					lam = 0.01 * (10 ** 2)
					curr_path = PATH + 'smoothgrad' + "/" + str(size) + "/run-" + str(j) + "-accuracy-curve-" + str(lam) + ".npy"
				curr_curve = np.load(curr_path)
				accuracies[j] = curr_curve[-1]				
			acc_curve[sz] = np.median(accuracies)
		if i == 0:
			label = 'baseline'
			plt.plot(np.arange(len(acc_curve)) * (TOTAL_TRAIN_SIZE // NUM_BLOCKS), acc_curve, label=label, color='black')	
		elif i == 1:
			label = 'middle-layer'
			plt.plot(np.arange(len(acc_curve)) * (TOTAL_TRAIN_SIZE // NUM_BLOCKS), acc_curve, label=label, color='red')	
		elif i == 2:
			label = 'edge-detector'
			plt.plot(np.arange(len(acc_curve)) * (TOTAL_TRAIN_SIZE // NUM_BLOCKS), acc_curve, label=label, color='blue')	
		elif i == 3:
			label = 'gradient'
			plt.plot(np.arange(len(acc_curve)) * (TOTAL_TRAIN_SIZE // NUM_BLOCKS), acc_curve, label=label, color='yellow')
		elif i == 4:
			label = 'smoothgrad'
			plt.plot(np.arange(len(acc_curve)) * (TOTAL_TRAIN_SIZE // NUM_BLOCKS), acc_curve, label=label, color='orange')	


	
	plt.ylim(0,70)
	plt.xticks(np.arange(NUM_BLOCKS+1) * (TOTAL_TRAIN_SIZE // NUM_BLOCKS))
	plt.xlabel('training set size')
	plt.ylabel('accuracy')
	plt.title('training set size vs accuracy')
	plt.legend(loc="lower right")
	plt.savefig(save_path)
	plt.close()

def plot_accuracy_data_size(explanation_type):
	plt.figure()
	save_path = PATH + explanation_type + "/data-size-accuracy-plot.pdf"

	TOTAL_TRAIN_SIZE = 7500
	NUM_BLOCKS = 5
	NUM_TRIALS = 20

	# print data size vs accuracy curves
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

		label = ''
		power = i-2
		if lam == 0:
			label = '0 (baseline)'
		elif power == 0:
			label = '1'
		else:
			label = "1e" + str(power)
		plt.plot(np.arange(len(acc_curve)) * (TOTAL_TRAIN_SIZE // NUM_BLOCKS), acc_curve, label=label, color=COLOR_MAP[label])	
	
	plt.ylim(0,70)
	plt.xticks(np.arange(NUM_BLOCKS+1) * (TOTAL_TRAIN_SIZE // NUM_BLOCKS))
	plt.xlabel('training set size')
	plt.ylabel('accuracy')
	plt.title('training set size vs accuracy for ' + str(explanation_type) + ' explanation')
	plt.legend(loc="lower right")
	plt.savefig(save_path)
	plt.close()

def plot_accuracy_time():
	TOTAL_TRAIN_SIZE = 7500
	NUM_BLOCKS = 5
	NUM_TRIALS = 5
	SAMPLE_PERIOD = 10

	# print data size vs accuracy curves
	for sz in range(1,NUM_BLOCKS+1,1):
		plt.figure()
		save_path = "./plots/time-accuracy-" + str(sz * (TOTAL_TRAIN_SIZE // NUM_BLOCKS)) + "-plot.png"
		size = (TOTAL_TRAIN_SIZE // NUM_BLOCKS) * sz
		for i in range(-10,3,2):
			acc_curves = np.zeros((NUM_TRIALS, SAMPLE_PERIOD * sz))
			lam = 0
			if i > -10:
				lam = 0.01 * (10 ** i)
			accuracies = np.zeros(NUM_TRIALS)
			for j in range(NUM_TRIALS):
				curr_path = "./results/" + str(size) + "/run-" + str(j) + "-accuracy-curve-" + str(lam) + ".npy"
				curr_curve = np.load(curr_path)
				acc_curves[j] = curr_curve
			
			mean_curve = np.mean(acc_curves, axis=0)
			label = ''
			power = i-2
			if lam == 0:
				label = '0 (baseline)'
			elif power == 0:
				label = '1'
			else:
				label = "1e" + str(power)
			plt.plot(np.arange(len(mean_curve)), mean_curve, label=label)	

		plt.xlabel('time')
		plt.ylabel('accuracy')
		plt.legend(loc="lower right")
		plt.title('time vs accuracy for training size ' + str(size))
		plt.savefig(save_path)
		plt.close()

if __name__ == "__main__":
	# parser = argparse.ArgumentParser(description="Parse explanation type")
	# parser.add_argument('explanation_type', type=str, help="A required string argument for the explanation type to graph learning curves for")
	# args = parser.parse_args()
	# explanation = args.explanation_type
	# plot_accuracy_data_size(explanation_type=explanation)
	plot_middle_and_edge()
	#plot_accuracy_time()
