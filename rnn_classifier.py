import pandas as pd
import numpy as np 
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.contrib import rnn


# Helper class to perform K-Folds Validation splitting
class CrossValidationFolds(object):
    
    def __init__(self, data, labels, num_folds, shuffle=True):
        self.data = data
        self.labels = labels
        self.num_folds = num_folds
        self.current_fold = 0
        
        # Shuffle Dataset
        if shuffle:
            perm = np.random.permutation(self.data.shape[0])
            data = data[perm]
            labels = labels[perm]

    
    def split(self):
        current = self.current_fold
        size = int(self.data.shape[0]/self.num_folds)
        
        index = np.arange(self.data.shape[0])
        lower_bound = index >= current*size
        upper_bound = index < (current + 1)*size
        cv_region = lower_bound*upper_bound

        cv_data = self.data[cv_region]
        train_data = self.data[~cv_region]
        
        cv_labels = self.labels[cv_region]
        train_labels = self.labels[~cv_region]
        
        self.current_fold += 1
        return (train_data, train_labels), (cv_data, cv_labels), (size, cv_region)

PATH = 'DATASET/'
TRAIN = 'traj_1_train_shuffled.csv'
TRAIN2 = 'traj_1_test_shuffled.csv'
TEST = 'traj_2_sta_un.csv'
'''

PATH = 'DATASET/'
TRAIN = 'traj_1_train_shuffled.csv'
TEST = 'traj_1_test_shuffled.csv'
'''
# Read Data
print('Reading CSV Data...')
df_traj1 = pd.read_csv(PATH + TRAIN)
df__traj1_add = pd.read_csv(PATH + TRAIN2)
df = pd.concat([df_traj1, df__traj1_add] , axis = 0)
test_df = pd.read_csv(PATH + TEST)

#Create a new feature for normal (non-fraudulent) transactions.
df.loc[df.label == 0, 'Slip'] = 1
df.loc[df.label == 1, 'Slip'] = 0

test_df.loc[test_df.label == 0, 'Slip'] = 1
test_df.loc[test_df.label == 1, 'Slip'] = 0

#Rename 'Class' to 'Fraud'.
df = df.rename(columns={'label': 'Stable'})
test_df = test_df.rename(columns={'label': 'Stable'})

#Create dataframes of only Fraud and Normal transactions.
Slip = df[df.Slip == 1]
Stable = df[df.Stable == 1]

test_Slip = test_df[test_df.Slip == 1]
test_Stable = test_df[test_df.Stable == 1]

# Set X_train equal to 80% of the fraudulent transactions.
#X_train = Slip.sample
#count_Slips = len(Slip)

# Add 80% of the normal transactions to X_train.
X_train = pd.concat([Slip, Stable], axis = 0)
X_test = pd.concat([test_Slip, test_Stable], axis = 0)

# X_test contains all the transaction not in X_train.
#X_test = df.loc[~df.index.isin(X_train.index)]

#Shuffle the dataframes so that the training is done in a random order.
X_train = shuffle(X_train)
X_test = shuffle(test_df)

#Add our target features to y_train and y_test.
y_train = X_train.Slip
y_train = pd.concat([y_train, X_train.Stable], axis=1)

y_test = X_test.Slip
y_test = pd.concat([y_test, X_test.Stable], axis=1)

#Drop target features from X_train and X_test.
X_train = X_train.drop(['Slip','Stable'], axis = 1)
X_test = X_test.drop(['Slip','Stable'], axis = 1)


#Select certain features
#cols = [c for c in X_train.columns if c.lower()[:5] != 'median']
X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='left')))]
X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='left')))]
X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='right')))]
X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='right')))]
'''X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='imu')))]
X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='imu')))]'''
X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='gripper')))]
X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='gripper')))]


'''X_train = X_train[X_train.columns.drop(list(X_train.filter(regex='pressure')))]
X_test = X_test[X_test.columns.drop(list(X_test.filter(regex='pressure')))]'''
print ("FEATURES: ", X_train.columns)
print ("FEATURES SIZE: ", X_train.columns.size)
#X_train = X_train[cols]
#X_test = X_test[cols]

#Names of all of the features in X_train.
features = X_train.columns.values

#Transform each feature in features so that it has a mean of 0 and standard deviation of 1; 
#this helps with training the neural network.
for feature in features:
    mean, std = df[feature].mean(), df[feature].std()
    X_train.loc[:, feature] = (X_train[feature] - mean) / std
    X_test.loc[:, feature] = (X_test[feature] - mean) / std



# Split the testing data into validation and testing sets
#split = int(len(y_test)/2)


inputX = X_train.as_matrix()
inputY = y_train.as_matrix()
#inputX_valid = X_test.as_matrix()[:split]
#inputY_valid = y_test.as_matrix()[:split]
inputX_test = X_test.as_matrix()
inputY_test = y_test.as_matrix()




# Number of input nodes.
time_steps=20
num_units=128
n_input=inputX[1, :].size
n_classes=2
# Parameters
training_epochs = 100 # should be 2000, it will timeout when uploading
training_dropout = 0.9
display_step = 1 # 10 
n_samples = y_train.shape[0]
batch_size = 20
learning_rate = 0.005
FOLDS = 2
trials = 5


out_weights=tf.Variable(tf.truncated_normal([num_units, n_classes]))
out_bias=tf.Variable(tf.truncated_normal([n_classes]))

x=tf.placeholder("float", [None, time_steps, n_input])
y=tf.placeholder("float", [None, n_classes])

input = tf.unstack(x, time_steps, 1)

lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")
prediction=tf.matmul(outputs[-1],out_weights)+out_bias
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))





accuracy_summary = [] # Record accuracy values for plot
cost_summary = [] # Record cost values for plot
valid_accuracy_summary = [] 
valid_cost_summary = [] 
test_accuracy_summary = [] 
test_cost_summary = [] 
stop_early = 0 # To keep track of the number of epochs before early stopping

# Save the best weights so that they can be used to make the final predictions
#checkpoint = "location_on_your_computer/best_model.ckpt"
saver = tf.train.Saver(max_to_keep=1)

# Initialize variables and tensorflow session
init = tf.global_variables_initializer()
with tf.Session() as sess:

	print ('Running...')

	for k in range(trials): 

		data = CrossValidationFolds(inputX, inputY, FOLDS)

		accuracy_summary.append([])
		cost_summary.append([])
		valid_accuracy_summary.append([])
		valid_cost_summary.append([])
		test_accuracy_summary.append([])
		test_cost_summary.append([])
		
		for i in range(FOLDS):
			
			#print('Current fold: {}\n'.format(data.current_fold + 1))
			(train_input, train_target), (cv_input, cv_target), (lower_bound, upper_bound) = data.split()

			sess.run(tf.global_variables_initializer())
			s = 0
			for epoch in range(training_epochs): 

				for batch in range(int(n_samples/batch_size)):
					batch_x = inputX[batch*batch_size : (1+batch)*batch_size]
					batch_y = inputY[batch*batch_size : (1+batch)*batch_size]

					sess.run(opt, feed_dict={x: batch_x, y: batch_y})
					
				# Display logs after every 10 epochs
				if (epoch) % display_step == 0:

					print ("Size ", batch_size.size)

					train_accuracy = sess.run(accuracy, feed_dict={x: batch_x, y:batch_y})	
					newCost = sess.run(loss, feed_dict={x:batch_x, y:batch_y})	

					valid_accuracy = sess.run(accuracy, feed_dict={x: cv_input, y:cv_target})	
					valid_newCost = sess.run(loss, feed_dict={x: cv_input, y:cv_target})

					test_accuracy = sess.run(accuracy, feed_dict={x: X_test, y: y_test})	
					test_newCost = sess.run(loss, feed_dict={x: X_test, y: y_test})												
												

				

					# Record the results of the model
					accuracy_summary[-1].append(train_accuracy)
					cost_summary[-1].append(newCost)

					valid_accuracy_summary[-1].append(valid_accuracy)
					valid_cost_summary[-1].append(valid_newCost)
					
					test_accuracy_summary[-1].append(test_accuracy)
					test_cost_summary[-1].append(test_newCost)

					

	print ("Number of iterations: ", s)
	accuracy_mean = np.array(accuracy_summary).mean(axis=0)  
	cost_mean = np.array(cost_summary).mean(axis=0)


	valid_accuracy_mean = np.array(valid_accuracy_summary).mean(axis=0)
	valid_cost_mean = np.array(valid_cost_summary).mean(axis=0)

	test_accuracy_mean = np.array(test_accuracy_summary).mean(axis=0)  
	test_cost_mean = np.array(test_cost_summary).mean(axis=0)
	

	print ('Done!')
	print ("Epoch:", epoch,
			"Train_Acc = ", (accuracy_mean), 
			#"Train_Cost = ", (cost_mean),
			"Valid_Acc = ", (valid_accuracy_mean),
			#"Valid_Cost = ", (valid_cost_mean)
			"Testing_Acc = ", (test_accuracy_mean))

	print()
	print("Optimization Finished!")
	print()

#with tf.Session() as sess:
	# Load the best weights and show its results
	#saver.restore(sess, checkpoint)
	#training_accuracy = sess.run(accuracy, feed_dict={x: inputX, y_: inputY, pkeep: training_dropout})
	#validation_accuracy = sess.run(accuracy, feed_dict={x: inputX_valid, y_: inputY_valid, pkeep: 1})

	#print("Results using the best Valid_Acc:")
	#print()
	#print("Training Accuracy =", training_accuracy)
	#print("Validation Accuracy =", validation_accuracy)


 # Plot the accuracy and cost summaries 
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10,4))

ax1.plot(accuracy_mean, color = 'green',  label = 'training') # green
ax1.plot(valid_accuracy_mean, color = 'orange', label = 'validation') # orange
ax1.plot(test_accuracy_mean, color = 'blue', label = 'testing') # blue

ax1.set_title('Accuracy')

ax2.plot(cost_mean, color = 'green',  label = 'training')
ax2.plot(valid_cost_mean, color = 'orange', label = 'validation')
ax2.plot(test_cost_mean, color = 'blue', label = 'testing') 

ax2.set_title('Cost')
plt.legend(bbox_to_anchor=(1.05, 1), loc=1, borderaxespad=0.)
plt.xlabel('Epochs (x10)')
plt.show()