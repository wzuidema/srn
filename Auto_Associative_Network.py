from random import uniform
from math import exp
from copy import deepcopy

	
data_set_12 = [[['ga'], ['li'], ['ni'], ['ta'], ['gi'], ['la'], ['na'], ['ti'], ['wo'], ['de'], ['fe'], ['ko']]]

data_set_4 = [[['bi'], ['bo'], ['bu'], ['ka']]]

data_set_1 = [[['bi']]]


################################################################################
### These are the parameters that have to be set before training or testing!!

training_set = data_set_12

# Choose the number of sweeps through the training set:
sweeps_train = 2000

# Choose the learning rate:
eta = 0.5

# Choose the number of input nodes:
input_nodes = 12

# Choose the number of encoding nodes:
encoding_nodes = 12

# Choose the number of output nodes:
output_nodes = 12

# Choose range for the initialization of the weights:
weight_range = 0.5

################################################################################

bi = [1, 1, 0, 0]
bo = [1, 0, 1, 0]
bu = [0, 1, 1, 0]
ka = [0, 0, 0, 1]


# Training syllables babies:
ga = [0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1]
li = [0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1]
ni = [0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1]
ta = [0, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1]
gi = [0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1]
la = [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
na = [0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1]
ti = [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
# Testing syllables babies:
wo = [0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]
de = [0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0]
fe = [0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0]
ko = [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1]


# Training syllables_network:
le = [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
wi = [0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
ji = [0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1]
de = [0, 0,	1, 1, 1, 0, 1, 1, 0, 0, 1, 0]
di = [0, 0,	1, 1, 1, 0, 1, 1, 1, 1, 1, 1]
je = [0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]
li = [0, 1, 0, 1, 1, 0,	1, 1, 1, 1, 1, 1]
we = [0, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0]
# Testing syllables_network:
ba = [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1]
ko = [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1]
po = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]
ga = [0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1]



################################################################################
# This function takes the training or testing set as input and replaces the syllables with their input vector according to the coding scheme described above.
def calc_synapses_in(t_set):
	synapses_in_list = []
	for part_set in t_set:
		for sequence in part_set:
			for syllable in sequence:
				if syllable == 'ga':
					synapses_in = deepcopy(ga)
				elif syllable == 'li':
					synapses_in = deepcopy(li)
				elif syllable == 'ni':
					synapses_in = deepcopy(ni)
				elif syllable == 'ta':
					synapses_in = deepcopy(ta)
				elif syllable == 'gi':
					synapses_in = deepcopy(gi)
				elif syllable == 'la':
					synapses_in = deepcopy(la)
				elif syllable == 'na':
					synapses_in = deepcopy(na)
				elif syllable == 'ti':
					synapses_in = deepcopy(ti)
				elif syllable == 'wo':
					synapses_in = deepcopy(wo)
				elif syllable == 'de':
					synapses_in = deepcopy(de)
				elif syllable == 'fe':
					synapses_in = deepcopy(fe)
				elif syllable == 'ko':
					synapses_in = deepcopy(ko)
				##############################	
				elif syllable == 'bi':
					synapses_in = deepcopy(bi)
				elif syllable == 'bo':
					synapses_in = deepcopy(bo)
				elif syllable == 'bu':
					synapses_in = deepcopy(bu)
				elif syllable == 'ka':
					synapses_in = deepcopy(ka)
				synapses_in_list.append(synapses_in)
	return synapses_in_list
## The last part of this function 'synapses_in.append(-1)' adds the bias node to each of the vectors. The activation of the bias node is always -1.

synapses_in_list = calc_synapses_in(training_set)


# Function that calculates the length of the dataset that is used.
def calc_length(t_set):
	length = 0
	for part_set in t_set:
		for sequence in part_set:
			for syllable in sequence:
				length += 1
	return length

	
################################################################################


def create_weights_in_enc():
	weights_list = []
	weight_vector = range(input_nodes)
	def create_vector():
		new_vector = deepcopy(weight_vector)
		for i in range(len(new_vector)):
			new_vector[i] = uniform(-weight_range, weight_range)
		return new_vector
	for i in range(encoding_nodes):
		node_vector = create_vector()
		weights_list.append(node_vector)
	return weights_list
	
weights_in_enc = create_weights_in_enc()

#weights_in_enc = [[0.1, 0.2, -0.1, 0.1], [0.3, 0.1, -0.1, 0.3], [-0.2, 0.2, -0.1, -0.2], [-0.3, 0.2, -0.1, -0.3]]

#weights_in_enc = [[0.9, 0.8, 0.7, 0.8], [0.6, 0.7, 0.8, 0.9], [0.7, 0.8, 0.6, 0.7], [0.8, 0.9, 0.9, 0.7]]


def create_weights_enc_out():
	weights_list = []
	weight_vector = range(encoding_nodes)
	def create_vector():
		new_vector = deepcopy(weight_vector)
		for i in range(len(new_vector)):
			new_vector[i] = uniform(-weight_range, weight_range)
		return new_vector
	for i in range(output_nodes):
		node_vector = create_vector()
		weights_list.append(node_vector)
	return weights_list

weights_enc_out = create_weights_enc_out()

#weights_enc_out = [[0.9, 0.8, 0.7, 0.8], [0.6, 0.7, 0.8, 0.9], [0.7, 0.8, 0.6, 0.7], [0.8, 0.9, 0.9, 0.7]]

# uniform(-0.5, 0.5) selects a random float between -0.5 and 0.5, so the weights are initialized randomly.



################################################################################
# Below the layers of the network are defined:



class Input_Layer(object):
	def __init__(self):
		self.activations = synapses_in_list[cycle]

class Layer(object):
	def __init__(self, synapse_list, weights_list):
		self.synapses = synapse_list
		self.weights = weights_list
	def calc_net(self):
		weighted_sum_list = []
		net_list = deepcopy(self.weights)
		for i in range (len(net_list)):
			for j in range (len(net_list[i])):
				net_list[i][j] = net_list[i][j]*self.synapses[j]
		for i in range (len(net_list)):
			weighted_sum = sum(net_list[i]) 
			weighted_sum_list.append(weighted_sum)
		self.weighted_sum_list = weighted_sum_list
		return True
	def calc_activations(self):
		self.calc_net()
		activations_list = deepcopy(self.weighted_sum_list)
		for i in range(len(activations_list)):
			activations_list[i] = (1/(1+exp(-activations_list[i])))
		self.activations = activations_list
		return True
# The calc_net function that is defined in this class takes the list of synapse activations and the list of connection weights as input.
# Then first multiplies each synapse activation with the weights of the connections going out from that node.
# Then sums all the s*w's that come in to a certain node, yielding the net_list, which holds a slot for every receiving node that was in the weights list.
# The calc_activation function then modifies these net activations with the sigmoid activation function.



class Encoding_Layer(Layer):
	def __init__(self, synapses_hid, weights_in_enc, weights_enc_out):
		self.synapses = synapses_hid
		self.weights = weights_in_enc
		self.weights_enc_out = weights_enc_out
		self.calc_activations()
	def calc_error(self):
		#First the error of the downstream nodes (output layer) are multiplied with the weights of the connections to those nodes:
		downstream_error_matrix = deepcopy(self.weights_enc_out)
		for i in range(len(output_error_list)):
			for j in range(len(downstream_error_matrix[i])):
				downstream_error_matrix[i][j] = downstream_error_matrix[i][j]*output_error_list[i]
		#Then these error*weight values (12 per encoding node) are summed for each node of the encoding layer
		sum_downstream_error_list = []
		sum_downstream_error = 0
		count_list = 0
		while count_list < len(downstream_error_matrix[0]):
			for i in range(len(downstream_error_matrix)):
				sum_downstream_error += downstream_error_matrix[i][count_list]
			sum_downstream_error_list.append(sum_downstream_error)
			count_list += 1
			sum_downstream_error = 0
		#Now the real error is calculated:
		error_list = deepcopy(self.activations)
		for i in range(len(error_list)):
			#First the derivation of the sigmoid activation function is calculated:
			error_list[i] = (error_list[i]*(1-error_list[i]))
			#Then this derivation is multiplied with the downstream error for that node:
			error_list[i] = error_list[i]*sum_downstream_error_list[i]
		self.error_list = error_list
		return True
	def update_weights(self):
		self.calc_error()		
		weights_list = deepcopy(self.weights)
		for i in range(len(weights_list)):
			for j in range(len(weights_list[i])):
				if weights_list[i][j] <= 1 and weights_list[i][j] >= -1:
					weights_list[i][j] = weights_list[i][j]+(eta*self.error_list[i]*self.synapses[j])
				else:
					weights_list[i][j] = weights_list[i][j]
		self.weights = weights_list
		return True
# The encoding layer borrows its calc_error and update_weights functions straight from the class Hidden_Layer, except that here there is only one layer (instead of two) that projects to the encoding layer.


class Output_Layer(Layer):
	def __init__(self, synapses_enc, weights_enc_out):
		self.synapses = synapses_enc
		self.weights = weights_enc_out
		self.calc_activations()
	def calc_error(self):
		error_list = deepcopy(self.activations)
		error = 0
		for i in range(len(error_list)):
			error_list[i] = error_list[i]*(1-error_list[i])*(target[i]-error_list[i])
		self.error_list = error_list
		return True
	def update_weights(self):
		self.calc_error()
		weights_list = deepcopy(self.weights)
		for i in range(len(weights_list)):
			for j in range(len(weights_list[i])):
				if weights_list[i][j] <= 1 and weights_list[i][j] >= -1:
					weights_list[i][j] = weights_list[i][j]+(eta*self.error_list[i]*self.synapses[j])
				else:
					weights_list[i][j] = weights_list[i][j]
		self.weights = weights_list
		return True
# The calc_error function is defined under the scope of this subclass because the error calculation for the output node works different than that of the hidden nodes (i.e. it doesn't take downstream units into account because there aren't any).
# Then the update_weights() function is also defined under the scope of this subclass because it has to work with the specific error list for the output nodes.


################################################################################
# Below the first input vector is initialized. 
# The initial activations of the context layer are initialized (this is 0.0 because the activations can vary between -1 and 1).
# Also the hidden layer is initialized here, taking the first input vector and the initial context layer activations as input. Later on in the while loop it is changed take the attribute '.activations' of the input layer and the context layer as input instead, so that these values will change with each iteration as they should.



init_input_vector = synapses_in_list[0]

encoding_layer = Encoding_Layer(init_input_vector, weights_in_enc, weights_enc_out)

output_layer = Output_Layer(encoding_layer.activations, weights_enc_out)



target_list_train = []
output_activations_matrix = []
output_activations_rounded = []
output_error_matrix = []

sweep = 0
while sweep < sweeps_train:
	cycle = 0
	while cycle < calc_length(training_set):

	
		print "This is cycle: "+str(cycle)
	
		print "The input vector is:"
		print synapses_in_list[cycle]
		
		
		target = synapses_in_list[cycle]
	
		print "The target is:"
		print target
		
		
		target_list_train.append(target)
	
	
		input_layer = Input_Layer()
	
	
		print "The input_layer_activations are:"
		print input_layer.activations
	

		print "The encoding_layer_activations are:"
		print encoding_layer.activations
	
	
		print "The output_layer_activations are:"
		print output_layer.activations
	
	
		output_activations_matrix.append(output_layer.activations)
	
	
		output_activations = deepcopy(output_layer.activations)
		for i in range(len(output_activations)):
			output_activations[i] = round(output_activations[i])
		
		print "output_activations (rounded) is:" 
		print output_activations
	
	
		output_activations_rounded.append(output_activations)
		

		

		output_layer.update_weights()
	
	
		print "output_layer.error_list is:"
		print output_layer.error_list
		
	
		print "output_layer.weights (new) are:"
		print output_layer.weights
	
	
		output_error_list = output_layer.error_list
		
		
				
		
		external_error_list = []
		for i in range(len(output_layer.activations)):
			external_error = (target[i]-output_layer.activations[i])**2
			external_error_list.append(external_error)
				
		
#		print "external_error_list is:"
#		print external_error_list

		
		output_error_matrix.append(external_error_list)

				
		average_error_list = []
		for list in output_error_matrix:
			average_error = sum(list)/len(list)
			average_error_list.append(average_error)
			
			
		
	
		encoding_layer.update_weights()
		
		print "encoding_layer.error_list is:"
		print encoding_layer.error_list
	
		print "encoding_layer.weights (new) are:"
		print encoding_layer.weights
		
		
		encoding_layer = Encoding_Layer(input_layer.activations, encoding_layer.weights, encoding_layer.weights_enc_out)

		output_layer = Output_Layer(encoding_layer.activations, output_layer.weights)


		cycle += 1
		
	sweep += 1
	
print ""
print ""
print "DONE"
print ""



################################################################################
# Here the results are evaluated by giving for each prediction the nearest neighbour of the syllables that are part of the data set.
#!!!Do this more efficiently??

def seek_neighbours(output_vector):
	ga_dstnce = 0
	li_dstnce = 0
	ni_dstnce = 0
	ta_dstnce = 0
	gi_dstnce = 0
	la_dstnce = 0
	na_dstnce = 0
	ti_dstnce = 0
	wo_dstnce = 0
	de_dstnce = 0
	fe_dstnce = 0
	ko_dstnce = 0
	for i in range(len(output_vector)):
		ga_dstnce += (ga[i]-output_vector[i])**2
		li_dstnce += (li[i]-output_vector[i])**2
		ni_dstnce += (ni[i]-output_vector[i])**2
		ta_dstnce += (ta[i]-output_vector[i])**2
		gi_dstnce += (gi[i]-output_vector[i])**2
		la_dstnce += (la[i]-output_vector[i])**2
		na_dstnce += (na[i]-output_vector[i])**2
		ti_dstnce += (ti[i]-output_vector[i])**2
		wo_dstnce += (wo[i]-output_vector[i])**2
		de_dstnce += (de[i]-output_vector[i])**2
		fe_dstnce += (fe[i]-output_vector[i])**2
		ko_dstnce += (ko[i]-output_vector[i])**2
	dstnce_list = [ga_dstnce, li_dstnce, ni_dstnce, ta_dstnce, gi_dstnce, la_dstnce, na_dstnce, ti_dstnce, wo_dstnce, de_dstnce, fe_dstnce, ko_dstnce]
	syllable_list = ['ga', 'li', 'ni', 'ta', 'gi', 'la', 'na', 'ti', 'wo', 'de', 'fe', 'ko']
	winner_value = min(dstnce_list)
	winner_index = dstnce_list.index(winner_value)
	winner = syllable_list[winner_index]
	return winner

def seek_neighbours_4(output_vector):
	bi_dstnce = 0
	bo_dstnce = 0
	bu_dstnce = 0
	ka_dstnce = 0
	for i in range(len(output_vector)):
		bi_dstnce += (bi[i]-output_vector[i])**2
		bo_dstnce += (bo[i]-output_vector[i])**2
		bu_dstnce += (bu[i]-output_vector[i])**2
		ka_dstnce += (ka[i]-output_vector[i])**2
	dstnce_list = [bi_dstnce, bo_dstnce, bu_dstnce, ka_dstnce]
	syllable_list = ['bi', 'bo', 'bu', 'ka']
	winner_value = min(dstnce_list)
	winner_index = dstnce_list.index(winner_value)
	winner = syllable_list[winner_index]
	return winner


def seek_neighbours_1(output_vector):
	bi_dstnce = 0
	for i in range(len(output_vector)):
		bi_dstnce += (bi[i]-output_vector[i])**2
	dstnce_list = [bi_dstnce]
	syllable_list = ['bi']
	winner_value = min(dstnce_list)
	winner_index = dstnce_list.index(winner_value)
	winner = syllable_list[winner_index]
	return winner
	




print "len(output_activations_matrix) is:"
print len(output_activations_matrix)




data_set_train = []
sweep = 0
while sweep < sweeps_train:
	for part_set in training_set:
		for sequence in part_set:
			for syllable in sequence:
				data_set_train.append(syllable)
	sweep += 1
				



print "len(data_set_train)"
print len(data_set_train)



	
	
	
neighbours_output_train = []
for i in range(len(output_activations_matrix)):
	neighbour = seek_neighbours(output_activations_matrix[i])
	neighbours_output_train.append(neighbour)



print "len(neighbours_output_train) is:"
print len(neighbours_output_train)





# Here the input syllables and the output that the network gave for that syllables are written to a file, with each input+output syllable on a separate line.
f = open("output_syllables_file_all_2.txt", "w")
for i in range((len(neighbours_output_train))-1):
	f.write(str(data_set_train[i]))
	f.write("\t"+str(neighbours_output_train[i])+"\n")
f.close()




# Here the average error of the output nodes is written to a text file, with every syllable on a new line. The transition to the test phase is denoted by an error of 0.5. (But because it averages over all the output nodes this error rate is only a very rough measure of the error.)
f = open("average_error_file.txt", "w")
count = 0
number = 2
for i in range((len(average_error_list))-2):
	if number % 3 == 0:
		f.write(str(count)+"\t"+str(average_error_list[number])+"\n")
		count += 1
	number += 1
number = 2
f.close()




