#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      Marieke
#
# Created:     10-04-2013
# Copyright:   (c) Marieke 2013
# Licence:     <your licence>
#-------------------------------------------------------------------------------


from Gen_Marcus_data import marcus_training_set_ABA
from Gen_Marcus_data import marcus_training_set_ABB
from Gen_Marcus_data import marcus_testing_set_ABA
from Gen_Marcus_data import marcus_testing_set_ABB
from Gen_Marcus_data import elman_training_set
from Gen_Marcus_data import elman_testing_set

from random import uniform
from math import exp
from copy import deepcopy

	

#################################################################################
### These are the parameters that have to be set before training or testing!!

training_set = marcus_training_set_ABA

testing_set = marcus_testing_set_ABA

# Choose the number of sweeps through the training set:
sweeps_train = 2

# Choose the number of sweeps through the testing set:
sweeps_test = 2

# Choose the bias:
bias = 0.0

# Choose the number of input nodes:
input_nodes = 12

# Choose the number of hidden nodes (number of context nodes will automatically be the same):
hidden_nodes = 16

# Choose the number of encoding nodes:
encoding_nodes = 8

# Choose the number of output nodes
output_nodes = 12


#################################################################################

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



############################################################################################################
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
				synapses_in.append(-1)
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

	
#####################################################################################

# Here the initialization of the connection weights takes place: These functions create matrices that work as such: each row stands for the receiving node and each column stands for the node from which the connections originate.

def create_weights_in_hid():
	weights_list = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0.3201239109, -1.0000000000, 0.9816257954, 1.0000000000, -0.9795650244, 0.9242663980, -0.4709381759, -0.3361683786, 1.0000000000, -0.7616758347, -0.4368487597, 1.0000000000], [-0.1864833534, -0.9844089746, -0.1537842155, -0.9713746905, 1.0000000000, -0.7814315557, -0.2449337989, 0.2928942144, -0.9792978764, 1.0000000000, 0.2787336707, 0.7263681889], [0.1247749254, 1.0000000000, -1.0000000000, 0.0712726936, 0.6636894941, -1.0000000000, -0.4356665313, -0.1356395781, -1.0000000000, 0.9804747701, -0.0445416309, 0.4038036466], [-0.3789025545, -0.9984577298, 0.9928669333, 0.1226094589, 1.0000000000, 1.0000000000, 0.2672019601, 0.4691120088, -0.9905810356, -0.9815300107, -0.1787955016, -0.9815300107], [0.0752433836, 0.9853855371, -1.0000000000, -1.0000000000, 0.9767683148, 1.0000000000, 0.4872715771, -0.1754383296, 0.9850196838, -0.9966840744, -0.3262308240, -0.3777721524], [-0.4638966024, -0.9876509309, 0.9904330373, 0.9904330373, -0.9990361333, -1.0000000000, 0.5289747119, 0.5598896742, -1.0000000000, -0.5410800576, 0.9817698002, -0.9958940148], [0.4946287274, 0.1378426701, -0.9890754819, 0.9984149933, 0.9886850119, -0.9957085848, -0.9891760945, -0.2072864622, 1.0000000000, 0.9461448789, -0.7592747808, 0.9905343056], [-0.4997558594, 0.8023796082, 1.0000000000, 0.9925118685, -0.9986522198, -1.0000000000, -0.9658523202, -0.7666271925, 1.0000000000, 0.9688275456, -0.9762956500, 0.8134099245]]
	for vector in weights_list:
		vector.append(bias)
	return weights_list

weights_in_hid = create_weights_in_hid()

print "weights_in_hid are:"
print weights_in_hid

print "len(weights_in_hid) is:"
print len(weights_in_hid)
print "len(weights_in_hid[0]) is:"
print len(weights_in_hid[0])


def create_weights_hid_con():
	weights_list = [[0.0]*hidden_nodes for i in range(hidden_nodes)]
	for j in range(hidden_nodes):
		weights_list[j][j] = 1.0
	return weights_list
	
weights_hid_con = create_weights_hid_con()

print "weights_hid_con are:"
print weights_hid_con

print "len(weights_hid_con) is:"
print len(weights_hid_con)
print "len(weights_hid_con[0]) is:"
print len(weights_hid_con[0])
	
	
def create_weights_con_hid():
	weights_list = [[0.0]*hidden_nodes for i in range(hidden_nodes)]
	for j in range(hidden_nodes/2):
		for k in range((hidden_nodes/2), hidden_nodes):
			weights_list[k][k] = 1.0
	weights_list_first = weights_list[(hidden_nodes/2):hidden_nodes]
	weights_list_second = weights_list[0:(hidden_nodes/2)]
	weights_list_new = weights_list_first+weights_list_second
	for vector in weights_list_new:
		vector.append(bias)
	return	weights_list_new

weights_con_hid = create_weights_con_hid()

print "weights_con_hid are:"
print weights_con_hid

print "len(weights_con_hid) is:"
print len(weights_con_hid)
print "len(weights_con_hid[0]) is:"
print len(weights_con_hid[0])
print "len(weights_con_hid[13]) is:"
print len(weights_con_hid[(hidden_nodes/2)])
	

def create_weights_hid_enc():
	weights_list = [[0.0]*hidden_nodes for i in range(encoding_nodes)]
	for j in range(encoding_nodes):
		weights_list[j][j] = -1.0
		weights_list[j][j+(hidden_nodes/2)] = 1.0
	for vector in weights_list:
		vector.append(bias)
	return weights_list
	
weights_hid_enc = create_weights_hid_enc()

print "weights_hid_enc are:"
print weights_hid_enc

print "len(weights_hid_enc) is:"
print len(weights_hid_enc)
print "len(weights_hid_enc[0]) is:"
print len(weights_hid_enc[0])


def create_weights_enc_out():
	weights_list = [[-0.7562103868,	-1.0000000000,	-0.9949653149,	-1.0000000000,	-0.9850166440, -1.0000000000,	-1.0000000000,	-1.0000000000], [-1.0000000000, -1.0000000000, 0.9675956964, -0.8155366778, 0.9844078422, -0.7923265696, 0.9316826463, 0.9419313669], [1.0000000000, 0.1610096544, -0.9878969789, 1.0000000000, -0.9941763282, 0.9991294146, -0.9744834900, 0.9967204332], [0.9969515800, -0.9936702251, -0.0665935799, -0.1798463017, -0.9981735945, 0.9857314825, 0.9475134611, 0.9878994226], [-0.9598040581, 1.0000000000, 0.9876647592, 1.0000000000, 1.0000000000, -0.9603825808, 0.9982374310, -0.9539791346], [0.8138707876, -0.8779259920, -1.0000000000, 0.9747119546, 0.9718333483, -1.0000000000, -1.0000000000, -1.0000000000], [0.8040011525, 1.0000000000, 1.0000000000, 1.0000000000, 0.9597618580, 1.0000000000, 1.0000000000, 0.7661055326], [1.0000000000, 1.0000000000, 1.0000000000, 1.0000000000, 0.8827033043, 0.9258756638, 1.0000000000, 0.8309329748], [1.0000000000, -0.9539121985, -0.9723631144, -0.9725161195, 0.9548552632, -0.9569956064, 0.9605460763, 1.0000000000], [-0.9249798059, 1.0000000000, 1.0000000000, -0.9538833499, -0.9562190771, -0.5541669726, 1.0000000000, 1.0000000000], [0.6869496703, 1.0000000000, 1.0000000000, 1.0000000000, 0.9417747855, 1.0000000000, 1.0000000000, 1.0000000000], [1.0000000000, 0.7529093027, 1.0000000000, -0.9672870040, -0.3287633657, -0.9385118484, 1.0000000000, 1.0000000000]]
	for vector in weights_list:
		vector.append(bias)
	return weights_list


weights_enc_out = create_weights_enc_out()

print "weights_enc_out are:"
print weights_enc_out

print "len(weights_enc_out) is:"
print len(weights_enc_out)
print "len(weights_enc_out[0]) is:"
print len(weights_enc_out[0])




#################################################################################
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



class Hidden_Layer(Layer):
	def __init__(self, synapses_in, synapses_con, weights_in_hid, weigths_con_hid, weights_hid_enc):
		self.synapses_in = synapses_in
		self.synapses_con = synapses_con
		self.weights_in_hid = weights_in_hid
		self.weights_con_hid = weights_con_hid
		self.weights_hid_enc = weights_hid_enc		
	def calc_net(self):
		#This is the calculation for the input from the input layer:
		weighted_sum_list_in = []
		net_list_in = deepcopy(self.weights_in_hid)
		for i in range (len(net_list_in)):
			for j in range (len(net_list_in[i])):
				net_list_in[i][j] = net_list_in[i][j]*self.synapses_in[j]
		for i in range (len(net_list_in)):
			weighted_sum_in = sum(net_list_in[i]) 
			weighted_sum_list_in.append(weighted_sum_in)
		#This is the calculation for the input from the context layer:
		weighted_sum_list_con = []
		net_list_con = deepcopy(self.weights_con_hid)
		for i in range (len(net_list_con)):
			for j in range (len(net_list_con[i])):
				net_list_con[i][j] = net_list_con[i][j]*self.synapses_con[j]
		for i in range (len(net_list_con)):
			weighted_sum_con = sum(net_list_con[i]) 
			weighted_sum_list_con.append(weighted_sum_con)
		#Here the input of the input layer and context layer are put together:	
		weighted_sum_list = []
		for i in range(len(weighted_sum_list_in)):
			weighted_sum_tot = weighted_sum_list_in[i]+weighted_sum_list_con[i]
			weighted_sum_list.append(weighted_sum_tot)
		self.weighted_sum_list = weighted_sum_list
		return True
	def calc_error(self):
		self.calc_net()
		self.calc_activations()
		#First the error of the downstream nodes (output layer) are multiplied with the weights of the connections to those nodes:		
		downstream_error_matrix = deepcopy(self.weights_hid_enc)
		for i in range(len(encoding_error_list)):
			for j in range(len(downstream_error_matrix[i])):
				downstream_error_matrix[i][j] = downstream_error_matrix[i][j]*encoding_error_list[i]
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
		#First the weights for the input layer to the hidden layer are changed:	
		weights_list_in = deepcopy(self.weights_in_hid)
		for i in range(len(weights_list_in)):
			for j in range((len(weights_list_in[i]))-1):
				weights_list_in[i][j] = weights_list_in[i][j]+(eta*self.error_list[i]*self.synapses_in[j])
		self.weights_in_hid = weights_list_in	
		#Then the weights for the context layer to the hidden layer are changed:
		weights_list_con = deepcopy(self.weights_con_hid)
		for i in range(len(weights_list_con)):
			for j in range((len(weights_list_con[i]))-1):
				weights_list_con[i][j] = weights_list_con[i][j]+(eta*self.error_list[i]*self.synapses_con[j])
		self.weights_con_hid = weights_list_con
		return True
# This subclass has its own function 'calc_net()', because it gets input from two layers (the input layer and the context layer) instead of one.
# It also has its own function calc_error because the error is calculated differently for hidden nodes than for output nodes (because for hidden nodes the error is based on the error of the downstream output node, multiplied by the weight of the connections to this node).
# It also has its own function update_weight() because this has to be calculated separately for the in_hid weights and the con_hid weights.


class Context_Layer(Layer):
	def __init__(self, synapses_hid, weights_hid_con):
		self.synapses = synapses_hid
		self.weights = weights_hid_con
		self.calc_net()
		self.weighted_sum_list.append(-1)
		self.activations = self.weighted_sum_list 
# Since the context nodes simply copy the activations of the hidden nodes, without any sigmoid activation function coming in between, the activation of these nodes is simply the net activation, calculated as: sum(synapses*weights). To avoid confusion about the activation levels of the context nodes I made these into a separate subclass.


class Encoding_Layer(Layer):
	def __init__(self, synapses_hid, weights_hid_enc, weights_enc_out):
		self.synapses = synapses_hid
		self.weights = weights_hid_enc
		self.weights_enc_out = weights_enc_out
	def calc_error(self):
		self.calc_activations()
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
			for j in range((len(weights_list[i]))-1):
				weights_list[i][j] = weights_list[i][j]+(eta*self.error_list[i]*self.synapses[j])
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
			for j in range((len(weights_list[i])-1)):
				weights_list[i][j] = weights_list[i][j]+(eta*self.error_list[i]*self.synapses[j])
		self.weights = weights_list
		return True
# The calc_error function is defined under the scope of this subclass because the error calculation for the output node works different than that of the hidden nodes (i.e. it doesn't take downstream units into account because there aren't any).
# Then the update_weights() function is also defined under the scope of this subclass because it has to work with the specific error list for the output nodes.





##############################################################################
# Below the first input vector is initialized. 
# The initial activations of the context layer are initialized (this is 0.0 because the activations can vary between -1 and 1).
# Also the hidden layer is initialized here, taking the first input vector and the initial context layer activations as input. Later on in the while loop it is changed take the attribute '.activations' of the input layer and the context layer as input instead, so that these values will change with each iteration as they should.

init_input_vector = synapses_in_list[0]

init_synapses_con = [0.0]*hidden_nodes
init_synapses_con.append(-1)
# The value -1 is appended for the bias node

output_error_list = []

hidden_layer = Hidden_Layer(init_input_vector, init_synapses_con, weights_in_hid, weights_con_hid, weights_hid_enc)
	
	
hidden_layer.calc_activations()
hidden_activations = hidden_layer.activations
hidden_activations.append(-1)
# The value -1 is appended for the bias node



encoding_layer = Encoding_Layer(hidden_activations, weights_hid_enc, weights_enc_out)
	
	
encoding_layer.calc_activations()
encoding_activations = encoding_layer.activations
encoding_activations.append(-1)
# The value -1 is appended for the bias node



context_layer = Context_Layer(hidden_activations, weights_hid_con)

	
		
output_layer = Output_Layer(encoding_activations, weights_enc_out)



###############################################################################
# While loop that iterates over the training (or testing) examples. If the variable 'sweeps' is set at 1 times the length of the set it goes through the set once, if sweeps is set at 2 times the length of the set it goes through the set twice, etc. 
# All relevant values are printed. The activation of the output node and all errors are only given for every 3rd syllable, since this should be based on the entire sequence.



output_activations_matrix = []
output_error_matrix = []
target_list_train = []
output_activations_matrix_rounded = []

sweep = 0	
while sweep < sweeps_train:
	cycle = 0
	cycle_plus_one = 1	
	while cycle_plus_one < calc_length(training_set):
	

		print ""
		print "This is training cycle: "+str(cycle_plus_one)+" of sweep: "+str(sweep)
		print ""
		print ""
	
		print "The input vector is:"
		print synapses_in_list[cycle]

		target = synapses_in_list[cycle_plus_one][0:(len(synapses_in_list[cycle_plus_one]))-1]
	
		print "The target vector is:"
		print target

		target_list_train.append(target)

		input_layer = Input_Layer()


		hidden_layer.calc_activations()
#		print "hidden_layer.activations are:"
#		print hidden_layer.activations
	
		hidden_activations = hidden_layer.activations
		hidden_activations.append(-1)
#		print "hidden_activations is:"
#		print hidden_activations


		encoding_layer.calc_activations()
#		print "encoding_layer.activations are:"
#		print encoding_layer.activations
	
		encoding_activations = encoding_layer.activations
		encoding_activations.append(-1)
#		print "encoding_activations is:"
#		print encoding_activations



#		print "hidden_layer.weights_in_hid are:"
#		print hidden_layer.weights_in_hid
	
#		print "hidden_layer.weights_con_hid are:"
#		print hidden_layer.weights_con_hid
	
#		print "context_layer.weights are:"
#		print context_layer.weights
	
#		print "encoding_layer.weights are:"
#		print encoding_layer.weights	
	
#		print "output_layer.weights are:"
#		print output_layer.weights


#		print "input_layer.activations are:"
#		print input_layer.activations
	
#		print "hidden_activations are:"
#		print hidden_activations
	
#		print "context_layer.activations are:"
#		print context_layer.activations

#		print "encoding_layer.activations are:"
#		print encoding_layer.activations
	
		print "output_layer.activations are:"
		print output_layer.activations
	
			
		output_activations_matrix.append(output_layer.activations)
		

		output_activations_rounded = deepcopy(output_layer.activations)
		for node in output_activations_rounded:
			node = round(node)

		
		output_activations_matrix_rounded.append(output_activations_rounded)
		
		
		
		output_layer.calc_error()

		output_error_list = output_layer.error_list
	
#		print "output_error_list is:"
#		print output_error_list
		
		

	
		context_layer = Context_Layer(hidden_activations, weights_hid_con)
	
		hidden_layer = Hidden_Layer(input_layer.activations, context_layer.activations, hidden_layer.weights_in_hid, hidden_layer.weights_con_hid, encoding_layer.weights)


		encoding_layer = Encoding_Layer(hidden_activations, encoding_layer.weights, output_layer.weights)

		
		output_layer = Output_Layer(encoding_activations, output_layer.weights)
	
	
	
		external_error_list = []
		for i in range(len(output_layer.activations)):
			external_error = (target[i]-output_layer.activations[i])**2
			external_error_list.append(external_error)
		
		
		print "external_error_list is:"
		print external_error_list
				
		
		output_error_matrix.append(external_error_list)
		

				
		average_error_list = []
		for list in output_error_matrix:
			average_error = sum(list)/len(list)
			average_error_list.append(average_error)
			
			
			
		print ""
		print ""
		print ""
		
		cycle += 1
		cycle_plus_one += 1
		
	sweep += 1
	

	
##############################################################################	
##############################################################################
# This is the test loop!



synapses_in_list = calc_synapses_in(testing_set)


target_list_test = []
output_activations_test = []
output_activations_test_rounded = []
output_error_test = []

sweep = 0
while sweep < sweeps_test:
	cycle = 0
	cycle_plus_one = 1	
	while cycle_plus_one < len(synapses_in_list):
	
		print ""
		print "This is the testing loop!! Cycle:"+str(cycle_plus_one)+" of sweep: "+str(sweep)
		print ""
		print ""
	
		print "The input vector is:"
		print synapses_in_list[cycle]

		target = synapses_in_list[cycle_plus_one][0:(len(synapses_in_list[cycle_plus_one]))-1]
	
	
		print "The target vector is:"
		print target


		target_list_test.append(target)
		
		input_layer = Input_Layer()
		
		hidden_layer = Hidden_Layer(input_layer.activations, context_layer.activations, hidden_layer.weights_in_hid, hidden_layer.weights_con_hid, hidden_layer.weights_hid_enc)



		hidden_layer.calc_activations()
#		print "hidden_layer.activations are:"
#		print hidden_layer.activations
	
		hidden_activations = hidden_layer.activations
		hidden_activations.append(-1)
#		print "hidden_activations is:"
#		print hidden_activations


		context_layer = Context_Layer(hidden_activations, context_layer.weights)

		encoding_layer = Encoding_Layer(hidden_activations, encoding_layer.weights, encoding_layer.weights_enc_out)
		

		encoding_layer.calc_activations()
#		print "encoding_layer.activations are:"
#		print encoding_layer.activations
	
		encoding_activations = encoding_layer.activations
		encoding_activations.append(-1)
#		print "encoding_activations is:"
#		print encoding_activations


		output_layer = Output_Layer(encoding_activations, output_layer.weights)



#		print "hidden_layer.weights_in_hid are:"
#		print hidden_layer.weights_in_hid
	
#		print "hidden_layer.weights_con_hid are:"
#		print hidden_layer.weights_con_hid
	
#		print "context_layer.weights are:"
#		print context_layer.weights

#		print "encoding_layer.weights are:"
#		print encoding_layer.weights
	
#		print "output_layer.weights are:"
#		print output_layer.weights


#		print "input_layer.activations are:"
#		print input_layer.activations
	
#		print "hidden_activations are:"
#		print hidden_activations
		
#		print "context_layer.activations are:"
#		print context_layer.activations

#		print "encoding_layer.activations are:"
#		print encoding_layer.activations
	
		print "output_layer.activations are:"
		print output_layer.activations


		
		output_activations_test.append(output_layer.activations)
				

		output_activations_rounded = deepcopy(output_layer.activations)
		for node in output_activations_rounded:
			node = round(node)

		
		output_activations_test_rounded.append(output_activations_rounded)


		output_layer.calc_error()

		output_error_list = output_layer.error_list
	
#		print "output_error_list is:"
#		print output_error_list
	

		external_error_list = []
		for i in range(len(output_layer.activations)):
			external_error = (target[i]-output_layer.activations[i])**2
			external_error_list.append(external_error)
				
		
		print "external_error_list is:"
		print external_error_list

		
		output_error_test.append(external_error_list)

				
		average_error_test_list = []
		for list in output_error_matrix:
			average_error = sum(list)/len(list)
			average_error_test_list.append(average_error)
		
		cycle += 1
		cycle_plus_one += 1
		
		
	sweep += 1

	
	
print ""	
print "DONE"
print ""
print ""
		


###############################################################################
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




print "len(output_activations_matrix) is:"
print len(output_activations_matrix)


print "len(output_activations_test) is:"
print len(output_activations_test)




#neighbours_data_train = [['dummy']]
#length_training_set = calc_length(training_set)
#count = 1
#count_dummies = 0
#for target_vector in target_list_train:
#	neighbour = seek_neighbours(target_vector)
#	neighbours_data_train.append(neighbour)
#	if count > 1 and count % (length_training_set-count_dummies) == 0:
#		neighbours_data_train.append(['dummy'])
#		count_dummies += 1
#	count += 1

#neighbours_data_test = [['dummy']]
#length_testing_set = calc_length(testing_set)
#count = 1
#for target_vector in target_list_test:
#	neighbour = seek_neighbours(target_vector)
#	neighbours_data_test.append(neighbour)
#	if count > 1 and count % (length_testing_set-1) == 0:
#		neighbours_data_test.append(['dummy'])
#	count += 1




data_set_train = []
sweep = 0
while sweep < sweeps_train:
	for part_set in training_set:
		for sequence in part_set:
			for syllable in sequence:
				data_set_train.append(syllable)
	sweep += 1
				
				
data_set_test = []
sweep = 0			
while sweep < sweeps_test:
	for part_set in testing_set:
		for sequence in part_set:
			for syllable in sequence:
				data_set_test.append(syllable)
	sweep += 1



print "len(data_set_train)"
print len(data_set_train)

print "len(data_set_test)"
print len(data_set_test)

	
	
	
neighbours_output_train = ["start_of_sweep"]
count = 0
for i in range(len(output_activations_matrix)):
	if output_activations_matrix[i] == "end_of_sweep":
		neighbours_output_train.append("end_of_sweep"+str(count))
		neighbours_output_train.append("start_of_sweep")
		count += 1
	else:
		neighbour = seek_neighbours(output_activations_matrix[i])
		neighbours_output_train.append(neighbour)


neighbours_output_test = ["start_of_sweep"]
count = 0
for i in range(len(output_activations_test)):
	if output_activations_test[i] == "end_of_sweep":
		neighbours_output_test.append("end_of_sweep"+str(count))
		neighbours_output_test.append("start_of_sweep")
		count += 1
	else:
		neighbour = seek_neighbours(output_activations_test[i])
		neighbours_output_test.append(neighbour)
	


print "len(neighbours_output_train) is:"
print len(neighbours_output_train)


print "len(neighbours_output_test) is:"
print len(neighbours_output_test)
	


# Here the input syllables and the output that the network gave for that syllables are written to a file, with each input+output syllable on a separate line.
f = open("output_syllables_file_all_2.txt", "w")
for i in range((len(neighbours_output_train))-1):
	f.write(str(data_set_train[i]))
	f.write("\t"+str(neighbours_output_train[i])+"\n")
f.write("\n"+"Testing set:"+"\n"+"\n")
for i in range((len(neighbours_output_test))-1):
	f.write(str(data_set_test[i]))
	f.write("\t"+str(neighbours_output_test[i])+"\n")
f.close()


# Same as above, but now only the output at the third syllable of each sequence is written to the file.
f = open("output_syllables_file_2.txt", "w")
count = 1
for i in range((len(neighbours_output_train))-1):
	f.write(str(data_set_train[i]))
	if count % 3 == 0:
		f.write("\t"+str(neighbours_output_train[i])+"\n")
	else:
		f.write("\n")
	count += 1
f.write("\n"+"Testing set:"+"\n"+"\n")
count = 1
for i in range((len(neighbours_output_test))-1):
	f.write(str(data_set_test[i]))
	if count % 3 == 0:
		f.write("\t"+str(neighbours_output_test[i])+"\n")
	else:
		f.write("\n")
	count += 1
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
f.write(str(count)+"\t"+str(0.5)+"\n")
for i in range((len(average_error_test_list))-2):
	if number % 3 == 0:
		f.write(str(count)+"\t"+str(average_error_test_list[number])+"\n")
		count += 1
	number += 1
f.close()







