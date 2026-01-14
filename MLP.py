import numpy as np
import pandas as pd
import time

# location of traning data => C:/Users/twilson763/OneDrive - Glow Scotland/projects/python/percep/ai learns to read training data.csv
# location on mac: /Users/toby/Library/CloudStorage/OneDrive-GlowScotland/projects/python/percep/ai learns to read training data.csv

np.random.seed(100)

class MLP():
    def __init__(self,start_layer: int,end_layer: int,location: str,hidden_layers: int, layer_sizes: np.ndarray):
        self.total = 0
        self.location = location
        self.layers: int = hidden_layers
        self.neurons: np.ndarray = [[0]]*(hidden_layers+1)
        self.num_neurons: np.ndarray = [start_layer]*(hidden_layers+1)
        if self.location != 'n':
            dO = pd.read_csv(location)
            self.d = dO.drop("label", axis=1)
            self.l = dO["label"]
            self.I: np.ndarray = [0]*end_layer
        
        self.hidden_layers: np.ndarray[np.ndarray] = [[]]*hidden_layers
        self.biases: np.ndarray[np.ndarray] = [[]]*hidden_layers
        for i in range(hidden_layers):
            if i == 0:
                n = start_layer
            else:
                n = m
            if i+1 == hidden_layers:
                m = end_layer
            else:
                #m = int(input('enter neurons in hidden layer ' + str(i+1) + ': '))
                m = layer_sizes[i] # alowes for the net shape to be given at net decleration, must give an array of size = hidden_layers - 1
            self.hidden_layers[i] = np.multiply(np.random.rand(m,n),0.5)-0.25
            self.biases[i] = np.multiply(np.random.rand(m),5)-2.5
            #if i == 0 or i == 1: # do not do this again it completely wrecks the nets eval capability
            #    self.hidden_layers[i] = np.ones((m,n))
            #    self.biases[i] = np.random.rand(m)*100 - 200
            self.neurons[i+1]: np.ndarray = [0 for i in range(m)]
            self.num_neurons[i+1]: int = m
            self.total += m

        self.velocity_array: np.ndarray[np.ndarray] = [np.zeros((self.num_neurons[i+1],self.num_neurons[i]+1)) for i in range(self.layers)]
        self.momentum_array: np.ndarray[np.ndarray] = [np.zeros((self.num_neurons[i+1],self.num_neurons[i]+1)) for i in range(self.layers)]
        self.loss = np.array([0.0])

        self.back_prop_range = range(self.layers-1,-1,-1)

        #self.GELU_const = np.sqrt(8/np.pi) to three sig fig is just 1.6

    def read_matrix(self,n: int,location: str):
        self.hidden_layers[n] = pd.read_csv(location, header=None, dtype=np.float64).to_numpy()

    def read_bias(self,n: int,location: str):
        self.biases[n] = pd.read_csv(location, header=None, dtype=np.float64).to_numpy().squeeze()
        if self.num_neurons[n+1] == 1:
            self.biases[n] = np.array([self.biases[n]])

    def read_input(self,idx: int,location: str):
        if location == 'n':
            self.input = self.d.iloc[idx]
        else:
            self.input = pd.read_csv(location, header=None, dtype=np.float64).to_numpy().squeeze()
    
    def wright_matrix(self,n: int,location: str):
        pd.DataFrame(self.hidden_layers[n]).to_csv(location, index=False, header=False)

    def wright_bias(self,n: int,location: str):
        pd.DataFrame(self.biases[n]).to_csv(location, index=False, header=False)

    def ReLU(self,a: np.ndarray,layer):
        if layer != self.layers - 1:
            return 1*a*(a>0) + 0.125*a*(a<0)
        else:
            return 1*a*(a>0) + 0.125*a*(a<0) #1*a

    def dReLU(self,a: np.ndarray,layer):
        if layer != self.layers - 1:
            return 1*(a>0) + 0.125*(a<0)
        else:
            return 1*(a>0) + 0.125*(a<0) #np.array([1.0]*len(a))

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-np.array(x, dtype=np.float64)))
    
    def dsigmoid(self,x):
        x = self.sigmoid(x)
        x = x*(1-x) 
        return x

    def GELU(self,x):# aproximation due to gausian
        return x*(self.sigmoid(1.6*x) + 0.5) # actual is x*self.sigmoid(1.6*x)
    
    def dGELU(self,x):# also aproximation but is the derivative of the above aproximation function
        x = 1.6*x
        y = self.sigmoid(x)
        return y*(x*(1 - y) + 1) + 0.5

    def softmax(self,x: np.ndarray):
        T = 0.1
        x = x*T
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x, axis=0)

    def derive(self,v,M,z,n):
        O = np.zeros(n)
        for k in range(n):
            O[k] = np.sum(np.prod(np.array([v,np.transpose(M)[k],self.dGELU(z)]),axis=0))
        return O

    def propigate(self,v_in: np.ndarray):# not been updated for GELU
        v_out: np.ndarray = v_in
        count = 0
        for matrix in self.hidden_layers:
            self.neurons[count]: np.ndarray = v_out
            v_out: np.ndarray = self.ReLU(np.add(np.dot(matrix,v_out),self.biases[count]),count)
            count +=1

        v_out = self.softmax(v_out)
        self.neurons[count]: np.ndarray = v_out
        return v_out

    def propigate_withought_softmax(self,v_in: np.ndarray):
        v_out: np.ndarray = np.array(v_in)
        self.neurons[0]: np.ndarray = np.array(v_out)
        for i in range(self.layers):
            self.neurons[i+1]: np.ndarray = np.add(np.dot(self.hidden_layers[i],v_out),self.biases[i])# neurons now store values before activation function rather than after for faster & simpler backprop as it means that derive doesnt need to be changed
            v_out: np.ndarray = self.GELU(self.neurons[i+1])

        return v_out
    
    def propigate_with_sigmoid(self,v_in: np.ndarray):
        v_out: np.ndarray = np.array(v_in)
        self.neurons[0]: np.ndarray = np.array(v_out)
        for i in range(self.layers):
            self.neurons[i+1]: np.ndarray = np.add(np.dot(self.hidden_layers[i],v_out),self.biases[i])# neurons now store values before activation function rather than after for faster & simpler backprop as it means that derive doesnt need to be changed
            if i+1 == self.layers:
                v_out: np.ndarray = self.sigmoid(self.neurons[i+1])
            else:
                v_out: np.ndarray = self.GELU(self.neurons[i+1])
        
        return v_out

    def back_propigate_epochs(self,epochs,len_epoch, start_idx):# not been updated for GELU
        # not been updated for layer dependent ReLU
        change_epoch = 1/len_epoch
        idx = start_idx
        correct = 0
        counter = 0
        for k in range(epochs):
            change_array = [np.zeros((self.num_neurons[i+1],self.num_neurons[i]+1)) for i in range(self.layers)]

            for j in range(len_epoch):
                self.read_input(idx,'n')
                if idx > 0:
                    self.I[self.l[idx-1]] = 0
                self.I[self.l[idx]] = -1

                self.output = self.propigate(self.input)

                ans = np.argmax(self.output)
                if ans == self.l[idx] and np.max(self.output) != ans:
                    correct += 1

                O = self.output + self.I
                self.cost = np.dot(O,O)

                derivative_vector = 2*O
                print(derivative_vector)
                for i in range(self.layers-1,-1,-1):
                    new_derivative_vector = self.derive(derivative_vector,self.hidden_layers[i],self.neurons[i+1],self.num_neurons[i])

                    O1 = np.stack([np.prod(np.transpose(np.array([derivative_vector,self.dReLU(self.neurons[i+1])])),axis=1) for _ in range(self.num_neurons[i])], axis=0)

                    O2 = np.stack([self.neurons[i] for _ in range(self.num_neurons[i+1])], axis=0)

                    change_array[i] += np.hstack([np.transpose(np.prod(np.array([O1,np.transpose(O2)]),axis = 0)),np.prod(np.transpose(np.array([derivative_vector,self.dReLU(self.neurons[i+1])])),axis=1)[:,None]])

                    derivative_vector = new_derivative_vector

                idx += 1
                counter += 1

            for i in range(self.layers):
                change = change_array[i]
                self.hidden_layers[i] -= change[::,:self.num_neurons[i]]*change_epoch
                self.biases[i] -= np.transpose(change[::,self.num_neurons[i]:])[0]*change_epoch
            
            if counter >= 100:
                print(self.cost)
                print()
                print(str(correct) + '/' + str(counter))
                print(ans)
                print(self.l[idx-1])
                print(idx)
                print('-------------------------------------------------------------')
                correct = 0
                counter = 0

    def back_propigate_once(self, Input, Output):# not been updated for GELU
        change_array = [np.zeros((self.num_neurons[i+1],self.num_neurons[i]+1)) for i in range(self.layers)]

        self.input = np.array(Input)
                
        self.I = -1*Output

        self.output = self.propigate_withought_softmax(self.input)

        O = self.output + self.I
        #self.cost = np.dot(O,O)

        derivative_vector = 2*O
        for i in range(self.layers-1,-1,-1):
            new_derivative_vector = self.derive(derivative_vector,self.hidden_layers[i],self.neurons[i+1],self.num_neurons[i],i)

            O1 = np.stack([np.prod(np.transpose(np.array([derivative_vector,self.dReLU(self.neurons[i+1],i)])),axis=1) for _ in range(self.num_neurons[i])], axis=0)

            O2 = np.stack([self.neurons[i] for _ in range(self.num_neurons[i+1])], axis=0)

            change_array[i] += np.hstack([np.transpose(np.prod(np.array([O1,np.transpose(O2)]),axis = 0)),np.prod(np.transpose(np.array([derivative_vector,self.dReLU(self.neurons[i+1],i)])),axis=1)[:,None]])

            derivative_vector = new_derivative_vector


        for i in range(self.layers):
            change = 0.00005*change_array[i]
            self.hidden_layers[i] -= change[::,:self.num_neurons[i]]
            self.biases[i] -= np.transpose(change[::,self.num_neurons[i]:])[0]

    def back_propigate_once_cross_entropy(self, Input, Output):# not been updated for GELU
        change_array = [np.zeros((self.num_neurons[i+1],self.num_neurons[i]+1)) for i in range(self.layers)]

        self.input = np.array(Input)
                
        self.I = Output

        self.output = self.propigate(self.input)

        # use cross entropy loss with softmax this can help i think and will be better than (o - i)^2
        # assume Output sums to 1 and has no negative values

        #O = np.prod(np.array([self.I,np.log(self.output)]),axis = 0)

        #self.cost = -1*np.sum(O,axis=0)

        # derivative vector is just the derivative of cost with regards to the output layer when its defined initaly though changes after that
        # should note that its the output layer before softmax not after as the rest of the code can acount for that
        derivative_vector = self.output - Output

        for i in range(self.layers-1,-1,-1):
            new_derivative_vector = self.derive(derivative_vector,self.hidden_layers[i],self.neurons[i+1],self.num_neurons[i],i)

            O1 = np.stack([np.prod(np.transpose(np.array([derivative_vector,self.dReLU(self.neurons[i+1],i)])),axis=1) for _ in range(self.num_neurons[i])], axis=0)

            O2 = np.stack([self.neurons[i] for _ in range(self.num_neurons[i+1])], axis=0)

            change_array[i] += np.hstack([np.transpose(np.prod(np.array([O1,np.transpose(O2)]),axis = 0)),np.prod(np.transpose(np.array([derivative_vector,self.dReLU(self.neurons[i+1],i)])),axis=1)[:,None]])

            derivative_vector = new_derivative_vector


        for i in range(self.layers):
            change = change_array[i]
            self.hidden_layers[i] -= change[::,:self.num_neurons[i]]*0.2
            self.biases[i] -= np.transpose(change[::,self.num_neurons[i]:])[0]*0.2

    def back_propigate_once_cross_entropy_RMSprop(self, Input, Output):# not been updated for GELU
        change_array = [np.zeros((self.num_neurons[i+1],self.num_neurons[i]+1)) for i in range(self.layers)]

        beta = 0.9
        alpha = 0.1

        self.input = np.array(Input)
                
        self.I = Output

        self.output = self.propigate(self.input)

        # assume Output sums to 1 and has no negative values

        #O = np.prod(np.array([self.I,np.log(self.output)]),axis = 0)

        #self.cost = -1*np.sum(O,axis=0)

        # derivative vector is just the derivative of cost with regards to the output layer when its defined initaly though changes after that
        # should note that its the output layer before softmax not after as the rest of the code can acount for that
        derivative_vector = self.output - Output

        for i in range(self.layers-1,-1,-1):
            new_derivative_vector = self.derive(derivative_vector,self.hidden_layers[i],self.neurons[i+1],self.num_neurons[i],i)

            O1 = np.stack([np.prod(np.transpose(np.array([derivative_vector,self.dReLU(self.neurons[i+1],i)])),axis=1) for _ in range(self.num_neurons[i])], axis=0)

            O2 = np.stack([self.neurons[i] for _ in range(self.num_neurons[i+1])], axis=0)

            change_array[i] += np.hstack([np.transpose(np.prod(np.array([O1,np.transpose(O2)]),axis = 0)),np.prod(np.transpose(np.array([derivative_vector,self.dReLU(self.neurons[i+1],i)])),axis=1)[:,None]])

            derivative_vector = new_derivative_vector


        for i in range(self.layers):
            self.velocity_array[i] = np.array(self.velocity_array[i])*beta + (1-beta)*np.prod(np.array([change_array[i],change_array[i]]),axis = 0)

            change = alpha*change_array[i]/(np.sqrt(self.velocity_array[i]) + 0.000000001)

            self.hidden_layers[i] -= change[::,:self.num_neurons[i]]
            self.biases[i] -= np.transpose(change[::,self.num_neurons[i]:])[0]

    def back_propigate_once_root_mean_squared_RMSprop(self, Input, Output):# not been updated for GELU
        change_array = [np.zeros((self.num_neurons[i+1],self.num_neurons[i]+1)) for i in range(self.layers)]

        beta = 0.9
        alpha = 0.08

        self.input = np.array(Input)
        self.output = self.propigate_withought_softmax(self.input)

        O = self.output - Output
        self.loss = np.concatenate([self.loss,[np.log10(np.dot(O,O))]],axis=0)
        derivative_vector = 2*O

        for i in range(self.layers-1,-1,-1):
            new_derivative_vector = self.derive(derivative_vector,self.hidden_layers[i],self.neurons[i+1],self.num_neurons[i],i)

            O1 = np.stack([np.prod(np.transpose(np.array([derivative_vector,self.dReLU(self.neurons[i+1],i)])),axis=1) for _ in range(self.num_neurons[i])], axis=0)

            O2 = np.stack([self.neurons[i] for _ in range(self.num_neurons[i+1])], axis=0)

            change_array[i] += np.hstack([np.transpose(np.prod(np.array([O1,np.transpose(O2)]),axis = 0)),np.prod(np.transpose(np.array([derivative_vector,self.dReLU(self.neurons[i+1],i)])),axis=1)[:,None]])

            derivative_vector = new_derivative_vector

        for i in range(self.layers):
            self.velocity_array[i] = np.array(self.velocity_array[i])*beta + (1-beta)*np.prod(np.array([change_array[i],change_array[i]]),axis = 0)

            change = alpha*change_array[i]/(np.sqrt(self.velocity_array[i]) + 0.000000001)

            self.hidden_layers[i] -= change[::,:self.num_neurons[i]]
            self.biases[i] -= np.transpose(change[::,self.num_neurons[i]:])[0]

    def back_propigate_once_cross_entropy_Adam(self, Input, Output):# not been updated for GELU
        change_array = [np.zeros((self.num_neurons[i+1],self.num_neurons[i]+1)) for i in range(self.layers)]

        beta_1 = 0.8
        beta_2 = 0.9
        alpha = 0.08

        self.input = np.array(Input)
                
        self.I = Output

        self.output = self.propigate(self.input)

        derivative_vector = self.output - Output

        for i in range(self.layers-1,-1,-1):
            new_derivative_vector = self.derive(derivative_vector,self.hidden_layers[i],self.neurons[i+1],self.num_neurons[i],i)

            O1 = np.stack([np.prod(np.transpose(np.array([derivative_vector,self.dReLU(self.neurons[i+1],i)])),axis=1) for _ in range(self.num_neurons[i])], axis=0)

            O2 = np.stack([self.neurons[i] for _ in range(self.num_neurons[i+1])], axis=0)

            change_array[i] += np.hstack([np.transpose(np.prod(np.array([O1,np.transpose(O2)]),axis = 0)),np.prod(np.transpose(np.array([derivative_vector,self.dReLU(self.neurons[i+1],i)])),axis=1)[:,None]])

            derivative_vector = new_derivative_vector

        for i in range(self.layers):
            self.momentum_array[i] = np.array(self.momentum_array[i])*beta_1 + (1-beta_1)*np.array(change_array[i])

            self.velocity_array[i] = np.array(self.velocity_array[i])*beta_2 + (1-beta_2)*np.prod(np.array([change_array[i],change_array[i]]),axis = 0)

            change = alpha*self.momentum_array[i]/(np.sqrt(self.velocity_array[i]) + 0.000000001)

            self.hidden_layers[i] -= change[::,:self.num_neurons[i]]
            self.biases[i] -= np.transpose(change[::,self.num_neurons[i]:])[0]

    def back_propigate_once_root_mean_squared_Adam(self, Input, Output):
        # changes have been made in this one to make the code faster
        # seems to be an issue where the net con only output one value when all its inputs are 0
        #start = time.time()

        beta_1 = 0.85
        beta_2 = 0.995
        alpha = 0.001

        self.input = np.array(Input)
        self.output = self.propigate_withought_softmax(self.input)

        O = self.output - Output
        self.loss = np.concatenate([self.loss,[np.log10(np.dot(O,O))]],axis=0)
        derivative_vector = 2*O


        for i in self.back_prop_range:
            new_derivative_vector = self.derive(derivative_vector,self.hidden_layers[i],self.neurons[i+1],self.num_neurons[i])

            O1 = np.stack([np.prod(np.array([derivative_vector,self.dGELU(self.neurons[i+1])]),axis=0) for _ in range(self.num_neurons[i])], axis=0)

            O2 = np.stack([self.GELU(self.neurons[i]) for _ in range(self.num_neurons[i+1])], axis=0)

            change_array = np.hstack([np.transpose(np.prod(np.array([O1,np.transpose(O2)]),axis = 0)),O1[0][:,None]]) # the [:,None] slicing is to give the arrays the same shape [0,0,0] -> [[0],[0],[0]]

            self.momentum_array[i] = self.momentum_array[i]*beta_1 + (1-beta_1)*change_array

            self.velocity_array[i] = self.velocity_array[i]*beta_2 + (1-beta_2)*(change_array**2)

            change = alpha*self.momentum_array[i]/(np.sqrt(np.array(self.velocity_array[i], dtype = np.float64)) + 0.00000001)

            self.hidden_layers[i] -= np.array(change[::,:self.num_neurons[i]], dtype = np.float64)
            self.biases[i] -= np.array(change[::,self.num_neurons[i]], dtype = np.float64)

            derivative_vector = new_derivative_vector

        #end = time.time()

        #print()
        #print(end - start)
