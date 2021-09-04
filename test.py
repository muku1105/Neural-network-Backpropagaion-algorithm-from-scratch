import numpy as np
x=np.array(([0,0],[0,1],[1,0],[1,1]),dtype= int)
y=np.array(([0],[1],[1],[0]),dtype=int)
x=x.flatten();
x = x[:, np.newaxis]
print(x.shape)
print(y)
class neuralnetwork():
    def __init__(self):
        self.input=2
        self.output=1
        self.hidden=3
        self.w1=np.random.randn(self.hidden,x.shape[0])
        
        self.w2=np.random.randn(self.output,self.hidden)
        print(self.w2.shape)
    def feedforward(self,x):
        self.z=np.dot(self.w1,x)
        print(self.z.shape);
        self.z2=self.sigmoid(self.z)
        self.z3=np.dot(self.w2,self.z)
        print(self.z3.shape);
        output=self.sigmoid(self.z3)
        return output
    
    def sigmoid(self,s, deriv=False):
        if (deriv==True):
            return s*(1-s)
        return 1/(1+np.exp(-s))
    def backprop(self,x,y,output):
        self.output_err= y - output
        self.output_delta= self.output_err* self.sigmoid(output, deriv=True)
        self.z2_err=self.output_delta.dot(self.w2.T)
        self.z2_delta=self.z2_err * self.sigmoid(self.z2, deriv=True)

        self.w1+=x.T.dot(self.z2_delta)
        self.w2+=self.z2.T.dot(self.output_delta)

    def train(self,x,y):
        output=self.feedforward(x)
        self.backprop(x,y,output)

nn=neuralnetwork()
for i in range(1):
    nn.train(x,y)
print("predicted:"+str(nn.feedforward(x)))
