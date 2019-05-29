import numpy as np
import random
import math

value_x = 5

EndFunction = np.sqrt(2*(value_x**2)+1)           #Function of problem
I_input = np.array([2.0])                   #Desired input
O_output= np.array([3.0])                   #Desired output
#Weight of input to hidden layer 1 
[wij1,wij2,wij3,wij4] = np.array([0.25,0.5,0.75,1.0])   
wij = [wij1,wij2,wij3,wij4]
#Weight of hidden layer 1 to hidden layer 2
[[wj1k1,wj1k2],[wj2k1,wj2k2],[wj3k1,wj3k2],[wj4k1,wj4k2]] = np.array([[1.0,0],[0.75,0.25],[0.5,0.5],[0.25,0.75]])
wjk = [[wj1k1,wj1k2],[wj2k1,wj2k2],[wj3k1,wj3k2],[wj4k1,wj4k2]] 
#Weight of hidden layer 2 to output layer
[[wk1o],[wk2o]] = np.array([[1.0],[0.5]])
wko = [[wk1o],[wk2o]]
#Bias of weight in input to hidden layer 1
[bij1,bij2,bij3,bij4] = np.array([1.0,1.0,1.0,1.0])
bij = [bij1,bij2,bij3,bij4]
#Bias of weight in hidden layer 1 to hidden layer 2
[bjk1,bjk2] = np.array([1.0,1.0])
bjk = [bjk1,bjk2]
#Bias of weight in output layer
bo = np.array([1.0])
#Finding value of input in hidden layer 1 based on its weight and bias
[j1in,j2in,j3in,j4in] = I_input*wij+bij
jin = [j1in,j2in,j3in,j4in]
print 'Hidden layer 1 input : ' , jin
#ReLU activation to find output in hidden layer 1
[j1out,j2out,j3out,j4out] = [max(0,j1in),max(0,j2in),max(0,j3in),max(0,j4in)] 
jout = np.array([j1out,j2out,j3out,j4out])
print 'Hidden layer 1 output: ' , jout

[k1in,k2in] = np.dot(jout,wjk)+bjk
kin = np.array([k1in,k2in])

print 'Hidden layer 2 input : ', kin
#Sigmoid function
kout = []
for k in kin:
    kout.append(1/(1+math.exp(-k)))
print 'Hidden layer 2 output: ', kout

Oin = np.dot(kout,wko )+ bo
print 'Output layer - input : ', Oin
#Linear activation
Oout = Oin
print 'Output layer - output: ', Oout

#Error of Prediction 
loss = 0.5 * (Oout-O_output)**2
print 'LOSS is : ', loss

#Back propagation start here
#Derivative of loss
Der_loss = Oout-O_output
print 'Derivative LOSS is : ', Der_loss
#Finding gradien to Oin
#It's a linear activation so
Grad_Oout_Oin = 1
print 'Gradien Oout for Oin: ',Grad_Oout_Oin
#Gradien Output in for bias out
Grad_Oin_bo = 1
print 'Gradien Output input for Bias Out: ',Grad_Oout_Oin
#Gradien weight of output layer to hidden layer 2
Grad_Oin_wko = kout
print 'Gradien Input in Output layer and Output in hidden layer 2: ',Grad_Oin_wko
#Gradient Loss for Weight Hidden Layer 2
Gradloss_wko = Der_loss*Grad_Oout_Oin*Grad_Oin_wko
print 'Gradient Loss for Weight Hidden Layer 2: ',Gradloss_wko
#Gradien loss for bias out
Gradloss_bo = Der_loss*Grad_Oout_Oin*Grad_Oin_bo
print 'Gradient Loss for bias out: ',Gradloss_bo

#Stochastic Gradient Descent (SGD) Update
#alpha = random.random()
alpha = 0.25 
wk1o = wk1o-alpha*Gradloss_wko[0]
wk2o = wk2o-alpha*Gradloss_wko[1]
new_wko = np.array([wk1o,wk2o])
new_bo = bo - alpha*Gradloss_bo

print 'New parameter after SGD'
print 'Weight on line hidden layer 2 to output layer: ',new_wko
print 'New Bias: ',bo
Gradloss_kout =[]
for i in range(len(wko)):
     Gradloss_kout.append(Gradloss_bo * Grad_Oout_Oin * kout[i] * wko[i])
     #print Gradloss_bo,Grad_Oout_Oin,kout[i], wko[i] >>>> Just for checking
print 'Gradient loss to hidden layer 2 output: ',Gradloss_kout

Grad_kout_kin =[]
for i in range(2):
    Grad_kout_kin.append((1/(1+math.exp(-kin[i])))*(1-(1/(1+math.exp(-kin[i])))))
print 'Gradien of Hidden layer 2 out to hidden layer 2 in: ',Grad_kout_kin
Grad_kin_wjk = [jout,jout]
print 'Dual jout is:', Grad_kin_wjk

Gradloss_wjk = []
for i in range(len(wjk)):
    x = 0
    y = 1
    Gradloss_wjk.append([(Gradloss_kout[x]*Grad_kout_kin[x]*jout[i]) , (Gradloss_kout[y]*Grad_kout_kin[y]*jout[i])])
    #Gradloss_wjk.append([1,0])
print 'Gradien Loss in weight of hidden layer 2 to hidden layer 1: ', Gradloss_wjk

Gradloss_bjk = []
for i in range(len(bjk)):
    Gradloss_bjk.append(Gradloss_kout[i]*Grad_kout_kin[i]*bjk[i])
print 'Gradien Loss in Bias of Hidden Layer 1 and Hidden Layer 2; ', Gradloss_bjk

new_wjk = []
for i in range(len(wjk)):
    new_wjk.append([wjk[i][0]-alpha*Gradloss_wjk[i][0] , wjk[i][1]-alpha*Gradloss_wjk[i][1]])
print 'NEW Gradien Loss in weight of hidden layer 2 to hidden layer 1', new_wjk

new_bjk = []
for i in range(len(bjk)):
    new_bjk.append(bjk[i]-alpha*Gradloss_bjk[i])
print 'NEW Gradien Loss in Bias of Hidden Layer 1 and Hidden Layer 2; ', new_bjk


Gradloss_jout = []
for i in range(len(jout)):
     a = (Gradloss_kout[0]+ Gradloss_kout[1])
     b = (Grad_kout_kin[0]+Grad_kout_kin[1])
     c = (Grad_kin_wjk[0][i]+Grad_kin_wjk[1][i])
     d = (wjk[i][0]+wjk[i][1])
     Gradloss_jout.append(a*b*c*d)
print 'Gradiel Loss for hidden layer 1 output session: ', Gradloss_jout

Grad_jout_jin = []
for i in jin:
     if i>0:
          Grad_jout_jin.append(1)
     else:
          Grad_jout_jin.append(0)
print 'Gradien of hidden layer 1 output and hidden layer 1 input: ', Grad_jout_jin

Grad_jin_wij = I_input
print 'Gradien of Hidden layer 1 input to it weight: ', Grad_jin_wij

Gradloss_wij = []
for i in range(len(Gradloss_jout)):
     Gradloss_wij.append(Gradloss_jout[i]*Grad_jout_jin[i]*Grad_jin_wij)
print 'Gradien loss of weight input to hidden layer 1: ', Gradloss_wij

Gradloss_bij = np.dot(Gradloss_wij,[0.5])
print 'Gradien loss of bias input to hidden layer 1: ', Gradloss_bij

new_wij = []
new_bij = []
for i in range(len(wij)):
     new_wij.append(wij[i]-alpha*Gradloss_wij[i])
     new_bij.append(bij[i]-alpha*Gradloss_bij[i])
print 'New Weight input to hidden layer 1: ',new_wij
print 'New Bias input to hidden layer 1: ',new_bij
print
print 
print '-----------------OLD------------------'
print 'Wij: ',wij
print 'Wjk: ',wjk
print 'Wko: ',wko
print 'Bij: ',bij
print 'Bjk: ',bjk
print 'Bo : ',bo
print '-----------------NEW------------------'
print 'Wij: ',new_wij
print 'Wjk: ',new_wjk
print 'Wko: ',new_wko
print 'Bij: ',new_bij
print 'Bjk: ',new_bjk
print 'Bo : ',new_bo
print '----------------\END/------------------'
print 'Used alpha: ', alpha