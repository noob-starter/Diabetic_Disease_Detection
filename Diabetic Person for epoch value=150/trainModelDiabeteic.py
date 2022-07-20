'''
The format of dataset for pima-indians-diabetics columns
   column - 1. Number of times pregnant
   column - 2. Plasma glucose concentration a 2 hours in an oral glucose tolerance test
   column - 3. Diastolic blood pressure (mm Hg)
   column - 4. Triceps skin fold thickness (mm)
   column - 5. 2-Hour serum insulin (mu U/ml)
   column - 6. Body mass index (weight in kg/(height in m)^2)
   column - 7. Diabetes pedigree function
   column - 8. Age (years)
   column - 9. Class variable (0 or 1)
'''

# import the necessary libraries 
# design neural network from scrach
from numpy import loadtxt # to handle the dataset csv file
from keras.models import Sequential  # create an empty stack of layers
from keras.layers import Dense  #specify the layer needed
from keras.models import model_from_json   #

dataset = loadtxt('newtextDiabetic.csv', delimiter=',')  #delimeter is ',' which is the field seperator   
x = dataset[:,0:8]  # to process the complete dataset for all column entries for the specific data
y = dataset[:,8]
# print(x)

# to initialise the model for data processing
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # to get the probability for given dataset
# model.summary()  # gives the complete summary of the dataset processing 

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x, y, epochs=150, batch_size=10) # line to train the model with dataset given
 

_, accuracy = model.evaluate(x, y)
print('Accuracy: %.2f' % (accuracy*100))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("Saved model to disk")
