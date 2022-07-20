from numpy import loadtxt
from keras.models import model_from_json
dataset = loadtxt('D:\\Study\\AI internship\\Day_10\\Diabetic Person for epoch value=300\\newtextDiabetic.csv', delimiter=',')
x = dataset[:,0:8]
y = dataset[:,8]

json_file = open('D:\\Study\\AI internship\\Day_10\\Diabetic Person for epoch value=300\\model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("D:\\Study\\AI internship\\Day_10\\Diabetic Person for epoch value=300\\model.h5")
print("Loaded model from disk")

predictions = model.predict_classes(x)
for i in range(5,10):
	print('%s => %d (expected %d)' % (x[i].tolist(), predictions[i], y[i]))
