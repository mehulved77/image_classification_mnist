if __name__=="__main__":

import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
    
    from DeepModel_Option import x_train, y_train, x_test, y_test
    from DeepModel_Option import large_cnn_model, batch_size
    
    
# Fit the model
large_cnn_model.fit(x_train, y_train, 
          validation_data=(x_test, y_test), 
          epochs=10, batch_size = batch_size, verbose=2)

# Final evaluation of the model
large_cnn_scores = large_cnn_model.evaluate(x_test, y_test, verbose=0)
print("Large CNN Accuracy: {:.2f}%".format(large_cnn_scores[1]*100))
print("Large CNN Error: {:.2f}%".format(100-large_cnn_scores[1]*100))