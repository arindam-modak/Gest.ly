def guessGesture(model, img):
    global output, get_output
    #Load image and flatten it
    image = np.array(img).flatten()
    
    # reshape it
    image = image.reshape(img_channels, img_rows,img_cols)
    
    # float32
    image = image.astype('float32') 
    
    # normalize it
    image = image / 255
    
    # reshape for NN
    rimage = image.reshape(1, img_channels, img_rows, img_cols)
    
    # Now feed it to the NN, to fetch the predictions
    #index = model.predict_classes(rimage)
    #prob_array = model.predict_proba(rimage)
    
    prob_array = get_output([rimage, 0])[0]
    
    #print prob_array
    
    d = {}
    i = 0
    for items in output:
        d[items] = prob_array[0][i] * 100
        i += 1
    
    # Get the output with maximum probability
    import operator
    
    guess = max(d.iteritems(), key=operator.itemgetter(1))[0]
    prob  = d[guess]

    if prob > 70.0:
        #print guess + "  Probability: ", prob

        #Enable this to save the predictions in a json file,
        #Which can be read by plotter app to plot bar graph
        #dump to the JSON contents to the file
        
        with open('gesturejson.txt', 'w') as outfile:
            json.dump(d, outfile)

        return output.index(guess)

    else:
        return 1



def visualizeLayers(model, img, layerIndex):
    imlist = modlistdir('./imgs')
    if img <= len(imlist):
        
        image = np.array(Image.open('./imgs/' + imlist[img - 1]).convert('L')).flatten()
        
        ## Predict
        guessGesture(model,image)
        
        # reshape it
        image = image.reshape(img_channels, img_rows,img_cols)
        
        # float32
        image = image.astype('float32')
        
        # normalize it
        image = image / 255
        
        # reshape for NN
        input_image = image.reshape(1, img_channels, img_rows, img_cols)
    else:
        X_train, X_test, Y_train, Y_test = initializers()
        
        # the input image
        input_image = X_test[:img+1]
    
    
    
        
    # visualizing intermediate layers
    #output_layer = model.layers[layerIndex].output
    #output_fn = theano.function([model.layers[0].input], output_layer)
    #output_image = output_fn(input_image)
    
    if layerIndex >= 1:
        visualizeLayer(model,img,input_image, layerIndex)
    else:
        tlayers = len(model.layers[:])
        print "Total layers - {}".format(tlayers)
        for i in range(1,tlayers):
             visualizeLayer(model,img, input_image,i)



def visualizeLayer(model, img, input_image, layerIndex):

    layer = model.layers[layerIndex]
    
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
    activations = get_activations([input_image, 0])[0]
    output_image = activations
    
    
    ## If 4 dimensional then take the last dimension value as it would be no of filters
    if output_image.ndim == 4:
        # Rearrange dimension so we can plot the result
        o1 = np.rollaxis(output_image, 3, 1)
        output_image = np.rollaxis(o1, 3, 1)
        
        print "Dumping filter data of layer{} - {}".format(layerIndex,layer.__class__.__name__)
        filters = len(output_image[0,0,0,:])
        
        fig=plt.figure(figsize=(8,8))
        # This loop will plot the 32 filter data for the input image
        for i in range(filters):
            ax = fig.add_subplot(6, 6, i+1)
            #ax.imshow(output_image[img,:,:,i],interpolation='none' ) #to see the first filter
            ax.imshow(output_image[0,:,:,i],'gray')
            #ax.set_title("Feature map of layer#{} \ncalled '{}' \nof type {} ".format(layerIndex,
            #                layer.name,layer.__class__.__name__))
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
        plt.tight_layout()
        #plt.show()
        fig.savefig("img_" + str(img) + "_layer" + str(layerIndex)+"_"+layer.__class__.__name__+".png")
        #plt.close(fig)
    else:
        print "Can't dump data of this layer{}- {}".format(layerIndex, layer.__class__.__name__)
                     
