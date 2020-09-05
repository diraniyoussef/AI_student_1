def predictdesease( features, classifier, input_fn, Deseases ) :
    print("Please type numeric values as prompted.")

    study_case_features_values = {}

    for feature_name in features:
      notvalid = True
      while notvalid:
        val = input(feature_name + ": ")
        if not val.isdigit():
            notvalid = False
      study_case_features_values[feature_name] = [float(val)]

    predictions = classifier.predict(
        input_fn = lambda: input_fn( study_case_features_values ) )
    print( predictions )
    for pred_dict in predictions :
        print( pred_dict )
        class_id = pred_dict['class_ids'][0] #AFAICT usually class_ids is returning the one with the highest probability
        probability = pred_dict['probabilities'][class_id]
        print('Prediction is "{}" ({:.1f}%)'.format(
            Deseases[class_id], 100 * probability))

        # class_id1 = pred_dict['class_ids'][1]
        # probability = pred_dict['probabilities'][class_id1]
        # print('Prediction of second class id is "{}" ({:.1f}%)'.format(
        #     Deseases[class_id1], 100 * probability))

    return
