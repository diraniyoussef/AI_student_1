import numpy as np

def change( dataFrame, oldValue, newValue, columnName = None) : # dataFrame here is sort of passed by reference so the changes take effect in the calling module.
    if columnName is None :
        column = dataFrame # it's a 1 column without a name at first. I.e. it simply starts with values. I made it if I wanted to change e.g. the value of train_y or test_y
    else :
        column = dataFrame[ columnName ] # it's like an array or table of multiple columns. Each column has a name at first then its values.
    index_where_head_ache_has_a_special_value = np.where( column == oldValue )
    column.loc[ index_where_head_ache_has_a_special_value ] = newValue
