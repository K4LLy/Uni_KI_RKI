
#Lara Ahrens

def print_Data_every_bundesland_onehot_encoded(labels_pred, str_to_predict, regression, kalenderwoche):
    print(regression +' '+str_to_predict + ' in Kalenderwoche '+str(kalenderwoche) +' :' + '\n' +
          str_to_predict + ' in Baden-Wüttemberg: '       + str(labels_pred[0])+'\n' +
          str_to_predict + ' in Bayern: '                 + str(labels_pred[1])+'\n' +
          str_to_predict + ' in Berlin: '                 + str(labels_pred[2])+'\n' +
          str_to_predict + ' in Brandenburg: '            + str(labels_pred[3])+'\n' +
          str_to_predict + ' in Bremen: '                 + str(labels_pred[4])+'\n' +
          str_to_predict + ' in Hamburg: '                + str(labels_pred[5])+'\n' +
          str_to_predict + ' in Hessen: '                 + str(labels_pred[6])+'\n' +
          str_to_predict + ' in Mecklenburg-Vorpommern: ' + str(labels_pred[7])+'\n' +
          str_to_predict + ' in Niedersachsen: '          + str(labels_pred[8])+'\n' +
          str_to_predict + ' in Nordrhein-Westfahlen: '   + str(labels_pred[9])+'\n' +
          str_to_predict + ' in Rheinland-Pfalz: '        + str(labels_pred[10])+'\n' +
          str_to_predict + ' in Saarland: '               + str(labels_pred[11])+'\n' +
          str_to_predict + ' in Sachsen: '                + str(labels_pred[12])+'\n' +
          str_to_predict + ' in Sachsen-Anhalt: '         + str(labels_pred[13])+'\n' +
          str_to_predict + ' in Schleswig-Holstein: '     + str(labels_pred[14])+'\n' +
          str_to_predict + ' in Thüringen: '              + str(labels_pred[15])
          )

def print_Data_for_one_bundesland(labels_pred, str_to_predict, regression, kalenderwoche, bundesland):
    print(regression +' '+str_to_predict + ' in KW '+str(kalenderwoche) +' :' + '\n' +
          str_to_predict + ' in '  +bundesland  + ': '   + str(labels_pred[0])
          )
    
    
    
def print_prediction_multi_label(column_to_predict, kalenderwoche, labels_pred, str_to_predict):
    str_to_predict = str_to_predict +' in Kalenderwoche '+str(kalenderwoche)
    print(
    str_to_predict + ' in Baden-Wüttemberg: '       + str(labels_pred[0][0])+'\n' +
    str_to_predict + ' in Bayern: '                 + str(labels_pred[0][1])+'\n' +
    str_to_predict + ' in Berlin: '                 + str(labels_pred[0][2])+'\n' +
    str_to_predict + ' in Brandenburg: '            + str(labels_pred[0][3])+'\n' +
    str_to_predict + ' in Bremen: '                 + str(labels_pred[0][4])+'\n' +
    str_to_predict + ' in Hamburg: '                + str(labels_pred[0][5])+'\n' +
    str_to_predict + ' in Hessen: '                 + str(labels_pred[0][6])+'\n' +
    str_to_predict + ' in Mecklenburg-Vorpommern: ' + str(labels_pred[0][7])+'\n' +
    str_to_predict + ' in Niedersachsen: '          + str(labels_pred[0][8])+'\n' +
    str_to_predict + ' in Nordrhein-Westfahlen: '   + str(labels_pred[0][9])+'\n' +
    str_to_predict + ' in Rheinland-Pfalz: '        + str(labels_pred[0][10])+'\n' +
    str_to_predict + ' in Saarland: '               + str(labels_pred[0][11])+'\n' +
    str_to_predict + ' in Sachsen: '                + str(labels_pred[0][12])+'\n' +
    str_to_predict + ' in Sachsen-Anhalt: '         + str(labels_pred[0][13])+'\n' +
    str_to_predict + ' in Schleswig-Holstein: '     + str(labels_pred[0][14])+'\n' +
    str_to_predict + ' in Thüringen: '              + str(labels_pred[0][15]))    
    