import csv
import numpy as np
import matplotlib.pyplot as plt

#covid_data = pd.read_csv('RKI_COVID19.csv')
#bundesland_data = pd.read_csv('bundesland.csv')

def generate():
    with open('Data\\RKI_COVID19.csv') as csvdatei:
        csv_reader_object = csv.reader(csvdatei, delimiter=',')
        
        altersgruppe = ["A00-A04", "A05-A14", "A15-A34", "A35-A59", "A60-A79", "A80+"]
        status = ["krank", "gesund", "tot"]
        
        rkiDaten = []
        krank = np.array([0, 0, 0, 0, 0, 0])
        gesund = np.array([0, 0, 0, 0, 0, 0])
        tot = np.array([0, 0, 0, 0, 0, 0])
        
        
        for row in csv_reader_object:
            rkiDaten = row
            if rkiDaten[6] != '0' and rkiDaten[6] != '-9' and rkiDaten[4] == 'A00-A04':
                krank[0] = krank[0] + 1
            if rkiDaten[6] != '0' and rkiDaten[6] != '-9' and rkiDaten[4] == 'A05-A14':
                krank[1] = krank[1] + 1
            if rkiDaten[6] != '0' and rkiDaten[6] != '-9' and rkiDaten[4] == 'A15-A34':
                krank[2] = krank[2] + 1
            if rkiDaten[6] != '0' and rkiDaten[6] != '-9' and rkiDaten[4] == 'A35-A59':
                krank[3] = krank[3] + 1
            if rkiDaten[6] != '0' and rkiDaten[6] != '-9' and rkiDaten[4] == 'A60-A79':
                krank[4] = krank[4] + 1    
            if rkiDaten[6] != '0' and rkiDaten[6] != '-9' and rkiDaten[4] == 'A80+':
                krank[5] = krank[5] + 1  
            
            if rkiDaten[16] != '0' and rkiDaten[16] != '-9' and rkiDaten[4] == 'A00-A04':
                gesund[0] = gesund[0] + 1
            if rkiDaten[16] != '0' and rkiDaten[16] != '-9' and rkiDaten[4] == 'A05-A14':
                gesund[1] = gesund[1] + 1
            if rkiDaten[16] != '0' and rkiDaten[16] != '-9' and rkiDaten[4] == 'A15-A34':
                gesund[2] = gesund[2] + 1
            if rkiDaten[16] != '0' and rkiDaten[16] != '-9' and rkiDaten[4] == 'A35-A59':
                gesund[3] = gesund[3] + 1
            if rkiDaten[16] != '0' and rkiDaten[16] != '-9' and rkiDaten[4] == 'A60-A79':
                gesund[4] = gesund[4] + 1    
            if rkiDaten[16] != '0' and rkiDaten[16] != '-9' and rkiDaten[4] == 'A80+':
                gesund[5] = gesund[5] + 1  
    
            if rkiDaten[7] != '0' and rkiDaten[7] != '-9' and rkiDaten[4] == 'A00-A04':
                tot[0] = tot[0] + 1
            if rkiDaten[7] != '0' and rkiDaten[7] != '-9' and rkiDaten[4] == 'A05-A14':
                tot[1] = tot[1] + 1
            if rkiDaten[7] != '0' and rkiDaten[7] != '-9' and rkiDaten[4] == 'A15-A34':
                tot[2] = tot[2] + 1
            if rkiDaten[7] != '0' and rkiDaten[7] != '-9' and rkiDaten[4] == 'A35-A59':
                tot[3] = tot[3] + 1
            if rkiDaten[7] != '0' and rkiDaten[7] != '-9' and rkiDaten[4] == 'A60-A79':
                tot[4] = tot[4] + 1    
            if rkiDaten[7] != '0' and rkiDaten[7] != '-9' and rkiDaten[4] == 'A80+':
                tot[5] = tot[5] + 1  
    
    covid_alter_abhaengigkeit = np.array([[krank[0], gesund[0], tot[0]],
                                          [krank[1], gesund[1], tot[1]],
                                          [krank[2], gesund[2], tot[2]],
                                          [krank[3], gesund[3], tot[3]],
                                          [krank[4], gesund[4], tot[4]],
                                          [krank[5], gesund[5], tot[5]]])
    
    fig, ax = plt.subplots()
    im = ax.imshow(covid_alter_abhaengigkeit)
    
    #Erstellen der Achsen + Beschriftung und Rotation dessen
    ax.set_xticks(np.arange(len(status)))
    ax.set_yticks(np.arange(len(altersgruppe)))
    
    ax.set_xticklabels(status)
    ax.set_yticklabels(altersgruppe)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    #Zellenbeschriftung
    for i in range(len(altersgruppe)):
        for j in range(len(status)):
            text = ax.text(j, i, covid_alter_abhaengigkeit[i, j], ha="center", va="center", color="w")
    
    fig.tight_layout()
    plt.show()


