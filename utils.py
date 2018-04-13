import matplotlib.pyplot as plt

def print_expl(T,start = 0, stop = 500, data_loc = '/media/sebastian/7B4861FD6D0F6AA2/O/annots.h5', feature_loc = '/media/sebastian/7B4861FD6D0F6AA2/O/fbanks.h5', phoneme_loc = '/media/sebastian/MYLINUXLIVE/PhonAd/CGN_speech_recognition/Preprocessing/feature_table.txt'):  

    T = h5py.File(data_loc,'r')
    X = h5py.File(feature_loc,'r')
        
        
    k = [k for k in X.keys()][50]
    
    X[k].shape
    
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.title('X : FBanks')
    plt.imshow(X[k][start:stop,:].T, aspect = 'auto')
    
    plt.subplot(2,1,2)
    t = np.array(T[k][start:stop,:])
    plt.title('T : one-hot phonemes')
    plt.imshow(t.T, aspect = 'auto')
    plt.hold()
    
    phone_dict= {}
    
    
    phonemes = open(phoneme_loc,'r')
    
    for line in phonemes:
        entries = line.split('\t')
        if len(entries)==11:
            n=int(entries[0])
            for p in entries[10].split():
                if not n in phone_dict.keys():
                    phone_dict[n]=p
            
    
    start = 0
    try:
        val = 37
    except:
        val = 37
    for i in range(t.shape[0]):
        try:
            if not np.where(t[i,:])[0][0] == val:
                x = int(start)
                plt.text(x,val-1,phone_dict[val],color='white')
                start = i
                val=np.where(t[i,:])[0][0]
        except:
            if not (np.where(t[i,:]) or (np.where(t[i,:]) == val)):
                x = int(start)
                plt.text(x,val-1,phone_dict[val],color='white')
                start = i
                val=np.where(t[i,:])
                
    
    plt.show()

def visualise(T, phoneme_loc = '/media/sebastian/MYLINUXLIVE/PhonAd/CGN_speech_recognition/Preprocessing/feature_table.txt', model_loc = 'model_final'):
    import numpy as np
    import keras
    
    from vis.visualization import visualize_activation
    
    from vis.utils import utils
    from keras import activations
    from matplotlib import pyplot as plt
    
    phone_dict= {}
    
    
    phonemes = open(phoneme_loc,'r')
    
    for line in phonemes:
        entries = line.split('\t')
        if len(entries)==11:
            n=int(entries[0])
            for p in entries[10].split():
                if not n in phone_dict.keys():
                    phone_dict[n]=p
    
    #model = net.model
    #layer_idx = utils.find_layer_idx(model, 'preds')
    layer_idx = 11
    model = load_model(model_loc)
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)
    
    #for i in range(5):
    filter_idx = 0
    
    for j in range(36):
        img = visualize_activation(model, layer_idx, filter_indices=j)
        plt.subplot(6,6,j+1)
        plt.imshow(img.T,aspect = 'auto')
        plt.axis('off')
        plt.title(phone_dict[j])
    
    plt.suptitle('Activation maximisation for different phonemes')    
    plt.show()


def print_gen(T):
    from matplotlib import pyplot as plt
    cnt = 0
    
    for i in gen.__getitem__():
        Xt,Tt = i
        cnt += 1;
        if cnt == 2:
            break
    
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.imshow(Xt[i,:,:].T,aspect = 'auto')
        plt.title(str(np.where(Tt[i,:])[0]))
    plt.show()
