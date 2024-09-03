import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def coherence(data, vectors, stop_list=None, entropy=None, norm=True,
              f_weight='logfreq', winsize=20, comp_gp='all', gc_no_window=False, output_wins=False):

    print(data.shape[1])
    if not isinstance(data, pd.DataFrame):
        raise ValueError('Input data must be a DataFrame')
    if data.shape[1] < 4:
        raise ValueError('Not enough columns in input')
    if data.shape[1] > 6:
        raise ValueError('Too many columns in input')
    if data.shape[1] == 5:
        print(data.iloc[:, 4].max(),data.iloc[:, 4].min() )
        if data.iloc[:, 4].max() > 1 or data.iloc[:, 4].min() < 0:
            raise ValueError('Filter column should only contain 0 or 1')
    if not isinstance(vectors, (pd.DataFrame, np.ndarray)):
        raise ValueError('Word vectors in incorrect format')
    print( " vectors.shape",vectors.shape,len(vectors))
    if vectors.shape[0] != len(vectors):
        raise ValueError('Word vectors are not labelled')
    
    print("stop_list",type(stop_list))
    if stop_list is not None:
        stop_list = stop_list.iloc[:, 0].tolist()
        print("stop_list",stop_list)
    
    if entropy is not None:
        entropy = entropy['x'].to_numpy()
        print('entropy',entropy)
        # print(len(entropy),vectors.shape[0])
        if len(entropy) != vectors.shape[0]:
            raise ValueError('Number of entropy values does not match number of words in vectors')
    
    if f_weight not in ['logfreq', 'freq', 'none']:
        raise ValueError('Invalid frequency option')
    if winsize < 1:
        raise ValueError('Invalid window size')
    if comp_gp != 'all' and comp_gp != 'own':
        if comp_gp not in data.iloc[:, 1].unique():
            raise ValueError('Invalid comparison group')
    
    vlength = vectors.shape[1]
    print("vlength",vlength)
    d2 = data.copy()
    print(d2)

    if d2.shape[1] == 4:
        d2.columns = ['pt', 'gp', 'prompt', 'response']
        data.columns = ['pt', 'gp', 'prompt', 'response']
    else:
        d2.columns = ['pt', 'gp', 'prompt', 'response', 'filter']
        data.columns = ['pt', 'gp', 'prompt', 'response', 'filter']

    d2['gc'] = np.nan
    d2['lc'] = np.nan
    print(d2['gc'])
    if output_wins:
        d2['window'] = np.nan
    
    speech = data['response'].str.lower().str.replace('[^\w\s]', '').str.strip()

    new = [0]

    try:
    # Iterate through rows, starting from the second row (index 1 in Python)
        for j in range(1, len(data)):
            # Check if the current row's 'prompt' or 'pt' values differ from the previous row
            if data['prompt'][j] != data['prompt'][j-1] or data['pt'][j] != data['pt'][j-1]:
                new.append(j)
    except Exception as e:
        # Custom error handling
        raise Exception("More than one response is needed!") from e
    

    nnew= len(new)
    print('Calculating vectors for each response in dataset...')
    allresp = np.zeros((nnew, vlength))
    startdata = d2.iloc[new, :]

    for j in range(nnew):
        #collect response
        if j < nnew - 1:
            currresp = speech[new[j]:new[j+1]]
        else:
            currresp = speech[new[j]:]

        words = [word for word in currresp if word in vectors.index]
        if not isinstance(words, pd.Series):
             words = pd.Series(words)

        word_counts = words.value_counts()

        if f_weight == 'logfreq':
            words_matrix = np.log10(word_counts + 1).to_frame()
        elif f_weight == 'none':
            words_matrix = pd.DataFrame(1, index=word_counts.index, columns=['count'])
        else:
            words_matrix = word_counts.to_frame()
        
        words_matrix = words_matrix.sort_index()

        if stop_list is None:
            w2 = words_matrix
        else:
            w2 = words_matrix.loc[~words_matrix.index.isin(stop_list)]


        if norm:
            vnorm = np.sqrt((vectors.loc[w2.index] ** 2).sum(axis=1))
        else:
            vnorm = 1

        mask = vectors.index.isin(w2.index)

        if entropy is None:
            entro = 1
        else:
            entro = entropy[mask]
        

        # Reshape w2, entro, and vnorm to be (58, 1) so they can broadcast correctly with (58, 300)
        w2_reshaped = w2.values.reshape(-1, 1)
        entro_reshaped = entro.reshape(-1, 1)
        vnorm_reshaped = vnorm.values.reshape(-1, 1)
        
        # Now perform the element-wise multiplication and division
        cvec_values = vectors.loc[w2.index].values * w2_reshaped * entro_reshaped / vnorm_reshaped

        # Convert cvec_values back to a DataFrame with the same index and columns as the original vectors DataFrame
        cvec = pd.DataFrame(cvec_values, index=w2.index, columns=vectors.columns)

        # Check if cvec has only one row (i.e., is a single numeric vector)
        if cvec.shape[0] == 1:
            # Assign cvec directly to the corresponding row of allresp
            allresp[j, :] = cvec.values.flatten()  # Use flatten to ensure it's a 1D array
        else:
            # Calculate the column means and assign to the corresponding row of allresp
            allresp[j, :] = cvec.mean(axis=0).values

    for j in range(nnew):
        start = new[j]

        if data.shape[1] == 5:
            if data.iloc[start, 4] == 0:
                continue

        currpromptid = data.iloc[start, 2]
        currpt = data.iloc[start, 0]
        currgp = data.iloc[start, 1]



        if j < nnew - 1:
            currresp = speech[new[j]:new[j+1]]
        else:
            currresp = speech[new[j]:]

            # Get vector for the whole response
        wholevec = allresp[(startdata['prompt'] == currpromptid) & 
                        (startdata['pt'] == currpt) & 
                        (startdata['gp'] == currgp) & 
                        ~np.isnan(allresp[:, 0])]
        
        

        # Convert to numeric (flatten the array)
        wholevec = wholevec.flatten()

        # Generate a typical response by averaging other responses to the prompt
        if comp_gp == 'all':
            otherresp = np.nanmean(allresp[(startdata['prompt'] == currpromptid) & 
                                        ((startdata['gp'] != currgp) | (startdata['pt'] != currpt)) & 
                                        ~np.isnan(allresp[:, 0])], axis=0)
        elif comp_gp == 'own':
            otherresp = np.nanmean(allresp[(startdata['prompt'] == currpromptid) & 
                                        (startdata['pt'] != currpt) & 
                                        (startdata['gp'] == currgp) & 
                                        ~np.isnan(allresp[:, 0])], axis=0)
        else:
            otherresp = np.nanmean(allresp[(startdata['prompt'] == currpromptid) & 
                                        (startdata['pt'] != currpt) & 
                                        (startdata['gp'] == comp_gp) & 
                                        ~np.isnan(allresp[:, 0])], axis=0)

        # Use `wholevec` and `otherresp` as needed for further coherence calculations

        # Create a window for each word in the sample and compute the vector for each window
        nwords = len(currresp)

        # wins will contain each window to be analyzed
        wins = np.full((nwords, winsize), '', dtype=object)

        # winvec will contain the vectors
        winvec = np.zeros((nwords, 300))

        for i in range(nwords):
            # Compute start and end points for the window
            winstart = max(0, i - winsize + 1)  # In Python, index starts from 0
            winend = i + 1  # Range end is exclusive, so add 1 to include winend


            # Get the words that will form this window
            wins[i, :winend - winstart] = currresp[winstart:winend]


            # Now make the vector for the window
            # Filter for words in the vector list
            words = [word for word in wins[i, :] if word in vectors.index]

            # Convert to (log) frequency table
            word_counts = pd.Series(words).value_counts()
            if f_weight == 'logfreq':
                words_matrix = np.log10(word_counts + 1).to_frame()
                # print("words_matrix")
                # print(words_matrix) 
            else:
                words_matrix = word_counts.to_frame()
        

            if f_weight == 'none':
                words_matrix[:] = 1  # Set all weights to 1

            # Remove stop words, if required
            if stop_list is None:
                w2 = words_matrix
            else:
                w2 = words_matrix.loc[~words_matrix.index.isin(stop_list)]
            

            # Custom function to calculate sqrt(sum(x^2))
            def calculate_vector_norm(vec):
                return np.sqrt(np.sum(vec**2))

            if norm:
                if len(w2) == 1:
                    # Special case for one word: compute sqrt(sum(x^2)) for a single vector (flattened)
                    vnorm = calculate_vector_norm(vectors.loc[w2.index].values.flatten())
                    # print("in if")
                else:
                    # General case for multiple words: compute sqrt(sum(x^2)) for each row (word vector)
                    vnorm = vectors.loc[w2.index].apply(calculate_vector_norm, axis=1)
            else:
                vnorm = 1

            # Handling entropy
            if entropy is None:
                entro = 1
            else:
                # Extract entropy values corresponding to the words in w2
                entro = entropy[vectors.index.isin(w2.index)]


            try:
                # print("start")
                # print(vectors.loc[w2.index].shape)  
                # print(len(w2.values.flatten()))     
                # print(len(entro))                   
                # print("end")
                
                # Reshape w2.values.flatten() and entro to align with the dimensions of vectors
                w2_reshaped = w2.values.flatten().reshape(-1, 1)  
                entro_reshaped = entro.reshape(-1, 1)             
                vnorm_reshpaed = np.array(vnorm).reshape(-1,1)
                w2_array = np.nan_to_num(w2_reshaped, nan=0)
                entro_array = np.nan_to_num(entro_reshaped, nan=0)
                vnorm_array = np.nan_to_num(vnorm_reshpaed, nan=0)
                # Calculate cvec
                cvec = pd.DataFrame(vectors.loc[w2.index].values * w2_array * entro_array / vnorm_array,index=w2.index, columns=vectors.columns)

            except Exception as e:
                print(f"Error occurred: {e}")
                continue


            

            # Aggregate: Check if cvec is a single numeric vector
            if cvec.ndim == 1:  # equivalent to 'numeric' check in R
                print("in the if")
                winvec[i, :] = cvec
            else:
                print("in the else")
                # Compute the column means of cvec
                winvec[i, :] = cvec.mean(axis=0)

   

        print("winvec@@@@@@@@@@@@@@@@@@")
        print(winvec)
        winvec = np.nan_to_num(winvec, nan=0)

        # Calculate the cosine similarity between whole response vector and combined vector from other participants
        no_win_gc = cosine_similarity(otherresp.reshape(1, -1), wholevec.reshape(1, -1))[0, 0]
        

        for i in range(nwords):
            # General Coherence (GC): Compare participant's vector with combined vector from other participants
            if not gc_no_window:
                
                d2.loc[start + i, 'gc'] = cosine_similarity(otherresp.reshape(1, -1), winvec[i, :].reshape(1, -1))[0, 0]

            else:
                # Use the vector for the entire response
                d2.loc[start + i, 'gc'] = no_win_gc

            # Local Coherence (LC): Only compute if there's a previous window (of any size) available
            if i < winsize:
                d2.loc[start + i, 'lc'] = None  # No previous window available, set to None
            else:
                d2.loc[start + i, 'lc'] = cosine_similarity(winvec[i, :].reshape(1, -1), winvec[i - winsize, :].reshape(1, -1))[0, 0]  # Compare with previous window (no overlap)

            # Optionally, output the window content
            if output_wins:
                d2.loc[start + i, 'window'] = ' '.join(wins[i, :])  # Convert the window to a space-separated string

            print("d2")
            print(d2)

            print(f"{currpt} {currpromptid} complete")

    # Replace NaN values with None (Python's equivalent of R's NA) in 'gc' and 'lc' columns
    d2['gc'] = d2['gc'].replace({np.nan: None})
    d2['lc'] = d2['lc'].replace({np.nan: None})

    return d2





from test_function import coherence
import pandas as pd
import numpy as np
import openpyxl

# import pyreadr

if __name__ =='__main__':
    v = pd.read_csv('new_LSA.csv',index_col=0)
    # print(v)
    en = pd.read_csv('Hoffman_entropy_53758.csv')
    print("en")
    print(en)
    
    stop = pd.read_table('stoplist.txt',header=None)
    # print(stop)
    speech = pd.read_table('sample_data.txt')
    speech.columns = ['V1', 'V2', 'V3', 'V4', 'V5']
    # print(speech)
    
    result = coherence(speech ,vectors=v, entropy=en, stop_list=stop)
    # print(result)
    result.to_excel("finald2python11.xlsx")
