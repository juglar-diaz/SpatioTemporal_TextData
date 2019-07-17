import pandas as pd
import numpy as np
from collections import Counter
from sklearn.cluster import MeanShift
from sklearn.neighbors import NearestNeighbors

data = ""



def buildIndexData(list_elements, start_index = 0):

    idx2data = {index+start_index:element  for index,element in enumerate(set(list_elements))}

    data2idx = {element:index for index,element in idx2data.items()}

    indexes = [data2idx[element] for element in list_elements]

    return indexes, data2idx, idx2data


class Discretize:
    def fit_transform(self):
        pass

    def transform(self):
        pass

    def updateIndexes(self, indexes, star_index):
        map_indexes = {value:counter+star_index for counter, value in enumerate(set(indexes))}


        new_indexes = [map_indexes[index] for index in indexes]

        idx2data = {map_indexes[index]:self.idx_data[index] for index in set(indexes)}
        data2idx = {val:key for (key, val) in idx2data.items()}

        self.idx_data = idx2data
        self.data_idx = data2idx

        return new_indexes
        
        
        
class Round(Discretize):
    def __init__(self, div= 10):
        self.div= div

    def fit_transform(self, latitudes, longitudes, start_index = 0, vocab_size = -1, vocab_min_count=0):
        
        lat = list(zip((latitudes).astype(int), ((latitudes * 1000).astype(int) - (latitudes).astype(int) * 1000)//self.div))
        
        lon = list(zip((longitudes).astype(int), ((longitudes * 1000).astype(int) - (longitudes).astype(int) * 1000)//self.div))

        discretizations = list(zip(lat, lon))

        counter = Counter(discretizations)
        if (vocab_size > 0):
            pairs = counter.most_common(vocab_size)
        else:
            pairs = list(counter.items())

        self.vocab = [keyword for keyword, count in pairs if count >= vocab_min_count]
        self.indexes, self.data_idx, self.idx_data = buildIndexData(self.vocab, start_index)

        return [self.data_idx.get(data, None) for data in discretizations], self.data_idx, self.idx_data



    def transform(self, latitudes, longitudes):
        lat = list(zip((latitudes).astype(int), ((latitudes * 1000).astype(int) - (latitudes).astype(int) * 1000)//self.div))
        lon = list(zip((longitudes).astype(int), ((longitudes * 1000).astype(int) - (longitudes).astype(int) * 1000)//self.div))
        discretizations = list(zip(lat, lon))
        return discretizations

class HourofDay_DayofWeak_Range(Discretize):
    def __init__(self, div= 1):
        self.div= div
        
  
    def transform(self, created_at):
        indi = pd.DatetimeIndex(created_at)
        if(self.div == 0):
            return list(indi.hour)
        else:
            return list(((indi.weekday*24)+indi.hour)//self.div)
               
        return discretizations       

      
        
def disc_time_to_sec_day(ts):
    return [(ts/10 ** 9)%(3600*24)]
    
class TMeanshiftClus(Discretize):
    def __init__(self, bandwidth):
		
        self.ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=10)
    
    def fit_transform(self, created_at):
        indi = pd.DatetimeIndex(created_at)
        lts = indi.values.astype(np.int64)
        
        dates = [disc_time_to_sec_day(ts) for ts in lts]
        
        
        dates = np.array(dates)
        self.ms.fit(dates)
        
        return ["temp_"+str(centroid) for centroid in  list(self.ms.labels_)]
		
    def transform(self, created_at):
        indi = pd.DatetimeIndex(created_at)
        lts = indi.values.astype(np.int64)
        
        dates = [disc_time_to_sec_day(ts) for ts in   lts]
        
        return ["temp_"+str(centroid) for centroid in  list(self.ms.predict(dates))]


class LMeanshiftClus():
    def __init__(self, bandwidth):
		
        self.ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=10)
    
    def fit_transform(self, X):
        X = np.array(X)
        self.ms.fit(X)
        print(len(list(self.ms.labels_)))
        return ["coord_"+str(centroid) for centroid in  list(self.ms.labels_)]
		
    def transform(self, x):
        return ["coord_"+str(centroid) for centroid in  list(self.ms.predict(x))]


class TMeanshiftClus(Discretize):
    def __init__(self, bandwidth):
		
        self.ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, n_jobs=10)
    
    def fit_transform(self, X):
        X = np.array(X)
        self.ms.fit(X)
        print(len(list(self.ms.labels_)))
        return ["coord_"+str(centroid) for centroid in  list(self.ms.labels_)]
		
    def transform(self, x):
        return ["coord_"+str(centroid) for centroid in  list(self.ms.predict(x))]
 
  

 








        
class Indexer():
    def __init__(self):
        pass

    def fit_transform(self,
            df,
            dates_vocab_size = 0, dates_vocab_mincount = 0,
            places_vocab_size = 0, places_vocab_mincount = 0,
            words_vocab_size = 0, words_vocab_mincount = 0,
            time_discretizer = 1,
            geo_granularity=100): 

        self.time_discretizer = HourofDay_DayofWeak_Range(time_discretizer)
        self.coor_discretizer = Round(geo_granularity)
       
        #file_csv has columns created_at, latitude, longitude, text
        dates = self.time_discretizer.transform(df['created_at'])
        places = self.coor_discretizer.transform(df['latitude'], df['longitude'])
        
        texts = df['texts'].astype(str)
        words = [word for list_words in texts for word in list_words.split()]
        #print(len(texts))


        counter_dates = Counter(dates)
        if (dates_vocab_size > 0):
            pairs = counter_dates.most_common(dates_vocab_size)
        else:
            pairs = list(counter_dates.items())
        self.vocab_dates = set([date for date, count in pairs if count >= dates_vocab_mincount])


        counter_places = Counter(places)
        if (places_vocab_size > 0):
            pairs = counter_places.most_common(places_vocab_size)
        else:
            pairs = list(counter_places.items())
        self.vocab_places = set([place for place, count in pairs if count >= places_vocab_mincount])


        counter_words = Counter(words)
        if (words_vocab_size > 0):
            pairs = counter_words.most_common(words_vocab_size)
        else:
            pairs = list(counter_words.items())

        self.vocab_words = set([keyword for keyword, count in pairs if count >= words_vocab_mincount])

        filtered_dates = set([i for i in range(len(dates)) if dates[i] in self.vocab_dates ])
        filtered_places = set([i for i in range(len(places)) if places[i] in self.vocab_places])
        filtered_words = set([i for i in range(len(texts)) if any([word in self.vocab_words for word in texts[i].split()]) ])

        #print(len(filtered_dates))
        #print(len(filtered_places))
        #print(len(filtered_words))

        filtered = list(filtered_dates.intersection(filtered_places).intersection(filtered_words))
        #print(len(filtered))

        dates = [dates[i] for i in filtered]
        places = [places[i] for i in filtered]
        texts = [texts[i] for i in filtered]


        idxsdates, self.date2idx, self.idx2date = buildIndexData(dates, start_index=0)
        idxsplaces, self.place2idx, self.idx2place = buildIndexData(places, start_index=max(idxsdates) + 1)

        
        
        
        start_index_words = max(idxsplaces) + 1
        idxs, self.word2idx, self.idx2word = buildIndexData(self.vocab_words, start_index = start_index_words+3)

        self.word2idx["<PAD>"] = start_index_words
        self.word2idx["<START>"] = start_index_words+2
        self.word2idx["<UNK>"] = start_index_words+1 
        
        self.idx2word[start_index_words] = "<PAD>"
        self.idx2word[start_index_words+2] = "<START>"
        self.idx2word[start_index_words+1] = "<UNK>"
        
        
        
        idxstexts = []

        for text in texts:
            indexed_text = [self.word2idx["<START>"]] + [self.word2idx[word] for word in text.split() if word in self.vocab_words]
            idxstexts.append(indexed_text)

        self.idx2item = {}
        self.idx2item.update(self.idx2word)
        self.idx2item.update(self.idx2place)
        self.idx2item.update(self.idx2date)

        self.item2idx = {}
        self.item2idx.update(self.word2idx)
        self.item2idx.update(self.place2idx)
        self.item2idx.update(self.date2idx)
        
        full_list = list(zip(idxsdates, idxsplaces, idxstexts))
        clean_list = [[x[0]]+ [x[1]]+ x[2] for x in full_list]
        return clean_list,self.item2idx,self.idx2item,max(idxsdates),max(idxsplaces)


    def transform(self, df):
        dates = self.time_discretizer.transform(df['created_at'])
        idxsdates = [self.date2idx.get(date, None) for date in dates]
        
        places = self.coor_discretizer.transform(df['latitude'], df['longitude'])
        idxsplaces = [self.place2idx.get(place, None) for place in places]
        
        idxstexts = []
        for text in df['texts'].astype(str):
            indexed_text = [self.word2idx["<START>"]] + [self.word2idx[word] for word in text.split() if word in self.vocab_words]
            idxstexts.append(indexed_text)

        full_list = list(zip(idxsdates,
                             idxsplaces,
                             idxstexts))

        clean_list = [[x[0]]+ [x[1]]+ x[2] for x in full_list if ((x[0] != None) and (x[1] != None) and (len(x[2]) > 1)) ]
        return clean_list

    
        
    def Item2idx(self, item):
        return self.item2idx.get(item, -1)

    def Idx2item(self, index):
        return self.idx2item.get(index, None)


def chargeData_SplitTrainTest(datacsv, 
                              dates_vocab_mincount=0, 
                              places_vocab_mincount=10, 
                              words_vocab_mincount=100, 
                              time_discretizer = 1, 
                              geo_granularity=100):
    df = pd.read_csv(data+datacsv)
    columns = list(df.columns.values)
    print('Done load data')
    
    if not ('created_at' in columns):
        print("Need a column name created_at with timestamps")
        return
                      
    if not ('latitude' in columns and 'longitude' in columns):
        print("Need a columns names latitude and longitude")
        return
                      
    if not ('texts' in columns):
    
        print("Need a column name texts with texts")
        return
                      
                      
    length = len(df)                   
                      
    train_range =  np.r_[0:int(0.6*length)]
    val_range =  np.r_[int(0.6*length):int(0.8*length)]
    test_range =  np.r_[int(0.8*length):length]
                      
    train = df.loc[train_range, :]
    val = df.loc[val_range, :]
    test = df.loc[test_range, :]
 
    indexer = Indexer()
    print('Start indexing ')
              
    train, item2idx, idx2item, max_idxsdates, max_idxsplaces = indexer.fit_transform(train, dates_vocab_mincount=dates_vocab_mincount, words_vocab_mincount=words_vocab_mincount, places_vocab_mincount=places_vocab_mincount,time_discretizer = time_discretizer, geo_granularity=geo_granularity)
    print('Done indexing train')
    
    val  = indexer.transform(val)
    print('Done indexing val')
    
    
    test = indexer.transform(test)
    print('Done indexing test')
    
    
    return (train, val, test, item2idx, idx2item, max_idxsdates, max_idxsplaces, indexer)






class Indexer_MS():
    def __init__(self):
        pass

    def fit_transform(self,
            df,
            dates_vocab_size = 0, dates_vocab_mincount = 0,
            places_vocab_size = 0, places_vocab_mincount = 0,
            words_vocab_size = 0, words_vocab_mincount = 0,
            time_bandwidth = 1000,
            geo_bandwidth = 0.002): 

        self.time_discretizer = TMeanshiftClus(time_bandwidth)
        self.coor_discretizer = LMeanshiftClus(geo_bandwidth)
       
        #file_csv has columns created_at, latitude, longitude, text
        indi = pd.DatetimeIndex(df['created_at'])
        df['ts'] = indi.values.astype(np.int64)
        
        dates = [[(ts/10 ** 9)%(3600*24)] for ts in   df['ts']]
        
        dates = self.time_discretizer.fit_transform(dates)
        print("Done MS ts")
        
        
        
        places = list(zip(df['latitude'],df['longitude']))
        places = [[lat,lon] for lat,lon in places]
        places = self.coor_discretizer.fit_transform(places)
        print("Done MS ps")
        texts = list(df['texts'].astype(str))
        words = [word for list_words in texts for word in list_words.split()]
        #print(len(texts))

        counter_dates = Counter(dates)
        if (dates_vocab_size > 0):
            pairs = counter_dates.most_common(dates_vocab_size)
        else:
            pairs = list(counter_dates.items())
        self.vocab_dates = set([date for date, count in pairs if count >= dates_vocab_mincount])


        counter_places = Counter(places)
        if (places_vocab_size > 0):
            pairs = counter_places.most_common(places_vocab_size)
        else:
            pairs = list(counter_places.items())
        self.vocab_places = set([place for place, count in pairs if count >= places_vocab_mincount])


        counter_words = Counter(words)
        if (words_vocab_size > 0):
            pairs = counter_words.most_common(words_vocab_size)
        else:
            pairs = list(counter_words.items())

        self.vocab_words = set([keyword for keyword, count in pairs if count >= words_vocab_mincount])

        filtered_dates = set([i for i in range(len(dates)) if dates[i] in self.vocab_dates ])
        filtered_places = set([i for i in range(len(places)) if places[i] in self.vocab_places])
        filtered_words = set([i for i in range(len(texts)) if any([word in self.vocab_words for word in texts[i].split()]) ])

        filtered = list(filtered_dates.intersection(filtered_places).intersection(filtered_words))

        dates = [dates[i] for i in filtered]
        places = [places[i] for i in filtered]
        texts = [texts[i] for i in filtered]


        idxsdates, self.date2idx, self.idx2date = buildIndexData(dates, start_index=0)
        idxsplaces, self.place2idx, self.idx2place = buildIndexData(places, start_index=max(idxsdates) + 1)

        
        
        
        start_index_words = max(idxsplaces) + 1
        idxs, self.word2idx, self.idx2word = buildIndexData(self.vocab_words, start_index = start_index_words+3)

        self.word2idx["<PAD>"] = start_index_words
        self.word2idx["<START>"] = start_index_words+2
        self.word2idx["<UNK>"] = start_index_words+1 
        
        self.idx2word[start_index_words] = "<PAD>"
        self.idx2word[start_index_words+2] = "<START>"
        self.idx2word[start_index_words+1] = "<UNK>"
        
        
        
        idxstexts = []

        for text in texts:
            indexed_text = [self.word2idx["<START>"]] + [self.word2idx[word] for word in text.split() if word in self.vocab_words]
            idxstexts.append(indexed_text)

        self.idx2item = {}
        self.idx2item.update(self.idx2word)
        self.idx2item.update(self.idx2place)
        self.idx2item.update(self.idx2date)

        self.item2idx = {}
        self.item2idx.update(self.word2idx)
        self.item2idx.update(self.place2idx)
        self.item2idx.update(self.date2idx)
        
        full_list = list(zip(idxsdates, idxsplaces, idxstexts))
        clean_list = [[x[0]]+ [x[1]]+ x[2] for x in full_list]
        return clean_list,self.item2idx,self.idx2item,max(idxsdates),max(idxsplaces)


    def transform(self, df):
        indi = pd.DatetimeIndex(df['created_at'])
        df['ts'] = indi.values.astype(np.int64)
        dates = [[(ts/10 ** 9)%(3600*24)] for ts in   df['ts']]
        dates = self.time_discretizer.transform(dates)
        idxsdates = [self.date2idx.get(date, None) for date in dates]
        
        
        places = list(zip(df['latitude'],df['longitude']))
        places = [[lat,lon] for lat,lon in places]
        places = self.coor_discretizer.transform(places)
        idxsplaces = [self.place2idx.get(place, None) for place in places]
        
        idxstexts = []
        for text in df['texts'].astype(str):
            indexed_text = [self.word2idx["<START>"]] + [self.word2idx[word] for word in text.split() if word in self.vocab_words]
            idxstexts.append(indexed_text)

        full_list = list(zip(idxsdates,
                             idxsplaces,
                             idxstexts))

        clean_list = [[x[0]]+ [x[1]]+ x[2] for x in full_list if ((x[0] != None) and (x[1] != None) and (len(x[2]) > 1)) ]
        return clean_list
    
    def transform_text(self, texts):
        
        
        idxstexts = []
        for text in texts:
            indexed_text = [self.word2idx["<START>"]] + [self.word2idx[word] for word in text.split() if word in self.vocab_words]
            idxstexts.append(indexed_text)

        
        return idxstexts
        
    def transform_time(self, times):
        
        indi = pd.DatetimeIndex(times)
        lts = indi.values.astype(np.int64)
        dates = [[(ts/10 ** 9)%(3600*24)] for ts in   lts]
        dates = self.time_discretizer.transform(dates)
        idxsdates = [self.date2idx.get(date, None) for date in dates]

        
        return idxsdates
    
    def transform_place(self, latitudes, longitudes):
        
        places = list(zip(latitudes,longitudes))
        
        places = [[lat,lon] for lat,lon in places]
        places = self.coor_discretizer.transform(places)
        idxsplaces = [self.place2idx.get(place, None) for place in places]

        
        return idxsplaces
        
    def Item2idx(self, item):
        return self.item2idx.get(item, -1)

    def Idx2item(self, index):
        return self.idx2item.get(index, None)


def chargeData_SplitTrainTest_MS(datacsv, 
                              dates_vocab_mincount= 0, 
                              places_vocab_mincount= 0, 
                              words_vocab_mincount= 100, 
                              time_bandwidth= 1000, 
                              geo_bandwidth= 0.002):
    df = pd.read_csv(data+datacsv)
    columns = list(df.columns.values)
    print('Done load data')
    
    if not ('created_at' in columns):
        print("Need a column name created_at with timestamps")
        return
                      
    if not ('latitude' in columns and 'longitude' in columns):
        print("Need a columns names latitude and longitude")
        return
                      
    if not ('texts' in columns):
    
        print("Need a column name texts with texts")
        return
                      
                      
    length = len(df)                   
                      
    train_range =  np.r_[0:int(0.6*length)]
    val_range =  np.r_[int(0.6*length):int(0.8*length)]
    test_range =  np.r_[int(0.8*length):length]
    
    df = df.sample(frac=1).reset_index(drop=True)
                  
    train = df.loc[train_range, :]
    val = df.loc[val_range, :]
    test = df.loc[test_range, :]
 
    indexer = Indexer_MS()
    print('Start indexing ')
              
    train, item2idx, idx2item, max_idxsdates, max_idxsplaces = indexer.fit_transform(train, dates_vocab_mincount=dates_vocab_mincount, words_vocab_mincount=words_vocab_mincount, places_vocab_mincount=places_vocab_mincount,time_bandwidth = time_bandwidth, geo_bandwidth=geo_bandwidth)
    print('Done indexing train')
    
    val  = indexer.transform(val)
    print('Done indexing val')
    
    
    test = indexer.transform(test)
    print('Done indexing test')
    
    
    return (train, val, test, item2idx, idx2item, max_idxsdates, max_idxsplaces, indexer)
