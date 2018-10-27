import pandas as pd
import numpy as np
from collections import Counter
import bisect

data = ""


def buildIndexData(list_elements, start_index = 0):

    idx2data = {index + start_index: discretization for index, discretization in enumerate(set(list_elements))}

    data2idx = {discretization: index for index, discretization in idx2data.items()}

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
        lat = (latitudes * self.div).astype(int)
        lon = (longitudes * self.div).astype(int)

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
        lat = (latitudes * self.div).astype(int)
        lon = (longitudes * self.div).astype(int)

        discretizations = list(zip(lat, lon))
        #indexes = [self.data_idx.get(data, None) for data in discretizations]
        #return indexes
        return discretizations
        
        
        
class HourofDay(Discretize):
    def fit_transform(self, created_at, start_index = 0):
        indi = pd.DatetimeIndex(created_at)
        discretizations = list(indi.hour)
        self.indexes, self.data_idx, self.idx_data = buildIndexData(discretizations, start_index)
        return self.indexes, self.data_idx, self.idx_data


    def transform(self, created_at):
        indi = pd.DatetimeIndex(created_at)
        discretizations = list(indi.hour)
        #indexes = [self.data_idx.get(data,None) for data in discretizations]
        #return indexes
        return discretizations

class DayofWeak(Discretize):
    def fit_transform(self, created_at, start_index = 0):
        indi = pd.DatetimeIndex(created_at)

        discretizations = list(indi.weekday)
        self.indexes, self.data_idx, self.idx_data = buildIndexData(discretizations, start_index)
        return self.indexes, self.data_idx, self.idx_data

    def transform(self, created_at):
        indi = pd.DatetimeIndex(created_at)
        discretizations = list(indi.weekday)
        indexes = [self.data_idx.get(data,None) for data in discretizations]
        return indexes


class HourofDay_DayofWeak(Discretize):
    def fit_transform(self, created_at, start_index = 0):
        indi = pd.DatetimeIndex(created_at)

        discretizations = list(zip(indi.weekday, indi.hour))
        self.indexes, self.data_idx, self.idx_data = buildIndexData(discretizations, start_index)
        return self.indexes, self.data_idx, self.idx_data

    def transform(self, created_at):
        indi = pd.DatetimeIndex(created_at)

        discretizations = list(zip(indi.weekday, indi.hour))
        #indexes = [self.data_idx.get(data, None) for data in discretizations]
        #return indexes        
        return discretizations
        
        
        
class Indexer():
    def __init__(self):
        pass

    def fit_transform(self,
            filename,
            #time_discretizer = HourofDay,
            
                      
           
            #represent_text = RepresentText,
            dates_vocab_size = 0, dates_vocab_mincount = 0,
            places_vocab_size = 0, places_vocab_mincount = 0,
            words_vocab_size = 0, words_vocab_mincount = 0,
            time_discretizer = 1,
            geo_granularity=100): #file_csv has columns created_at, latitude, longitude, text

        #self.datapath = "Data" + sep
        if (filename.split('.')[1] == 'csv'):
            df = pd.read_csv(filename)
        elif (filename.split('.')[1] == 'p'):
            df = pd.read_pickle(filename)
        
        map_time_granularity = [HourofDay_DayofWeak, HourofDay,DayofWeak]
        
        self.time_discretizer = map_time_granularity[time_discretizer]()
        self.coor_discretizer = Round(geo_granularity)
        #self.represent_text = represent_text()

        #date_out, self.date_idx, self.idx_date = self.time_discretizer.fit_transform(df['created_at'], start_index = 0)
        dates = self.time_discretizer.transform(df['created_at'])

        #self.coor_out, self.coor_idx, self.idx_coor = self.coor_discretizer.fit_transform(df['latitude'], df['longitude'],start_index=max(self.date_out) + 1)

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

        idxs, self.word2idx, self.idx2word = buildIndexData(self.vocab_words, start_index = max(idxsplaces) + 1)

        idxstexts = []

        for text in texts:
            indexed_text = [self.word2idx[word] for word in text.split() if word in self.vocab_words]
            idxstexts.append(indexed_text)

        self.idx2item = {}
        self.idx2item.update(self.idx2word)
        self.idx2item.update(self.idx2place)
        self.idx2item.update(self.idx2date)

        self.item2idx = {}
        self.item2idx.update(self.word2idx)
        self.item2idx.update(self.place2idx)
        self.item2idx.update(self.date2idx)

        return list(zip(idxsdates,
                             idxsplaces,
                             idxstexts)),self.word2idx


    def transform(self, filename):
        if (filename.split('.')[1] == 'csv'):
            df = pd.read_csv( filename)
        elif (filename.split('.')[1] == 'p'):
            df = pd.read_pickle( filename)

        dates = self.time_discretizer.transform(df['created_at'])
        idxsdates = [self.date2idx.get(date, None) for date in dates]

        places = self.coor_discretizer.transform(df['latitude'], df['longitude'])
        idxsplaces = [self.place2idx.get(place, None) for place in places]

        idxstexts = []
        for text in df['texts'].astype(str):
            indexed_text = [self.word2idx[word] for word in text.split() if word in self.vocab_words]
            idxstexts.append(indexed_text)


        full_list = list(zip(idxsdates,
                             idxsplaces,
                             idxstexts))
        #print(len(full_list))
        clean_list = [(x[0], x[1], x[2]) for x in full_list if ((x[0] != None) and (x[1] != None) and (x[2] != [])) ]
        return clean_list


    def Item2index(self, item):
        return self.item2idx.get(item, -1)

    def Index2item(self, index):
        return self.idx2item.get(index, None)

    #def indexes(self):
    #    return (self.coor_out[0], self.coor_out[-1], self.texts_out[-1])
        

def chargeDataTrainTest(train, test, dates_vocab_mincount=0, words_vocab_mincount=100, places_vocab_mincount=10,time_discretizer = 1, geo_granularity=100):
    indexer = Indexer(time_discretizer = 1, geo_granularity=100)
    train,word2idx = indexer.fit_transform(data+train, dates_vocab_mincount=dates_vocab_mincount, words_vocab_mincount=words_vocab_mincount, places_vocab_mincount=places_vocab_mincount)
    test = indexer.transform(data+test)
    return train,word2idx,test
    


def chargeData_SplitTrainTest(datacsv, dates_vocab_mincount=0, words_vocab_mincount=100, places_vocab_mincount=10, time_discretizer = 1, geo_granularity=100):
    indexer = Indexer()
    df = pd.read_csv(data+datacsv)
    
    test_size = int(0.2*len(df))

    train, test = np.split(df.sample(frac=1), [len(df)-test_size])
    train.to_csv(data+'train.csv')
    test.to_csv(data+'test.csv')
    train,word2idx = indexer.fit_transform(data+'train.csv', dates_vocab_mincount=dates_vocab_mincount, words_vocab_mincount=words_vocab_mincount, places_vocab_mincount=places_vocab_mincount,time_discretizer = time_discretizer, geo_granularity=geo_granularity)
    test = indexer.transform(data+'test.csv')
    return train,word2idx,test

class QuantitativeEvaluator:
    def __init__(self, fake_num=10):
        self.ranks = []

        self.fake_num = fake_num


    def get_ranks(self, test, predictor, predict_type = 'w'):
        self.predict_type = predict_type
        noiseList = np.random.choice(len(test), self.fake_num*len(test)).tolist()
        count = 5
        for example in test:



            scores = []
            score = predictor.predict(example[0], example[1], example[2])
            scores.append(score)


            for i in range(self.fake_num):
                noise = test[noiseList.pop()]
                if self.predict_type == 't':
                    noise_score = predictor.predict(noise[0], example[1], example[2])
                elif self.predict_type=='l':
                    noise_score = predictor.predict(example[0], noise[1], example[2])
                elif self.predict_type=='w':
                    noise_score = predictor.predict(example[0], example[1], noise[2])
                scores.append(noise_score)
            scores.sort()


            # handle ties
            rank = len(scores)+1-(bisect.bisect_left(scores,score)+bisect.bisect_right(scores,score)+1)/2.0
            self.ranks.append(rank)


    def get_ranks_old(self, np_xytest, predictor, predict_type = 'w'):
        self.predict_type = predict_type
        noiseList = np.random.choice(np_xytest.shape[0], self.fake_num*np_xytest.shape[0]).tolist()

        for i in range(np_xytest.shape[0]):
            example = list(np_xytest[i])
            scores = []
            score = predictor.predict(example[0], example[1], example[2])
            scores.append(score)

            for i in range(self.fake_num):
                noise = np_xytest[noiseList.pop()]
                if self.predict_type == 'l':
                    noise_score = predictor.predict(noise[0], example[1], example[2])
                elif self.predict_type=='t':
                    noise_score = predictor.predict(example[0], noise[1], example[2])
                elif self.predict_type=='w':
                    noise_score = predictor.predict(example[0], example[1], noise[2])
                scores.append(noise_score)
            scores.sort()
            # handle ties
            rank = len(scores)+1-(bisect.bisect_left(scores,score)+bisect.bisect_right(scores,score)+1)/2.0
            self.ranks.append(rank)

    def compute_mrr(self):
        r = self.ranks
        reciprocal_ranks = [1/rank for rank in r]
        mrr = sum(reciprocal_ranks)/len(reciprocal_ranks)
        mr = sum(r)/len(r)
        return round(mrr,4), round(mr,4)

