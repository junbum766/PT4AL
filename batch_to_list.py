import numpy as np

total_num_list = []

for i in range(5):
    num_list = []
    with open(f'./loss/batch_{i}.txt', 'r') as f:
        while True:
            data = f.readline()
            if not data :
                break
            num = int(data.split('/')[9].split('.')[0])
            num_list.append(num)
        if i == 0 :
            print('i=0')
            num_list = [num_list[i*5] for i in range(1000)]
        else :
            num_list = [num_list[i] for i in range(1000)]

    f.close()
    print(i, num_list)

    total_num_list.extend(num_list)
        
        
np.save('./loss/lSet_5000.npy', total_num_list)

load_data = np.load('./loss/lSet_5000.npy', allow_pickle=True)
print(len(load_data))