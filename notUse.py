import numpy as np

with open('/home/ubuntu/junbeom/repo/PT4AL/loss_all_batch/batch_typicality.txt', 'r') as f: 
    atypicality = f.readlines()

for i in range(10):
    batch = atypicality[i*5000:(i+1)*5000]
    for b in batch:
        with open(f'/home/ubuntu/junbeom/repo/PT4AL/typicality_batch/batch_{i}.txt', 'a') as f: 
            f.write(b)

# a = np.load("/home/ubuntu/junbeom/repo/PT4AL/checkpoint_b50_base_Atypicality/lSet.npy", allow_pickle=True)
# print(a)

# a = [1, 2, 3, 4, 5]
# a = np.array(a)
# b = np.arange(3)
# b = np.array(b)
# a = np.delete(a, b)
# print(a, b)